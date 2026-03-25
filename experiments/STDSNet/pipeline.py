"""
STDSNet 训练主流程

包含完整的训练循环、验证、日志记录、模型保存等
"""

import os
import sys
import random
import logging
from datetime import datetime, timedelta

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb

from config import config
from model import STDSNet
from dataprocess import get_loaders
from loss_function import stdsnet_loss
from metrics import SegmentationMetrics
from checkpoint import save_checkpoint
from visualization import create_sample_images
from message2lark import send_message


def set_seed(seed: int) -> None:
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, optimizer, device, metrics, epoch, scaler):
    """训练一个 epoch"""
    model.train()
    metrics.reset()
    loss_meter = {"total": 0.0, "seg": 0.0, "shape": 0.0}

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images)
            total_loss_batch, seg_loss_batch, shape_loss_batch = stdsnet_loss(
                outputs, labels,
                shape_weight=config.get("shape_loss_weight", 0.4),
                dice_ratio=config.get("dice_ratio", 0.5),
            )

        scaler.scale(total_loss_batch).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter["total"] += total_loss_batch.item()
        loss_meter["seg"] += seg_loss_batch.item()
        loss_meter["shape"] += shape_loss_batch.item()

        global_pred = outputs[0]
        preds = torch.argmax(global_pred, dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        metrics.update(preds, targets)

        pbar.set_postfix({
            "Total": f"{total_loss_batch.item():.4f}",
            "Seg": f"{seg_loss_batch.item():.4f}",
            "Shape": f"{shape_loss_batch.item():.4f}",
            "IoU": f"{metrics.get_metrics()['iou']:.4f}",
        })

    avg_losses = {k: v / len(train_loader) for k, v in loss_meter.items()}
    train_metrics_results = metrics.get_metrics()
    return avg_losses, train_metrics_results


def validate(model, val_loader, device, metrics, epoch):
    """验证"""
    model.eval()
    metrics.reset()
    loss_meter = {"total": 0.0, "seg": 0.0, "shape": 0.0}

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                total_loss_batch, seg_loss_batch, shape_loss_batch = stdsnet_loss(
                    outputs, labels,
                    shape_weight=config.get("shape_loss_weight", 0.4),
                    dice_ratio=config.get("dice_ratio", 0.5),
                )

            loss_meter["total"] += total_loss_batch.item()
            loss_meter["seg"] += seg_loss_batch.item()
            loss_meter["shape"] += shape_loss_batch.item()

            global_pred = outputs[0]
            preds = torch.argmax(global_pred, dim=1).cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(preds, targets)

            pbar.set_postfix({
                "Total": f"{total_loss_batch.item():.4f}",
                "IoU": f"{metrics.get_metrics()['iou']:.4f}",
            })

    avg_losses = {k: v / len(val_loader) for k, v in loss_meter.items()}
    val_metrics_results = metrics.get_metrics()
    return avg_losses, val_metrics_results


def main():
    """训练主流程"""
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = f"{config['model_name']}_{datetime.now().strftime('%m%d')}"
    wandb.init(
        project=config.get("project_name", "Building-Segmentation-3Bands"),
        name=experiment_name,
        config=config,
        notes=config.get("description", ""),
        tags=config.get("tags", []),
    )

    try:
        # 数据加载
        train_loader_strong, train_loader_mild, val_loader, test_loader, vis_loader = get_loaders(config)

        if vis_loader is None:
            vis_loader = val_loader

        # 模型初始化
        model = STDSNet(
            num_classes=config["num_classes"],
            image_size=config["image_size"],
            pretrained=config.get("encoder_pretrained", True),
        ).to(device)
        wandb.watch(model, log="all", log_freq=100)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update({
            "model_total_params": total_params,
            "model_trainable_params": trainable_params,
        })

        # 优化器和调度器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, total_iters=config["warmup_epochs"]
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config["num_epochs"] - config["warmup_epochs"],
            eta_min=config["min_lr"],
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config["warmup_epochs"]],
        )
        scaler = GradScaler(init_scale=2. ** 16, enabled=(device.type == 'cuda'))

        # 评估指标
        train_metrics = SegmentationMetrics(config["num_classes"])
        val_metrics = SegmentationMetrics(config["num_classes"])
        test_metrics = SegmentationMetrics(config["num_classes"])

        # 检查点和日志目录
        checkpoint_dir = os.path.join(config["save_dir"], experiment_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        local_log_path = os.path.join(checkpoint_dir, "train_log.txt")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(local_log_path, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )

        logging.info(f"实验开始: {experiment_name}")
        logging.info(f"模型: {config['model_name']}, 总参数量: {total_params:,}, 可训练参数量: {trainable_params:,}")

        start_time = datetime.now()
        send_message(
            title=f"实验开始: {experiment_name}",
            content=(
                f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M')}\n"
                f"模型: {config['model_name']}\n"
                f"总参数: {total_params:,}\n"
                f"可训练参数: {trainable_params:,}\n"
                f"训练轮数: {config['num_epochs']}\n"
                f"学习率: {config['learning_rate']}\n"
            ),
        )

        # 训练循环
        best_iou, report_iou = 0.0, 0.0
        best_epoch = -1
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

        for epoch in range(config["num_epochs"]):
            train_losses, train_result = train_one_epoch(
                model, train_loader_strong, optimizer, device, train_metrics, epoch + 1, scaler
            )
            val_losses, val_result = validate(
                model, val_loader, device, val_metrics, epoch + 1
            )
            scheduler.step()

            wandb.log({
                "Comparison Board/IoU": val_result["iou"],
                "Comparison Board/F1": val_result["f1"],
                "Train_info/Loss/Train": train_losses["total"],
                "Train_info/Loss/Val": val_losses["total"],
                "Train_info/IoU/Train": train_result["iou"],
                "Train_info/IoU/Val": val_result["iou"],
                "Train_info/Learning_Rate": optimizer.param_groups[0]["lr"],
            })

            logging.info(
                f"Epoch {epoch+1} | Train Loss: {train_losses['total']:.4f}, Train IoU: {train_result['iou']:.4f} | "
                f"Val Loss: {val_losses['total']:.4f}, Val IoU: {val_result['iou']:.4f}"
            )

            # 定期可视化
            if (epoch + 1) % 10 == 0:
                figure = create_sample_images(model, vis_loader, device, epoch + 1, num_samples=4)
                if figure:
                    wandb.log({"Prediction_Summary": wandb.Image(figure)})
                    import matplotlib.pyplot as plt
                    plt.close(figure)

            # 保存最佳模型
            if val_result["iou"] > best_iou:
                if val_result["iou"] > 0.73 and val_result["iou"] - report_iou > 0.01:
                    report_iou = val_result["iou"]
                    elapsed = (datetime.now() - start_time).total_seconds()
                    eta_seconds = (elapsed / (epoch + 1)) * (config["num_epochs"] - (epoch + 1))
                    eta_time = datetime.now() + timedelta(seconds=eta_seconds)
                    send_message(
                        title=f"{experiment_name}：模型最佳指标更新",
                        content=(
                            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                            f"Epoch: {best_epoch}\n"
                            f"Val IoU: {report_iou:.4f}\n"
                            f"Cur lr: {optimizer.param_groups[0]['lr']:.6g}\n"
                            f"预计结束时间: {eta_time.strftime('%Y-%m-%d %H:%M')}"
                        ),
                    )
                best_iou = val_result["iou"]
                best_epoch = epoch + 1
                save_checkpoint(model, optimizer, epoch, best_iou, best_model_path)
                logging.info(f"New best model saved at epoch {best_epoch} with IoU: {best_iou:.4f}")

        wandb.summary["best_iou"] = best_iou
        wandb.summary["best_epoch"] = best_epoch

        # 测试集评估
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            logging.info(f"Loaded best model from epoch {checkpoint['epoch']}")

            _, test_result = validate(model, test_loader, device, test_metrics, best_epoch)

            wandb.log({
                "Result Board/IoU": test_result["iou"],
                "Result Board/F1": test_result["f1"],
            })
            logging.info(f"Test results - IoU: {test_result['iou']:.4f}, F1: {test_result['f1']:.4f}")

            send_message(
                title=f"实验结束: {experiment_name}",
                content=(
                    f"训练完成!\n"
                    f"最佳 Val IoU: {best_iou:.4f} (at epoch {best_epoch})\n"
                    f"测试集 IoU: {test_result['iou']:.4f}"
                ),
            )

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        send_message(title=f"实验失败: {experiment_name}", content=f"错误信息: \n{e}")
        raise

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
