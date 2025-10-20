import os
import sys
import logging
import random
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

# 导入项目模块
from configs.hdnet_config import config
from models.HDNet import HDNet
from utils4train.data_process import create_dataloaders, create_vis_dataloader
from utils4train.metrics import SegmentationMetrics
from utils4train.losses import hdnet_loss
from utils4train.checkpoint import save_checkpoint
from utils4train.visualization import create_hdnet_sample_images
from utils4train.trainer import test  # 假设 test 函数在 trainer.py 中
from utils4train.alerts_by_lark import send_message


def set_seed(seed: int) -> None:
    """设置随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, optimizer, device, metrics, epoch, scaler):
    model.train()
    metrics.reset()
    loss_meter = {"total": 0.0, "seg": 0.0, "bd": 0.0}

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            total_loss_batch, seg_loss_batch, bd_loss_batch = hdnet_loss(
                outputs, labels
            )

        scaler.scale(total_loss_batch).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter["total"] += total_loss_batch.item()
        loss_meter["seg"] += seg_loss_batch.item()
        loss_meter["bd"] += bd_loss_batch.item()

        x_seg = outputs[0]
        preds = torch.argmax(x_seg, dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        metrics.update(preds, targets)

        pbar.set_postfix(
            {
                "Total": f"{total_loss_batch.item():.4f}",
                "Seg": f"{seg_loss_batch.item():.4f}",
                "BD": f"{bd_loss_batch.item():.4f}",
                "IoU": f"{metrics.get_metrics()['iou']:.4f}",
            }
        )

    avg_losses = {k: v / len(train_loader) for k, v in loss_meter.items()}
    train_metrics_results = metrics.get_metrics()
    return avg_losses, train_metrics_results


def validate(model, val_loader, device, metrics, epoch):
    model.eval()
    metrics.reset()
    loss_meter = {"total": 0.0, "seg": 0.0, "bd": 0.0}

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast():
                outputs = model(images)
                total_loss_batch, seg_loss_batch, bd_loss_batch = hdnet_loss(
                    outputs, labels
                )

            loss_meter["total"] += total_loss_batch.item()
            loss_meter["seg"] += seg_loss_batch.item()
            loss_meter["bd"] += bd_loss_batch.item()

            x_seg = outputs[0]
            preds = torch.argmax(x_seg, dim=1).cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(preds, targets)

            pbar.set_postfix(
                {
                    "Total": f"{total_loss_batch.item():.4f}",
                    "IoU": f"{metrics.get_metrics()['iou']:.4f}",
                }
            )

    avg_losses = {k: v / len(val_loader) for k, v in loss_meter.items()}
    val_metrics_results = metrics.get_metrics()
    return avg_losses, val_metrics_results


def main():
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = f"{config['model_name']}_{datetime.now().strftime('%m%d')}"
    wandb.init(
        project=config["project_name"],
        name=experiment_name,
        config=config,
        notes=config["description"],
        tags=config["tags"],
    )

    try:
        # 数据加载
        train_loader, val_loader, test_loader = create_dataloaders(
            root_dir=config["data_root"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            image_size=config["image_size"],
            augment=True,
            use_nir=config["use_nir"],
        )
        vis_loader = create_vis_dataloader(
            root_dir=config["data_root"],
            image_size=config["image_size"],
            num_workers=config["num_workers"],
            use_nir=config["use_nir"],
        )
        if vis_loader is None:
            vis_loader = val_loader

        # 模型初始化
        model = HDNet(
            base_channel=config["base_channel"], num_classes=config["num_classes"]
        ).to(device)
        wandb.watch(model, log="all", log_freq=100)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update(
            {
                "model_total_params": total_params,
                "model_trainable_params": trainable_params,
            }
        )

        # 优化器和学习率调度器 (升级版)
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
        scaler = GradScaler()

        # 初始化指标计算类
        train_metrics = SegmentationMetrics(config["num_classes"])
        val_metrics = SegmentationMetrics(config["num_classes"])
        test_metrics = SegmentationMetrics(config["num_classes"])

        # 创建检查点目录和日志
        checkpoint_dir = os.path.join("./checkpoints", experiment_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(checkpoint_dir, "train_log.txt")),
                logging.StreamHandler(),
            ],
        )

        logging.info(f"实验开始: {experiment_name}")
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

        # --- 训练循环 ---
        best_iou, report_iou = 0.0, 0.0
        best_epoch = -1
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
        start_time = datetime.now()

        for epoch in range(config["num_epochs"]):
            train_losses, train_result = train_one_epoch(
                model, train_loader, optimizer, device, train_metrics, epoch + 1, scaler
            )
            val_losses, val_result = validate(
                model, val_loader, device, val_metrics, epoch + 1
            )
            scheduler.step()

            wandb.log(
                {
                    "Comparison Board/IoU": val_result["iou"],
                    "Comparison Board/F1": val_result["f1"],
                    "Comparison Board/Precision": val_result["precision"],
                    "Comparison Board/Recall": val_result["recall"],
                    "Train_info/Loss/Train": train_losses["total"],
                    "Train_info/Loss/Val": val_losses["total"],
                    "Train_info/Seg_Loss/Train": train_losses["seg"],
                    "Train_info/Seg_Loss/Val": val_losses["seg"],
                    "Train_info/Boundary_Loss/Train": train_losses["bd"],
                    "Train_info/Boundary_Loss/Val": val_losses["bd"],
                    "Train_info/IoU/Train": train_result["iou"],
                    "Train_info/IoU/Val": val_result["iou"],
                    "Train_info/F1/Train": train_result["f1"],
                    "Train_info/F1/Val": val_result["f1"],
                    "Train_info/Learning_Rate": optimizer.param_groups[0]["lr"],
                }
            )
            logging.info(
                f"Epoch {epoch+1} | Train Loss: {train_losses['total']:.4f}, Train IoU: {train_result['iou']:.4f} | Val Loss: {val_losses['total']:.4f}, Val IoU: {val_result['iou']:.4f}"
            )

            if (epoch + 1) % 10 == 0:
                sample_figure = create_hdnet_sample_images(
                    model, vis_loader, device, epoch + 1, num_samples=4
                )
                wandb.log({"Prediction Summary": wandb.Image(sample_figure)})
                plt.close(sample_figure)

            if val_result["iou"] > best_iou:
                if val_result["iou"] > 0.73 and val_result["iou"] - report_iou > 0.01:
                    report_iou = val_result["iou"]
                    elapsed = (datetime.now() - start_time).total_seconds()
                    eta_seconds = (elapsed / (epoch + 1)) * (
                        config["num_epochs"] - (epoch + 1)
                    )
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
                logging.info(
                    f"New best model saved at epoch {best_epoch} with IoU: {best_iou:.4f}"
                )

        wandb.summary["best_iou"] = best_iou
        wandb.summary["best_epoch"] = best_epoch

        # --- 测试阶段 ---
        logging.info("开始测试最佳模型...")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])

            # HDNet的test函数在trainer.py中没有区分，我们复用validate
            _, test_result = validate(
                model, test_loader, device, test_metrics, best_epoch
            )

            wandb.log(
                {
                    "Result Board/IoU": test_result["iou"],
                    "Result Board/F1": test_result["f1"],
                    "Result Board/Precision": test_result["precision"],
                    "Result Board/Recall": test_result["recall"],
                }
            )
            logging.info(
                f"测试结果 - IoU: {test_result['iou']:.4f}, F1: {test_result['f1']:.4f}"
            )

            end_message = (
                f"训练完成!\n"
                f"最佳 Val IoU: {best_iou:.4f} (at epoch {best_epoch})\n"
                f"测试集 IoU: {test_result['iou']:.4f}"
            )
        else:
            logging.warning("未找到最佳模型文件，跳过测试。")
            end_message = f"训练完成! 最佳 Val IoU: {best_iou:.4f} (at epoch {best_epoch}). 未找到模型文件进行测试。"

        send_message(title=f"实验结束: {experiment_name}", content=end_message)

    except Exception as e:
        logging.error(f"实验发生错误: {e}", exc_info=True)
        send_message(title=f"实验失败: {experiment_name}", content=f"错误信息: \n{e}")
        raise

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
