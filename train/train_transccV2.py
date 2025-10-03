import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import random
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb
from configs.transcc_v2_config import config
from utils4train.data_process import create_dataloaders, create_vis_dataloader
from utils4train.metrics import SegmentationMetrics
from models.TransCCV2 import create_transcc_v2
from utils4train.checkpoint import save_checkpoint
from utils4train.losses import transcc_v2_loss
from utils4train.trainer import test
from utils4train.visualization import create_sample_images
from utils4train.alerts_by_lark import send_message


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, optimizer, device, metrics, epoch):
    model.train()
    total_loss = 0.0
    seg_loss_total = 0.0
    boundary_loss_total = 0.0
    metrics.reset()

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss, seg_loss, boundary_loss = transcc_v2_loss(
            outputs, labels, seg_weight=1.0, boundary_weight=1.5, aux_weight=0.4
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        seg_loss_total += seg_loss.item()
        boundary_loss_total += boundary_loss.item()

        main_output = outputs[0]
        preds = torch.argmax(main_output, dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        metrics.update(preds, targets)

        pbar.set_postfix(
            {
                "Total": f"{loss.item():.4f}",
                "Seg": f"{seg_loss.item():.4f}",
                "Bd": f"{boundary_loss.item():.4f}",
                "Avg": f"{total_loss / (batch_idx + 1):.4f}",
            }
        )

    avg_total = total_loss / len(train_loader)
    avg_seg = seg_loss_total / len(train_loader)
    avg_bd = boundary_loss_total / len(train_loader)
    metric_values = metrics.get_metrics()

    return avg_total, avg_seg, avg_bd, metric_values


def validate(model, val_loader, device, metrics, epoch):
    model.eval()
    total_loss = 0.0
    seg_loss_total = 0.0
    boundary_loss_total = 0.0
    metrics.reset()

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss, seg_loss, boundary_loss = transcc_v2_loss(outputs, labels)

            total_loss += loss.item()
            seg_loss_total += seg_loss.item()
            boundary_loss_total += boundary_loss.item()

            main_output = outputs[0]
            preds = torch.argmax(main_output, dim=1).cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(preds, targets)

            pbar.set_postfix(
                {
                    "Total": f"{loss.item():.4f}",
                    "Seg": f"{seg_loss.item():.4f}",
                    "Bd": f"{boundary_loss.item():.4f}",
                    "Avg": f"{total_loss / (batch_idx + 1):.4f}",
                }
            )

    avg_total = total_loss / len(val_loader)
    avg_seg = seg_loss_total / len(val_loader)
    avg_bd = boundary_loss_total / len(val_loader)
    metric_values = metrics.get_metrics()

    return avg_total, avg_seg, avg_bd, metric_values


def main():
    # 设置随机种子
    set_seed(config["seed"])
    # 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # W&B实验看板初始化
    experiment_name = f"{config['model_name']}_{datetime.now().strftime('%m%d')}"
    wandb.init(
        project="Building-Segmentation-3Bands",
        name=experiment_name,
        config=config,
        notes="TransCCV2建筑物分割实验",
        tags=["TransCCV2", "building-segmentation", "RGB"],
    )

    try:
        # 创建数据加载器
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
            use_nir=config["use_nir"]
        )
        if vis_loader is None:
            vis_loader = val_loader

        # 模型初始化
        model = create_transcc_v2(
            {
                "img_size": config["image_size"],
                "patch_size": 16,
                "in_chans": config["input_channels"],
                "num_classes": config["num_classes"],
            }
        )
        model = model.to(device)
        wandb.watch(model, log="all", log_freq=100)

        # 加载ViT预训练权重
        from utils4train.weights import load_pretrained_weights
        pretrained_path = config.get("pretrained_weights", None)
        fusion_strategy = config.get("fusion_strategy", "interpolate")
        if pretrained_path:
            model = load_pretrained_weights(model, pretrained_path, fusion_strategy)

        # 计算并记录模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update(
            {
                "model_total_params": total_params,
                "model_trainable_params": trainable_params,
                "model_input_channels": config["input_channels"],
            }
        )

        # 优化器和学习率调度器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=config["warmup_epochs"],
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

        # 初始化评价指标
        train_metrics = SegmentationMetrics(config["num_classes"])
        val_metrics = SegmentationMetrics(config["num_classes"])
        test_metrics = SegmentationMetrics(config["num_classes"])

        # 创建检查点目录
        checkpoint_dir = os.path.join("./checkpoints", experiment_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 本地日志配置
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
        logging.info(
            f"模型: {config['model_name']}, 总参数量: {total_params:,}, 可训练参数量: {trainable_params:,}"
        )
        send_message(
            title=f"实验开始: {experiment_name}",
            content=f"模型: {config['model_name']}\n总参数量: {total_params:,}",
        )

        # 训练循环
        best_iou, report_iou = 0.0, 0.0
        best_epoch = -1
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

        for epoch in range(config["num_epochs"]):
            train_total, train_seg, train_bd, train_result = train_one_epoch(
                model, train_loader, optimizer, device, train_metrics, epoch + 1
            )

            val_total, val_seg, val_bd, val_result = validate(
                model, val_loader, device, val_metrics, epoch + 1
            )

            scheduler.step()

            # 使用 wandb.log 记录训练和验证指标
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/total_loss": train_total,
                    "train/seg_loss": train_seg,
                    "train/bd_loss": train_bd,
                    "train/iou": train_result["iou"],
                    "train/precision": train_result["precision"],
                    "train/recall": train_result["recall"],
                    "train/f1": train_result["f1"],
                    "val/total_loss": val_total,
                    "val/seg_loss": val_seg,
                    "val/bd_loss": val_bd,
                    "val/iou": val_result["iou"],
                    "val/precision": val_result["precision"],
                    "val/recall": val_result["recall"],
                    "val/f1": val_result["f1"],
                }
            )

            logging.info(
                f"epoch: {epoch+1}, "
                f"train_iou: {train_result['iou']:.4f}, val_iou: {val_result['iou']:.4f}, "
                f"train_loss: {train_total:.4f}, val_loss: {val_total:.4f}, "
                f"lr: {optimizer.param_groups[0]['lr']:.6g}"
            )

            # 每10个epoch保存一次可视化结果
            if (epoch + 1) % 10 == 0:
                figure = create_sample_images(model, vis_loader, device, epoch + 1, num_samples=len(vis_loader))
                if figure:
                    wandb.log({"Prediction_Summary": wandb.Image(figure)})
                    plt.close(figure)

            if val_result["iou"] > best_iou:
                if val_result["iou"] > 0.73 and val_result["iou"] - report_iou > 0.01:
                    report_iou = val_result["iou"]
                    send_message(
                        title=f"{experiment_name}：模型最佳指标更新",
                        content=(
                            f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            f"Epoch: {best_epoch}\n"
                            f"Val IoU: {report_iou:.4f}\n"
                            f"Cur lr: {optimizer.param_groups[0]['lr']:.6g}\n"
                        ),
                    )
                best_iou = val_result["iou"]
                best_epoch = epoch + 1
                save_checkpoint(model, optimizer, epoch, best_iou, best_model_path)
                logging.info(f"New best model saved at epoch {best_epoch} with IoU: {best_iou:.4f}")

            if (epoch + 1) % 10 == 0 and epoch > 100:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"
                )
                save_checkpoint(model, optimizer, epoch, best_iou, checkpoint_path)

        wandb.summary["best_iou"] = best_iou
        wandb.summary["best_epoch"] = best_epoch

        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

        test_result = test(model, test_loader, device, test_metrics)
        wandb.log(
            {
                "test/iou": test_result["iou"],
                "test/precision": test_result["precision"],
                "test/recall": test_result["recall"],
                "test/f1": test_result["f1"],
            }
        )
        logging.info(
            f"Test results - IoU: {test_result['iou']:.4f}, F1: {test_result['f1']:.4f}"
        )
        
        send_message(
            title=f"实验结束: {experiment_name}",
            content=f"训练完成!\n最佳 Val IoU: {best_iou:.4f} (at epoch {best_epoch})\n测试集 IoU: {test_result['iou']:.4f}",
        )

    except Exception as exc:
        logging.error(f"An error occurred: {exc}", exc_info=True)
        send_message(
            title=f"实验失败: {experiment_name}",
            content=f"实验运行时发生错误: \n{exc}",
        )
        raise

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()