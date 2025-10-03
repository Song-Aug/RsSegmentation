import os
import sys
import logging
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb

# 将项目根目录添加到系统路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.mssdmpanet_config import config
from models.MSSDMPA_Net import MSSDMPA_Net
from utils4train.alerts_by_lark import send_message
from utils4train.checkpoint import save_checkpoint
from utils4train.data_process import create_dataloaders, create_vis_dataloader
from utils4train.losses import mssdmpanet_dice_coeff, mssdmpanet_y_bce_loss, MSSDMPA_IoU
from utils4train.metrics import SegmentationMetrics
from utils4train.trainer import test
from utils4train.visualization import create_sample_images


def set_seed(seed: int) -> None:
    """设置随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, criterion, optimizer, device, metrics, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    metrics.reset()
    iou_metric = MSSDMPA_IoU(threshold=0.5)

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if len(labels.shape) == 3:
            labels = labels.unsqueeze(1).float()
        else:
            labels = labels.float()

        optimizer.zero_grad()
        pred1, pred2, pred3, pred4, pred5 = model(images)
        loss = criterion(pred1, pred2, pred3, pred4, pred5, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            # 指标计算时，对logits应用sigmoid
            pred1_logits = pred1.detach().cpu()
            labels_cpu = labels.cpu()
            
            dice = mssdmpanet_dice_coeff(labels_cpu, pred1_logits)
            iou = iou_metric(labels_cpu, pred1_logits)[3].item()
            
            pred_binary = (torch.sigmoid(pred1_logits) > 0.5).float().squeeze(1).numpy().astype(np.int64)
            target_binary = labels_cpu.squeeze(1).numpy().astype(np.int64)
            metrics.update(pred_binary, target_binary)

        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{total_loss / (batch_idx + 1):.4f}",
                "Dice": f"{dice.item():.4f}",
                "IoU": f"{iou:.4f}",
            }
        )

    avg_loss = total_loss / len(train_loader)
    metric_values = metrics.get_metrics()
    return avg_loss, metric_values


def validate(model, val_loader, criterion, device, metrics, epoch):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    metrics.reset()
    iou_metric = MSSDMPA_IoU(threshold=0.5)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            if len(labels.shape) == 3:
                labels = labels.unsqueeze(1).float()
            else:
                labels = labels.float()

            pred1, pred2, pred3, pred4, pred5 = model(images)
            loss = criterion(pred1, pred2, pred3, pred4, pred5, labels)
            total_loss += loss.item()

            # 指标计算时，对logits应用sigmoid
            pred1_logits = pred1.detach().cpu()
            labels_cpu = labels.cpu()

            dice = mssdmpanet_dice_coeff(labels_cpu, pred1_logits)
            iou = iou_metric(labels_cpu, pred1_logits)[3].item()

            pred_binary = (torch.sigmoid(pred1_logits) > 0.5).float().squeeze(1).numpy().astype(np.int64)
            target_binary = labels_cpu.squeeze(1).numpy().astype(np.int64)
            metrics.update(pred_binary, target_binary)

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{total_loss / (batch_idx + 1):.4f}",
                    "Dice": f"{dice.item():.4f}",
                    "IoU": f"{iou:.4f}",
                }
            )

    avg_loss = total_loss / len(val_loader)
    metric_values = metrics.get_metrics()
    return avg_loss, metric_values


def main():
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = f"{config['model_name']}_{datetime.now().strftime('%m%d')}"
    wandb.init(
        project="Building-Segmentation-3Bands",
        name=experiment_name,
        config=config,
        notes="MSSDMPA-Net 建筑物分割实验 (更新版)",
        tags=["MSSDMPA-Net", "building-segmentation", "RGB"],
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
            use_nir=config["use_nir"],
        )
        if vis_loader is None:
            vis_loader = val_loader

        # 模型初始化
        model = MSSDMPA_Net(
            input_channels=config["input_channels"], num_classes=config["num_classes"]
        ).to(device)
        wandb.watch(model, log="all", log_freq=100)
        
        # 计算并记录模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update({
            "model_total_params": total_params,
            "model_trainable_params": trainable_params,
        })

        # 损失函数, 优化器和学习率调度器
        criterion = mssdmpanet_y_bce_loss
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        # 确保配置文件中有warmup_epochs和min_lr
        warmup_epochs = config.get("warmup_epochs", 10)
        min_lr = config.get("min_lr", 1e-6)
        
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"] - warmup_epochs, eta_min=min_lr)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

        # 初始化评价指标
        train_metrics = SegmentationMetrics(2) # 二分类
        val_metrics = SegmentationMetrics(2)
        test_metrics = SegmentationMetrics(2)

        # 创建检查点目录和日志
        checkpoint_dir = os.path.join("./checkpoints", experiment_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        local_log_path = os.path.join(checkpoint_dir, "train_log.txt")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
            handlers=[logging.FileHandler(local_log_path, encoding="utf-8"), logging.StreamHandler()],
        )

        logging.info(f"实验开始: {experiment_name}")
        logging.info(f"模型: {config['model_name']}, 总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
        send_message(
            title=f"实验开始: {experiment_name}",
            content=(
                f"模型: {config['model_name']}\n"
                f"总参数: {total_params:,}\n"
                f"可训练参数: {trainable_params:,}\n"
                f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"训练轮数: {config['num_epochs']}\n"
                f"学习率: {config['learning_rate']}\n"
            )
        )

        # 训练循环
        best_iou,report_iou = 0.0, 0.0
        best_epoch = -1
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

        for epoch in range(config["num_epochs"]):
            train_loss, train_result = train_one_epoch(
                model, train_loader, criterion, optimizer, device, train_metrics, epoch + 1
            )
            val_loss, val_result = validate(
                model, val_loader, criterion, device, val_metrics, epoch + 1
            )
            scheduler.step()

            wandb.log({
                "epoch": epoch + 1,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/loss": train_loss,
                "train/iou": train_result["iou"],
                "train/precision": train_result["precision"],
                "train/recall": train_result["recall"],
                "train/f1": train_result["f1"],
                "val/loss": val_loss,
                "val/iou": val_result["iou"],
                "val/precision": val_result["precision"],
                "val/recall": val_result["recall"],
                "val/f1": val_result["f1"],
            })

            logging.info(
                f"epoch: {epoch+1}, train_iou: {train_result['iou']:.4f}, val_iou: {val_result['iou']:.4f}, "
                f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, lr: {optimizer.param_groups[0]['lr']:.6g}"
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
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                save_checkpoint(model, optimizer, epoch, best_iou, checkpoint_path)

        wandb.summary["best_iou"] = best_iou
        wandb.summary["best_epoch"] = best_epoch

        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

        test_result = test(model, test_loader, device, test_metrics)
        wandb.log({
            "test/iou": test_result["iou"],
            "test/precision": test_result["precision"],
            "test/recall": test_result["recall"],
            "test/f1": test_result["f1"],
        })
        logging.info(f"Test results - IoU: {test_result['iou']:.4f}, F1: {test_result['f1']:.4f}")

        send_message(
            title=f"实验结束: {experiment_name}",
            content=f"训练完成!\n最佳 Val IoU: {best_iou:.4f} (at epoch {best_epoch})\n测试集 IoU: {test_result['iou']:.4f}",
        )

    except Exception as exc:
        logging.error(f"An error occurred: {exc}", exc_info=True)
        send_message(title=f"实验失败: {experiment_name}", content=f"实验运行时发生错误: \n{exc}")
        raise

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()