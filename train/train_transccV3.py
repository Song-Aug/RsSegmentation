import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from datetime import timedelta
import random
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb
from configs.transcc_v2_config import config
from utils4train.data_process import *
from utils4train.metrics import SegmentationMetrics
from models.TransCCV3 import create_transcc_v3
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


def train_one_epoch(model, train_loader, optimizer, device, metrics, epoch, scaler):
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
        with autocast():
            outputs = model(images)
            loss, seg_loss, boundary_loss = transcc_v2_loss(
                outputs, labels, seg_weight=1.0, boundary_weight=1.3, aux_weight=0.4
            )
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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

            with autocast():
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
        notes="TransCCV3建筑物分割实验",
        tags=["TransCCV3"],
    )

    try:
        _, val_loader, test_loader = create_dataloaders(
            root_dir=config["data_root"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            image_size=config["image_size"],
            augment=False,
            use_nir=config["use_nir"],
        )
        strong_aug = get_train_augmentations(config["image_size"], config["use_nir"])
        mild_aug = get_mild_augmentations(config["image_size"], config["use_nir"])
        train_dataset_strong = BuildingSegmentationDataset(
            root_dir=config["data_root"], split='Train', transform=strong_aug, use_nir=config["use_nir"]
        )
        train_dataset_mild = BuildingSegmentationDataset(
            root_dir=config["data_root"], split='Train', transform=mild_aug, use_nir=config["use_nir"]
        )
        # 4. 创建两个对应的训练数据加载器
        train_loader_strong = torch.utils.data.DataLoader(
            train_dataset_strong, batch_size=config["batch_size"], shuffle=True,
            num_workers=config["num_workers"], pin_memory=True, drop_last=True
        )
        train_loader_mild = torch.utils.data.DataLoader(
            train_dataset_mild, batch_size=config["batch_size"], shuffle=True,
            num_workers=config["num_workers"], pin_memory=True, drop_last=True
        )
        logging.info(f"强增强训练将在前 {config['mild_aug_epoch']} 个epochs执行。")
        logging.info(f"温和增强训练将在第 {config['mild_aug_epoch']} 个epoch后执行。")
        vis_loader = create_vis_dataloader(
            root_dir=config["data_root"],
            image_size=config["image_size"],
            num_workers=config["num_workers"],
            use_nir=config["use_nir"]
        )
        if vis_loader is None:
            vis_loader = val_loader

        # 模型初始化
        model = create_transcc_v3(
            {
                "img_size": config["image_size"],
                "patch_size": 16,
                "in_chans": config["input_channels"],
                "num_classes": config["num_classes"],
                "depth": 9,
                "hdnet_base_channel": 48
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
        scaler = GradScaler()

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
            )
        )

        # 训练循环
        best_iou, report_iou = 0.0, 0.0
        best_epoch = -1
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

        for epoch in range(config["num_epochs"]):
            train_loader = train_loader_mild if epoch + 1 > config['mild_aug_epoch'] else train_loader_strong
            train_total, train_seg, train_bd, train_result = train_one_epoch(
                model, train_loader, optimizer, device, train_metrics, epoch + 1, scaler=scaler
            )

            val_total, val_seg, val_bd, val_result = validate(
                model, val_loader, device, val_metrics, epoch + 1
            )

            scheduler.step()

            # 使用 wandb.log 记录训练和验证指标
            wandb.log(
                {   
                    "Comparison Board/IoU": val_result["iou"],
                    "Comparison Board/F1": val_result["f1"],
                    "Comparison Board/Precision": val_result["precision"],
                    "Comparison Board/Recall": val_result["recall"],

                    "Train_info/Loss/Train": train_total,
                    "Train_info/Loss/Val": val_total,
                    "Train_info/Seg_Loss/Train": train_seg,
                    "Train_info/Seg_Loss/Val": val_seg,
                    "Train_info/Boundary_Loss/Train": train_bd,
                    "Train_info/Boundary_Loss/Val": val_bd,
                    "Train_info/IoU/Train": train_result["iou"],
                    "Train_info/IoU/Val": val_result["iou"],
                    "Train_info/F1/Train": train_result["f1"],
                    "Train_info/F1/Val": val_result["f1"],

                    "Train_info/Learning_Rate": optimizer.param_groups[0]["lr"]
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

        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

        test_result = test(model, test_loader, device, test_metrics)
        wandb.log(
            {
                "Result Board/IoU": test_result["iou"],
                "Result Board/F1": test_result["f1"],
                "Result Board/Precision": test_result["precision"],
                "Result Board/Recall": test_result["recall"]
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