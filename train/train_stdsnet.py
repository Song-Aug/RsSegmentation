import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb

from configs.stdsnet_config import config
from models.STDSNet import STDSNet
from utils4train.alerts_by_lark import send_message
from utils4train.checkpoint import save_checkpoint
from utils4train.data_process import (
    BuildingSegmentationDataset,
    create_dataloaders,
    create_vis_dataloader,
)
from utils4train.losses import stdsnet_loss
from utils4train.metrics import SegmentationMetrics
from utils4train.trainer import test
from utils4train.visualization import create_sample_images


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model, train_loader, optimizer, device, metrics, epoch, loss_params
): # 移除了 scaler 参数
    model.train()
    total_loss = 0.0
    global_loss_total = 0.0
    shape_loss_total = 0.0
    metrics.reset()

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss, global_loss, shape_loss = stdsnet_loss(
            outputs,
            labels,
            seg_weight=1.0,
            shape_weight=loss_params["shape_loss_weight"],
            dice_ratio=loss_params["dice_ratio"],
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        global_loss_total += global_loss.item()
        shape_loss_total += shape_loss.item()

        main_output = outputs[0]
        preds = torch.argmax(main_output, dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        metrics.update(preds, targets)

        pbar.set_postfix(
            {
                "Total": f"{loss.item():.4f}",
                "Global": f"{global_loss.item():.4f}",
                "Shape": f"{shape_loss.item():.4f}",
                "Avg": f"{total_loss / (batch_idx + 1):.4f}",
            }
        )

    avg_total = total_loss / len(train_loader)
    avg_global = global_loss_total / len(train_loader)
    avg_shape = shape_loss_total / len(train_loader)
    metric_values = metrics.get_metrics()

    return avg_total, avg_global, avg_shape, metric_values


def validate(model, val_loader, device, metrics, epoch, loss_params):
    model.eval()
    total_loss = 0.0
    global_loss_total = 0.0
    shape_loss_total = 0.0
    metrics.reset()

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # 移除了 with autocast(...)
            outputs = model(images)
            loss, global_loss, shape_loss = stdsnet_loss(
                outputs,
                labels,
                seg_weight=1.0,
                shape_weight=loss_params["shape_loss_weight"],
                dice_ratio=loss_params["dice_ratio"],
            )

            total_loss += loss.item()
            global_loss_total += global_loss.item()
            shape_loss_total += shape_loss.item()

            main_output = outputs[0]
            preds = torch.argmax(main_output, dim=1).cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(preds, targets)

            pbar.set_postfix(
                {
                    "Total": f"{loss.item():.4f}",
                    "Global": f"{global_loss.item():.4f}",
                    "Shape": f"{shape_loss.item():.4f}",
                    "Avg": f"{total_loss / (batch_idx + 1):.4f}",
                }
            )

    avg_total = total_loss / len(val_loader)
    avg_global = global_loss_total / len(val_loader)
    avg_shape = shape_loss_total / len(val_loader)
    metric_values = metrics.get_metrics()

    return avg_total, avg_global, avg_shape, metric_values


def main():
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = f"{config['model_name']}_{datetime.now().strftime('%m%d')}"
    wandb.init(
        project="Building-Segmentation-3Bands",
        name=experiment_name,
        config=config,
        notes="STDSNet建筑物分割实验",
        tags=["STDSNet", "building-segmentation", "RGB"],
    )

    loss_params = {
        "shape_loss_weight": config.get("shape_loss_weight", 0.4),
        "dice_ratio": config.get("dice_ratio", 0.5),
    }

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

        model = STDSNet(
            num_classes=config["num_classes"],
            image_size=config["image_size"],
            pretrained=config.get("encoder_pretrained", False),
        )
        model = model.to(device)
        wandb.watch(model, log="all", log_freq=100)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update(
            {
                "model_total_params": total_params,
                "model_trainable_params": trainable_params,
                "model_input_channels": config["input_channels"],
            }
        )

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
        # 移除了 scaler = GradScaler(...)

        train_metrics = SegmentationMetrics(config["num_classes"])
        val_metrics = SegmentationMetrics(config["num_classes"])
        test_metrics = SegmentationMetrics(config["num_classes"])

        checkpoint_dir = os.path.join("./checkpoints", experiment_name)
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
            ),
        )

        best_iou, report_iou = 0.0, 0.0
        best_epoch = -1
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

        for epoch in range(config["num_epochs"]):
            train_total, train_global, train_shape, train_result = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                train_metrics,
                epoch + 1,
                # 移除了 scaler=scaler
                loss_params=loss_params,
            )

            val_total, val_global, val_shape, val_result = validate(
                model,
                val_loader,
                device,
                val_metrics,
                epoch + 1,
                loss_params=loss_params,
            )

            scheduler.step()

            wandb.log(
                {
                    "Comparison Board/IoU": val_result["iou"],
                    "Comparison Board/F1": val_result["f1"],
                    "Comparison Board/Precision": val_result["precision"],
                    "Comparison Board/Recall": val_result["recall"],
                    "Train_info/Loss/Train": train_total,
                    "Train_info/Loss/Val": val_total,
                    "Train_info/Seg_Loss/Train": train_global,
                    "Train_info/Seg_Loss/Val": val_global,
                    "Train_info/Boundary_Loss/Train": train_shape,
                    "Train_info/Boundary_Loss/Val": val_shape,
                    "Train_info/IoU/Train": train_result["iou"],
                    "Train_info/IoU/Val": val_result["iou"],
                    "Train_info/F1/Train": train_result["f1"],
                    "Train_info/F1/Val": val_result["f1"],
                    "Train_info/Learning_Rate": optimizer.param_groups[0]["lr"],
                }
            )

            logging.info(
                f"epoch: {epoch+1}, "
                f"train_iou: {train_result['iou']:.4f}, val_iou: {val_result['iou']:.4f}, "
                f"train_loss: {train_total:.4f}, val_loss: {val_total:.4f}, "
                f"lr: {optimizer.param_groups[0]['lr']:.6g}"
            )

            if (epoch + 1) % 10 == 0:
                figure = create_sample_images(
                    model,
                    vis_loader,
                    device,
                    epoch + 1,
                    num_samples=len(vis_loader),
                )
                if figure:
                    wandb.log({"Prediction_Summary": wandb.Image(figure)})
                    plt.close(figure)

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

        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

        test_result = test(model, test_loader, device, test_metrics)
        wandb.log(
            {
                "Result Board/IoU": test_result["iou"],
                "Result Board/F1": test_result["f1"],
                "Result Board/Precision": test_result["precision"],
                "Result Board/Recall": test_result["recall"],
            }
        )
        logging.info(
            f"Test results - IoU: {test_result['iou']:.4f}, F1: {test_result['f1']:.4f}"
        )

        send_message(
            title=f"实验结束: {experiment_name}",
            content=(
                f"训练完成!\n最佳 Val IoU: {best_iou:.4f} (at epoch {best_epoch})\n"
                f"测试集 IoU: {test_result['iou']:.4f}"
            ),
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