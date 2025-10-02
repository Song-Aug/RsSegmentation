import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
from tqdm import tqdm
import swanlab
import matplotlib.pyplot as plt

from models.MSSDMPA_Net import MSSDMPA_Net
from utils.data_process import create_dataloaders
from utils.metrics import SegmentationMetrics
from configs.mssdmpanet_config import config
from utils.losses import mssdmpanet_y_bce_loss, MSSDMPA_IoU, mssdmpanet_dice_coeff
from utils.checkpoint import save_checkpoint
from utils.visualization import create_sample_images
from utils.trainer import test


def train_one_epoch(model, train_loader, criterion, optimizer, device, metrics, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    metrics.reset()

    # 使用MSSDMPA-Net自带的IoU计算类
    iou_metric = MSSDMPA_IoU(threshold=0.5)

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # 确保标签维度正确 - 添加通道维度如果不存在
        if len(labels.shape) == 3:  # [B, H, W]
            labels = labels.unsqueeze(1).float()  # [B, 1, H, W]
        else:
            labels = labels.float()

        optimizer.zero_grad()

        # 前向传播 - MSSDMPA-Net返回5个输出
        pred1, pred2, pred3, pred4, pred5 = model(images)

        # 计算多尺度损失
        loss = criterion(pred1, pred2, pred3, pred4, pred5, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 计算指标（使用主输出pred1）
        with torch.no_grad():
            dice = mssdmpanet_dice_coeff(labels.cpu(), pred1.detach().cpu())
            total_dice += dice.item()

            iou_metrics = iou_metric(labels.cpu(), pred1.detach().cpu())
            total_iou += iou_metrics[3].item()  # IoU值

            pred_binary = (
                (pred1 > 0.5).float().squeeze(1).cpu().numpy().astype(np.int64)
            )
            target_binary = labels.squeeze(1).cpu().numpy().astype(np.int64)
            metrics.update(pred_binary, target_binary)

        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{total_loss/(batch_idx+1):.4f}",
                "Dice": f"{dice.item():.4f}",
                "IoU": f"{iou_metrics[3].item():.4f}",
            }
        )

    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    train_metrics_results = metrics.get_metrics()

    return avg_loss, train_metrics_results, avg_dice, avg_iou


def validate(model, val_loader, criterion, device, metrics, epoch):
    """验证模型"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
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

            dice = mssdmpanet_dice_coeff(labels.cpu(), pred1.detach().cpu())
            total_dice += dice.item()

            iou_metrics = iou_metric(labels.cpu(), pred1.detach().cpu())
            total_iou += iou_metrics[3].item()

            pred_binary = (
                (pred1 > 0.5).float().squeeze(1).cpu().numpy().astype(np.int64)
            )
            target_binary = labels.squeeze(1).cpu().numpy().astype(np.int64)
            metrics.update(pred_binary, target_binary)

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{total_loss/(batch_idx+1):.4f}",
                    "Dice": f"{dice.item():.4f}",
                    "IoU": f"{iou_metrics[3].item():.4f}",
                }
            )

    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    val_metrics_results = metrics.get_metrics()

    return avg_loss, val_metrics_results, avg_dice, avg_iou


def main():
    seed = config["seed"]
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    experiment_name = f"{config['model_name']}_{datetime.now().strftime('%m%d%H%M')}"
    swanlab.init(
        project="Building-Segmentation-3Bands",
        experiment_name=experiment_name,
        config=config,
        description="MSSDMPA-Net建筑物分割实验",
        tags=["MSSDMPA-Net", "building-segmentation", "RGB", "multi-scale"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            root_dir=config["data_root"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            image_size=config["image_size"],
            augment=True,
            use_nir=config["use_nir"],
        )

        swanlab.log(
            {
                "dataset/train_batches": swanlab.Text(str(len(train_loader))),
                "dataset/val_batches": swanlab.Text(str(len(val_loader))),
                "dataset/test_batches": swanlab.Text(str(len(test_loader))),
            }
        )

        model = MSSDMPA_Net(
            input_channels=config["input_channels"], num_classes=config["num_classes"]
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        swanlab.log(
            {
                "model/total_params": swanlab.Text(str(total_params)),
                "model/trainable_params": swanlab.Text(str(trainable_params)),
            }
        )

        criterion = mssdmpanet_y_bce_loss
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["num_epochs"]
        )

        train_metrics = SegmentationMetrics(2)
        val_metrics = SegmentationMetrics(2)
        test_metrics = SegmentationMetrics(2)

        best_iou = 0
        best_epoch = 0

        for epoch in range(config["num_epochs"]):
            swanlab.log({"train/learning_rate": optimizer.param_groups[0]["lr"]})

            train_loss, train_result, train_dice, train_iou = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                train_metrics,
                epoch + 1,
            )

            val_loss, val_result, val_dice, val_iou = validate(
                model, val_loader, criterion, device, val_metrics, epoch + 1
            )

            scheduler.step()

            swanlab.log(
                {
                    "train/loss": train_loss,
                    "train/dice": train_dice,
                    "train/iou_custom": train_iou,
                    "train/iou": train_result["iou"],
                    "train/precision": train_result["precision"],
                    "train/recall": train_result["recall"],
                    "train/f1": train_result["f1"],
                    "val/loss": val_loss,
                    "val/dice": val_dice,
                    "val/iou_custom": val_iou,
                    "val/iou": val_result["iou"],
                    "val/precision": val_result["precision"],
                    "val/recall": val_result["recall"],
                    "val/f1": val_result["f1"],
                }
            )

            if (epoch + 1) % 10 == 0:
                sample_figures = create_sample_images(
                    model, val_loader, device, epoch + 1
                )
                for i, fig in enumerate(sample_figures):
                    swanlab.log(
                        {
                            f"Example_Images/epoch_{epoch+1}_sample_{i}": swanlab.Image(
                                fig
                            )
                        }
                    )
                    plt.close(fig)

            checkpoint_dir = f"./checkpoints/{experiment_name}"
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            if val_iou > best_iou:
                best_iou = val_iou
                best_epoch = epoch + 1
                best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
                save_checkpoint(model, optimizer, epoch, best_iou, best_model_path)

            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"
                )
                save_checkpoint(model, optimizer, epoch, best_iou, checkpoint_path)

        swanlab.log(
            {
                "best/iou": swanlab.Text(str(best_iou)),
                "best/epoch": swanlab.Text(str(best_epoch)),
            }
        )

        if "best_model_path" in locals() and os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint["model_state_dict"])

        test_result, test_dice, test_iou = test(
            model, test_loader, device, test_metrics
        )

        swanlab.log(
            {
                "test/dice": test_dice,
                "test/iou_custom": test_iou,
                "test/iou": test_result["iou"],
                "test/precision": test_result["precision"],
                "test/recall": test_result["recall"],
                "test/f1": test_result["f1"],
            }
        )

    except Exception as e:
        try:
            swanlab.log({"error": swanlab.Text(str(e))})
        except:
            pass
        raise

    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
