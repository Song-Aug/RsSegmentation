import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import swanlab

from models.UNetFormer import UNetFormer
from utils.data_process import create_dataloaders
from utils.metrics import SegmentationMetrics

# 导入重构后的模块
from configs.unetformer_config import config
from utils.trainer import train_one_epoch, validate, test
from utils.checkpoint import save_checkpoint
from utils.visualization import create_sample_images


def main():
    # 初始化SwanLab实验看板
    experiment_name = f"{config['model_name']}_{datetime.now().strftime('%m%d')}"
    swanlab.init(
        project=config["project_name"],
        experiment_name=experiment_name,
        config=config,
        description=config["description"],
        tags=config["tags"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # 记录数据集信息到SwanLab
        swanlab.log(
            {
                "dataset/train_batches": len(train_loader),
                "dataset/val_batches": len(val_loader),
                "dataset/test_batches": len(test_loader),
                "dataset/train_samples": len(train_loader.dataset),
                "dataset/val_samples": len(val_loader.dataset),
                "dataset/test_samples": len(test_loader.dataset),
            }
        )

        # 创建模型并记录模型信息
        model = UNetFormer(
            backbone_name=config["backbone"],
            pretrained=config["pretrained"],
            num_classes=config["num_classes"],
            in_channels=config["input_channels"],
        )
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        swanlab.log(
            {
                "model/total_params": total_params,
                "model/trainable_params": trainable_params,
                "model/input_channels": config["input_channels"],
            }
        )

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["num_epochs"]
        )

        # 定义指标计算类
        train_metrics = SegmentationMetrics(config["num_classes"])
        val_metrics = SegmentationMetrics(config["num_classes"])
        test_metrics = SegmentationMetrics(config["num_classes"])

        # 训练循环
        best_iou = 0
        best_epoch = 0

        # 创建检查点保存目录
        checkpoint_dir = os.path.join(config["save_dir"], experiment_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(config["num_epochs"]):
            # 记录学习率到SwanLab
            swanlab.log(
                {
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch + 1,
                }
            )

            # 训练
            train_loss, train_result = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                train_metrics,
                epoch + 1,
                use_aux_loss=config["use_aux_loss"],
                aux_loss_weight=config["aux_loss_weight"],
            )

            # 验证
            val_loss, val_result = validate(
                model, val_loader, criterion, device, val_metrics, epoch + 1
            )

            # 更新学习率
            scheduler.step()

            # 记录训练和验证指标到SwanLab
            swanlab.log(
                {
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
                    "epoch": epoch + 1,
                }
            )

            # 每10个epoch创建样本图像
            if (epoch + 1) % 10 == 0:
                sample_images_log = create_sample_images(
                    model, val_loader, device, epoch + 1, num_samples=4
                )
                for log_item in sample_images_log:
                    swanlab.log(log_item)

            # 保存最佳模型
            if val_result["iou"] > best_iou:
                best_iou = val_result["iou"]
                best_epoch = epoch + 1
                best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
                save_checkpoint(model, optimizer, epoch + 1, best_iou, best_model_path)
                swanlab.log({"best/iou": best_iou, "best/epoch": best_epoch})

            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"
                )
                save_checkpoint(model, optimizer, epoch + 1, best_iou, checkpoint_path)

        # 测试
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best model from {best_model_path} for testing.")

        test_result = test(model, test_loader, device, test_metrics)

        # 记录测试结果到SwanLab
        swanlab.log(
            {
                "test/iou": test_result["iou"],
                "test/precision": test_result["precision"],
                "test/recall": test_result["recall"],
                "test/f1": test_result["f1"],
            }
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        # 记录错误到SwanLab
        try:
            swanlab.log({"error": str(e)})
        except:
            pass
        raise

    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
