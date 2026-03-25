"""
TransCCV3 训练主流程

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
from model import create_transcc_v3
from dataprocess import get_loaders
from loss_function import transcc_v2_loss
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
    total_loss = 0.0
    seg_loss_total = 0.0
    boundary_loss_total = 0.0
    metrics.reset()

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images)
            loss, seg_loss, boundary_loss = transcc_v2_loss(
                outputs, labels,
                seg_weight=config.get("seg_weight", 1.0),
                boundary_weight=config.get("boundary_weight", 1.0),
                aux_weight=config.get("aux_weight", 0.4)
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

        pbar.set_postfix({
            "Total": f"{loss.item():.4f}",
            "Seg": f"{seg_loss.item():.4f}",
            "Bd": f"{boundary_loss.item():.4f}",
            "Avg": f"{total_loss / (batch_idx + 1):.4f}",
        })

    avg_total = total_loss / len(train_loader)
    avg_seg = seg_loss_total / len(train_loader)
    avg_bd = boundary_loss_total / len(train_loader)
    metric_values = metrics.get_metrics()

    return avg_total, avg_seg, avg_bd, metric_values


def validate(model, val_loader, device, metrics, epoch):
    """验证"""
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

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss, seg_loss, boundary_loss = transcc_v2_loss(outputs, labels, boundary_weight=0.0)

            total_loss += loss.item()
            seg_loss_total += seg_loss.item()
            boundary_loss_total += boundary_loss.item()

            main_output = outputs[0]
            preds = torch.argmax(main_output, dim=1).cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(preds, targets)

            pbar.set_postfix({
                "Total": f"{loss.item():.4f}",
                "Seg": f"{seg_loss.item():.4f}",
                "Bd": f"{boundary_loss.item():.4f}",
                "Avg": f"{total_loss / (batch_idx + 1):.4f}",
            })

    avg_total = total_loss / len(val_loader)
    avg_seg = seg_loss_total / len(val_loader)
    avg_bd = boundary_loss_total / len(val_loader)
    metric_values = metrics.get_metrics()

    return avg_total, avg_seg, avg_bd, metric_values


def test(model, test_loader, device, metrics):
    """测试"""
    model.eval()
    metrics.reset()

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)

            main_output = outputs[0]
            preds = torch.argmax(main_output, dim=1).cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(preds, targets)

    metric_values = metrics.get_metrics()
    return metric_values


def load_pretrained_weights(model, pretrained_path, fusion_strategy='interpolate', encoder_name='transformer_encoder'):
    """
    加载 ViT 预训练权重到 TransCC 模型的编码器部分

    Args:
        model: TransCC 模型
        pretrained_path: 预训练权重文件路径
        fusion_strategy: 权重融合策略
        encoder_name: 模型中编码器模块的名称
    """
    if not os.path.exists(pretrained_path):
        print(f"警告: 预训练权重文件不存在: {pretrained_path}")
        return model

    print(f"正在加载预训练权重: {pretrained_path}")
    print(f"使用融合策略: {fusion_strategy}")

    try:
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')

        if 'model' in pretrained_dict:
            pretrained_dict = pretrained_dict['model']
        elif 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']

        model_dict = model.state_dict()

        model_layers = 0
        for key in model_dict.keys():
            if f'{encoder_name}.blocks.' in key:
                layer_num = int(key.split('.')[2]) + 1
                model_layers = max(model_layers, layer_num)

        pretrained_layers = 0
        for key in pretrained_dict.keys():
            if 'blocks.' in key:
                layer_num = int(key.split('.')[1]) + 1
                pretrained_layers = max(pretrained_layers, layer_num)

        matched_dict = {}

        if fusion_strategy == 'interpolate':
            for i in range(model_layers):
                src_pos = i * (pretrained_layers - 1) / (model_layers - 1) if model_layers > 1 else 0
                src_layer_low = int(src_pos)
                src_layer_high = min(src_layer_low + 1, pretrained_layers - 1)
                weight_high = src_pos - src_layer_low
                weight_low = 1.0 - weight_high

                layer_params = [
                    'norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
                    'attn.proj.weight', 'attn.proj.bias', 'norm2.weight', 'norm2.bias',
                    'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias'
                ]

                for param in layer_params:
                    key_low = f'blocks.{src_layer_low}.{param}'
                    key_high = f'blocks.{src_layer_high}.{param}'
                    target_key = f'{encoder_name}.blocks.{i}.{param}'

                    if key_low in pretrained_dict and key_high in pretrained_dict and target_key in model_dict:
                        if src_layer_low == src_layer_high:
                            interpolated_weight = pretrained_dict[key_low]
                        else:
                            interpolated_weight = (weight_low * pretrained_dict[key_low] +
                                                  weight_high * pretrained_dict[key_high])
                        matched_dict[target_key] = interpolated_weight

        if matched_dict:
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
            print(f"成功加载 {len(matched_dict)} 个预训练权重")
        else:
            print("警告: 没有找到匹配的预训练权重")

    except Exception as e:
        print(f"加载预训练权重时出错: {e}")

    return model


def main():
    """训练主流程"""
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = f"{config['model_name']}_{datetime.now().strftime('%m%d')}"
    wandb.init(
        project=config["wandb_project"],
        name=experiment_name,
        config=config,
        notes="TransCCV3 建筑物分割实验",
        tags=["TransCCV3"],
    )

    try:
        # 数据加载
        train_loader_strong, train_loader_mild, val_loader, test_loader, vis_loader = get_loaders(config)
        logging.info(f"强增强训练将在前 {config['mild_aug_epoch']} 个 epochs 执行")
        logging.info(f"温和增强训练将在第 {config['mild_aug_epoch']} 个 epoch 后执行")

        if vis_loader is None:
            vis_loader = val_loader

        # 模型初始化
        model = create_transcc_v3({
            "img_size": config["image_size"],
            "patch_size": config["patch_size"],
            "in_chans": config["input_channels"],
            "num_classes": config["num_classes"],
            "depth": config["depth"],
            "hdnet_base_channel": config["hdnet_base_channel"],
            "drop_rate": config["drop_rate"],
            "attn_drop_rate": config["attn_drop_rate"],
            "drop_path_rate": config["drop_path_rate"],
        })
        model = model.to(device)
        wandb.watch(model, log="all", log_freq=100)

        # 加载预训练权重
        pretrained_path = config.get("pretrained_weights", None)
        fusion_strategy = config.get("fusion_strategy", "interpolate")
        if pretrained_path:
            model = load_pretrained_weights(model, pretrained_path, fusion_strategy, 'transformer_encoder')

        # 参数统计
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

            # WandB 日志
            wandb.log({
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
            })

            logging.info(
                f"epoch: {epoch + 1}, "
                f"train_iou: {train_result['iou']:.4f}, val_iou: {val_result['iou']:.4f}, "
                f"train_loss: {train_total:.4f}, val_loss: {val_total:.4f}, "
                f"lr: {optimizer.param_groups[0]['lr']:.6g}"
            )

            # 定期可视化
            if (epoch + 1) % 10 == 0:
                figure = create_sample_images(model, vis_loader, device, epoch + 1, num_samples=len(vis_loader))
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

        # 测试集评估
        wandb.summary["best_iou"] = best_iou
        wandb.summary["best_epoch"] = best_epoch

        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            logging.info(f"Loaded best model from epoch {checkpoint['epoch']} with IoU: {checkpoint['best_iou']:.4f}")

        test_result = test(model, test_loader, device, test_metrics)
        wandb.log({
            "Result Board/IoU": test_result["iou"],
            "Result Board/F1": test_result["f1"],
            "Result Board/Precision": test_result["precision"],
            "Result Board/Recall": test_result["recall"]
        })
        logging.info(f"Test results - IoU: {test_result['iou']:.4f}, F1: {test_result['f1']:.4f}")

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
