import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import random


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import swanlab

# webhook配置，与transcc一致
from swanlab.plugin.notification import LarkCallback
lark_callback = LarkCallback(
    webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/5cd99837-5be3-4438-975f-89697bb5250c",
    secret="wzn4LqIgfwN4TRk2Mecc1b"
)

from configs.transcc_v2_config import config
from data_process import create_dataloaders
from metrics import SegmentationMetrics
from models.TransCCV2 import create_transcc_v2
from utils.checkpoint import save_checkpoint
from utils.losses import transcc_v2_loss
from utils.trainer import test
from utils.visualization import create_sample_images


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

    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss, seg_loss, boundary_loss = transcc_v2_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        seg_loss_total += seg_loss.item()
        boundary_loss_total += boundary_loss.item()

        main_output = outputs[0]
        preds = torch.argmax(main_output, dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        metrics.update(preds, targets)

        pbar.set_postfix({
            'Total': f'{loss.item():.4f}',
            'Seg': f'{seg_loss.item():.4f}',
            'Bd': f'{boundary_loss.item():.4f}',
            'Avg': f'{total_loss / (batch_idx + 1):.4f}'
        })

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
        pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            loss, seg_loss, boundary_loss = transcc_v2_loss(outputs, labels)

            total_loss += loss.item()
            seg_loss_total += seg_loss.item()
            boundary_loss_total += boundary_loss.item()

            main_output = outputs[0]
            preds = torch.argmax(main_output, dim=1).cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(preds, targets)

            pbar.set_postfix({
                'Total': f'{loss.item():.4f}',
                'Seg': f'{seg_loss.item():.4f}',
                'Bd': f'{boundary_loss.item():.4f}',
                'Avg': f'{total_loss / (batch_idx + 1):.4f}'
            })

    avg_total = total_loss / len(val_loader)
    avg_seg = seg_loss_total / len(val_loader)
    avg_bd = boundary_loss_total / len(val_loader)
    metric_values = metrics.get_metrics()

    return avg_total, avg_seg, avg_bd, metric_values


def main():
    # 设置随机种子
    set_seed(config['seed'])

    # swanlab实验看版配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_name = f"{config['model_name']}_{datetime.now().strftime('%m%d')}"
    swanlab.init(
        project="Building-Segmentation-3Bands",
        experiment_name=experiment_name,
        config=config,
        description="TransCCV2建筑物分割实验",
        tags=["TransCCV2", "building-segmentation", "RGB"],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[lark_callback]
    )


    try:
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_dataloaders(
            root_dir=config['data_root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            image_size=config['image_size'],
            augment=True,
            use_nir=config['use_nir']
        )

        # 模型初始化
        model = create_transcc_v2({
            'img_size': config['image_size'],
            'patch_size': 16,
            'in_chans': config['input_channels'],
            'num_classes': config['num_classes'],
        })
        model = model.to(device)

        # 加载ViT预训练权重
        from utils.weights import load_pretrained_weights
        pretrained_path = config.get('pretrained_weights', None)
        fusion_strategy = config.get('fusion_strategy', 'interpolate')
        if pretrained_path:
            model = load_pretrained_weights(model, pretrained_path, fusion_strategy)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        swanlab.log({
            'model/total_params': swanlab.Text(str(total_params)),
            'model/trainable_params': swanlab.Text(str(trainable_params)),
            'model/input_channels': swanlab.Text(str(config['input_channels'])),
        })

        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config['warmup_epochs']
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'] - config['warmup_epochs'],
            eta_min=config['min_lr']
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config['warmup_epochs']]
        )

        train_metrics = SegmentationMetrics(config['num_classes'])
        val_metrics = SegmentationMetrics(config['num_classes'])
        test_metrics = SegmentationMetrics(config['num_classes'])

        checkpoint_dir = os.path.join('./checkpoints', experiment_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_iou = 0.0
        best_epoch = -1
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')


        # 训练开始通知
        lark_callback.send_msg(
            content=f"{experiment_name} - 训练开始",
        )
        for epoch in range(config['num_epochs']):
            swanlab.log({'train/learning_rate': optimizer.param_groups[0]['lr']})

            train_total, train_seg, train_bd, train_result = train_one_epoch(
                model, train_loader, optimizer, device, train_metrics, epoch + 1
            )

            val_total, val_seg, val_bd, val_result = validate(
                model, val_loader, device, val_metrics, epoch + 1
            )

            scheduler.step()

            # 记录训练和验证指标到SwanLab，字段与transcc保持一致
            swanlab.log({
                'train/total_loss': train_total,
                'train/seg_loss': train_seg,
                'train/bd_loss': train_bd,
                'train/iou': train_result['iou'],
                'train/precision': train_result['precision'],
                'train/recall': train_result['recall'],
                'train/f1': train_result['f1'],
                'val/total_loss': val_total,
                'val/seg_loss': val_seg,
                'val/bd_loss': val_bd,
                'val/iou': val_result['iou'],
                'val/precision': val_result['precision'],
                'val/recall': val_result['recall'],
                'val/f1': val_result['f1']
            })

            if (epoch + 1) % 10 == 0:
                figures = create_sample_images(model, val_loader, device, epoch + 1)
                for idx, fig in enumerate(figures):
                    swanlab.log({f'Example_Images/epoch_{epoch+1}_sample_{idx}': swanlab.Image(fig)})
                    plt.close(fig)

            if val_result['iou'] > best_iou:
                best_iou = val_result['iou']
                best_epoch = epoch + 1
                save_checkpoint(model, optimizer, epoch, best_iou, best_model_path)
                # IoU大于0.6时发送通知
                if val_result['iou'] > 0.6:
                    lark_callback.send_msg(
                        content=f"Current IoU: {val_result['iou']}, New best model is saved",
                    )

            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(model, optimizer, epoch, best_iou, checkpoint_path)


        swanlab.log({
            'best/iou': swanlab.Text(str(best_iou)),
            'best/epoch': swanlab.Text(str(best_epoch))
        })

        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])


        test_result = test(model, test_loader, device, test_metrics)
        swanlab.log({
            'test/iou': test_result['iou'],
            'test/precision': test_result['precision'],
            'test/recall': test_result['recall'],
            'test/f1': test_result['f1']
        })

    except Exception as exc:
        try:
            swanlab.log({'error': str(exc)})
        except Exception:
            pass
        raise

    finally:
        swanlab.finish()


if __name__ == '__main__':
    main()
