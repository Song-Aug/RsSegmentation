import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from tqdm import tqdm
import swanlab
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import PIL.Image
import sys
sys.path.append('..')

from models.HDNet import HDNet
from data_process import create_dataloaders
from metrics import SegmentationMetrics


def hdnet_loss(outputs, labels, weights=None):
    """
    HDNet的复合损失函数
    
    Args:
        outputs: 模型输出，包含主输出和深度监督输出
        labels: 真实标签 (B, H, W) 取值为 0 或 1
        weights: 各个输出的权重
    """
    if weights is None:
        # 默认权重：主输出权重最大，深度监督权重递减
        weights = [1.0, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    
    # 解包输出
    x_seg, x_bd, seg1, seg2, seg3, seg4, seg5, seg6, bd1, bd2, bd3, bd4, bd5, bd6 = outputs
    
    # 使用 CrossEntropyLoss，与 TransCC 和 UNetFormer 保持一致
    criterion = nn.CrossEntropyLoss()
    
    # 确保标签为 long 类型，并且值在 [0, 1] 范围内
    labels = labels.long()
    labels = torch.clamp(labels, 0, 1)
    
    # 计算各个输出的损失
    losses = []
    
    # 主分割损失
    seg_loss = criterion(x_seg, labels)
    losses.append(seg_loss)
    
    # 边界损失
    bd_loss = criterion(x_bd, labels)
    losses.append(bd_loss)
    
    # 深度监督损失
    deep_outputs = [seg1, seg2, seg3, seg4, seg5, seg6, bd1, bd2, bd3, bd4, bd5, bd6]
    
    for i, output in enumerate(deep_outputs):
        # 调整输出尺寸以匹配标签
        if output.size()[2:] != labels.size()[1:]:
            output = torch.nn.functional.interpolate(
                output, size=labels.size()[1:], mode='bilinear', align_corners=False
            )
        
        deep_loss = criterion(output, labels)
        losses.append(deep_loss)
    
    # 计算加权总损失
    total_loss = sum(w * loss for w, loss in zip(weights, losses))
    
    return total_loss, seg_loss, bd_loss


def train_one_epoch(model, train_loader, optimizer, device, metrics, epoch):
    model.train()
    total_loss = 0
    total_seg_loss = 0
    total_bd_loss = 0
    metrics.reset()
    
    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # 调试：检查第一个batch的标签
        if batch_idx == 0:
            print(f"标签形状: {labels.shape}")
            print(f"标签数据类型: {labels.dtype}")
            print(f"标签值范围: min={labels.min()}, max={labels.max()}")
            print(f"标签中的唯一值: {torch.unique(labels)}")
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        total_loss_batch, seg_loss_batch, bd_loss_batch = hdnet_loss(outputs, labels)
    
        # 反向传播
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        total_seg_loss += seg_loss_batch.item()
        total_bd_loss += bd_loss_batch.item()
        
        # 计算指标（使用主分割输出）
        x_seg = outputs[0]  # 主分割输出
        pred = torch.argmax(x_seg, dim=1).cpu().numpy()
        target = labels.cpu().numpy()
        # 确保target在[0,1]范围内
        target = np.clip(target, 0, 1)
        metrics.update(pred, target)
        
        # 更新进度条
        pbar.set_postfix({
            'Total Loss': f'{total_loss_batch.item():.4f}',
            'Seg Loss': f'{seg_loss_batch.item():.4f}',
            'BD Loss': f'{bd_loss_batch.item():.4f}',
            'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    avg_total_loss = total_loss / len(train_loader)
    avg_seg_loss = total_seg_loss / len(train_loader)
    avg_bd_loss = total_bd_loss / len(train_loader)
    train_metrics = metrics.get_metrics()
    
    return avg_total_loss, avg_seg_loss, avg_bd_loss, train_metrics


def validate(model, val_loader, device, metrics, epoch):
    model.eval()
    total_loss = 0
    total_seg_loss = 0
    total_bd_loss = 0
    metrics.reset()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            total_loss_batch, seg_loss_batch, bd_loss_batch = hdnet_loss(outputs, labels)
            
            total_loss += total_loss_batch.item()
            total_seg_loss += seg_loss_batch.item()
            total_bd_loss += bd_loss_batch.item()
            
            # 计算指标（使用主分割输出）
            x_seg = outputs[0]  # 主分割输出
            pred = torch.argmax(x_seg, dim=1).cpu().numpy()
            target = labels.cpu().numpy()
            # 确保target在[0,1]范围内
            target = np.clip(target, 0, 1)
            metrics.update(pred, target)
            
            # 更新进度条
            pbar.set_postfix({
                'Total Loss': f'{total_loss_batch.item():.4f}',
                'Seg Loss': f'{seg_loss_batch.item():.4f}',
                'BD Loss': f'{bd_loss_batch.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
    
    avg_total_loss = total_loss / len(val_loader)
    avg_seg_loss = total_seg_loss / len(val_loader)
    avg_bd_loss = total_bd_loss / len(val_loader)
    val_metrics = metrics.get_metrics()
    
    return avg_total_loss, avg_seg_loss, avg_bd_loss, val_metrics


def test(model, test_loader, device, metrics):
    model.eval()
    metrics.reset()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(images)
            x_seg = outputs[0]  # 主分割输出

            pred = torch.argmax(x_seg, dim=1).cpu().numpy()
            target = labels.cpu().numpy()
            # 确保target在[0,1]范围内
            target = np.clip(target, 0, 1)
            metrics.update(pred, target)
    
    test_metrics = metrics.get_metrics()
    return test_metrics


def save_checkpoint(model, optimizer, epoch, best_iou, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': best_iou,
    }
    torch.save(checkpoint, save_path)


def create_sample_images(model, val_loader, device, epoch, num_samples=4):
    """创建样本预测图像用于SwanLab可视化"""
    model.eval()
    
    # 收集所有样本数据
    samples_data = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_samples:
                break
                
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            x_seg = outputs[0]  # 主分割输出
            x_bd = outputs[1]   # 边界输出
            
            pred_seg = torch.argmax(x_seg, dim=1)
            pred_bd = torch.argmax(x_bd, dim=1)
            
            # 只取第一张图像
            img = images[0].cpu()
            label = labels[0].cpu()
            prediction_seg = pred_seg[0].cpu()
            prediction_bd = pred_bd[0].cpu()
            
            # 处理原图
            original_img = img[:3]  # 取RGB三个通道
            
            # 反标准化
            if original_img.min() < 0:  # 检查是否经过标准化
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                original_img = original_img * std + mean
            
            # 确保像素值在[0,1]范围内
            original_img = torch.clamp(original_img, 0, 1)
            original_img = original_img.permute(1, 2, 0).numpy()
            
            # 标签和预测转换为灰度图像
            label_gray = label.numpy().astype(np.uint8)
            pred_seg_gray = prediction_seg.numpy().astype(np.uint8)
            pred_bd_gray = prediction_bd.numpy().astype(np.uint8)
            
            samples_data.append((original_img, label_gray, pred_seg_gray, pred_bd_gray))
    
    # 创建一个综合的figure
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    fig.suptitle(f'HDNet Results - Epoch {epoch}', fontsize=16, fontweight='bold')
    
    for i, (original_img, label_gray, pred_seg_gray, pred_bd_gray) in enumerate(samples_data):
        # 原图
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f'Sample {i} - Original', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # 真实标签
        axes[i, 1].imshow(label_gray, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Sample {i} - Ground Truth', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
        
        # 分割预测结果
        axes[i, 2].imshow(pred_seg_gray, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Sample {i} - Segmentation', fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')
        
        # 边界预测结果
        axes[i, 3].imshow(pred_bd_gray, cmap='gray', vmin=0, vmax=1)
        axes[i, 3].set_title(f'Sample {i} - Boundary', fontsize=12, fontweight='bold')
        axes[i, 3].axis('off')
    
    # 添加图例
    legend_elements = [
        patches.Patch(color='black', label='Background'),
        patches.Patch(color='white', label='Building')
    ]
    fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.05)
    
    return fig


def main():
    # 设置随机种子确保实验可重现
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # 设置确定性算法（可选，但会影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 配置实验参数
    config = {
        'data_root': '/mnt/data1/rove/asset/GF7_Building/3BandsSample',
        'batch_size': 1,
        'num_workers': 4,
        'image_size': 512,
        'input_channels': 3,  # RGB
        'use_nir': False,
        'num_epochs': 120,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'save_dir': './runs',
        'model_name': 'hdnet_3bands',
        'base_channel': 48,
        'num_classes': 2,  # 2类：背景(0)和建筑物(1)
        'seed': seed
    }
    
    # 初始化SwanLab实验看板
    experiment_name = f"{config['model_name']}_{datetime.now().strftime('%m%d%H%M')}"
    swanlab.init(
        project="Building-Segmentation-HDNet",
        experiment_name=experiment_name,
        config=config,
        description="HDNet模型用于3波段建筑物分割实验",
        tags=["HDNet", "building-segmentation", "RGB", "high-resolution"],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # 数据加载
        train_loader, val_loader, test_loader = create_dataloaders(
            root_dir=config['data_root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            image_size=config['image_size'],
            augment=True,
            use_nir=config['use_nir']
        )
        
        # 记录数据集信息到SwanLab
        swanlab.log({
            'dataset/train_batches': swanlab.Text(str(len(train_loader))),
            'dataset/val_batches': swanlab.Text(str(len(val_loader))),
            'dataset/test_batches': swanlab.Text(str(len(test_loader))),
            'dataset/train_samples': swanlab.Text(str(len(train_loader) * config['batch_size'])),
            'dataset/val_samples': swanlab.Text(str(len(val_loader) * config['batch_size'])),
            'dataset/test_samples': swanlab.Text(str(len(test_loader) * config['batch_size'])),
        })

        # 创建模型并记录模型信息
        model = HDNet(
            base_channel=config['base_channel'],
            num_classes=config['num_classes']
        )
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        swanlab.log({
            'model/total_params': swanlab.Text(str(total_params)),
            'model/trainable_params': swanlab.Text(str(trainable_params)),
            'model/base_channel': swanlab.Text(str(config['base_channel'])),
            'model/input_channels': swanlab.Text(str(config['input_channels'])),
        })
        
        # 优化器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
        
        # 定义指标计算类
        train_metrics = SegmentationMetrics(config['num_classes'])
        val_metrics = SegmentationMetrics(config['num_classes'])
        test_metrics = SegmentationMetrics(config['num_classes'])
        
        # 创建检查点保存目录
        checkpoint_dir = f'./checkpoints/{experiment_name}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 训练循环
        best_iou = 0
        best_epoch = 0
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        
        for epoch in range(config['num_epochs']):       
            # 记录学习率到SwanLab
            swanlab.log({
                'train/learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # 训练
            train_total_loss, train_seg_loss, train_bd_loss, train_result = train_one_epoch(
                model, train_loader, optimizer, device, train_metrics, epoch + 1
            )
            
            # 验证
            val_total_loss, val_seg_loss, val_bd_loss, val_result = validate(
                model, val_loader, device, val_metrics, epoch + 1
            )
            
            # 更新学习率
            scheduler.step()
            
            # 记录训练和验证指标到SwanLab
            swanlab.log({
                'train/total_loss': train_total_loss,
                'train/seg_loss': train_seg_loss,
                'train/bd_loss': train_bd_loss,
                'train/iou': train_result['iou'],
                'train/precision': train_result['precision'],
                'train/recall': train_result['recall'],
                'train/f1': train_result['f1'],
                'val/total_loss': val_total_loss,
                'val/seg_loss': val_seg_loss,
                'val/bd_loss': val_bd_loss,
                'val/iou': val_result['iou'],
                'val/precision': val_result['precision'],
                'val/recall': val_result['recall'],
                'val/f1': val_result['f1']
            })
            
            # 每10个epoch创建样本图像
            if (epoch + 1) % 10 == 0:
                sample_figure = create_sample_images(model, val_loader, device, epoch + 1)
                swanlab.log({"Example_Images": swanlab.Image(sample_figure)})
                plt.close(sample_figure)  # 关闭figure释放内存
            
            # 保存最佳模型
            if val_result['iou'] > best_iou:
                best_iou = val_result['iou']
                best_epoch = epoch + 1
                save_checkpoint(model, optimizer, epoch, best_iou, best_model_path)
                print(f"New best model saved at epoch {epoch+1} with IoU: {best_iou:.4f}")

            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(model, optimizer, epoch, best_iou, checkpoint_path)
        
        # 记录最佳指标到SwanLab
        swanlab.log({
            'best/iou': swanlab.Text(str(best_iou)),
            'best/epoch': swanlab.Text(str(best_epoch))
        })
        
        print(f"Training completed. Best IoU: {best_iou:.4f} at epoch {best_epoch}")
        
        # 测试
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Best model loaded for testing")
        
        test_result = test(model, test_loader, device, test_metrics)
        
        # 记录测试结果到SwanLab
        swanlab.log({
            'test/iou': test_result['iou'],
            'test/precision': test_result['precision'],
            'test/recall': test_result['recall'],
            'test/f1': test_result['f1']
        })
        
        print("Test Results:")
        print(f"IoU: {test_result['iou']:.4f}")
        print(f"Precision: {test_result['precision']:.4f}")
        print(f"Recall: {test_result['recall']:.4f}")
        print(f"F1: {test_result['f1']:.4f}")
        
    except Exception as e:
        # 记录错误到SwanLab
        try:
            swanlab.log({'error': str(e)})
        except:
            pass
        print(f"Error occurred: {e}")
        raise
    
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
