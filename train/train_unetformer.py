import os
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

from models.UNetFormer import UNetFormer
from data_process import create_dataloaders
from metrics import SegmentationMetrics

def train_one_epoch(model, train_loader, criterion, optimizer, device, metrics, epoch):
    model.train()
    total_loss = 0
    metrics.reset()
    
    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        
        # 处理输出
        if isinstance(outputs, tuple):
            # 训练模式：(main_output, aux_output)
            main_output, aux_output = outputs
            main_loss = criterion(main_output, labels)
            aux_loss = criterion(aux_output, labels)
            # 主损失权重更大，辅助损失用于提供额外监督
            loss = main_loss + 0.4 * aux_loss
            
            # 记录分别的损失到SwanLab
            if batch_idx % 50 == 0:  # 每50个batch记录一次
                swanlab.log({
                    'train/main_loss_step': main_loss.item(),
                    'train/aux_loss_step': aux_loss.item(),
                    'train/total_loss_step': loss.item(),
                    'step': epoch * len(train_loader) + batch_idx
                })
        else:
            # 推理模式：只有main_output
            main_output = outputs
            loss = criterion(main_output, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算指标（只使用主输出）
        pred = torch.argmax(main_output, dim=1).cpu().numpy()
        target = labels.cpu().numpy()
        metrics.update(pred, target)
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    train_metrics = metrics.get_metrics()
    
    return avg_loss, train_metrics


def validate(model, val_loader, criterion, device, metrics, epoch):
    model.eval()
    total_loss = 0
    metrics.reset()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 处理UNetFormer的输出
            if isinstance(outputs, tuple):
                # 如果验证时也返回tuple（不应该发生，但防止意外）
                main_output, aux_output = outputs
                main_loss = criterion(main_output, labels)
                aux_loss = criterion(aux_output, labels)
                loss = main_loss + 0.4 * aux_loss
            else:
                # 正常情况：eval模式下只返回main_output
                main_output = outputs
                loss = criterion(main_output, labels)
            
            total_loss += loss.item()
            
            # 计算指标
            pred = torch.argmax(main_output, dim=1).cpu().numpy()
            target = labels.cpu().numpy()
            metrics.update(pred, target)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
    
    avg_loss = total_loss / len(val_loader)
    val_metrics = metrics.get_metrics()
    
    return avg_loss, val_metrics


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
            
            # 处理输出（测试时应该只有main_output）
            if isinstance(outputs, tuple):
                main_output = outputs[0]
            else:
                main_output = outputs
            
            # 计算指标
            pred = torch.argmax(main_output, dim=1).cpu().numpy()
            target = labels.cpu().numpy()
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
    images_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_samples:
                break
                
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                main_output = outputs[0]
            else:
                main_output = outputs
            
            pred = torch.argmax(main_output, dim=1)
            
            # 只取第一张图像
            img = images[0].cpu()
            label = labels[0].cpu()
            prediction = pred[0].cpu()
            
            # 处理原图 - 将图像转换为可视化格式
            original_img = img[:3]  # 取RGB三个通道
            
            # 反标准化
            # 假设使用了ImageNet标准化参数
            if original_img.min() < 0:  # 检查是否经过标准化
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                original_img = original_img * std + mean
            
            # 确保像素值在[0,1]范围内
            original_img = torch.clamp(original_img, 0, 1)
            original_img = original_img.permute(1, 2, 0).numpy()
            
            # 标签和预测转换为灰度图像 (建筑物=1, 背景=0)
            label_gray = label.numpy().astype(np.uint8)
            pred_gray = prediction.numpy().astype(np.uint8)
            
            # 创建matplotlib图表
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Epoch {epoch} - Sample {batch_idx}', fontsize=16, fontweight='bold')
            
            # 原图
            axes[0].imshow(original_img)
            axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            # 真实标签
            axes[1].imshow(label_gray, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            # 预测结果
            axes[2].imshow(pred_gray, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title('Prediction', fontsize=12, fontweight='bold')
            axes[2].axis('off')
            
            legend_elements = [
                patches.Patch(color='black', label='Background'),
                patches.Patch(color='white', label='Building')
            ]
            fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, 0.02))
        
            plt.tight_layout()
            plt.subplots_adjust(top=0.85, bottom=0.1)
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            combined_img = PIL.Image.open(buf)
            combined_img_array = np.array(combined_img)
            images_list.append({
                f'samples/epoch_{epoch}_sample_{batch_idx}_combined': swanlab.Image(combined_img_array)
            })

            plt.close(fig)
            buf.close()
    
    return images_list


def main():
    # 配置实验参数
    config = {
        'data_root': '/mnt/sda1/songyufei/asset/GF7_Building/3Bands',
        'batch_size': 12,
        'num_workers': 4,
        'image_size': 512,
        'input_channels': 3,  # RGB + NIR
        'use_nir': False,
        'num_epochs': 200,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'save_dir': './runs',
        'model_name': 'unetformer_3bands',
        'backbone': 'efficientnet_b0',
        'pretrained': False,
        'num_classes': 2
    }
    
    # 初始化SwanLab实验看板
    experiment_name = f"{config['model_name']}_{datetime.now().strftime('%m%d')}"
    swanlab.init(
        project="Building-Segmentation-3Bands",
        experiment_name=experiment_name,
        config=config,
        description="3波段建筑物分割实验 - UNetFormer模型",
        tags=["UNetFormer", "building-segmentation", "RGB"],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
        model = UNetFormer(
            backbone_name=config['backbone'],
            pretrained=config['pretrained'],
            num_classes=config['num_classes'],
            in_channels=config['input_channels']
        ) 
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        swanlab.log({
            'model/total_params': swanlab.Text(str(total_params)),
            'model/trainable_params': swanlab.Text(str(trainable_params)),
            'model/input_channels': swanlab.Text(str(config['input_channels'])),
        })
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
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
        
        # 训练循环
        best_iou = 0
        best_epoch = 0
        
        for epoch in range(config['num_epochs']):       
            # 记录学习率到SwanLab
            swanlab.log({
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch + 1
            })
            
            # 训练
            train_loss, train_result = train_one_epoch(
                model, train_loader, criterion, optimizer, device, train_metrics, epoch + 1
            )
            
            # 验证
            val_loss, val_result = validate(
                model, val_loader, criterion, device, val_metrics, epoch + 1
            )
            
            # 更新学习率
            scheduler.step()
            
            # 记录训练和验证指标到SwanLab
            swanlab.log({
                'train/loss': train_loss,
                'train/iou': train_result['iou'],
                'train/precision': train_result['precision'],
                'train/recall': train_result['recall'],
                'train/f1': train_result['f1'],
                'val/loss': val_loss,
                'val/iou': val_result['iou'],
                'val/precision': val_result['precision'],
                'val/recall': val_result['recall'],
                'val/f1': val_result['f1'],
                'epoch': epoch + 1
            })
            
            # 每10个epoch创建样本图像
            if (epoch + 1) % 10 == 0:
                sample_image = create_sample_images(model, val_loader, device, epoch + 1)
                swanlab.log({"Example Images": swanlab.Image(sample_image)})
            
            # 保存最佳模型
            if val_result['iou'] > best_iou:
                best_iou = val_result['iou']
                best_epoch = epoch + 1
                os.mkdir(f'./checkpoints/{experiment_name}', exist_ok=True)
                best_model_path = os.path.join(f'./checkpoints/{experiment_name}/', 'best_model.pth')
                save_checkpoint(model, optimizer, epoch, best_iou, best_model_path)

            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                os.mkdir(f'./checkpoints/{experiment_name}', exist_ok=True)
                checkpoint_path = os.path.join(f'./checkpoints/{experiment_name}', f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(model, optimizer, epoch, best_iou, checkpoint_path)
            
        
        # 记录最佳指标到SwanLab
        swanlab.log({
            'best/iou': swanlab.Text(str(best_iou)),
            'best/epoch': swanlab.Text(str(best_epoch))
        })
        
        # 测试
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        test_result = test(model, test_loader, device, test_metrics)
        
        # 记录测试结果到SwanLab
        swanlab.log({
            'test/iou': test_result['iou'],
            'test/precision': test_result['precision'],
            'test/recall': test_result['recall'],
            'test/f1': test_result['f1']
        })
        
    except Exception as e:
        # 记录错误到SwanLab
        try:
            swanlab.log({'error': str(e)})
        except:
            pass
        raise
    
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()