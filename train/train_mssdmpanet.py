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

from models.MSSDMPA_Net import MSSDMPA_Net, y_bce_loss, IoU, dice_coeff
from data_process import create_dataloaders
from metrics import SegmentationMetrics

from swanlab.plugin.notification import LarkCallback
lark_callback = LarkCallback(
    webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/5cd99837-5be3-4438-975f-89697bb5250c",
    secret="wzn4LqIgfwN4TRk2Mecc1b"
)



def train_one_epoch(model, train_loader, criterion, optimizer, device, metrics, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    metrics.reset()
    
    # 使用MSSDMPA-Net自带的IoU计算类
    iou_metric = IoU(threshold=0.5)
    
    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # 确保标签维度正确 - 添加通道维度如果不存在
        if len(labels.shape) == 3:  # [B, H, W]
            labels = labels.unsqueeze(1).float()  # [B, 1, H, W]
        elif len(labels.shape) == 4 and labels.shape[1] != 1:  # [B, H, W, C] 或其他格式
            labels = labels.float()
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
            # 计算Dice系数 - 确保维度匹配
            dice = dice_coeff(labels.cpu(), pred1.detach().cpu())
            total_dice += dice.item()
            
            # 计算IoU等指标 - 确保维度匹配
            iou_metrics = iou_metric(labels.cpu(), pred1.detach().cpu())
            total_iou += iou_metrics[3].item()  # IoU值
            
            # 为SegmentationMetrics计算预测结果 - 移除通道维度并转换为整数
            pred_binary = (pred1 > 0.5).float().squeeze(1).cpu().numpy().astype(np.int64)  # 转换为整数
            target_binary = labels.squeeze(1).cpu().numpy().astype(np.int64)  # 转换为整数
            metrics.update(pred_binary, target_binary)
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/(batch_idx+1):.4f}',
            'Dice': f'{dice.item():.4f}',
            'IoU': f'{iou_metrics[3].item():.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    train_metrics = metrics.get_metrics()
    
    return avg_loss, train_metrics, avg_dice, avg_iou


def validate(model, val_loader, criterion, device, metrics, epoch):
    """验证模型"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    metrics.reset()
    
    # 使用MSSDMPA-Net自带的IoU计算类
    iou_metric = IoU(threshold=0.5)
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 确保标签维度正确 - 添加通道维度如果不存在
            if len(labels.shape) == 3:  # [B, H, W]
                labels = labels.unsqueeze(1).float()  # [B, 1, H, W]
            elif len(labels.shape) == 4 and labels.shape[1] != 1:  # [B, H, W, C] 或其他格式
                labels = labels.float()
            else:
                labels = labels.float()
            
            # 前向传播 - MSSDMPA-Net返回5个输出
            pred1, pred2, pred3, pred4, pred5 = model(images)
            
            # 计算多尺度损失
            loss = criterion(pred1, pred2, pred3, pred4, pred5, labels)
            
            total_loss += loss.item()
            
            # 计算指标（使用主输出pred1）
            # 计算Dice系数 - 确保维度匹配
            dice = dice_coeff(labels.cpu(), pred1.detach().cpu())
            total_dice += dice.item()
            
            # 计算IoU等指标 - 确保维度匹配
            iou_metrics = iou_metric(labels.cpu(), pred1.detach().cpu())
            total_iou += iou_metrics[3].item()  # IoU值
            
            # 为SegmentationMetrics计算预测结果 - 移除通道维度并转换为整数
            pred_binary = (pred1 > 0.5).float().squeeze(1).cpu().numpy().astype(np.int64)  # 转换为整数
            target_binary = labels.squeeze(1).cpu().numpy().astype(np.int64)  # 转换为整数
            metrics.update(pred_binary, target_binary)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}',
                'Dice': f'{dice.item():.4f}',
                'IoU': f'{iou_metrics[3].item():.4f}'
            })
    
    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    val_metrics = metrics.get_metrics()
    
    return avg_loss, val_metrics, avg_dice, avg_iou


def test(model, test_loader, device, metrics):
    """测试模型"""
    model.eval()
    total_dice = 0
    total_iou = 0
    metrics.reset()
    
    # 使用MSSDMPA-Net自带的IoU计算类
    iou_metric = IoU(threshold=0.5)
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 确保标签维度正确 - 添加通道维度如果不存在
            if len(labels.shape) == 3:  # [B, H, W]
                labels = labels.unsqueeze(1).float()  # [B, 1, H, W]
            elif len(labels.shape) == 4 and labels.shape[1] != 1:  # [B, H, W, C] 或其他格式
                labels = labels.float()
            else:
                labels = labels.float()
            
            # 前向传播 - 只使用主输出pred1
            pred1, _, _, _, _ = model(images)
            
            # 计算指标（使用主输出pred1）
            # 计算Dice系数 - 确保维度匹配
            dice = dice_coeff(labels.cpu(), pred1.detach().cpu())
            total_dice += dice.item()
            
            # 计算IoU等指标 - 确保维度匹配
            iou_metrics = iou_metric(labels.cpu(), pred1.detach().cpu())
            total_iou += iou_metrics[3].item()  # IoU值
            
            # 为SegmentationMetrics计算预测结果 - 移除通道维度并转换为整数
            pred_binary = (pred1 > 0.5).float().squeeze(1).cpu().numpy().astype(np.int64)  # 转换为整数
            target_binary = labels.squeeze(1).cpu().numpy().astype(np.int64)  # 转换为整数
            metrics.update(pred_binary, target_binary)
    
    avg_dice = total_dice / len(test_loader)
    avg_iou = total_iou / len(test_loader)
    test_metrics = metrics.get_metrics()
    
    return test_metrics, avg_dice, avg_iou


def save_checkpoint(model, optimizer, epoch, best_iou, save_path):
    """保存模型检查点"""
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
    figures = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_samples:
                break
                
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 只使用主输出pred1进行可视化
            pred1, _, _, _, _ = model(images)
            pred = (pred1 > 0.5).float()
            
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
            label_gray = label.squeeze().numpy().astype(np.uint8)
            pred_gray = prediction.squeeze().numpy().astype(np.uint8)
            
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
            
            # 直接添加figure对象到列表
            figures.append(fig)
    
    return figures


def main():
    # 设置随机种子
    seed = 42
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # 配置实验参数
    config = {
        'data_root': '/mnt/sda1/songyufei/asset/GF7_Building/3Bands',
        'batch_size': 4,
        'num_workers': 4,
        'image_size': 512,
        'input_channels': 3,  # RGB图像
        'use_nir': False,
        'num_epochs': 200,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'save_dir': './runs',
        'model_name': 'MSSDMPA_Net',
        'num_classes': 1,  # MSSDMPA-Net使用1个类别（二分类）
        'seed': seed
    }
    
    # 初始化SwanLab实验看板
    experiment_name = f"{config['model_name']}_{datetime.now().strftime('%m%d%H%M')}"
    swanlab.init(
        project="Building-Segmentation-3Bands",
        experiment_name=experiment_name,
        config=config,
        description="MSSDMPA-Net建筑物分割实验",
        tags=["MSSDMPA-Net", "building-segmentation", "RGB", "multi-scale"],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[lark_callback]
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
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
        
        print(f"数据加载完成:")
        print(f"训练集: {len(train_loader)} batches")
        print(f"验证集: {len(val_loader)} batches")
        print(f"测试集: {len(test_loader)} batches")
        
        # 记录数据集信息到SwanLab
        swanlab.log({
            'dataset/train_batches': swanlab.Text(str(len(train_loader))),
            'dataset/val_batches': swanlab.Text(str(len(val_loader))),
            'dataset/test_batches': swanlab.Text(str(len(test_loader))),
            'dataset/train_samples': swanlab.Text(str(len(train_loader) * config['batch_size'])),
            'dataset/val_samples': swanlab.Text(str(len(val_loader) * config['batch_size'])),
            'dataset/test_samples': swanlab.Text(str(len(test_loader) * config['batch_size'])),
        })

        # 创建模型
        model = MSSDMPA_Net(
            input_channels=config['input_channels'],
            num_classes=config['num_classes']
        )
        model = model.to(device)
        
        # 计算模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"模型创建完成:")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        # 记录模型信息到SwanLab
        swanlab.log({
            'model/total_params': swanlab.Text(str(total_params)),
            'model/trainable_params': swanlab.Text(str(trainable_params)),
            'model/input_channels': swanlab.Text(str(config['input_channels'])),
        })
        
        # 损失函数和优化器
        criterion = y_bce_loss  # 使用MSSDMPA-Net的多尺度损失函数
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
        
        # 定义指标计算类
        train_metrics = SegmentationMetrics(2)  # 二分类，但metrics需要2个类别
        val_metrics = SegmentationMetrics(2)
        test_metrics = SegmentationMetrics(2)
        
        # 训练循环
        best_iou = 0
        best_epoch = 0
        
        print("开始训练...")
        lark_callback.send_msg(
            content=f"{experiment_name} - 训练开始",  # 通知内容
        )

        for epoch in range(config['num_epochs']):       
            # 记录学习率到SwanLab
            swanlab.log({
                'train/learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # 训练
            train_loss, train_result, train_dice, train_iou = train_one_epoch(
                model, train_loader, criterion, optimizer, device, train_metrics, epoch + 1
            )
            
            # 验证
            val_loss, val_result, val_dice, val_iou = validate(
                model, val_loader, criterion, device, val_metrics, epoch + 1
            )
            
            # 更新学习率
            scheduler.step()
            
            # 打印训练进度
            print(f'Epoch {epoch + 1}/{config["num_epochs"]}:')
            print(f'  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}')
            print(f'  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}')
            
            # 记录训练和验证指标到SwanLab
            swanlab.log({
                'train/loss': train_loss,
                'train/dice': train_dice,
                'train/iou_custom': train_iou,
                'train/iou': train_result['iou'],
                'train/precision': train_result['precision'],
                'train/recall': train_result['recall'],
                'train/f1': train_result['f1'],
                'val/loss': val_loss,
                'val/dice': val_dice,
                'val/iou_custom': val_iou,
                'val/iou': val_result['iou'],
                'val/precision': val_result['precision'],
                'val/recall': val_result['recall'],
                'val/f1': val_result['f1']
            })
            
            # 每10个epoch创建样本图像
            if (epoch + 1) % 10 == 0:
                sample_figures = create_sample_images(model, val_loader, device, epoch + 1)
                
                # 逐个记录每个figure
                for i, fig in enumerate(sample_figures):
                    swanlab.log({f"Example_Images/epoch_{epoch+1}_sample_{i}": swanlab.Image(fig)})
                    plt.close(fig)
            

            # 保存模型
            if not os.path.exists(f'./checkpoints/{experiment_name}'):
                os.makedirs(f'./checkpoints/{experiment_name}')
                
            # 使用val_iou（自定义IoU）作为最佳模型指标
            if val_iou > best_iou:
                best_iou = val_iou
                best_epoch = epoch + 1
                best_model_path = os.path.join(f'./checkpoints/{experiment_name}/', 'best_model.pth')
                save_checkpoint(model, optimizer, epoch, best_iou, best_model_path)
                print(f"  新的最佳模型保存！IoU: {best_iou:.4f}")
                
                if val_iou > 0.8:
                    lark_callback.send_msg(
                        content=f"Current IoU: {val_iou}, New best model is saved",  # 通知内容
                    )

                
            # 每10个epoch保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(f'./checkpoints/{experiment_name}', f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(model, optimizer, epoch, best_iou, checkpoint_path)
            
        
        # 记录最佳指标到SwanLab
        swanlab.log({
            'best/iou': swanlab.Text(str(best_iou)),
            'best/epoch': swanlab.Text(str(best_epoch))
        })
        
        print(f"\n训练完成！最佳IoU: {best_iou:.4f} (Epoch {best_epoch})")
        
        # 测试
        if os.path.exists(best_model_path):
            print("加载最佳模型进行测试...")
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        test_result, test_dice, test_iou = test(model, test_loader, device, test_metrics)
        
        print(f"测试结果:")
        print(f"  Dice: {test_dice:.4f}")
        print(f"  IoU (Custom): {test_iou:.4f}")
        print(f"  IoU (Metrics): {test_result['iou']:.4f}")
        print(f"  Precision: {test_result['precision']:.4f}")
        print(f"  Recall: {test_result['recall']:.4f}")
        print(f"  F1: {test_result['f1']:.4f}")
        
        # 记录测试结果到SwanLab
        swanlab.log({
            'test/dice': test_dice,
            'test/iou_custom': test_iou,
            'test/iou': test_result['iou'],
            'test/precision': test_result['precision'],
            'test/recall': test_result['recall'],
            'test/f1': test_result['f1']
        })
        
    except Exception as e:
        # 记录错误到SwanLab - 使用文本类型
        try:
            swanlab.log({'error': swanlab.Text(str(e))})
        except:
            pass
        print(f"训练过程中出现错误: {e}")
        raise
    
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
