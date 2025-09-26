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

from models.HDNet import HDNet
from data_process import create_dataloaders
from metrics import SegmentationMetrics
from configs.hdnet_config import config
from utils.losses import hdnet_loss
from utils.checkpoint import save_checkpoint
from utils.visualization import create_hdnet_sample_images
from utils.trainer import test


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


def main():
    # 设置随机种子确保实验可重现
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # 设置确定性算法（可选，但会影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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
                sample_figure = create_hdnet_sample_images(model, val_loader, device, epoch + 1)
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
