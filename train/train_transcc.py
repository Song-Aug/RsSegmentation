import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import numpy as np
from datetime import datetime
from tqdm import tqdm
import swanlab
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models.TransCC import TransCC
from models.TransCC_V2 import create_transcc_model
from data_process import create_dataloaders
from metrics import SegmentationMetrics

from swanlab.plugin.notification import LarkCallback
lark_callback = LarkCallback(
    webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/5cd99837-5be3-4438-975f-89697bb5250c",
    secret="wzn4LqIgfwN4TRk2Mecc1b"
)

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
        output = model(images)
        loss = criterion(output, labels)
    
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算指标（只使用主输出）
        pred = torch.argmax(output, dim=1).cpu().numpy()
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
            output = model(images)
            loss = criterion(output, labels)
            
            total_loss += loss.item()
            
            # 计算指标
            pred = torch.argmax(output, dim=1).cpu().numpy()
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
            output = model(images)

            pred = torch.argmax(output, dim=1).cpu().numpy()
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


def load_pretrained_weights(model, pretrained_path, fusion_strategy='interpolate'):
    """
    加载ViT预训练权重到TransCC模型的编码器部分
    
    Args:
        model: TransCC模型
        pretrained_path: 预训练权重文件路径
        fusion_strategy: 权重融合策略 ('direct', 'interpolate', 'average_pairs', 'skip')
    """
    if not os.path.exists(pretrained_path):
        print(f"警告: 预训练权重文件不存在: {pretrained_path}")
        return model
    
    print(f"正在加载预训练权重: {pretrained_path}")
    print(f"使用融合策略: {fusion_strategy}")
    
    try:
        # 加载预训练权重
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        
        # 如果预训练权重是完整的检查点，提取state_dict
        if 'model' in pretrained_dict:
            pretrained_dict = pretrained_dict['model']
        elif 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        
        print(f"预训练权重包含 {len(pretrained_dict)} 个键")
        
        # 获取模型的state_dict
        model_dict = model.state_dict()
        
        # 自动检测模型的transformer层数
        model_layers = 0
        for key in model_dict.keys():
            if 'encoder.blocks.' in key:
                layer_num = int(key.split('.')[2]) + 1
                model_layers = max(model_layers, layer_num)
        
        print(f"检测到TransCC模型有 {model_layers} 层transformer blocks")
        
        # 检测预训练模型的层数
        pretrained_layers = 0
        for key in pretrained_dict.keys():
            if 'blocks.' in key:
                layer_num = int(key.split('.')[1]) + 1
                pretrained_layers = max(pretrained_layers, layer_num)
        
        print(f"检测到预训练模型有 {pretrained_layers} 层transformer blocks")
        
        # 创建匹配的权重字典
        matched_dict = {}
        unmatched_keys = []
        
        # 首先加载非transformer层的权重（patch_embed, pos_embed, cls_token, norm）
        basic_mapping = {
            'patch_embed.proj.weight': 'encoder.patch_embed.proj.weight',
            'patch_embed.proj.bias': 'encoder.patch_embed.proj.bias',
            'pos_embed': 'encoder.pos_embed',
            'cls_token': 'encoder.cls_token',
            'norm.weight': 'encoder.norm.weight',
            'norm.bias': 'encoder.norm.bias',
        }
        
        for pretrained_key, model_key in basic_mapping.items():
            if pretrained_key in pretrained_dict and model_key in model_dict:
                pretrained_weight = pretrained_dict[pretrained_key]
                model_weight = model_dict[model_key]
                
                if pretrained_weight.shape == model_weight.shape:
                    matched_dict[model_key] = pretrained_weight
                    print(f"✓ 基础组件匹配: {pretrained_key} -> {model_key}")
        
        # 根据融合策略处理transformer层
        if fusion_strategy == 'direct':
            # 策略1: 直接加载前N层
            print(f"使用直接加载策略：加载前{model_layers}层")
            for i in range(model_layers):
                layer_mapping = {
                    f'blocks.{i}.norm1.weight': f'encoder.blocks.{i}.norm1.weight',
                    f'blocks.{i}.norm1.bias': f'encoder.blocks.{i}.norm1.bias',
                    f'blocks.{i}.attn.qkv.weight': f'encoder.blocks.{i}.attn.qkv.weight',
                    f'blocks.{i}.attn.qkv.bias': f'encoder.blocks.{i}.attn.qkv.bias',
                    f'blocks.{i}.attn.proj.weight': f'encoder.blocks.{i}.attn.proj.weight',
                    f'blocks.{i}.attn.proj.bias': f'encoder.blocks.{i}.attn.proj.bias',
                    f'blocks.{i}.norm2.weight': f'encoder.blocks.{i}.norm2.weight',
                    f'blocks.{i}.norm2.bias': f'encoder.blocks.{i}.norm2.bias',
                    f'blocks.{i}.mlp.fc1.weight': f'encoder.blocks.{i}.mlp.fc1.weight',
                    f'blocks.{i}.mlp.fc1.bias': f'encoder.blocks.{i}.mlp.fc1.bias',
                    f'blocks.{i}.mlp.fc2.weight': f'encoder.blocks.{i}.mlp.fc2.weight',
                    f'blocks.{i}.mlp.fc2.bias': f'encoder.blocks.{i}.mlp.fc2.bias',
                }
                
                for pretrained_key, model_key in layer_mapping.items():
                    if pretrained_key in pretrained_dict and model_key in model_dict:
                        matched_dict[model_key] = pretrained_dict[pretrained_key]
        
        elif fusion_strategy == 'skip':
            # 策略2: 隔层采样 (0,2,4,6,8,10) -> (0,1,2,3,4,5)
            print(f"使用隔层采样策略：从12层中采样到{model_layers}层")
            skip_ratio = pretrained_layers // model_layers
            for i in range(model_layers):
                src_layer = i * skip_ratio
                if src_layer < pretrained_layers:
                    layer_mapping = {
                        f'blocks.{src_layer}.norm1.weight': f'encoder.blocks.{i}.norm1.weight',
                        f'blocks.{src_layer}.norm1.bias': f'encoder.blocks.{i}.norm1.bias',
                        f'blocks.{src_layer}.attn.qkv.weight': f'encoder.blocks.{i}.attn.qkv.weight',
                        f'blocks.{src_layer}.attn.qkv.bias': f'encoder.blocks.{i}.attn.qkv.bias',
                        f'blocks.{src_layer}.attn.proj.weight': f'encoder.blocks.{i}.attn.proj.weight',
                        f'blocks.{src_layer}.attn.proj.bias': f'encoder.blocks.{i}.attn.proj.bias',
                        f'blocks.{src_layer}.norm2.weight': f'encoder.blocks.{i}.norm2.weight',
                        f'blocks.{src_layer}.norm2.bias': f'encoder.blocks.{i}.norm2.bias',
                        f'blocks.{src_layer}.mlp.fc1.weight': f'encoder.blocks.{i}.mlp.fc1.weight',
                        f'blocks.{src_layer}.mlp.fc1.bias': f'encoder.blocks.{i}.mlp.fc1.bias',
                        f'blocks.{src_layer}.mlp.fc2.weight': f'encoder.blocks.{i}.mlp.fc2.weight',
                        f'blocks.{src_layer}.mlp.fc2.bias': f'encoder.blocks.{i}.mlp.fc2.bias',
                    }
                    
                    for pretrained_key, model_key in layer_mapping.items():
                        if pretrained_key in pretrained_dict and model_key in model_dict:
                            matched_dict[model_key] = pretrained_dict[pretrained_key]
                            print(f"✓ 隔层映射: layer{src_layer} -> layer{i}")
        
        elif fusion_strategy == 'average_pairs':
            # 策略3: 相邻层平均 (0+1)/2 -> 0, (2+3)/2 -> 1, ...
            print(f"使用相邻层平均策略：将12层合并为{model_layers}层")
            for i in range(model_layers):
                src_layer1 = i * 2
                src_layer2 = i * 2 + 1
                
                if src_layer1 < pretrained_layers and src_layer2 < pretrained_layers:
                    layer_params = [
                        'norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
                        'attn.proj.weight', 'attn.proj.bias', 'norm2.weight', 'norm2.bias',
                        'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias'
                    ]
                    
                    for param in layer_params:
                        key1 = f'blocks.{src_layer1}.{param}'
                        key2 = f'blocks.{src_layer2}.{param}'
                        target_key = f'encoder.blocks.{i}.{param}'
                        
                        if key1 in pretrained_dict and key2 in pretrained_dict and target_key in model_dict:
                            # 平均两层的权重
                            averaged_weight = (pretrained_dict[key1] + pretrained_dict[key2]) / 2.0
                            matched_dict[target_key] = averaged_weight
                    
                    print(f"✓ 平均融合: layer{src_layer1}+layer{src_layer2} -> layer{i}")
        
        elif fusion_strategy == 'interpolate':
            # 策略4: 线性插值
            print(f"使用线性插值策略：将{pretrained_layers}层插值为{model_layers}层")
            
            # 为每个目标层计算对应的源层索引（浮点数）
            for i in range(model_layers):
                # 计算在源层中的位置
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
                    target_key = f'encoder.blocks.{i}.{param}'
                    
                    if key_low in pretrained_dict and key_high in pretrained_dict and target_key in model_dict:
                        # 线性插值
                        if src_layer_low == src_layer_high:
                            interpolated_weight = pretrained_dict[key_low]
                        else:
                            interpolated_weight = (weight_low * pretrained_dict[key_low] + 
                                                 weight_high * pretrained_dict[key_high])
                        matched_dict[target_key] = interpolated_weight
                
                print(f"✓ 插值映射: layer{src_pos:.1f} -> layer{i}")
        
        # 加载匹配的权重
        if matched_dict:
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
            print(f"✓ 成功加载 {len(matched_dict)} 个预训练权重")
        else:
            print("警告: 没有找到匹配的预训练权重")
        
        print("✓ 预训练权重加载完成")
        
    except Exception as e:
        print(f"加载预训练权重时出错: {e}")
        print("继续使用随机初始化权重")
    
    return model


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
            
            # # 确保像素值在[0,1]范围内
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
            
            # 直接添加figure对象到列表
            figures.append(fig)
    
    return figures


def main():
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
        'input_channels': 3,  # RGB + NIR
        'use_nir': False,
        'num_epochs': 200,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'save_dir': './runs',
        'model_name': 'TransCC',
        # 'backbone': 'efficientnet_b0',
        # 'pretrained': False,
        'num_classes': 2,
        'seed': seed
    }
    
    # 初始化SwanLab实验看板
    # experiment_name = f"{config['model_name']}_{datetime.now().strftime('%m%d%H%M')}"
    experiment_name = f"{config['model_name']}_loadpretrain"
    swanlab.init(
        project="Building-Segmentation-3Bands",
        experiment_name=experiment_name,
        config=config,
        description="3波段建筑物分割实验",
        tags=["UNetFormer", "building-segmentation", "RGB"],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[lark_callback]
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
        # model = TransCC(
        #     in_chans=config['input_channels'],
        #     num_classes=config['num_classes']
        # )
        model = create_transcc_model({'patch_size':16, 'num_classes':2})
        
        # 加载预训练权重
        pretrained_path = './pretrained_weights/vit_base_patch16_224.pth'
        # 可选的融合策略: 'direct', 'interpolate', 'average_pairs', 'skip'
        model = load_pretrained_weights(model, pretrained_path, fusion_strategy='interpolate')
        
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

        # Warm-up阶段
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=5
        )
        # 主调度器
        main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,          # 第一个重启周期
            T_mult=2,        # 周期倍增因子
            eta_min=1e-6,    # 最小学习率
            last_epoch=-1
        )
        # 组合调度器
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[5]
        )
        
        # 定义指标计算类
        train_metrics = SegmentationMetrics(config['num_classes'])
        val_metrics = SegmentationMetrics(config['num_classes'])
        test_metrics = SegmentationMetrics(config['num_classes'])
        
        # 训练循环
        best_iou = 0
        best_epoch = 0
        
        lark_callback.send_msg(
            content=f"{experiment_name} - 训练开始",  # 通知内容
        )
        for epoch in range(config['num_epochs']):       
            # 记录学习率到SwanLab
            swanlab.log({
                'train/learning_rate': optimizer.param_groups[0]['lr']
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
            scheduler.step(val_result['iou'])
            
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
                os.mkdir(f'./checkpoints/{experiment_name}')
            if val_result['iou'] > best_iou:
                best_iou = val_result['iou']
                best_epoch = epoch + 1
                best_model_path = os.path.join(f'./checkpoints/{experiment_name}/', 'best_model.pth')
                save_checkpoint(model, optimizer, epoch, best_iou, best_model_path)
                if val_result['iou'] > 0.7:
                    lark_callback.send_msg(
                        content=f"Current IoU: {val_result['iou']}, New best model is saved",  # 通知内容
                    )
            
            if (epoch + 1) % 10 == 0:
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