import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_hdnet_sample_images(model, val_loader, device, epoch, num_samples=4):
    """创建HDNet样本预测图像用于SwanLab可视化（包含分割和边界）"""
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
