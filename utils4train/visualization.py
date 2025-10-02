import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_hdnet_sample_images(model, val_loader, device, epoch, num_samples=4):
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


def create_sample_images(model, vis_loader, device, epoch, num_samples=4):
    model.eval()
    
    actual_num_samples = min(num_samples, len(vis_loader.dataset))
    if actual_num_samples == 0:
        return None

    samples_data = []
    
    with torch.no_grad():
        for i, batch in enumerate(vis_loader):
            if i >= actual_num_samples:
                break
            
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            
            main_output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            pred_seg = torch.argmax(main_output, dim=1)
            
            boundary_output = None
            if isinstance(outputs, (list, tuple)) and len(outputs) > 1 and outputs[1].shape[1] == 1:
                boundary_output = outputs[1]

            img_tensor = images[0].cpu()
            original_img = img_tensor[:3] 
            if original_img.min() < 0:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                original_img = original_img * std + mean
            original_img = torch.clamp(original_img, 0, 1).permute(1, 2, 0).numpy()

            label_gray = labels[0].cpu().numpy().astype(np.uint8)
            pred_seg_gray = pred_seg[0].cpu().numpy().astype(np.uint8)
            
            boundary_pred = torch.sigmoid(boundary_output[0, 0]).cpu().numpy() if boundary_output is not None else None

            samples_data.append((original_img, label_gray, pred_seg_gray, boundary_pred))

    if not samples_data:
        return None

    has_boundary = any(s[3] is not None for s in samples_data)
    num_rows = 4 if has_boundary else 3
    num_cols = actual_num_samples
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    fig.suptitle(f'Epoch {epoch} - Model Predictions', fontsize=16, fontweight='bold')
    
    # Ensure axes is a 2D array for consistent indexing
    if num_cols == 1:
        axes = axes.reshape(-1, 1)
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    # --- Set Row Titles ---
    row_titles = ["Original Image", "Ground Truth", "Segmentation", "Boundary"]
    for i in range(num_rows):
        axes[i, 0].set_ylabel(row_titles[i], fontsize=12, fontweight='bold')

    # --- Populate the Grid ---
    for i in range(num_cols): # Iterate through columns (samples)
        original_img, label_gray, pred_seg_gray, boundary_pred = samples_data[i]
        
        # Column 1: Original Image
        axes[0, i].imshow(original_img)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        # Column 2: Ground Truth
        axes[1, i].imshow(label_gray, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

        # Column 3: Segmentation Prediction
        axes[2, i].imshow(pred_seg_gray, cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_xticks([])
        axes[2, i].set_yticks([])

        # Column 4: Boundary Prediction (if available)
        if has_boundary:
            axes[3, i].imshow(boundary_pred, cmap='hot', vmin=0, vmax=1)
            axes[3, i].set_xticks([])
            axes[3, i].set_yticks([])
            
    plt.tight_layout(rect=[0.05, 0, 1, 0.95]) # Adjust layout
    
    return fig