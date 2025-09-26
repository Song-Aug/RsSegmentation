import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_boundary_labels(labels):
    """
    从分割标签动态生成边界标签。
    
    Args:
        labels (torch.Tensor): 分割标签 (B, H, W)，值为0或1。
        
    Returns:
        torch.Tensor: 边界标签 (B, H, W)，值为0或1。
    """
    # 确保标签是浮点型以便进行卷积操作
    labels_float = labels.float().unsqueeze(1)  # (B, 1, H, W)
    
    # 使用一个简单的3x3核进行最大池化和最小池化
    kernel = torch.ones((1, 1, 3, 3), device=labels.device, dtype=torch.float32)
    
    # 使用padding='same'需要PyTorch 1.9+，这里手动padding
    padded_labels = F.pad(labels_float, (1, 1, 1, 1), mode='replicate')
    
    # 最大池化模拟膨胀
    dilated = F.conv2d(padded_labels, kernel, padding=0) > 0
    dilated = dilated.squeeze(1).long() # (B, H, W)
    
    # 最小池化模拟腐蚀
    eroded = F.conv2d(padded_labels, kernel, padding=0) == 9 # 只有当3x3窗口内全是1时，结果才是9
    eroded = eroded.squeeze(1).long() # (B, H, W)
    
    # 边界 = 膨胀结果 - 腐蚀结果
    boundary = dilated - eroded
    return boundary.long()


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
        weights = [1.0, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    
    # 解包输出
    x_seg, x_bd, seg1, seg2, seg3, seg4, seg5, seg6, bd1, bd2, bd3, bd4, bd5, bd6 = outputs
    
    # 使用 CrossEntropyLoss
    criterion_seg = nn.CrossEntropyLoss()
    # 边界损失使用BCE，因为边界预测是单通道的
    criterion_bd = nn.BCEWithLogitsLoss() 
    
    # 确保分割标签为 long 类型
    labels = labels.long()
    
    # 动态生成边界标签
    boundary_labels = generate_boundary_labels(labels).float() # BCE需要float类型的标签

    # --- 1. 计算分割损失 ---
    seg_losses = []
    seg_outputs = [x_seg, seg1, seg2, seg3, seg4, seg5, seg6]
    
    for output in seg_outputs:
        if output.size()[2:] != labels.size()[1:]:
            output = F.interpolate(output, size=labels.size()[1:], mode='bilinear', align_corners=False)
        seg_losses.append(criterion_seg(output, labels))

    # --- 2. 计算边界损失 ---
    bd_losses = []
    bd_outputs = [x_bd, bd1, bd2, bd3, bd4, bd5, bd6]

    for output in bd_outputs:
        # 边界预测是单通道的，所以标签也需要是 (B, 1, H, W)
        if output.size()[2:] != boundary_labels.size()[1:]:
            output = F.interpolate(output, size=boundary_labels.size()[1:], mode='bilinear', align_corners=False)
        
        # 确保标签有正确的形状
        bd_labels_reshaped = boundary_labels.unsqueeze(1)
        bd_losses.append(criterion_bd(output, bd_labels_reshaped))

    # --- 3. 组合并加权 ---
    # 权重顺序：x_seg, x_bd, seg1-6, bd1-6
    all_losses = [seg_losses[0], bd_losses[0]] + seg_losses[1:] + bd_losses[1:]
    
    # 计算加权总损失
    total_loss = sum(w * loss for w, loss in zip(weights, all_losses))
    
    # 返回主要损失用于监控
    main_seg_loss = seg_losses[0]
    main_bd_loss = bd_losses[0]
    
    return total_loss, main_seg_loss, main_bd_loss

# =================================================================
# MSSDMPA-Net Losses and Metrics
# =================================================================

def mssdmpanet_y_bce_loss(pred1, pred2, pred3, pred4, pred5, y):
    """MSSDMPA-Net的BCE损失函数"""
    bce = nn.BCELoss()
    loss1 = bce(pred1, y)
    loss2 = bce(pred2, y)
    loss3 = bce(pred3, y)
    loss4 = bce(pred4, y)
    loss5 = bce(pred5, y)
    return loss1 + loss2 + loss3 + loss4 + loss5

class MSSDMPA_IoU(object):
    """MSSDMPA-Net的IoU计算类"""
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, y_true, y_pred):
        y_pred = (y_pred > self.threshold).float()
        
        intersection = torch.sum(y_true * y_pred)
        union = torch.sum(y_true) + torch.sum(y_pred) - intersection
        
        iou = (intersection + 1e-7) / (union + 1e-7)
        
        recall = intersection / (torch.sum(y_true) + 1e-7)
        precision = intersection / (torch.sum(y_pred) + 1e-7)
        
        return [recall, precision, iou, iou] # 返回iou两次以匹配原始代码

def mssdmpanet_dice_coeff(y_true, y_pred):
    """MSSDMPA-Net的Dice系数计算"""
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
