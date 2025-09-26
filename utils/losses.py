import torch
import torch.nn as nn

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
