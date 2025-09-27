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


def _dice_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """计算 softmax Dice 损失。"""
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    labels_onehot = F.one_hot(labels.long(), num_classes).permute(0, 3, 1, 2).float()
    intersection = torch.sum(probs * labels_onehot, dim=(0, 2, 3))
    union = torch.sum(probs + labels_onehot, dim=(0, 2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def transcc_v2_loss(
    outputs,
    labels,
    dice_weight: float = 0.1,
    aux_weight: float = 0.4,
    boundary_weight: float = 0.5,
):
    """TransCC V2 复合损失函数。"""

    seg_main, boundary_main, seg_aux1, seg_aux2, boundary_aux = outputs

    criterion_ce = nn.CrossEntropyLoss()
    criterion_bd = nn.BCEWithLogitsLoss()

    labels_long = labels.long()

    # 主分割损失（CE + Dice）
    seg_ce = criterion_ce(seg_main, labels_long)
    seg_dice = _dice_loss_from_logits(seg_main, labels_long)
    seg_loss = seg_ce + dice_weight * seg_dice

    # 辅助分割损失
    aux_losses = []
    for aux_seg in (seg_aux1, seg_aux2):
        aux_losses.append(criterion_ce(aux_seg, labels_long))
    if aux_losses:
        aux_loss = sum(aux_losses) / len(aux_losses)
    else:
        aux_loss = torch.zeros(1, device=seg_main.device, dtype=seg_main.dtype)

    # 边界损失
    boundary_labels = generate_boundary_labels(labels).float().unsqueeze(1)
    boundary_losses = []
    for bd_out in (boundary_main, boundary_aux):
        boundary_losses.append(criterion_bd(bd_out, boundary_labels))
    boundary_loss = sum(boundary_losses) / len(boundary_losses)

    total_loss = seg_loss + aux_weight * aux_loss + boundary_weight * boundary_loss

    return total_loss, seg_loss, boundary_loss

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
