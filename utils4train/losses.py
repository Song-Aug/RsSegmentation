import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice损失函数，用于分割任务，能有效处理类别不均衡"""
    def __init__(self, smooth=1e-6, from_logits=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        
        if self.from_logits:
            probs = torch.softmax(logits, dim=1)
        else:
            probs = logits

        if num_classes > 1:
            targets_onehot = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()
        else: # for binary case
            targets_onehot = targets.unsqueeze(1).float()
            
        intersection = torch.sum(probs * targets_onehot, dim=(0, 2, 3))
        union = torch.sum(probs, dim=(0, 2, 3)) + torch.sum(targets_onehot, dim=(0, 2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 返回1 - dice_score的平均值
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss，用于处理难易样本不均衡问题"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', from_logits=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.from_logits = from_logits
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce_loss(logits, targets.long())
        
        if self.from_logits:
            probs = torch.softmax(logits, dim=1)
        else:
            probs = logits
        
        pt = probs.gather(1, targets.long().unsqueeze(1)).squeeze(1)
        
        loss = self.alpha * torch.pow(1 - pt, self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 现有函数
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def generate_boundary_labels(labels, kernel_size=3):
    """
    从分割标签动态生成边界标签。
    Args:
        labels (torch.Tensor): 分割标签 (B, H, W)，值为0或1。
        kernel_size (int): 用于膨胀和腐蚀的核大小。
    Returns:
        torch.Tensor: 边界标签 (B, 1, H, W)，值为0或1。
    """
    labels_float = labels.float().unsqueeze(1)
    padding = (kernel_size - 1) // 2
    
    dilated = F.max_pool2d(labels_float, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-labels_float, kernel_size=kernel_size, stride=1, padding=padding)
    
    boundary = dilated - eroded
    return boundary.float()


def transcc_v2_loss(
    outputs,
    labels,
    seg_weight: float = 1.0,
    boundary_weight: float = 1.0,
    aux_weight: float = 0.4,
    dice_focal_ratio: float = 0.5, # Dice和Focal在分割损失中的比例
    bce_dice_ratio: float = 0.5,   # BCE和Dice在边界损失中的比例
):
    """
    TransCC V2 复合损失函数 (优化版)
    - 分割损失: Dice Loss + Focal Loss
    - 边界损失: BCE Loss + Dice Loss
    """
    seg_main, boundary_main, seg_aux1, seg_aux2, boundary_aux = outputs

    # --- 实例化损失函数 ---
    loss_focal = FocalLoss(alpha=0.25, gamma=2.0)
    loss_dice_seg = DiceLoss()
    loss_bce_bd = nn.BCEWithLogitsLoss()
    loss_dice_bd = DiceLoss(from_logits=False) # 边界是sigmoid输出，不是logits

    labels_long = labels.long()

    # --- 1. 主分割损失 (Focal + Dice) ---
    seg_focal = loss_focal(seg_main, labels_long)
    seg_dice = loss_dice_seg(seg_main, labels_long)
    main_seg_loss = (1 - dice_focal_ratio) * seg_focal + dice_focal_ratio * seg_dice

    # --- 2. 辅助分割损失 (Focal + Dice) ---
    aux_losses = []
    for aux_seg in (seg_aux1, seg_aux2):
        aux_focal = loss_focal(aux_seg, labels_long)
        aux_dice = loss_dice_seg(aux_seg, labels_long)
        aux_losses.append((1 - dice_focal_ratio) * aux_focal + dice_focal_ratio * aux_dice)
    
    if aux_losses:
        total_aux_loss = sum(aux_losses) / len(aux_losses)
    else:
        total_aux_loss = torch.zeros(1, device=seg_main.device, dtype=seg_main.dtype)

    # --- 3. 边界损失 (BCE + Dice) ---
    boundary_labels = generate_boundary_labels(labels) # (B, 1, H, W)
    
    # 主边界损失
    bdy_main_bce = loss_bce_bd(boundary_main, boundary_labels)
    bdy_main_dice = loss_dice_bd(torch.sigmoid(boundary_main), boundary_labels)
    main_bdy_loss = (1 - bce_dice_ratio) * bdy_main_bce + bce_dice_ratio * bdy_main_dice
    
    # 辅助边界损失
    bdy_aux_bce = loss_bce_bd(boundary_aux, boundary_labels)
    bdy_aux_dice = loss_dice_bd(torch.sigmoid(boundary_aux), boundary_labels)
    aux_bdy_loss = (1 - bce_dice_ratio) * bdy_aux_bce + bce_dice_ratio * bdy_aux_dice

    total_boundary_loss = main_bdy_loss + aux_bdy_loss
    
    # --- 4. 计算总损失 ---
    total_loss = (
        seg_weight * main_seg_loss + 
        boundary_weight * total_boundary_loss + 
        aux_weight * total_aux_loss
    )

    return total_loss, main_seg_loss, total_boundary_loss


# =================================================================
# 其他模型的损失函数（保持不变）
# =================================================================
def hdnet_loss(outputs, labels, weights=None):
    """
    HDNet的复合损失函数
    """
    if weights is None:
        weights = [1.0, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    
    x_seg, x_bd, seg1, seg2, seg3, seg4, seg5, seg6, bd1, bd2, bd3, bd4, bd5, bd6 = outputs
    
    criterion_seg = nn.CrossEntropyLoss()
    criterion_bd = nn.BCEWithLogitsLoss() 
    
    labels = labels.long()
    boundary_labels = generate_boundary_labels(labels).float() 

    seg_losses = []
    seg_outputs = [x_seg, seg1, seg2, seg3, seg4, seg5, seg6]
    
    for output in seg_outputs:
        if output.size()[2:] != labels.size()[1:]:
            output = F.interpolate(output, size=labels.size()[1:], mode='bilinear', align_corners=False)
        seg_losses.append(criterion_seg(output, labels))

    bd_losses = []
    bd_outputs = [x_bd, bd1, bd2, bd3, bd4, bd5, bd6]

    for output in bd_outputs:
        if output.size()[2:] != boundary_labels.size()[1:]:
            output = F.interpolate(output, size=boundary_labels.size()[1:], mode='bilinear', align_corners=False)
        
        bd_labels_reshaped = boundary_labels.unsqueeze(1) if boundary_labels.dim() == 3 else boundary_labels
        bd_losses.append(criterion_bd(output, bd_labels_reshaped))

    all_losses = [seg_losses[0], bd_losses[0]] + seg_losses[1:] + bd_losses[1:]
    total_loss = sum(w * loss for w, loss in zip(weights, all_losses))
    
    main_seg_loss = seg_losses[0]
    main_bd_loss = bd_losses[0]
    
    return total_loss, main_seg_loss, main_bd_loss


def mssdmpanet_y_bce_loss(pred1, pred2, pred3, pred4, pred5, y):
    bce = nn.BCELoss()
    # 确保标签尺寸与预测匹配
    loss1 = bce(pred1, F.interpolate(y, size=pred1.shape[2:], mode='nearest'))
    loss2 = bce(pred2, F.interpolate(y, size=pred2.shape[2:], mode='nearest'))
    loss3 = bce(pred3, F.interpolate(y, size=pred3.shape[2:], mode='nearest'))
    loss4 = bce(pred4, F.interpolate(y, size=pred4.shape[2:], mode='nearest'))
    loss5 = bce(pred5, F.interpolate(y, size=pred5.shape[2:], mode='nearest'))
    return loss1 + loss2 + loss3 + loss4 + loss5

class MSSDMPA_IoU(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, y_true, y_pred):
        y_pred = (y_pred > self.threshold).float()
        intersection = torch.sum(y_true * y_pred)
        union = torch.sum(y_true) + torch.sum(y_pred) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        recall = intersection / (torch.sum(y_true) + 1e-7)
        precision = intersection / (torch.sum(y_pred) + 1e-7)
        return [recall, precision, iou, iou]

def mssdmpanet_dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)