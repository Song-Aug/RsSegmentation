import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


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

    def forward(self, logits, targets, weights=None):
        ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')
        
        if self.from_logits:
            probs = torch.softmax(logits, dim=1)
        else:
            probs = logits
        
        pt = probs.gather(1, targets.long().unsqueeze(1)).squeeze(1)
        
        loss = self.alpha * torch.pow(1 - pt, self.gamma) * ce_loss
        
        if weights is not None:
            loss = loss * weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def generate_boundary_and_weight_maps(labels, kernel_size=3, w0=10, sigma=5):
    """
    从分割标签动态生成边界标签和像素权重图。
    Args:
        labels (torch.Tensor): 分割标签 (B, H, W)，值为0或1。
        kernel_size (int): 用于生成边界的核大小。
        w0 (float): 权重图的基础权重。
        sigma (float): 高斯函数中的sigma，控制权重衰减速度。
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 边界标签 (B, 1, H, W) 和 权重图 (B, H, W)。
    """
    # 1. 生成边界图
    labels_float = labels.float().unsqueeze(1)
    padding = (kernel_size - 1) // 2
    
    dilated = F.max_pool2d(labels_float, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-labels_float, kernel_size=kernel_size, stride=1, padding=padding)
    
    boundary = dilated - eroded  # Shape: (B, 1, H, W)
    
    # 2. 生成权重图 (在CPU上使用numpy计算更高效)
    labels_np = labels.cpu().numpy()
    weights = np.zeros_like(labels_np, dtype=np.float32)
    
    for i in range(labels_np.shape[0]):
        # 寻找所有前景(1)和背景(0)像素
        foreground = labels_np[i] > 0
        background = ~foreground
        
        # 如果图像中没有前景或背景，则不进行加权
        if np.sum(foreground) == 0 or np.sum(background) == 0:
            weight_map = np.ones_like(labels_np[i], dtype=np.float32)
        else:
            # 计算每个像素到最近背景像素的距离
            dist_map = distance_transform_edt(foreground)
            # 使用高斯函数和平滑因子计算权重
            weight_map = w0 * np.exp(-((dist_map)**2) / (2 * sigma**2))
        
        weights[i] = weight_map

    # 类别平衡权重 (可选，简单实现)
    class_weights = np.ones_like(labels_np, dtype=np.float32)
    # final_weights = torch.from_numpy(weights + class_weights).to(labels.device)
    final_weights = torch.from_numpy(weights).to(labels.device) + 1 # 基础权重为1

    return boundary.float(), final_weights


def transcc_v2_loss(
    outputs,
    labels,
    seg_weight: float = 1.0,
    boundary_weight: float = 1.5, # 默认提高边界权重
    aux_weight: float = 0.4,
    dice_focal_ratio: float = 0.5,
    bce_dice_ratio: float = 0.5,
):
    """
    TransCC V2 复合损失函数 (优化版)
    - 分割损失: Dice Loss + 加权的 Focal Loss
    - 边界损失: BCE Loss + Dice Loss
    """
    seg_main, boundary_main, seg_aux1, seg_aux2, boundary_aux = outputs

    # --- 实例化损失函数 ---
    loss_focal = FocalLoss(alpha=0.25, gamma=2.0) # 使用带权重的版本
    loss_dice_seg = DiceLoss()
    loss_bce_bd = nn.BCEWithLogitsLoss()
    loss_dice_bd = DiceLoss(from_logits=False)

    labels_long = labels.long()

    # --- 动态生成边界和权重图 ---
    boundary_labels, weight_map = generate_boundary_and_weight_maps(labels)

    # --- 1. 主分割损失 (Focal + Dice)，Focal应用权重 ---
    seg_focal = loss_focal(seg_main, labels_long, weights=weight_map)
    seg_dice = loss_dice_seg(seg_main, labels_long)
    main_seg_loss = (1 - dice_focal_ratio) * seg_focal + dice_focal_ratio * seg_dice

    # --- 2. 辅助分割损失 (Focal + Dice) ---
    aux_losses = []
    for aux_seg in (seg_aux1, seg_aux2):
        # 辅助损失不加权以简化
        aux_focal = F.cross_entropy(aux_seg, labels_long) # 使用简单的交叉熵
        aux_dice = loss_dice_seg(aux_seg, labels_long)
        aux_losses.append((1 - dice_focal_ratio) * aux_focal + dice_focal_ratio * aux_dice)
    
    total_aux_loss = sum(aux_losses) / len(aux_losses) if aux_losses else torch.tensor(0.0).to(seg_main.device)

    # --- 3. 边界损失 (BCE + Dice) ---
    bdy_main_bce = loss_bce_bd(boundary_main, boundary_labels)
    bdy_main_dice = loss_dice_bd(torch.sigmoid(boundary_main), boundary_labels)
    main_bdy_loss = (1 - bce_dice_ratio) * bdy_main_bce + bce_dice_ratio * bdy_main_dice
    
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


def hdnet_loss(outputs, labels, weights=None):
    if weights is None:
        weights = [1.0, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    
    x_seg, x_bd, seg1, seg2, seg3, seg4, seg5, seg6, bd1, bd2, bd3, bd4, bd5, bd6 = outputs
    
    criterion_seg = nn.CrossEntropyLoss()
    criterion_bd = nn.BCEWithLogitsLoss() 
    
    labels_long = labels.long()
    boundary_labels, _ = generate_boundary_and_weight_maps(labels) 

    seg_losses = []
    seg_outputs = [x_seg, seg1, seg2, seg3, seg4, seg5, seg6]
    
    for output in seg_outputs:
        if output.size()[2:] != labels.size()[1:]:
            output = F.interpolate(output, size=labels.size()[1:], mode='bilinear', align_corners=False)
        seg_losses.append(criterion_seg(output, labels_long))

    bd_losses = []
    bd_outputs = [x_bd, bd1, bd2, bd3, bd4, bd5, bd6]

    for output in bd_outputs:
        if output.size()[2:] != boundary_labels.size()[2:]: # 边界标签现在是 (B, 1, H, W)
            output = F.interpolate(output, size=boundary_labels.size()[2:], mode='bilinear', align_corners=False)
        bd_losses.append(criterion_bd(output, boundary_labels))

    all_losses = [seg_losses[0], bd_losses[0]] + seg_losses[1:] + bd_losses[1:]
    total_loss = sum(w * loss for w, loss in zip(weights, all_losses))
    
    main_seg_loss = seg_losses[0]
    main_bd_loss = bd_losses[0]
    
    return total_loss, main_seg_loss, main_bd_loss


def mssdmpanet_y_bce_loss(pred1, pred2, pred3, pred4, pred5, y):
    bce = nn.BCELoss()
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