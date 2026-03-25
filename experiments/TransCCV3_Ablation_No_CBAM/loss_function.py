"""
损失函数模块

TransCCV3 使用的复合损失函数:
- 分割损失: Dice Loss + 加权的 Focal Loss
- 边界损失: BCE Loss + Dice Loss
"""

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
        else:
            targets_onehot = targets.unsqueeze(1).float()

        intersection = torch.sum(probs * targets_onehot, dim=(0, 2, 3))
        union = torch.sum(probs, dim=(0, 2, 3)) + torch.sum(targets_onehot, dim=(0, 2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss，用于处理难易样本不均衡问题"""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean", from_logits=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, logits, targets, weights=None):
        ce_loss = F.cross_entropy(logits, targets.long(), reduction="none")

        if self.from_logits:
            probs = torch.softmax(logits, dim=1)
        else:
            probs = logits

        pt = probs.gather(1, targets.long().unsqueeze(1)).squeeze(1)
        loss = self.alpha * torch.pow(1 - pt, self.gamma) * ce_loss

        if weights is not None:
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def generate_boundary_and_weight_maps(labels, kernel_size=5, w0=10, sigma=5):
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
    labels_float = labels.float().unsqueeze(1)
    padding = (kernel_size - 1) // 2

    dilated = F.max_pool2d(labels_float, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-labels_float, kernel_size=kernel_size, stride=1, padding=padding)
    boundary = dilated - eroded

    labels_np = labels.cpu().numpy()
    weights = np.zeros_like(labels_np, dtype=np.float32)

    for i in range(labels_np.shape[0]):
        foreground = labels_np[i] > 0
        background = ~foreground

        if np.sum(foreground) == 0 or np.sum(background) == 0:
            weight_map = np.ones_like(labels_np[i], dtype=np.float32)
        else:
            dist_map = distance_transform_edt(foreground)
            weight_map = w0 * np.exp(-((dist_map) ** 2) / (2 * sigma ** 2))

        weights[i] = weight_map

    final_weights = torch.from_numpy(weights).to(labels.device) + 1

    return boundary.float(), final_weights


def transcc_v2_loss(
    outputs,
    labels,
    seg_weight: float = 1.0,
    boundary_weight: float = 1.5,
    aux_weight: float = 0.4,
    dice_focal_ratio: float = 0.5,
    bce_dice_ratio: float = 0.5,
):
    """
    TransCC V3 复合损失函数

    Args:
        outputs: 模型输出 (seg_main, boundary_main, seg_aux1, seg_aux2, boundary_aux)
        labels: 分割标签
        seg_weight: 主分割损失权重
        boundary_weight: 边界损失权重
        aux_weight: 辅助损失权重
        dice_focal_ratio: Dice/Focal 比例
        bce_dice_ratio: BCE/Dice 比例

    Returns:
        total_loss, main_seg_loss, total_boundary_loss
    """
    seg_main, boundary_main, seg_aux1, seg_aux2, boundary_aux = outputs

    loss_focal = FocalLoss(alpha=0.25, gamma=2.0)
    loss_dice_seg = DiceLoss()
    loss_bce_bd = nn.BCEWithLogitsLoss()
    loss_dice_bd = DiceLoss(from_logits=False)

    labels_long = labels.long()
    boundary_labels, weight_map = generate_boundary_and_weight_maps(labels)

    # 主分割损失
    seg_focal = loss_focal(seg_main, labels_long, weights=weight_map)
    seg_dice = loss_dice_seg(seg_main, labels_long)
    main_seg_loss = (1 - dice_focal_ratio) * seg_focal + dice_focal_ratio * seg_dice

    # 辅助分割损失
    aux_losses = []
    for aux_seg in (seg_aux1, seg_aux2):
        aux_focal = F.cross_entropy(aux_seg, labels_long)
        aux_dice = loss_dice_seg(aux_seg, labels_long)
        aux_losses.append((1 - dice_focal_ratio) * aux_focal + dice_focal_ratio * aux_dice)

    total_aux_loss = (
        sum(aux_losses) / len(aux_losses)
        if aux_losses
        else torch.tensor(0.0).to(seg_main.device)
    )

    # 边界损失
    bdy_main_bce = loss_bce_bd(boundary_main, boundary_labels)
    bdy_main_dice = loss_dice_bd(torch.sigmoid(boundary_main), boundary_labels)
    main_bdy_loss = (1 - bce_dice_ratio) * bdy_main_bce + bce_dice_ratio * bdy_main_dice

    bdy_aux_bce = loss_bce_bd(boundary_aux, boundary_labels)
    bdy_aux_dice = loss_dice_bd(torch.sigmoid(boundary_aux), boundary_labels)
    aux_bdy_loss = (1 - bce_dice_ratio) * bdy_aux_bce + bce_dice_ratio * bdy_aux_dice

    total_boundary_loss = main_bdy_loss + aux_bdy_loss

    # 总损失
    total_loss = (
        seg_weight * main_seg_loss
        + boundary_weight * total_boundary_loss
        + aux_weight * total_aux_loss
    )

    return total_loss, main_seg_loss, total_boundary_loss


if __name__ == "__main__":
    # 损失函数测试
    batch_size = 2
    num_classes = 2
    height, width = 512, 512

    # 模拟输出
    seg_main = torch.randn(batch_size, num_classes, height, width)
    boundary_main = torch.randn(batch_size, 1, height, width)
    seg_aux1 = torch.randn(batch_size, num_classes, height, width)
    seg_aux2 = torch.randn(batch_size, num_classes, height, width)
    boundary_aux = torch.randn(batch_size, 1, height, width)
    outputs = (seg_main, boundary_main, seg_aux1, seg_aux2, boundary_aux)

    # 模拟标签
    labels = torch.randint(0, 2, (batch_size, height, width)).long()

    # 计算损失
    total_loss, seg_loss, bd_loss = transcc_v2_loss(outputs, labels)
    print(f"总损失: {total_loss.item():.4f}")
    print(f"分割损失: {seg_loss.item():.4f}")
    print(f"边界损失: {bd_loss.item():.4f}")
