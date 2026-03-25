"""
STDSNet 损失函数模块

STDSNet 使用的复合损失函数:
- 全局分割损失: CE + Dice
- 形状分割损失: CE + Dice
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice损失函数"""

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


def stdsnet_loss(
    outputs,
    labels,
    seg_weight: float = 1.0,
    shape_weight: float = 0.4,
    dice_ratio: float = 0.5,
):
    """
    STDSNet 复合损失函数

    Args:
        outputs: 模型输出 (global_pred, shape_pred)
        labels: 分割标签
        seg_weight: 主分割损失权重
        shape_weight: 形状损失权重
        dice_ratio: Dice/CE 比例

    Returns:
        total_loss, main_seg_loss, shape_loss
    """
    global_pred, shape_pred = outputs
    labels_long = labels.long()

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()

    # 全局分割损失
    seg_ce = ce_loss(global_pred, labels_long)
    seg_dice = dice_loss(global_pred, labels_long)
    main_seg_loss = (1 - dice_ratio) * seg_ce + dice_ratio * seg_dice

    # 形状分割损失
    shape_ce = ce_loss(shape_pred, labels_long)
    shape_dice = dice_loss(shape_pred, labels_long)
    shape_seg_loss = (1 - dice_ratio) * shape_ce + dice_ratio * shape_dice

    total_loss = seg_weight * main_seg_loss + shape_weight * shape_seg_loss

    return total_loss, main_seg_loss, shape_seg_loss


if __name__ == "__main__":
    # 损失函数测试
    batch_size = 2
    num_classes = 2
    height, width = 512, 512

    global_pred = torch.randn(batch_size, num_classes, height, width)
    shape_pred = torch.randn(batch_size, num_classes, height, width)
    outputs = (global_pred, shape_pred)

    labels = torch.randint(0, 2, (batch_size, height, width)).long()

    total_loss, seg_loss, shape_loss = stdsnet_loss(outputs, labels)
    print(f"总损失: {total_loss.item():.4f}")
    print(f"全局分割损失: {seg_loss.item():.4f}")
    print(f"形状损失: {shape_loss.item():.4f}")
