"""
UNetFormer 损失函数模块

UNetFormer 使用的复合损失函数:
- 主分割损失: CE + Dice
- 辅助损失: CE + Dice
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


def unetformer_loss(
    outputs,
    labels,
    seg_weight: float = 1.0,
    aux_weight: float = 0.4,
    dice_ratio: float = 0.5,
):
    """
    UNetFormer 复合损失函数

    Args:
        outputs: 模型输出 (main_pred, aux_pred1, aux_pred2, ...) 或单个输出
        labels: 分割标签
        seg_weight: 主分割损失权重
        aux_weight: 辅助损失权重
        dice_ratio: Dice/CE 比例

    Returns:
        total_loss, main_seg_loss, aux_loss
    """
    if isinstance(outputs, (list, tuple)):
        main_pred = outputs[0]
        aux_preds = outputs[1:] if len(outputs) > 1 else []
    else:
        main_pred = outputs
        aux_preds = []

    labels_long = labels.long()

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()

    # 主分割损失
    main_ce = ce_loss(main_pred, labels_long)
    main_dice = dice_loss(main_pred, labels_long)
    main_seg_loss = (1 - dice_ratio) * main_ce + dice_ratio * main_dice

    # 辅助损失
    aux_losses = []
    for aux_pred in aux_preds:
        if aux_pred.size()[2:] != labels.size()[1:]:
            aux_pred = F.interpolate(aux_pred, size=labels.size()[1:], mode="bilinear", align_corners=False)
        aux_ce = ce_loss(aux_pred, labels_long)
        aux_dice = dice_loss(aux_pred, labels_long)
        aux_losses.append((1 - dice_ratio) * aux_ce + dice_ratio * aux_dice)

    if aux_losses:
        aux_loss = sum(aux_losses) / len(aux_losses)
    else:
        aux_loss = torch.tensor(0.0, device=main_pred.device)

    total_loss = seg_weight * main_seg_loss + aux_weight * aux_loss

    return total_loss, main_seg_loss, aux_loss


if __name__ == "__main__":
    # 损失函数测试
    batch_size = 2
    num_classes = 2
    height, width = 512, 512

    main_pred = torch.randn(batch_size, num_classes, height, width)
    aux_pred = torch.randn(batch_size, num_classes, height // 2, width // 2)
    outputs = (main_pred, aux_pred)

    labels = torch.randint(0, 2, (batch_size, height, width)).long()

    total_loss, seg_loss, aux_loss = unetformer_loss(outputs, labels)
    print(f"总损失: {total_loss.item():.4f}")
    print(f"主分割损失: {seg_loss.item():.4f}")
    print(f"辅助损失: {aux_loss.item():.4f}")
