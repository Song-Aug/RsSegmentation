"""
MSSDMPA-Net 损失函数模块

MSSDMPA-Net 使用的多尺度 BCE 损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mssdmpanet_bce_loss(pred1, pred2, pred3, pred4, pred5, y):
    """
    MSSDMPA-Net 的多尺度损失函数

    Args:
        pred1-pred5: 5个尺度的预测输出
        y: 标签

    Returns:
        总损失
    """
    bce_with_logits = nn.BCEWithLogitsLoss()

    loss1 = bce_with_logits(pred1, F.interpolate(y, size=pred1.shape[2:], mode="nearest"))
    loss2 = bce_with_logits(pred2, F.interpolate(y, size=pred2.shape[2:], mode="nearest"))
    loss3 = bce_with_logits(pred3, F.interpolate(y, size=pred3.shape[2:], mode="nearest"))
    loss4 = bce_with_logits(pred4, F.interpolate(y, size=pred4.shape[2:], mode="nearest"))
    loss5 = bce_with_logits(pred5, F.interpolate(y, size=pred5.shape[2:], mode="nearest"))

    return loss1 + loss2 + loss3 + loss4 + loss5


def mssdmpanet_dice_coeff(y_true, y_pred):
    """
    Dice 系数计算

    Args:
        y_true: 真实标签
        y_pred: 预测结果 (logits)

    Returns:
        Dice 系数
    """
    smooth = 1.0
    y_true_f = y_true.flatten()
    y_pred_f = torch.sigmoid(y_pred).flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


class MSSDMPA_IoU:
    """MSSDMPA-Net IoU 计算类"""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, y_true, y_pred):
        y_pred_probs = torch.sigmoid(y_pred)
        y_pred_binary = (y_pred_probs > self.threshold).float()

        intersection = torch.sum(y_true * y_pred_binary)
        union = torch.sum(y_true) + torch.sum(y_pred_binary) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        recall = intersection / (torch.sum(y_true) + 1e-7)
        precision = intersection / (torch.sum(y_pred_binary) + 1e-7)
        return recall, precision, iou, iou


def mssdmpanet_loss(outputs, labels):
    """
    MSSDMPA-Net 损失函数封装

    Args:
        outputs: 模型输出 (pred1, pred2, pred3, pred4, pred5)
        labels: 分割标签 (B, 1, H, W) 或 (B, H, W)

    Returns:
        total_loss, main_loss, aux_loss
    """
    pred1, pred2, pred3, pred4, pred5 = outputs

    # 确保标签格式正确
    if labels.dim() == 3:
        labels = labels.unsqueeze(1).float()
    else:
        labels = labels.float()

    # 主损失使用 pred5 (最高分辨率)
    main_loss = F.binary_cross_entropy_with_logits(
        pred5, F.interpolate(labels, size=pred5.shape[2:], mode="nearest")
    )

    # 多尺度损失
    total_loss = mssdmpanet_bce_loss(pred1, pred2, pred3, pred4, pred5, labels)

    # 辅助损失 (其他尺度的平均)
    aux_loss = (
        F.binary_cross_entropy_with_logits(pred1, F.interpolate(labels, size=pred1.shape[2:], mode="nearest")) +
        F.binary_cross_entropy_with_logits(pred2, F.interpolate(labels, size=pred2.shape[2:], mode="nearest")) +
        F.binary_cross_entropy_with_logits(pred3, F.interpolate(labels, size=pred3.shape[2:], mode="nearest")) +
        F.binary_cross_entropy_with_logits(pred4, F.interpolate(labels, size=pred4.shape[2:], mode="nearest"))
    ) / 4

    return total_loss, main_loss, aux_loss


if __name__ == "__main__":
    # 损失函数测试
    batch_size = 2
    height, width = 512, 512

    # 模拟多尺度输出
    pred1 = torch.randn(batch_size, 1, height // 16, width // 16)
    pred2 = torch.randn(batch_size, 1, height // 8, width // 8)
    pred3 = torch.randn(batch_size, 1, height // 4, width // 4)
    pred4 = torch.randn(batch_size, 1, height // 2, width // 2)
    pred5 = torch.randn(batch_size, 1, height, width)
    outputs = (pred1, pred2, pred3, pred4, pred5)

    labels = torch.randint(0, 2, (batch_size, 1, height, width)).float()

    total_loss, main_loss, aux_loss = mssdmpanet_loss(outputs, labels)
    print(f"总损失: {total_loss.item():.4f}")
    print(f"主损失: {main_loss.item():.4f}")
    print(f"辅助损失: {aux_loss.item():.4f}")
