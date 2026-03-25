"""
HDNet 损失函数模块

HDNet 使用的复合损失函数:
- 分割损失: CrossEntropyLoss
- 边界损失: BCEWithLogitsLoss
- 深度监督
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np


def generate_boundary_and_weight_maps(labels, kernel_size=5, w0=10, sigma=5):
    """
    从分割标签动态生成边界标签和像素权重图。
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


def hdnet_loss(outputs, labels, weights=None):
    """
    HDNet 复合损失函数

    Args:
        outputs: 模型输出 (x_seg, x_bd, seg1, seg2, seg3, seg4, seg5, seg6, bd1, bd2, bd3, bd4, bd5, bd6)
        labels: 分割标签
        weights: 各输出权重

    Returns:
        total_loss, main_seg_loss, main_bd_loss
    """
    if weights is None:
        weights = [1.0, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

    x_seg, x_bd, seg1, seg2, seg3, seg4, seg5, seg6, bd1, bd2, bd3, bd4, bd5, bd6 = outputs

    criterion_seg = nn.CrossEntropyLoss()
    criterion_bd = nn.BCEWithLogitsLoss()

    labels_long = labels.long()
    boundary_labels, _ = generate_boundary_and_weight_maps(labels)

    # 分割损失
    seg_losses = []
    seg_outputs = [x_seg, seg1, seg2, seg3, seg4, seg5, seg6]

    for output in seg_outputs:
        if output.size()[2:] != labels.size()[1:]:
            output = F.interpolate(output, size=labels.size()[1:], mode="bilinear", align_corners=False)
        seg_losses.append(criterion_seg(output, labels_long))

    # 边界损失
    bd_losses = []
    bd_outputs = [x_bd, bd1, bd2, bd3, bd4, bd5, bd6]

    for output in bd_outputs:
        if output.size()[2:] != boundary_labels.size()[2:]:
            output = F.interpolate(output, size=boundary_labels.size()[2:], mode="bilinear", align_corners=False)
        bd_losses.append(criterion_bd(output, boundary_labels))

    all_losses = [seg_losses[0], bd_losses[0]] + seg_losses[1:] + bd_losses[1:]
    total_loss = sum(w * loss for w, loss in zip(weights, all_losses))

    main_seg_loss = seg_losses[0]
    main_bd_loss = bd_losses[0]

    return total_loss, main_seg_loss, main_bd_loss


if __name__ == "__main__":
    # 损失函数测试
    batch_size = 2
    num_classes = 2
    height, width = 512, 512

    # 模拟 HDNet 输出 (14个输出)
    outputs = [
        torch.randn(batch_size, num_classes, height, width),  # x_seg
        torch.randn(batch_size, 1, height, width),  # x_bd
    ] + [torch.randn(batch_size, num_classes, height // (2 ** i), width // (2 ** i)) for i in range(6) for _ in range(2)]  # 简化测试

    labels = torch.randint(0, 2, (batch_size, height, width)).long()

    total_loss, seg_loss, bd_loss = hdnet_loss(outputs, labels)
    print(f"总损失: {total_loss.item():.4f}")
    print(f"分割损失: {seg_loss.item():.4f}")
    print(f"边界损失: {bd_loss.item():.4f}")
