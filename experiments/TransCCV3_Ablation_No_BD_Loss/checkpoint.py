"""
检查点管理模块

保存和加载模型检查点
"""

import torch
import os


def save_checkpoint(model, optimizer, epoch, best_iou, save_path):
    """
    保存模型检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮次
        best_iou: 最佳IoU
        save_path: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': best_iou,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model, checkpoint_path, optimizer=None, device='cpu'):
    """
    加载模型检查点

    Args:
        model: 模型
        checkpoint_path: 检查点路径
        optimizer: 优化器（可选）
        device: 设备

    Returns:
        model, optimizer, epoch, best_iou
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    best_iou = checkpoint.get('best_iou', 0.0)

    return model, optimizer, epoch, best_iou


if __name__ == "__main__":
    print("检查点管理模块")
