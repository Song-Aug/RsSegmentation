"""
评估指标模块

计算分割任务的评估指标：IoU, F1, Precision, Recall 等
"""

import numpy as np
import torch
from typing import Dict, Union, Optional


class SegmentationMetrics:
    """分割任务评价指标计算类"""

    def __init__(self, num_classes: int = 2, ignore_index: Optional[int] = None):
        """
        Args:
            num_classes (int): 类别数量
            ignore_index (int, optional): 忽略的类别索引
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """重置混淆矩阵"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray]):
        """
        更新混淆矩阵

        Args:
            pred: 预测结果，形状为 (N,) 或 (B, H, W)
            target: 真实标签，形状为 (N,) 或 (B, H, W)
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        pred = pred.flatten()
        target = target.flatten()

        if self.ignore_index is not None:
            mask = target != self.ignore_index
        else:
            mask = (target >= 0) & (target < self.num_classes)

        pred = pred[mask]
        target = target[mask]
        pred = np.clip(pred, 0, self.num_classes - 1)

        for i in range(len(target)):
            self.confusion_matrix[target[i], pred[i]] += 1

    def get_confusion_matrix(self) -> np.ndarray:
        """获取混淆矩阵"""
        return self.confusion_matrix.copy()

    def get_metrics(self) -> Dict[str, float]:
        """计算所有指标"""
        if self.num_classes == 2:
            return self._get_binary_metrics()
        else:
            return self._get_multiclass_metrics()

    def _get_binary_metrics(self) -> Dict[str, float]:
        """计算二分类指标"""
        tn, fp, fn, tp = self.confusion_matrix.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        dice = f1

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'dice': dice,
            'iou': iou,
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn)
        }

    def _get_multiclass_metrics(self) -> Dict[str, float]:
        """计算多分类指标"""
        metrics = {}

        total_correct = np.trace(self.confusion_matrix)
        total_samples = np.sum(self.confusion_matrix)
        metrics['accuracy'] = total_correct / total_samples if total_samples > 0 else 0.0

        class_metrics = {'precision': [], 'recall': [], 'f1': [], 'iou': []}

        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            fn = np.sum(self.confusion_matrix[i, :]) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            class_metrics['precision'].append(precision)

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            class_metrics['recall'].append(recall)

            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            class_metrics['f1'].append(f1)

            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            class_metrics['iou'].append(iou)

        metrics['macro_precision'] = np.mean(class_metrics['precision'])
        metrics['macro_recall'] = np.mean(class_metrics['recall'])
        metrics['macro_f1'] = np.mean(class_metrics['f1'])
        metrics['macro_iou'] = np.mean(class_metrics['iou'])

        class_support = np.sum(self.confusion_matrix, axis=1)
        total_support = np.sum(class_support)

        if total_support > 0:
            weights = class_support / total_support
            metrics['weighted_precision'] = np.average(class_metrics['precision'], weights=weights)
            metrics['weighted_recall'] = np.average(class_metrics['recall'], weights=weights)
            metrics['weighted_f1'] = np.average(class_metrics['f1'], weights=weights)
            metrics['weighted_iou'] = np.average(class_metrics['iou'], weights=weights)
        else:
            metrics['weighted_precision'] = 0.0
            metrics['weighted_recall'] = 0.0
            metrics['weighted_f1'] = 0.0
            metrics['weighted_iou'] = 0.0

        for i in range(self.num_classes):
            metrics[f'class_{i}_precision'] = class_metrics['precision'][i]
            metrics[f'class_{i}_recall'] = class_metrics['recall'][i]
            metrics[f'class_{i}_f1'] = class_metrics['f1'][i]
            metrics[f'class_{i}_iou'] = class_metrics['iou'][i]

        return metrics

    def get_iou_per_class(self) -> np.ndarray:
        """获取每个类别的IoU"""
        ious = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            fn = np.sum(self.confusion_matrix[i, :]) - tp
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            ious.append(iou)
        return np.array(ious)

    def get_mean_iou(self) -> float:
        """获取平均IoU"""
        return np.mean(self.get_iou_per_class())


def calculate_metrics(pred: Union[torch.Tensor, np.ndarray],
                      target: Union[torch.Tensor, np.ndarray],
                      num_classes: int = 2) -> Dict[str, float]:
    """
    快速计算分割指标的辅助函数

    Args:
        pred: 预测结果
        target: 真实标签
        num_classes: 类别数量

    Returns:
        dict: 指标字典
    """
    metrics = SegmentationMetrics(num_classes)
    metrics.update(pred, target)
    return metrics.get_metrics()


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    打印指标的辅助函数

    Args:
        metrics: 指标字典
        prefix: 前缀字符串
    """
    print(f"{prefix} Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    # 指标测试
    metrics = SegmentationMetrics(num_classes=2)

    # 模拟预测和标签
    pred = torch.randint(0, 2, (4, 512, 512))
    target = torch.randint(0, 2, (4, 512, 512))

    metrics.update(pred, target)
    results = metrics.get_metrics()

    print("二分类指标:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
