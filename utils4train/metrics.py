import numpy as np
import torch
import torch.nn.functional as F
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
        # 转换为numpy数组
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        # 展平
        pred = pred.flatten()
        target = target.flatten()
        
        # 过滤有效像素
        if self.ignore_index is not None:
            mask = target != self.ignore_index
        else:
            mask = (target >= 0) & (target < self.num_classes)
        
        pred = pred[mask]
        target = target[mask]
        
        # 确保预测值在有效范围内
        pred = np.clip(pred, 0, self.num_classes - 1)
        
        # 更新混淆矩阵
        for i in range(len(target)):
            self.confusion_matrix[target[i], pred[i]] += 1
    
    def get_confusion_matrix(self) -> np.ndarray:
        """获取混淆矩阵"""
        return self.confusion_matrix.copy()
    
    def get_metrics(self) -> Dict[str, float]:
        """
        计算所有指标
        Returns:
            dict: 包含各种指标的字典
        """
        if self.num_classes == 2:
            return self._get_binary_metrics()
        else:
            return self._get_multiclass_metrics()
    
    def _get_binary_metrics(self) -> Dict[str, float]:
        """计算二分类指标"""
        tn, fp, fn, tp = self.confusion_matrix.ravel()
        
        # 基础指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # IoU (Intersection over Union)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        # F1 Score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        # Dice Score (等同于 F1)
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
        
        # 总体准确率
        total_correct = np.trace(self.confusion_matrix)
        total_samples = np.sum(self.confusion_matrix)
        metrics['accuracy'] = total_correct / total_samples if total_samples > 0 else 0.0
        
        # 每类指标
        class_metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'iou': []
        }
        
        for i in range(self.num_classes):
            # 当前类的 TP, FP, FN
            tp = self.confusion_matrix[i, i]
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            fn = np.sum(self.confusion_matrix[i, :]) - tp
            
            # 精确率
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            class_metrics['precision'].append(precision)
            
            # 召回率
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            class_metrics['recall'].append(recall)
            
            # F1 分数
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            class_metrics['f1'].append(f1)
            
            # IoU
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            class_metrics['iou'].append(iou)
        
        # 宏平均
        metrics['macro_precision'] = np.mean(class_metrics['precision'])
        metrics['macro_recall'] = np.mean(class_metrics['recall'])
        metrics['macro_f1'] = np.mean(class_metrics['f1'])
        metrics['macro_iou'] = np.mean(class_metrics['iou'])
        
        # 加权平均（按类别样本数量加权）
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
        
        # 每类详细指标
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


class LossMetrics:
    """损失值统计类"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置统计"""
        self.values = []
        self.total = 0.0
        self.count = 0
    
    def update(self, value: float, n: int = 1):
        """
        更新损失值
        Args:
            value: 损失值
            n: 样本数量
        """
        self.values.append(value)
        self.total += value * n
        self.count += n
    
    def get_average(self) -> float:
        """获取平均损失"""
        return self.total / self.count if self.count > 0 else 0.0
    
    def get_current(self) -> float:
        """获取当前损失"""
        return self.values[-1] if self.values else 0.0


class MetricsTracker:
    """训练过程中的指标跟踪器"""
    
    def __init__(self, num_classes: int = 2, track_loss: bool = True):
        """
        Args:
            num_classes: 类别数量
            track_loss: 是否跟踪损失
        """
        self.num_classes = num_classes
        self.track_loss = track_loss
        
        # 指标计算器
        self.seg_metrics = SegmentationMetrics(num_classes)
        
        if track_loss:
            self.loss_metrics = LossMetrics()
    
    def reset(self):
        """重置所有指标"""
        self.seg_metrics.reset()
        if self.track_loss:
            self.loss_metrics.reset()
    
    def update(self, pred: Union[torch.Tensor, np.ndarray], 
               target: Union[torch.Tensor, np.ndarray],
               loss: Optional[float] = None):
        """
        更新指标
        
        Args:
            pred: 预测结果
            target: 真实标签
            loss: 损失值（可选）
        """
        # 如果pred是logits，转换为类别预测
        if isinstance(pred, torch.Tensor) and pred.dim() > 1:
            if pred.shape[1] > 1:  # 多分类输出
                pred = torch.argmax(pred, dim=1)
        
        # 更新分割指标
        self.seg_metrics.update(pred, target)
        
        # 更新损失指标
        if loss is not None and self.track_loss:
            batch_size = target.shape[0] if hasattr(target, 'shape') else 1
            self.loss_metrics.update(loss, batch_size)
    
    def get_metrics(self) -> Dict[str, float]:
        """获取所有指标"""
        metrics = self.seg_metrics.get_metrics()
        
        if self.track_loss:
            metrics['loss'] = self.loss_metrics.get_average()
        
        return metrics
    
    def get_summary_string(self) -> str:
        """获取指标摘要字符串"""
        metrics = self.get_metrics()
        
        if self.num_classes == 2:
            summary = (
                f"Loss: {metrics.get('loss', 0):.4f}, "
                f"Acc: {metrics['accuracy']:.4f}, "
                f"IoU: {metrics['iou']:.4f}, "
                f"F1: {metrics['f1']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}"
            )
        else:
            summary = (
                f"Loss: {metrics.get('loss', 0):.4f}, "
                f"Acc: {metrics['accuracy']:.4f}, "
                f"mIoU: {metrics['macro_iou']:.4f}, "
                f"mF1: {metrics['macro_f1']:.4f}"
            )
        
        return summary


# 辅助函数
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


# 用于向后兼容的类别别名
class BuildingSegmentationMetrics(SegmentationMetrics):
    """建筑物分割指标类（向后兼容）"""
    
    def __init__(self):
        super().__init__(num_classes=2)