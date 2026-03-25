"""
TransCCV3 推理脚本

支持大图滑窗推理，保持地理参考信息
"""

import os
import sys
import torch
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from config import config
from model import create_transcc_v3


# ========== 推理配置 ==========
# 从训练日志复制统计信息
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 推理参数
MODEL_PATH = "/home/rove/BuindingsSeg/checkpoints/TransCC_V3_1105/best_model.pth"
INPUT_PATH = "/home/rove/BuindingsSeg/infer/TJ_BIG_bwd_ps.tif"
OUTPUT_PATH = "/home/rove/BuindingsSeg/infer/TJ_BIG_bwd_ps_result.tif"
THRESHOLD = 0.5


def get_gaussian_window(size, sigma=1 / 4):
    """生成高斯权重矩阵，用于平滑窗口边缘"""
    temp = np.zeros((size, size))
    temp[size // 2, size // 2] = 1
    window = gaussian_filter(temp, sigma=size * sigma)
    return window / (window.max() + 1e-8)


def check_geo_alignment(original_path, mask_path):
    """地理对齐自动化验证"""
    print("\n" + "=" * 40)
    print("正在执行地理参考对齐验证...")
    with rasterio.open(original_path) as src_orig, rasterio.open(mask_path) as src_mask:
        size_check = (
            src_orig.width == src_mask.width and src_orig.height == src_mask.height
        )
        crs_check = src_orig.crs == src_mask.crs
        trans_check = src_orig.transform == src_mask.transform

        print(f"1. 像素尺寸一致性: {'通过' if size_check else '失败'}")
        print(f"   - 原始尺寸: {src_orig.width}x{src_orig.height}")
        print(f"   - Mask尺寸: {src_mask.width}x{src_mask.height}")
        print(f"2. 坐标系(CRS)一致性: {'通过' if crs_check else '失败'}")
        print(f"3. 地理变换(Transform)一致性: {'通过' if trans_check else '失败'}")

        if size_check and crs_check and trans_check:
            print("结论: 结果图与原始图像已完美对齐，Padding已消除。")
        else:
            print("警告: 对齐存在偏差，请确认元数据写入逻辑。")
    print("=" * 40 + "\n")


def load_model(model_path, device, config):
    """加载模型"""
    model = create_transcc_v3({
        "img_size": config["image_size"],
        "patch_size": config["patch_size"],
        "in_chans": config["input_channels"],
        "num_classes": config["num_classes"],
        "depth": config["depth"],
        "hdnet_base_channel": config["hdnet_base_channel"],
        # 消融实验开关
        "use_transformer": config.get("use_transformer", True),
        "use_cnn": config.get("use_cnn", True),
        "use_cbam": config.get("use_cbam", True),
    })

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    return model


def inference_sliding_window(
    model_path: str,
    input_path: str,
    output_path: str,
    config: dict,
    mean: np.ndarray = MEAN,
    std: np.ndarray = STD,
    threshold: float = THRESHOLD,
):
    """
    滑动窗口推理

    Args:
        model_path: 模型权重路径
        input_path: 输入图像路径
        output_path: 输出结果路径
        config: 配置字典
        mean: 归一化均值
        std: 归一化标准差
        threshold: 二值化阈值
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = config["image_size"]
    stride = img_size // 2

    # 加载模型
    print(f"加载模型: {model_path}")
    model = load_model(model_path, device, config)

    # 读取输入图像
    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        orig_h, orig_w = src.height, src.width
        orig_transform = src.transform

        # 读取 RGB 通道 (3, 2, 1 对应 BGR -> RGB)
        full_img = src.read([3, 2, 1])
        full_img = np.transpose(full_img, (1, 2, 0))

    # 初始化输出
    full_probs = np.zeros((orig_h, orig_w), dtype=np.float32)
    count_map = np.zeros((orig_h, orig_w), dtype=np.float32)
    window_weight = get_gaussian_window(img_size)

    # 计算滑窗位置
    rows = list(range(0, orig_h - img_size + 1, stride))
    if (orig_h - img_size) % stride != 0:
        rows.append(orig_h - img_size)
    cols = list(range(0, orig_w - img_size + 1, stride))
    if (orig_w - img_size) % stride != 0:
        cols.append(orig_w - img_size)

    print(f"开始滑动窗口推理... 图像尺寸: {orig_w}x{orig_h}")
    with torch.no_grad():
        for r in tqdm(rows):
            for c in cols:
                # 提取切片
                tile = full_img[r: r + img_size, c: c + img_size, :]

                # 归一化到 [0, 1]
                if tile.max() > 255 or tile.dtype != np.uint8:
                    tile = ((tile - tile.min()) / (tile.max() - tile.min() + 1e-6) * 255).astype(np.uint8)

                # 标准化
                input_tensor = tile.astype(np.float32) / 255.0
                input_tensor = (input_tensor - mean) / std
                input_tensor = (
                    torch.from_numpy(input_tensor)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                )

                # 推理
                outputs = model(input_tensor)
                main_output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

                # 提取建筑物类别概率
                tile_probs = torch.softmax(main_output, dim=1)[0, 1].cpu().numpy()

                # 高斯加权累加
                full_probs[r: r + img_size, c: c + img_size] += tile_probs * window_weight
                count_map[r: r + img_size, c: c + img_size] += window_weight

    # 归一化
    full_probs /= count_map + 1e-8

    # 二值化
    final_mask = (full_probs[:orig_h, :orig_w] > threshold).astype(np.uint8) * 255

    # 写入结果
    out_meta = meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": "uint8",
        "nodata": None,
        "width": orig_w,
        "height": orig_h,
        "transform": orig_transform,
        "compress": "lzw",
    })

    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(final_mask, 1)
        dst.update_tags(1, description="Building Segmentation Mask (TransCCV3)")

    print(f"推理完成！结果已保存至: {output_path}")

    # 地理对齐验证
    check_geo_alignment(input_path, output_path)

    return final_mask


def evaluate(pred_mask, label_path):
    """
    评估预测结果

    Args:
        pred_mask: 预测的 mask
        label_path: 标签文件路径

    Returns:
        dict: 评估指标
    """
    with rasterio.open(label_path) as src:
        label = src.read(1)

    # 二值化
    pred_binary = (pred_mask > 0).astype(np.uint8)
    label_binary = (label > 0).astype(np.uint8)

    # 计算指标
    tp = np.sum(pred_binary * label_binary)
    fp = np.sum(pred_binary * (1 - label_binary))
    fn = np.sum((1 - pred_binary) * label_binary)
    tn = np.sum((1 - pred_binary) * (1 - label_binary))

    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    oa = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "oa": oa,
    }


def main():
    """推理主流程"""
    import argparse

    parser = argparse.ArgumentParser(description="TransCCV3 推理脚本")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="模型权重路径")
    parser.add_argument("--input", type=str, default=INPUT_PATH, help="输入图像路径")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH, help="输出结果路径")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="二值化阈值")
    parser.add_argument("--label", type=str, default=None, help="标签路径（用于评估）")

    args = parser.parse_args()

    # 执行推理
    pred_mask = inference_sliding_window(
        model_path=args.model,
        input_path=args.input,
        output_path=args.output,
        config=config,
        threshold=args.threshold,
    )

    # 评估（如果提供了标签）
    if args.label:
        results = evaluate(pred_mask, args.label)
        print("\n评估结果:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
