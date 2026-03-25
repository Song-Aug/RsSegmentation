"""
Checkpoint 分析脚本

分析模型权重文件，判断是哪个消融实验的模型
- 检测是否包含 CBAM 参数
- 检测是否包含 boundary_head 参数
"""

import torch
import argparse
import re


def analyze_checkpoint(ckpt_path):
    """
    分析 checkpoint 结构

    Args:
        ckpt_path: checkpoint 文件路径

    Returns:
        dict: 分析结果
    """
    print(f"加载 checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if not isinstance(state_dict, dict):
        print("错误: 无法解析 state_dict")
        return

    keys = list(state_dict.keys())
    total_params = sum(p.numel() for p in state_dict.values())

    print(f"\n{'='*60}")
    print(f"Checkpoint 分析报告")
    print(f"{'='*60}")
    print(f"总参数数量: {total_params:,}")
    print(f"总键数量: {len(keys)}")

    # 检测 CBAM 相关参数
    cbam_keys = [k for k in keys if 'cbam' in k.lower()]
    has_cbam = len(cbam_keys) > 0

    # 检测 boundary_head 相关参数
    boundary_keys = [k for k in keys if 'boundary' in k.lower()]
    has_boundary = len(boundary_keys) > 0

    # 检测 transformer 编码器参数
    transformer_keys = [k for k in keys if 'transformer_encoder' in k]
    has_transformer = len(transformer_keys) > 0

    # 检测 CNN 编码器参数
    cnn_keys = [k for k in keys if 'cnn_encoder' in k]
    has_cnn = len(cnn_keys) > 0

    print(f"\n{'='*60}")
    print("模块检测结果:")
    print(f"{'='*60}")
    print(f"  Transformer 编码器: {'✅ 存在' if has_transformer else '❌ 不存在'} ({len(transformer_keys)} 个参数)")
    print(f"  CNN 编码器: {'✅ 存在' if has_cnn else '❌ 不存在'} ({len(cnn_keys)} 个参数)")
    print(f"  CBAM 模块: {'✅ 存在' if has_cbam else '❌ 不存在'} ({len(cbam_keys)} 个参数)")
    print(f"  Boundary Head: {'✅ 存在' if has_boundary else '❌ 不存在'} ({len(boundary_keys)} 个参数)")

    # 判断实验类型
    print(f"\n{'='*60}")
    print("实验类型判断:")
    print(f"{'='*60}")

    if has_transformer and has_cnn:
        if has_cbam and has_boundary:
            experiment_type = "Full 模型 (完整 TransCCV3)"
        elif not has_cbam and has_boundary:
            experiment_type = "无 CBAM 消融"
        elif has_cbam and not has_boundary:
            experiment_type = "无 Boundary Loss 消融"
        else:
            experiment_type = "无 CBAM + 无 Boundary Loss 消融"
    elif has_cnn and not has_transformer:
        experiment_type = "仅 CNN 编码器消融"
    elif has_transformer and not has_cnn:
        experiment_type = "仅 Transformer 编码器消融"
    else:
        experiment_type = "未知类型"

    print(f"  ➜ {experiment_type}")

    # 详细参数列表
    print(f"\n{'='*60}")
    print("CBAM 相关参数详情:")
    print(f"{'='*60}")
    if cbam_keys:
        for k in cbam_keys[:10]:  # 只显示前10个
            print(f"  {k}")
        if len(cbam_keys) > 10:
            print(f"  ... 还有 {len(cbam_keys) - 10} 个参数")
    else:
        print("  (无)")

    print(f"\n{'='*60}")
    print("Boundary 相关参数详情:")
    print(f"{'='*60}")
    if boundary_keys:
        for k in boundary_keys[:10]:  # 只显示前10个
            print(f"  {k}")
        if len(boundary_keys) > 10:
            print(f"  ... 还有 {len(boundary_keys) - 10} 个参数")
    else:
        print("  (无)")

    return {
        "total_params": total_params,
        "has_transformer": has_transformer,
        "has_cnn": has_cnn,
        "has_cbam": has_cbam,
        "has_boundary": has_boundary,
        "experiment_type": experiment_type,
    }


def main():
    parser = argparse.ArgumentParser(description="分析 checkpoint 结构")
    parser.add_argument("checkpoint", type=str, help="Checkpoint 文件路径")
    args = parser.parse_args()

    analyze_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
