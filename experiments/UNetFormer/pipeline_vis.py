"""
UNetFormer 可视化推理脚本

用于对比实验的可视化
- 对 Vis 数据集进行推理
- 只保存预测结果 mask
"""

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import config
from model import UNetFormer
from dataprocess import get_val_augmentations


def load_model(model_path, device, config):
    """加载模型"""
    model = UNetFormer(
        decode_channels=64,
        dropout=0.1,
        backbone_name='swsl_resnet18',
        pretrained=False,  # 推理时不加载预训练权重
        window_size=8,
        num_classes=config.get("num_classes", 2),
        in_channels=config.get("input_channels", 3),
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    return model


def inference_vis_dataset(
    model_path: str,
    vis_dir: str,
    output_dir: str,
    config: dict,
):
    """
    对 Vis 数据集进行推理并保存预测结果

    Args:
        model_path: 模型权重路径
        vis_dir: Vis 数据集目录 (包含 Vis/image/ 和 Vis/label/ 子目录)
        output_dir: 输出目录
        config: 配置字典
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    print(f"加载模型: {model_path}")
    model = load_model(model_path, device, config)

    # 获取 Vis 数据集路径
    image_dir = os.path.join(vis_dir, "Vis", "image")

    if not os.path.exists(image_dir):
        print(f"错误: Vis 图像目录不存在: {image_dir}")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取图像列表
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.tif', '.tiff', '.png', '.jpg'))])

    print(f"找到 {len(image_files)} 张 Vis 图像")

    # 数据增强 (仅标准化)
    transform = get_val_augmentations(config["image_size"], config.get("use_nir", False))

    # 推理
    with torch.no_grad():
        for img_file in tqdm(image_files, desc="推理中"):
            # 读取图像
            img_path = os.path.join(image_dir, img_file)

            image = Image.open(img_path).convert('RGB')
            image_np = np.array(image)

            # 预处理
            augmented = transform(image=image_np, mask=np.zeros_like(image_np[:, :, 0]))
            input_tensor = augmented['image'].unsqueeze(0).to(device)

            # 推理
            outputs = model(input_tensor)

            # UNetFormer 在 eval 模式下只返回单个输出
            main_output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            pred = torch.argmax(main_output, dim=1)[0].cpu().numpy()

            # 转换为 0-255
            pred_mask = (pred * 255).astype(np.uint8)

            # 保存预测结果
            save_name = os.path.splitext(img_file)[0] + '_pred.png'
            save_path = os.path.join(output_dir, save_name)

            Image.fromarray(pred_mask).save(save_path)

    print(f"预测结果已保存到: {output_dir}")


def main():
    """可视化主流程"""
    import argparse

    parser = argparse.ArgumentParser(description="UNetFormer 可视化推理脚本")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    parser.add_argument("--vis_dir", type=str, default=None, help="Vis 数据集目录 (默认使用 config 中的 data_root)")
    parser.add_argument("--output", type=str, default=None, help="输出目录")

    args = parser.parse_args()

    # 设置路径
    vis_dir = args.vis_dir if args.vis_dir else config["data_root"]

    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join("./vis_results", config.get("model_name", "UNetFormer"))

    # 执行可视化推理
    inference_vis_dataset(
        model_path=args.model,
        vis_dir=vis_dir,
        output_dir=output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
