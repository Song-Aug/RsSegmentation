"""
TransCCV3 可视化推理脚本

用于对比实验和消融实验的可视化
- 对 Vis 数据集进行推理
- 生成可视化结果（原图、GT、预测对比）
- 保存到指定文件夹
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import config
from model import create_transcc_v3
from dataprocess import get_val_augmentations, MEAN_3BANDS, STD_3BANDS
from PIL import Image


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


def denormalize(image_tensor, mean, std):
    """反归一化图像用于显示"""
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image


def create_visualization(image, label, pred, boundary_pred=None, save_path=None):
    """
    创建单张可视化图

    Args:
        image: 原图 (C, H, W) tensor
        label: 标签 (H, W)
        pred: 预测结果 (H, W)
        boundary_pred: 边界预测 (H, W), optional
        save_path: 保存路径
    """
    # 反归一化
    image_np = denormalize(image, MEAN_3BANDS, STD_3BANDS)

    # 创建图
    if boundary_pred is not None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 原图
    axes[0].imshow(image_np)
    axes[0].set_title('Image', fontsize=12)
    axes[0].axis('off')

    # GT
    axes[1].imshow(label, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth', fontsize=12)
    axes[1].axis('off')

    # 预测
    axes[2].imshow(pred, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Prediction', fontsize=12)
    axes[2].axis('off')

    # 边界预测
    if boundary_pred is not None:
        axes[3].imshow(boundary_pred, cmap='hot', vmin=0, vmax=1)
        axes[3].set_title('Boundary', fontsize=12)
        axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    else:
        return fig


def inference_vis_dataset(
    model_path: str,
    vis_dir: str,
    output_dir: str,
    config: dict,
):
    """
    对 Vis 数据集进行推理并保存可视化结果

    Args:
        model_path: 模型权重路径
        vis_dir: Vis 数据集目录 (包含 image/ 和 label/ 子目录)
        output_dir: 输出目录
        config: 配置字典
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    print(f"加载模型: {model_path}")
    model = load_model(model_path, device, config)

    # 获取 Vis 数据集路径
    image_dir = os.path.join(vis_dir, "Vis", "image")
    label_dir = os.path.join(vis_dir, "Vis", "label")

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
            label_path = os.path.join(label_dir, img_file)

            # 读取图像
            image = Image.open(img_path).convert('RGB')
            image_np = np.array(image)

            # 读取标签
            if os.path.exists(label_path):
                label = Image.open(label_path).convert('L')
                label_np = np.array(label)
                if label_np.max() > 1:
                    label_np = (label_np > 0).astype(np.uint8)
            else:
                label_np = np.zeros(image_np.shape[:2], dtype=np.uint8)

            # 预处理
            augmented = transform(image=image_np, mask=label_np)
            input_tensor = augmented['image'].unsqueeze(0).to(device)
            label_tensor = augmented['mask']

            # 推理
            outputs = model(input_tensor)

            # 获取主输出
            main_output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            pred = torch.argmax(main_output, dim=1)[0].cpu().numpy()

            # 获取边界预测
            boundary_pred = None
            if isinstance(outputs, (list, tuple)) and len(outputs) > 1:
                boundary_output = outputs[1]
                if boundary_output.shape[1] == 1:
                    boundary_pred = torch.sigmoid(boundary_output)[0, 0].cpu().numpy()

            # 保存可视化
            save_name = os.path.splitext(img_file)[0] + '_vis.png'
            save_path = os.path.join(output_dir, save_name)

            create_visualization(
                image=augmented['image'],
                label=label_tensor,
                pred=pred,
                boundary_pred=boundary_pred,
                save_path=save_path
            )

    print(f"可视化结果已保存到: {output_dir}")


def main():
    """可视化主流程"""
    import argparse

    parser = argparse.ArgumentParser(description="TransCCV3 可视化推理脚本")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    parser.add_argument("--vis_dir", type=str, default=None, help="Vis 数据集目录 (默认使用 config 中的 data_root)")
    parser.add_argument("--output", type=str, default=None, help="输出目录 (默认: ./vis_results/{model_name})")

    args = parser.parse_args()

    # 设置路径
    vis_dir = args.vis_dir if args.vis_dir else config["data_root"]

    if args.output:
        output_dir = args.output
    else:
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        output_dir = os.path.join("./vis_results", config.get("model_name", "TransCCV3"))

    # 执行可视化推理
    inference_vis_dataset(
        model_path=args.model,
        vis_dir=vis_dir,
        output_dir=output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
