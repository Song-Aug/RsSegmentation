"""
数据处理模块

定义数据集类和数据加载器，处理数据增强和标准化。
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import rasterio
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ========== 数据归一化统计 ==========
# 基于 GF7_Building 数据集计算
MEAN_3BANDS = [0.485, 0.456, 0.406]
STD_3BANDS = [0.229, 0.224, 0.225]
MEAN_4BANDS = [0.485, 0.456, 0.406, 0.5]
STD_4BANDS = [0.229, 0.224, 0.225, 0.25]


def get_train_augmentations(image_size=512, use_nir=False):
    """为训练集创建强大的数据增强管道"""
    mean = MEAN_4BANDS if use_nir else MEAN_3BANDS
    std = STD_4BANDS if use_nir else STD_3BANDS

    return A.Compose([
        A.Resize(image_size, image_size, interpolation=Image.BILINEAR),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(p=0.3),
        A.HueSaturationValue(p=0.3, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        A.GaussNoise(p=0.2),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=0.5, p=1),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=int(image_size * 0.1), max_width=int(image_size * 0.1), p=0.3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_mild_augmentations(image_size=512, use_nir=False):
    """为训练后期创建温和的数据增强管道"""
    mean = MEAN_4BANDS if use_nir else MEAN_3BANDS
    std = STD_4BANDS if use_nir else STD_3BANDS

    return A.Compose([
        A.Resize(image_size, image_size, interpolation=Image.BILINEAR),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_augmentations(image_size=512, use_nir=False):
    """验证/测试集数据增强（仅标准化）"""
    mean = MEAN_4BANDS if use_nir else MEAN_3BANDS
    std = STD_4BANDS if use_nir else STD_3BANDS

    return A.Compose([
        A.Resize(image_size, image_size, interpolation=Image.BILINEAR),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


class BuildingSegmentationDataset(Dataset):
    """建筑物分割数据集"""

    def __init__(self, root_dir, split="Train", transform=None, use_nir=True):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.use_nir = use_nir

        self.image_dir = os.path.join(root_dir, split, "image")
        self.label_dir = os.path.join(root_dir, split, "label")

        if not os.path.exists(self.image_dir):
            raise ValueError(f"图像目录不存在: {self.image_dir}")
        if not os.path.exists(self.label_dir):
            raise ValueError(f"标签目录不存在: {self.label_dir}")

        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".tif", ".tiff"))
        ])

        self.valid_pairs = []
        for img_file in self.image_files:
            label_path = os.path.join(self.label_dir, img_file)
            if os.path.exists(label_path):
                self.valid_pairs.append((img_file, img_file))

        print(f"{split} 数据集: 找到 {len(self.valid_pairs)} 对有效的图像-标签对")

    def __len__(self):
        return len(self.valid_pairs)

    def read_multispectral_image(self, img_path):
        with rasterio.open(img_path) as src:
            image = src.read()
            image = np.transpose(image, (1, 2, 0))
            if self.use_nir and image.shape[2] >= 4:
                return image[:, :, :4].astype(np.uint8)
            else:
                return image[:, :, :3].astype(np.uint8)

    def __getitem__(self, idx):
        img_file, label_file = self.valid_pairs[idx]
        img_path = os.path.join(self.image_dir, img_file)
        label_path = os.path.join(self.label_dir, label_file)

        try:
            image = self.read_multispectral_image(img_path)
            label = np.array(Image.open(label_path).convert("L"))

            if label.max() > 1:
                label[label > 0] = 1

            if self.transform:
                augmented = self.transform(image=image, mask=label)
                image = augmented["image"]
                label = augmented["mask"].long()

            return {"image": image, "label": label, "filename": img_file}

        except Exception as e:
            print(f"读取文件出错 {img_path}: {e}")
            channels = 4 if self.use_nir else 3
            return {
                "image": torch.zeros((channels, 512, 512), dtype=torch.float32),
                "label": torch.zeros((512, 512), dtype=torch.long),
                "filename": img_file,
            }


def get_loaders(config):
    """
    创建数据加载器

    Args:
        config: 配置字典

    Returns:
        train_loader_strong, train_loader_mild, val_loader, test_loader, vis_loader
    """
    image_size = config["image_size"]
    use_nir = config["use_nir"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    data_root = config["data_root"]

    # 强增强（训练前期）
    strong_aug = get_train_augmentations(image_size, use_nir)
    train_dataset_strong = BuildingSegmentationDataset(
        root_dir=data_root, split='Train', transform=strong_aug, use_nir=use_nir
    )
    train_loader_strong = DataLoader(
        train_dataset_strong, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    # 温和增强（训练后期）
    mild_aug = get_mild_augmentations(image_size, use_nir)
    train_dataset_mild = BuildingSegmentationDataset(
        root_dir=data_root, split='Train', transform=mild_aug, use_nir=use_nir
    )
    train_loader_mild = DataLoader(
        train_dataset_mild, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    # 验证集
    val_aug = get_val_augmentations(image_size, use_nir)
    val_dataset = BuildingSegmentationDataset(
        root_dir=data_root, split='Val', transform=val_aug, use_nir=use_nir
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    # 测试集
    test_dataset = BuildingSegmentationDataset(
        root_dir=data_root, split='Test', transform=val_aug, use_nir=use_nir
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    # 可视化集
    vis_loader = create_vis_dataloader(data_root, image_size, num_workers, use_nir)
    if vis_loader is None:
        vis_loader = val_loader

    return train_loader_strong, train_loader_mild, val_loader, test_loader, vis_loader


def create_vis_dataloader(root_dir, image_size, num_workers, use_nir):
    """创建专用的可视化数据加载器"""
    vis_dir = os.path.join(root_dir, "Vis")

    if not os.path.exists(vis_dir):
        print(f"警告: 未找到可视化数据集目录 {vis_dir}")
        return None

    vis_transform = get_val_augmentations(image_size, use_nir)
    vis_dataset = BuildingSegmentationDataset(
        root_dir=root_dir, split="Vis", transform=vis_transform, use_nir=use_nir
    )
    vis_loader = DataLoader(
        vis_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return vis_loader


if __name__ == "__main__":
    # 数据加载测试
    from config import config

    train_strong, train_mild, val, test, vis = get_loaders(config)

    print(f"强增强训练集: {len(train_strong.dataset)} 样本")
    print(f"温和增强训练集: {len(train_mild.dataset)} 样本")
    print(f"验证集: {len(val.dataset)} 样本")
    print(f"测试集: {len(test.dataset)} 样本")

    # 测试单个样本
    batch = next(iter(val))
    print(f"\n样本形状 - 图像: {batch['image'].shape}, 标签: {batch['label'].shape}")
