import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import rasterio
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_augmentations(image_size=512, use_nir=False):
    """为训练集创建强大的数据增强管道"""
    mean = [0.485, 0.456, 0.406, 0.5] if use_nir else [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225, 0.25] if use_nir else [0.229, 0.224, 0.225]
    
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
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1),
        ], p=0.3),

        # 随机擦除 (模拟遮挡)
        A.CoarseDropout(max_holes=8, max_height=int(image_size*0.1), max_width=int(image_size*0.1),
                        min_holes=1, min_height=8, min_width=8, p=0.3, fill_value=0, mask_fill_value=0),
        
        # 归一化和转换为Tensor
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

def get_mild_augmentations(image_size=512, use_nir=False):
    """为训练后期创建温和的数据增强管道"""
    mean = [0.485, 0.456, 0.406, 0.5] if use_nir else [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225, 0.25] if use_nir else [0.229, 0.224, 0.225]
    
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=Image.BILINEAR),
        
        # 只保留基础的几何变换
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # 可以保留轻微的颜色增强
        A.RandomBrightnessContrast(p=0.2),
        
        # 移除了 ElasticTransform, GridDistortion, CoarseDropout 等强力增强
        
        # 归一化和转换为Tensor
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

def get_val_augmentations(image_size=512, use_nir=False):
    mean = [0.485, 0.456, 0.406, 0.5] if use_nir else [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225, 0.25] if use_nir else [0.229, 0.224, 0.225]

    return A.Compose([
        A.Resize(image_size, image_size, interpolation=Image.BILINEAR),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


class BuildingSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='Train', transform=None, use_nir=True):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.use_nir = use_nir
        
        self.image_dir = os.path.join(root_dir, split, 'image')
        self.label_dir = os.path.join(root_dir, split, 'label')
        
        if not os.path.exists(self.image_dir): raise ValueError(f"图像目录不存在: {self.image_dir}")
        if not os.path.exists(self.label_dir): raise ValueError(f"标签目录不存在: {self.label_dir}")
        
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(('.tif', '.tiff'))])
        
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
            label = np.array(Image.open(label_path).convert('L'))
            
            if label.max() > 1:
                label[label > 0] = 1

            if self.transform:
                augmented = self.transform(image=image, mask=label)
                image = augmented['image']
                label = augmented['mask'].long() # 确保是LongTensor

            return {'image': image, 'label': label, 'filename': img_file}
            
        except Exception as e:
            print(f"读取文件出错 {img_path}: {e}")
            channels = 4 if self.use_nir else 3
            return {
                'image': torch.zeros((channels, 512, 512), dtype=torch.float32),
                'label': torch.zeros((512, 512), dtype=torch.long),
                'filename': img_file
            }


def create_dataloaders(root_dir, batch_size=16, num_workers=4, image_size=512, 
                      augment=True, use_nir=True):
    train_transform = get_train_augmentations(image_size, use_nir) if augment else get_val_augmentations(image_size, use_nir)
    val_transform = get_val_augmentations(image_size, use_nir)
    
    train_dataset = BuildingSegmentationDataset(
        root_dir=root_dir, split='Train', transform=train_transform, use_nir=use_nir
    )
    val_dataset = BuildingSegmentationDataset(
        root_dir=root_dir, split='Val', transform=val_transform, use_nir=use_nir
    )
    test_dataset = BuildingSegmentationDataset(
        root_dir=root_dir, split='Test', transform=val_transform, use_nir=use_nir
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    
    return train_loader, val_loader, test_loader

def create_vis_dataloader(root_dir, image_size, num_workers, use_nir):
    """创建专用的可视化数据加载器"""
    vis_dir = os.path.join(root_dir, 'Vis')
    
    if not os.path.exists(vis_dir):
        print(f"警告: 未找到可视化数据集目录 {vis_dir}。")
        return None
        
    vis_transform = get_val_augmentations(image_size, use_nir)

    vis_dataset = BuildingSegmentationDataset(
        root_dir=root_dir, 
        split='Vis', 
        transform=vis_transform,
        use_nir=use_nir
    )
    
    vis_loader = DataLoader(
        vis_dataset,
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    return vis_loader