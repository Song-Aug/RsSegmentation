import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import rasterio
import random


class BuildingSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='Train', transform=None, augment=False, use_nir=True):
        """
        Args:
            root_dir (str): 数据集根目录路径
            split (str): 数据集分割 ('Train', 'Val', 'Test')
            transform (callable, optional): 图像变换
            augment (bool): 是否使用数据增强
            use_nir (bool): 是否使用NIR波段
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.augment = augment
        self.use_nir = use_nir
        
        # 构建图像和标签路径
        self.image_dir = os.path.join(root_dir, split, 'image')
        self.label_dir = os.path.join(root_dir, split, 'label')
        
        # 检查路径是否存在
        if not os.path.exists(self.image_dir):
            raise ValueError(f"图像目录不存在: {self.image_dir}")
        if not os.path.exists(self.label_dir):
            raise ValueError(f"标签目录不存在: {self.label_dir}")
        
        # 获取所有图像文件名
        self.image_files = [f for f in os.listdir(self.image_dir) 
                           if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
        self.image_files.sort()
        
        # 验证图像和标签文件的对应关系
        self.valid_pairs = []
        for img_file in self.image_files:
            label_file = img_file  # 假设图像和标签文件名相同
            label_path = os.path.join(self.label_dir, label_file)
            if os.path.exists(label_path):
                self.valid_pairs.append((img_file, label_file))
            else:
                print(f"警告: 未找到对应的标签文件 {label_file}")
        
        print(f"{split} 数据集: 找到 {len(self.valid_pairs)} 对有效的图像-标签对")
        print(f"使用波段: {'RGB + NIR (4通道)' if use_nir else 'RGB (3通道)'}")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def read_multispectral_image(self, img_path):
        """读取多光谱图像"""
        try:
            # 使用rasterio读取多波段TIFF
            with rasterio.open(img_path) as src:
                image = src.read()  # shape: (bands, height, width)
                
                # 转换为 (height, width, bands)
                image = np.transpose(image, (1, 2, 0))
                
                # 检查波段数
                if image.shape[2] < 3:
                    raise ValueError(f"图像波段数不足: {image.shape[2]}")
                
                if self.use_nir and image.shape[2] >= 4:
                    image = image[:, :, :4]
                else:
                    image = image[:, :, :3]
                
                return image
                
        except Exception as e:
            print(f"使用rasterio读取失败，尝试PIL: {e}")
            try:
                image = Image.open(img_path)
                image = np.array(image)
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.shape[2] > 4:
                    image = image[:, :, :4]
                if self.use_nir and image.shape[2] >= 4:
                    image = image[:, :, :4]
                else:
                    image = image[:, :, :3]
                return image
            except Exception as e2:
                print(f"PIL读取也失败: {e2}")
                channels = 4 if self.use_nir else 3
                return np.zeros((256, 256, channels), dtype=np.uint8)
    
    def process_label(self, label):
        """处理标签，确保为二分类格式"""
        if isinstance(label, Image.Image):
            label = np.array(label)
        if label.max() > 1:
            label = (label > 0).astype(np.uint8)
        label = np.clip(label, 0, 1)
        return label.astype(np.int64)  # 使用int64确保兼容性
    
    def apply_augmentation(self, image, label):
        """数据增强"""
        if not self.augment or self.split != 'Train':
            return image, label
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label_tensor = torch.from_numpy(label).long().unsqueeze(0)
        
        # 随机水平翻转
        if random.random() > 0.5:
            image_tensor = torch.flip(image_tensor, [2])
            label_tensor = torch.flip(label_tensor, [2])
        
        # 随机垂直翻转
        if random.random() > 0.5:
            image_tensor = torch.flip(image_tensor, [1])
            label_tensor = torch.flip(label_tensor, [1])
        
        # 随机90度旋转
        if random.random() > 0.5:
            k = random.randint(1, 3)
            image_tensor = torch.rot90(image_tensor, k, [1, 2])
            label_tensor = torch.rot90(label_tensor, k, [1, 2])
        
        # 随机亮度和对比度调整（仅对RGB通道）
        if random.random() > 0.7:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            
            # 只对RGB通道应用亮度和对比度调整
            rgb_channels = min(3, image_tensor.shape[0])
            image_tensor[:rgb_channels] = torch.clamp(
                image_tensor[:rgb_channels] * contrast_factor + brightness_factor - 1.0,
                0.0, 255.0
            )
        
        # 添加高斯噪声
        if random.random() > 0.8:
            noise = torch.randn_like(image_tensor) * 5.0
            image_tensor = torch.clamp(image_tensor + noise, 0.0, 255.0)
        
        # 转换回numpy
        image = image_tensor.permute(1, 2, 0).numpy()
        label = label_tensor.squeeze(0).numpy()
        
        return image, label

    def __getitem__(self, idx):
        img_file, label_file = self.valid_pairs[idx]
        img_path = os.path.join(self.image_dir, img_file)
        label_path = os.path.join(self.label_dir, label_file)
        
        try:
            # 读取多光谱图像和标签
            image = self.read_multispectral_image(img_path)
            label = Image.open(label_path)
            if label.mode != 'L':
                label = label.convert('L')
            label = np.array(label)
            
            # 关键修复：确保标签值为0或1
            if label.max() > 1:
                label[label > 0] = 1
            
            # 确保图像和标签尺寸匹配
            if image.shape[:2] != label.shape:
                label = Image.fromarray(label.astype(np.uint8)).resize(
                    (image.shape[1], image.shape[0]), Image.NEAREST
                )
                label = np.array(label)
                if label.max() > 1: # 重新检查
                    label[label > 0] = 1
            
            # # 数据增强
            image, label = self.apply_augmentation(image, label)
            
            # 应用变换
            if self.transform is not None:
                image, label = self.transform(image, label)
            else:
                # 默认转换为tensor
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
                label = torch.from_numpy(label).long()
            
            # 最终检查标签范围和类型
            label = torch.clamp(label, 0, 1).long()
            
            return {
                'image': image,
                'label': label,
                'filename': img_file,
                'channels': image.shape[0]  # 记录通道数
            }
            
        except Exception as e:
            print(f"读取文件出错 {img_path}: {e}")
            # 返回一个安全的样本
            channels = 4 if self.use_nir else 3
            return {
                'image': torch.zeros((channels, 256, 256)),
                'label': torch.zeros((256, 256), dtype=torch.long),
                'filename': img_file,
                'channels': channels
            }


class CustomTransform:
    """自定义标准化config"""
    def __init__(self, image_size=512, use_nir=True, normalize=True):
        self.image_size = image_size
        self.use_nir = use_nir
        self.normalize = normalize
        
        if use_nir:
            # 4通道的归一化参数（RGB + NIR）
            self.mean = torch.tensor([0.485, 0.456, 0.406, 0.5])  # R, G, B, NIR
            self.std = torch.tensor([0.229, 0.224, 0.225, 0.25])   # R, G, B, NIR
        else:
            # 3通道RGB
            self.mean = torch.tensor([0.485, 0.456, 0.406])
            self.std = torch.tensor([0.229, 0.224, 0.225])
    
    def __call__(self, image, label):
        # 转换为tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).long()
        
        # 调整尺寸
        if image.shape[1] != self.image_size or image.shape[2] != self.image_size:
            image = F.interpolate(
                image.unsqueeze(0), 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            label = F.interpolate(
                label.unsqueeze(0).unsqueeze(0).float(), 
                size=(self.image_size, self.image_size), 
                mode='nearest'
            ).squeeze(0).squeeze(0).long()
        
        # 先归一化到[0,1]范围
        image = image / 255.0
        
        # 再进行标准化
        if self.normalize:
            mean = self.mean.view(-1, 1, 1)
            std = self.std.view(-1, 1, 1)
            image = (image - mean) / std
        
        return image, label


def get_transforms(image_size=512, split='Train', use_nir=True):
    """获取数据变换（使用自定义变换）"""
    return CustomTransform(image_size=image_size, use_nir=use_nir, normalize=True)


def create_dataloaders(root_dir, batch_size=16, num_workers=4, image_size=512, 
                      augment=True, use_nir=True):
    """创建数据加载器"""
    
    # 获取变换
    train_transform = get_transforms(image_size, 'Train', use_nir)
    val_transform = get_transforms(image_size, 'Val', use_nir)
    test_transform = get_transforms(image_size, 'Test', use_nir)
    
    # 创建数据集
    train_dataset = BuildingSegmentationDataset(
        root_dir=root_dir,
        split='Train',
        transform=train_transform,
        augment=augment,
        use_nir=use_nir
    )
    
    val_dataset = BuildingSegmentationDataset(
        root_dir=root_dir,
        split='Val',
        transform=val_transform,
        augment=False,
        use_nir=use_nir
    )
    
    test_dataset = BuildingSegmentationDataset(
        root_dir=root_dir,
        split='Test',
        transform=test_transform,
        augment=False,
        use_nir=use_nir
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def check_dataset(root_dir, use_nir=True):
    """检查数据集完整性（包括多光谱信息）"""
    print("=" * 50)
    print("多光谱数据集检查报告")
    print("=" * 50)
    
    splits = ['Train', 'Val', 'Test']
    
    for split in splits:
        print(f"\n{split} 数据集:")
        
        image_dir = os.path.join(root_dir, split, 'image')
        label_dir = os.path.join(root_dir, split, 'label')
        
        if not os.path.exists(image_dir):
            print(f"  ❌ 图像目录不存在: {image_dir}")
            continue
            
        if not os.path.exists(label_dir):
            print(f"  ❌ 标签目录不存在: {label_dir}")
            continue
        
        # 统计文件数量
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
        label_files = [f for f in os.listdir(label_dir) 
                      if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
        
        print(f"  图像文件数量: {len(image_files)}")
        print(f"  标签文件数量: {len(label_files)}")
        
        # 检查波段信息
        if image_files:
            sample_img_path = os.path.join(image_dir, image_files[0])
            
            try:
                # 使用rasterio检查波段信息
                with rasterio.open(sample_img_path) as src:
                    bands = src.count
                    width = src.width
                    height = src.height
                    dtype = src.dtypes[0]
                
                print(f"  样本图像波段数: {bands}")
                print(f"  样本图像尺寸: {width} x {height}")
                print(f"  样本图像数据类型: {dtype}")
                
                if bands >= 4:
                    print(f"  ✓ 支持4波段模式 (RGB + NIR)")
                else:
                    print(f"  ⚠️  波段数不足，只能使用RGB模式")
                
            except Exception as e:
                print(f"  无法读取波段信息: {e}")
                # 尝试PIL
                try:
                    with Image.open(sample_img_path) as img:
                        if hasattr(img, 'n_frames'):
                            print(f"  PIL检测到的图像信息: 尺寸{img.size}, 模式{img.mode}")
                except:
                    print(f"  PIL也无法读取图像")


def calculate_band_statistics(root_dir, split='Train', use_nir=True):
    """计算各波段的统计信息，用于归一化"""
    print(f"\n计算 {split} 数据集的波段统计信息...")
    
    dataset = BuildingSegmentationDataset(
        root_dir=root_dir,
        split=split,
        transform=None,
        augment=False,
        use_nir=use_nir
    )
    
    channels = 4 if use_nir else 3
    means = np.zeros(channels)
    stds = np.zeros(channels)
    total_pixels = 0
    
    for i in range(min(100, len(dataset))):  # 只处理前100张图像
        sample = dataset[i]
        image = sample['image']
        
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        if len(image.shape) == 3 and image.shape[0] <= 4:
            # (C, H, W) format
            image = image.transpose(1, 2, 0)
        
        # 计算每个通道的均值和标准差
        for c in range(min(channels, image.shape[2])):
            channel_data = image[:, :, c].flatten()
            means[c] += channel_data.mean()
            stds[c] += channel_data.std()
        
        total_pixels += 1
        
        if (i + 1) % 20 == 0:
            print(f"  处理了 {i + 1} 张图像...")
    
    means /= total_pixels
    stds /= total_pixels
    
    band_names = ['Red', 'Green', 'Blue', 'NIR'] if use_nir else ['Red', 'Green', 'Blue']
    
    print(f"\n波段统计信息:")
    for i, name in enumerate(band_names[:channels]):
        print(f"  {name}: 均值={means[i]:.4f}, 标准差={stds[i]:.4f}")
    
    return means, stds


# 使用示例
if __name__ == "__main__":
    # 数据集路径
    root_dir = "/path/to/your/dataset"
    
    # 检查数据集
    check_dataset(root_dir, use_nir=True)
    
    # 计算波段统计信息
    means, stds = calculate_band_statistics(root_dir, 'Train', use_nir=True)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=root_dir,
        batch_size=16,
        num_workers=4,
        image_size=512,
        augment=True,
        use_nir=True  # 使用4波段
    )
    
    # 测试数据加载
    print("\n测试4波段数据加载:")
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image']
        labels = batch['label']
        filenames = batch['filename']
        channels = batch['channels']
        
        print(f"Batch {batch_idx}:")
        print(f"  图像形状: {images.shape}")
        print(f"  标签形状: {labels.shape}")
        print(f"  通道数: {channels[0]}")
        print(f"  文件名: {filenames[0]}")
        
        if batch_idx >= 2:  # 只测试前3个batch
            break