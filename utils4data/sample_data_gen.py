import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def create_sample_dataset(source_root, target_root, sample_ratio=0.1):
    """
    从原始数据集中随机抽取指定比例的数据创建简易数据集
    
    Args:
        source_root: 原始数据集路径
        target_root: 目标数据集路径
        sample_ratio: 抽样比例，默认0.1(10%)
    """
    source_path = Path(source_root)
    target_path = Path(target_root)
    
    
    splits = ['Train', 'Val', 'Test']
    sub_dirs = ['image', 'label']
    
    for split in splits:
        for sub_dir in sub_dirs:
            target_dir = target_path / split / sub_dir
            target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"创建数据集样本: {source_root} -> {target_root}")
    print(f"抽样比例: {sample_ratio * 100}%")
    
    total_copied = 0
    
    for split in splits:
        print(f"\n处理 {split} 数据集...")
        
        
        image_dir = source_path / split / 'image'
        if not image_dir.exists():
            print(f"警告: {image_dir} 不存在，跳过")
            continue
            
        
        image_files = []
        for img_file in image_dir.glob('*'):
            if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                image_files.append(img_file.stem)  
        
        if not image_files:
            print(f"警告: {image_dir} 中没有找到图像文件")
            continue
        
        
        sample_size = max(1, int(len(image_files) * sample_ratio))  
        sampled_files = random.sample(image_files, sample_size)
        
        print(f"  总文件数: {len(image_files)}")
        print(f"  抽样文件数: {sample_size}")
        
        
        source_image_dir = source_path / split / 'image'
        source_label_dir = source_path / split / 'label'
        target_image_dir = target_path / split / 'image'
        target_label_dir = target_path / split / 'label'
        
        copied_count = 0
        failed_count = 0
        
        for file_stem in tqdm(sampled_files, desc=f"  复制{split}文件"):
            
            image_file = None
            label_file = None
            
            
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                potential_image = source_image_dir / f"{file_stem}{ext}"
                if potential_image.exists():
                    image_file = potential_image
                    break
            
            
            for ext in ['.png', '.tif', '.tiff']:
                potential_label = source_label_dir / f"{file_stem}{ext}"
                if potential_label.exists():
                    label_file = potential_label
                    break
            
            
            if image_file and label_file:
                try:
                    
                    target_image_file = target_image_dir / image_file.name
                    shutil.copy2(image_file, target_image_file)
                    
                    
                    target_label_file = target_label_dir / label_file.name
                    shutil.copy2(label_file, target_label_file)
                    
                    copied_count += 1
                except Exception as e:
                    print(f"    复制失败 {file_stem}: {e}")
                    failed_count += 1
            else:
                print(f"    找不到对应文件 {file_stem} (image: {image_file is not None}, label: {label_file is not None})")
                failed_count += 1
        
        print(f"  成功复制: {copied_count} 对文件")
        if failed_count > 0:
            print(f"  失败: {failed_count} 对文件")
        
        total_copied += copied_count
    
    print(f"\n数据集创建完成!")
    print(f"总共复制了 {total_copied} 对图像-标签文件")
    
    
    print("\n目标数据集统计:")
    for split in splits:
        image_dir = target_path / split / 'image'
        if image_dir.exists():
            image_count = len(list(image_dir.glob('*')))
            print(f"  {split}: {image_count} 个文件")


def main():
    
    random.seed(42)
    
    
    source_root = "/mnt/data1/rove/asset/GF7_Building/3Bands"
    target_root = "/mnt/data1/rove/asset/GF7_Building/3BandsSample"
    sample_ratio = 0.01  
    
    
    if not os.path.exists(source_root):
        print(f"错误: 源路径不存在 - {source_root}")
        return
    
    
    create_sample_dataset(source_root, target_root, sample_ratio)


if __name__ == "__main__":
    main()