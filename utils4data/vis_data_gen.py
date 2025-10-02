import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def create_city_stratified_dataset(source_root, target_root, samples_per_city=2):
    """
    从数据集中按城市分层抽样，创建专用的可视化数据集。

    Args:
        source_root (str): 原始数据集路径 (例如 .../3Bands/Test)
        target_root (str): 目标数据集的根目录 (例如 .../3Bands)
        samples_per_city (int): 每个城市抽取的样本数量
    """
    source_path = Path(source_root)
    # 目标目录将在源数据的上一级创建，名为 'Vis'
    target_path = Path(target_root) / 'Vis'
    
    print(f"创建可视化数据集: {target_path}")
    print(f"从源目录: {source_path}")
    print(f"每个城市抽样数量: {samples_per_city}")

    # --- 1. 扫描并按城市分组 ---
    source_image_dir = source_path / 'image'
    if not source_image_dir.exists():
        print(f"错误: 图像目录不存在 -> {source_image_dir}")
        return

    city_files = defaultdict(list)
    for img_file in source_image_dir.glob('*.tif'):
        city_name = img_file.stem.split('_')[0]
        city_files[city_name].append(img_file.name)

    print(f"\n找到了 {len(city_files)} 个城市: {list(city_files.keys())}")

    # --- 2. 从每个城市抽样 ---
    selected_files = []
    for city, files in city_files.items():
        if len(files) >= samples_per_city:
            sampled = random.sample(files, samples_per_city)
            print(f"  从 '{city}' 中抽样 {len(sampled)} 张图像.")
        else:
            sampled = files
            print(f"  '{city}' 图像不足 {samples_per_city} 张, 使用全部 {len(sampled)} 张.")
        selected_files.extend(sampled)

    print(f"\n总共选中了 {len(selected_files)} 张图像用于可视化。")

    # --- 3. 复制文件到新目录 ---
    target_image_dir = target_path / 'image'
    target_label_dir = target_path / 'label'
    
    # 清理旧目录（如果存在）
    if target_path.exists():
        print(f"清理已存在的可视化目录: {target_path}")
        shutil.rmtree(target_path)
        
    target_image_dir.mkdir(parents=True, exist_ok=True)
    target_label_dir.mkdir(parents=True, exist_ok=True)

    source_label_dir = source_path / 'label'
    
    for file_name in tqdm(selected_files, desc="复制文件中"):
        # 复制图像
        shutil.copy2(source_image_dir / file_name, target_image_dir / file_name)
        # 复制标签
        shutil.copy2(source_label_dir / file_name, target_label_dir / file_name)
        
    print("\n可视化数据集创建成功！")


if __name__ == '__main__':
    random.seed(42)
    
    # --- 配置路径 ---
    # 源 Test 目录
    source_test_dir = "/mnt/data1/rove/asset/GF7_Building/3Bands/Test"
    # 整个数据集的根目录
    dataset_root = "/mnt/data1/rove/asset/GF7_Building/3Bands"
    
    create_city_stratified_dataset(
        source_root=source_test_dir,
        target_root=dataset_root,
        samples_per_city=2
    )