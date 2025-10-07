import os
import random
import numpy as np
import rasterio
from PIL import Image

def create_image_grid():
    """
    从文件夹中随机抽取图像和掩码，筛选后创建一个带间隙的网格图像并保存。
    掩码将以彩色（黑+蓝）显示。
    """
    # 1. 定义路径和参数
    img_dir = '/home/rove/SegSAR/data/Turkana/timedata/edge_data'
    mask_dir = '/home/rove/SegSAR/data/Turkana/timedata/edge_mask'
    output_path = '/home/rove/BuindingsSeg/image_mask_grid_8x4_color_mask.png'
    gap = 15  # 图像之间的间隙（像素）
    
    # 新的布局和筛选参数
    grid_rows, grid_cols = 4, 8
    num_samples = 16 # 8列 * 2行图像
    min_mask_ratio = 0.2 # 掩码中白色像素的最小比例
    max_mask_ratio = 0.8 # 掩码中白色像素的最大比例
    water_color = [30, 76, 162] # 水体颜色 (蓝色)

    # 2. 获取文件列表并筛选
    try:
        all_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
        if not all_files:
            print(f"错误: 在 '{img_dir}' 中没有找到 .tif 文件。")
            return
        
        random.shuffle(all_files) # 打乱文件列表以实现随机性
        
        selected_files = []
        print("正在筛选符合条件的图像和掩码...")
        for filename in all_files:
            if len(selected_files) >= num_samples:
                break # 已找到足够数量的样本

            mask_path = os.path.join(mask_dir, filename)
            if not os.path.exists(mask_path):
                continue

            with rasterio.open(mask_path) as src:
                mask_data = src.read(1)
                ratio = np.count_nonzero(mask_data) / mask_data.size
                
                if min_mask_ratio < ratio < max_mask_ratio:
                    selected_files.append(filename)

        if len(selected_files) < num_samples:
            print(f"警告: 只找到了 {len(selected_files)} 个符合条件的样本，少于所需的 {num_samples} 个。")
            if not selected_files:
                return
            num_samples = len(selected_files)
            grid_cols = min(grid_cols, num_samples)
            grid_rows = 2 * ((num_samples -1) // grid_cols + 1)

    except FileNotFoundError:
        print(f"错误: 找不到目录 '{img_dir}' 或 '{mask_dir}'。请检查路径是否正确。")
        return

    print(f"已选择 {len(selected_files)} 对符合条件的图像和掩码。")

    # 3. 读取第一张图像以确定尺寸
    with rasterio.open(os.path.join(img_dir, selected_files[0])) as src:
        sample_img = src.read(1)
        img_h, img_w = sample_img.shape

    # 4. 创建带间隙的RGB画布
    canvas_h = grid_rows * img_h + (grid_rows - 1) * gap
    canvas_w = grid_cols * img_w + (grid_cols - 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8) # 3通道RGB，白色背景

    # 5. 填充画布
    for i in range(num_samples):
        img_filename = selected_files[i]
        
        row_in_grid = (i // grid_cols) * 2
        col_in_grid = i % grid_cols

        # 读取原始灰度图像
        with rasterio.open(os.path.join(img_dir, img_filename)) as src:
            img_data = src.read(1)
        # 将灰度图转换为3通道RGB图，以便放入画布
        img_data_rgb = np.stack([img_data]*3, axis=-1)

        # 读取掩码并转换为彩色RGB图
        with rasterio.open(os.path.join(mask_dir, img_filename)) as src:
            mask_data = src.read(1) # 值为 0 或 1
        # 创建一个黑色的RGB画布
        mask_rgb = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        # 将掩码中值为1的像素（水体）设置为蓝色
        mask_rgb[mask_data == 1] = water_color

        # 计算在画布上的位置
        y_start_img = row_in_grid * (img_h + gap)
        x_start_img = col_in_grid * (img_w + gap)
        canvas[y_start_img : y_start_img + img_h, x_start_img : x_start_img + img_w] = img_data_rgb

        y_start_mask = (row_in_grid + 1) * (img_h + gap)
        x_start_mask = col_in_grid * (img_w + gap)
        canvas[y_start_mask : y_start_mask + img_h, x_start_mask : x_start_mask + img_w] = mask_rgb

    # 6. 保存结果
    try:
        result_image = Image.fromarray(canvas, 'RGB') # 保存为RGB图像
        result_image.save(output_path)
        print(f"筛选后的彩色掩码网格图像已成功保存到: {output_path}")
    except Exception as e:
        print(f"保存图像时出错: {e}")

if __name__ == '__main__':
    create_image_grid()