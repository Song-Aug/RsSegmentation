import os
import sys
import torch
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.TransCCV3 import create_transcc_v3
from configs.transcc_v3_config import config


def get_gaussian_window(size, sigma=1 / 4):
    """生成高斯权重矩阵，用于平滑窗口边缘"""
    temp = np.zeros((size, size))
    temp[size // 2, size // 2] = 1
    window = gaussian_filter(temp, sigma=size * sigma)
    return window / (window.max() + 1e-8)


def check_geo_alignment(original_path, mask_path):
    """地理对齐自动化验证"""
    print("\n" + "=" * 40)
    print("🌍 正在执行地理参考对齐验证...")
    with rasterio.open(original_path) as src_orig, rasterio.open(mask_path) as src_mask:
        size_check = (
            src_orig.width == src_mask.width and src_orig.height == src_mask.height
        )
        crs_check = src_orig.crs == src_mask.crs
        trans_check = src_orig.transform == src_mask.transform

        print(f"1. 像素尺寸一致性: {'✅ 通过' if size_check else '❌ 失败'}")
        print(f"   - 原始尺寸: {src_orig.width}x{src_orig.height}")
        print(f"   - Mask尺寸: {src_mask.width}x{src_mask.height}")
        print(f"2. 坐标系(CRS)一致性: {'✅ 通过' if crs_check else '❌ 失败'}")
        print(
            f"3. 地理变换(Transform)一致性: {'✅ 通过' if trans_check else '❌ 失败'}"
        )

        if size_check and crs_check and trans_check:
            print("🎉 结论: 结果图与原始图像已完美对齐，Padding已消除。")
        else:
            print("⚠️ 结论: 对齐存在偏差，请确认元数据写入逻辑。")
    print("=" * 40 + "\n")


def inference_sliding_window():
    
    model_path = "/home/rove/BuindingsSeg/checkpoints/TransCC_V3_1105/best_model.pth"
    
    
    input_path = "/home/rove/BuindingsSeg/infer/TJ_BIG_bwd_ps.tif"
    output_path = "/home/rove/BuindingsSeg/infer/TJ_BIG_bwd_ps_result.tif"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = config["image_size"]  
    stride = img_size // 2  
    threshold = 0.5  

    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    
    model = create_transcc_v3(
        {
            "img_size": img_size,
            "patch_size": 16,
            "in_chans": 3,
            "num_classes": config["num_classes"],
            "depth": 9,
            "hdnet_base_channel": 48,
        }
    )

    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    
    with rasterio.open(input_path) as src:
        
        meta = src.meta.copy()
        orig_h, orig_w = src.height, src.width
        orig_transform = src.transform

        
        full_img = src.read([3, 2, 1])
        full_img = np.transpose(full_img, (1, 2, 0))  

    
    full_probs = np.zeros((orig_h, orig_w), dtype=np.float32)
    count_map = np.zeros((orig_h, orig_w), dtype=np.float32)
    window_weight = get_gaussian_window(img_size)

    
    rows = list(range(0, orig_h - img_size + 1, stride))
    if (orig_h - img_size) % stride != 0:
        rows.append(orig_h - img_size)
    cols = list(range(0, orig_w - img_size + 1, stride))
    if (orig_w - img_size) % stride != 0:
        cols.append(orig_w - img_size)

    
    print(f"🚀 开始滑动窗口推理... 图像尺寸: {orig_w}x{orig_h}")
    with torch.no_grad():
        for r in tqdm(rows):
            for c in cols:
                
                tile = full_img[r : r + img_size, c : c + img_size, :]

                
                if tile.max() > 255 or tile.dtype != np.uint8:
                    tile = (
                        (tile - tile.min()) / (tile.max() - tile.min() + 1e-6) * 255
                    ).astype(np.uint8)

                
                input_tensor = tile.astype(np.float32) / 255.0
                input_tensor = (input_tensor - mean) / std
                input_tensor = (
                    torch.from_numpy(input_tensor)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                )

                
                outputs = model(input_tensor)
                main_output = (
                    outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                )

                
                tile_probs = torch.softmax(main_output, dim=1)[0, 1].cpu().numpy()

                
                full_probs[r : r + img_size, c : c + img_size] += (
                    tile_probs * window_weight
                )
                count_map[r : r + img_size, c : c + img_size] += window_weight

    
    full_probs /= count_map + 1e-8

    
    
    final_mask = (full_probs[:orig_h, :orig_w] > threshold).astype(np.uint8) * 255

    
    out_meta = meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "uint8",
            "nodata": None,  
            "width": orig_w,
            "height": orig_h,
            "transform": orig_transform,
            "compress": "lzw",  
        }
    )

    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(final_mask, 1)
        dst.update_tags(1, description="Building Segmentation Mask (TransCCV3)")

    print(f"✅ 推理完成！结果已保存至: {output_path}")

    
    check_geo_alignment(input_path, output_path)


if __name__ == "__main__":
    inference_sliding_window()
