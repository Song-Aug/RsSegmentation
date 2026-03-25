"""
MSSDMPA-Net 实验配置

集中管理所有实验超参数和路径配置，确保实验可复现。
"""

config = {
    # ========== 可复现性 ==========
    "seed": 42,
    "device": "cuda",

    # ========== 项目信息 ==========
    "project_name": "Building-Segmentation-3Bands",
    "model_name": "MSSDMPA_Net",
    "description": "MSSDMPA-Net 建筑物分割实验",
    "tags": ["MSSDMPA-Net", "building-segmentation", "RGB"],

    # ========== 数据配置 ==========
    "data_root": "/mnt/data1/rove/asset/GF7_Building/3Bands",
    "batch_size": 4,
    "num_workers": 4,
    "image_size": 512,
    "input_channels": 3,
    "use_nir": False,
    "num_classes": 1,  # MSSDMPA-Net 使用单通道输出

    # ========== 训练配置 ==========
    "num_epochs": 300,
    "learning_rate": 1e-3,
    "min_lr": 1e-6,
    "weight_decay": 1e-4,
    "warmup_epochs": 10,

    # ========== 输出配置 ==========
    "save_dir": "./runs",
}
