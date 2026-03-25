"""
HDNet 实验配置

集中管理所有实验超参数和路径配置，确保实验可复现。
"""

config = {
    # ========== 可复现性 ==========
    "seed": 42,
    "device": "cuda",

    # ========== 项目信息 ==========
    "project_name": "Building-Segmentation-3Bands",
    "model_name": "HDNet",
    "description": "HDNet model for 3-band building segmentation",
    "tags": ["HDNet", "building-segmentation", "RGB", "high-resolution"],

    # ========== 数据配置 ==========
    "data_root": "/mnt/data1/rove/asset/GF7_Building/3Bands",
    "batch_size": 6,
    "num_workers": 4,
    "image_size": 512,
    "input_channels": 3,
    "use_nir": False,
    "num_classes": 2,

    # ========== 模型配置 ==========
    "base_channel": 48,

    # ========== 训练配置 ==========
    "num_epochs": 300,
    "learning_rate": 5e-4,
    "min_lr": 1e-6,
    "weight_decay": 1e-4,
    "warmup_epochs": 1,

    # ========== 损失函数配置 ==========
    "loss_weights": [1.0, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],

    # ========== 输出配置 ==========
    "save_dir": "./runs",
}
