"""
STDSNet 实验配置

集中管理所有实验超参数和路径配置，确保实验可复现。
"""

config = {
    # ========== 可复现性 ==========
    "seed": 3047,
    "device": "cuda",

    # ========== 项目信息 ==========
    "project_name": "Building-Segmentation-3Bands",
    "model_name": "STDSNet",
    "description": "STDSNet model for 3-band building segmentation",
    "tags": ["STDSNet", "building-segmentation", "RGB", "Swin"],

    # ========== 数据配置 ==========
    "data_root": "/mnt/data1/rove/asset/GF7_Building/3Bands",
    "batch_size": 1,
    "num_workers": 4,
    "image_size": 512,
    "input_channels": 3,
    "use_nir": False,
    "num_classes": 2,

    # ========== 模型配置 ==========
    "encoder_pretrained": True,

    # ========== 训练配置 ==========
    "num_epochs": 400,
    "learning_rate": 5e-4,
    "min_lr": 1e-6,
    "weight_decay": 1e-3,
    "warmup_epochs": 10,

    # ========== 损失函数配置 ==========
    "shape_loss_weight": 0.4,
    "dice_ratio": 0.5,

    # ========== 输出配置 ==========
    "save_dir": "./runs",
}
