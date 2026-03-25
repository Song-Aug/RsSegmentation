"""
TransCCV3 实验配置

集中管理所有实验超参数和路径配置，确保实验可复现。
"""

config = {
    # ========== 可复现性 ==========
    "seed": 3047,
    "device": "cuda",

    # ========== 数据配置 ==========
    "data_root": "/mnt/data1/rove/asset/GF7_Building/3Bands",
    "batch_size": 4,
    "num_workers": 4,
    "image_size": 512,
    "input_channels": 3,
    "use_nir": False,

    # ========== 模型配置 ==========
    "model_name": "TransCC_V3",
    "num_classes": 2,
    "patch_size": 16,
    "embed_dim": 768,
    "depth": 9,
    "num_heads": 12,
    "hdnet_base_channel": 48,

    # ========== 预训练权重 ==========
    "pretrained_weights": "./pretrained_weights/vit_base_patch16_224.pth",
    "fusion_strategy": "interpolate",

    # ========== 正则化 ==========
    "drop_rate": 0.1,
    "attn_drop_rate": 0.1,
    "drop_path_rate": 0.15,

    # ========== 训练配置 ==========
    "num_epochs": 500,
    "mild_aug_epoch": 400,
    "learning_rate": 5e-4,
    "min_lr": 1e-6,
    "weight_decay": 1e-3,
    "warmup_epochs": 10,

    # ========== 输出配置 ==========
    "save_dir": "./runs",
    "wandb_project": "Building-Segmentation-3Bands",
}
