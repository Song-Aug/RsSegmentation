"""
TransCCV3 消融实验: 无 CBAM 注意力模块

验证 CBAM 注意力融合机制的有效性
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
    "model_name": "TransCCV3_Ablation_No_CBAM",
    "num_classes": 2,
    "patch_size": 16,
    "embed_dim": 768,
    "depth": 9,
    "num_heads": 12,
    "hdnet_base_channel": 48,

    # ========== 预训练权重 ==========
    "pretrained_weights": None,  # 消融实验不加载预训练
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

    # ========== 损失函数配置 ==========
    "seg_weight": 1.0,
    "boundary_weight": 1.0,
    "aux_weight": 0.4,
    "dice_focal_ratio": 0.5,
    "bce_dice_ratio": 0.5,

    # ========== 编码器开关 ==========
    "use_transformer": True,
    "use_cnn": True,

    # ========== CBAM 开关（消融实验）==========
    "use_cbam": False,  # 关闭 CBAM

    # ========== 输出配置 ==========
    "save_dir": "./runs",
    "wandb_project": "Building-Segmentation-3Bands",
}
