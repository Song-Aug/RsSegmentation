# -*- coding: utf-8 -*-
# 配置实验参数
config = {
    # --- Project Info ---
    'project_name': 'Building-Segmentation-3Bands',
    'description': 'MSSDMPA-Net 建筑物分割实验 (更新版)',
    'tags': ["MSSDMPA-Net", "building-segmentation", "RGB"],
    
    # --- Data Paths ---
    'data_root': '/mnt/data1/rove/asset/GF7_Building/3Bands',
    
    # --- Model & Training Params ---
    'model_name': 'MSSDMPA_Net',
    'input_channels': 3,  # RGB图像
    'num_classes': 1,
    'image_size': 512,
    'batch_size': 4,
    'num_epochs': 300,
    'use_nir': False,
    'seed': 42,

    # --- Optimizer & Scheduler Params ---
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'warmup_epochs': 10,
    'min_lr': 1e-6,

    # --- System ---
    'num_workers': 4,
    'save_dir': './runs',
}