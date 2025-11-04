# -*- coding: utf-8 -*-
# 配置实验参数
config = {
    # --- Project Info (与新范式统一) ---
    'project_name': 'Building-Segmentation-3Bands',
    'description': 'UNetFormer 建筑物分割实验 (3波段)',
    'tags': ["UNetFormer", "building-segmentation", "RGB"],
    'model_name': 'unetformer_3bands',
    
    # --- Data Paths & Settings ---
    'data_root': '/mnt/sda1/songyufei/GF7_Building/3Bands',
    'image_size': 512,
    'input_channels': 3,
    'use_nir': False,
    'num_classes': 2,

    # --- Training Hyperparameters ---
    'batch_size': 4,
    'num_epochs': 200,
    'seed': 42,

    # --- Optimizer & Scheduler Params (新范式) ---
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'warmup_epochs': 10,
    'min_lr': 1e-6,

    # --- Model Specific ---
    'backbone': 'efficientnet_b0',
    'pretrained': False,
    'use_aux_loss': True,
    'aux_loss_weight': 0.4,
    
    # --- System ---
    'num_workers': 4,
    'save_dir': './runs', 
}