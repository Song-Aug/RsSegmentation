# -*- coding: utf-8 -*-
# 配置实验参数
config = {
    'data_root': '/mnt/sda1/songyufei/asset/GF7_Building/3Bands',
    'batch_size': 4,
    'num_workers': 4,
    'image_size': 512,
    'input_channels': 3,  # RGB图像
    'use_nir': False,
    'num_epochs': 200,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'save_dir': './runs',
    'model_name': 'MSSDMPA_Net',
    'num_classes': 1,
    'seed': 42
}
