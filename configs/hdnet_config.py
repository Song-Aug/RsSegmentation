# -*- coding: utf-8 -*-
# 配置实验参数
config = {
    'data_root': '/mnt/data1/rove/asset/GF7_Building/3Bands',
    'batch_size': 6,
    'num_workers': 4,
    'image_size': 512,
    'input_channels': 3,  # RGB
    'use_nir': False,
    'num_epochs': 300,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'save_dir': './runs',
    'model_name': 'hdnet_3bands',
    'base_channel': 48,
    'num_classes': 2,
    'seed': 42
}
