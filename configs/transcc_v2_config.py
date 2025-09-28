# -*- coding: utf-8 -*-
# TransCC V2 训练配置
config = {
    'data_root': '/mnt/data1/rove/asset/GF7_Building/3Bands',
    'batch_size': 16,
    'num_workers': 4,
    'image_size': 512,
    'input_channels': 3,
    'use_nir': False,
    'num_epochs': 200,
    'learning_rate': 5e-4,
    'weight_decay': 1e-4,
    'save_dir': './runs',
    'model_name': 'TransCC_V2',
    'num_classes': 2,
    'seed': 42,
    'warmup_epochs': 5,
    'min_lr': 1e-6
}
