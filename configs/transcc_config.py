# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/26 21:21
@Auth ： Song
@File ：transcc_config.py
@IDE ：PyCharm
@Motto：No pains, no gains
"""
# 配置实验参数
config = {
    'data_root': '/mnt/data1/rove/asset/GF7_Building/3Bands',
    'batch_size': 16,
    'num_workers': 4,
    'image_size': 512,
    'input_channels': 3,  # RGB + NIR
    'use_nir': False,
    'num_epochs': 450,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'save_dir': './runs',
    'model_name': 'TransCC',
    # 'backbone': 'efficientnet_b0',
    # 'pretrained': False,
    'num_classes': 2,
    'seed': 42
}
