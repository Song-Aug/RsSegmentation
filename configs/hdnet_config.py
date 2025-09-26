# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/27 10:00
@Auth ： Song
@File ：hdnet_config.py
@IDE ：PyCharm
@Motto：No pains, no gains
"""
# 配置实验参数
config = {
    'data_root': '/mnt/data1/rove/asset/GF7_Building/3BandsSample',
    'batch_size': 1,
    'num_workers': 4,
    'image_size': 512,
    'input_channels': 3,  # RGB
    'use_nir': False,
    'num_epochs': 120,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'save_dir': './runs',
    'model_name': 'hdnet_3bands',
    'base_channel': 48,
    'num_classes': 2,  # 2类：背景(0)和建筑物(1)
    'seed': 42
}
