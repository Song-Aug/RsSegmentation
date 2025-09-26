# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/27 10:30
@Auth ： Song
@File ：unetformer_config.py
@IDE ：PyCharm
@Motto：No pains, no gains
"""
# 配置实验参数
config = {
    'data_root': '/mnt/sda1/songyufei/asset/GF7_Building/3Bands',
    'batch_size': 12,
    'num_workers': 4,
    'image_size': 512,
    'input_channels': 3,  # RGB + NIR
    'use_nir': False,
    'num_epochs': 200,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'save_dir': './runs',
    'model_name': 'unetformer_3bands',
    'backbone': 'efficientnet_b0',
    'pretrained': False,
    'num_classes': 2,
    'seed': 42,
    'use_aux_loss': True, # UNetFormer specific
    'aux_loss_weight': 0.4 # UNetFormer specific
}
