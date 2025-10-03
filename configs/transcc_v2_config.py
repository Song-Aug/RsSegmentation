config = {
    'data_root': '/mnt/data1/rove/asset/GF7_Building/3Bands',
    'batch_size': 12,
    'num_workers': 4,

    'image_size': 512,
    'input_channels': 3,
    'use_nir': False,
    'pretrained_weights': './pretrained_weights/vit_base_patch16_224.pth',
    'fusion_strategy': 'interpolate',
    'drop_rate': 0.1,
    'attn_drop_rate': 0.1,
    'drop_path_rate': 0.15,

    'num_epochs': 300,
    'learning_rate': 5e-4,
    'weight_decay': 5e-4,
    'save_dir': './runs',
    'model_name': 'TransCC_V2',
    'num_classes': 2,
    'seed': 3047,
    'warmup_epochs': 10,
    'min_lr': 1e-6
}
