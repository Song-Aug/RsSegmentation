config = {
    'data_root': '/mnt/data1/rove/asset/GF7_Building/3Bands',
    'batch_size': 12,
    'num_workers': 4,

    'image_size': 512,
    'input_channels': 3,
    'use_nir': False,
    'pretrained_weights': './pretrained_weights/vit_base_patch16_224.pth',
    'fusion_strategy': 'interpolate',

    'num_epochs': 300,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'save_dir': './runs',
    'model_name': 'TransCC_V2',
    'num_classes': 2,
    'seed': 42,
    'warmup_epochs': 5,
    'min_lr': 1e-6
}
