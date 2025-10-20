config = {
    # --- Project & Experiment Info ---
    "project_name": "Building-Segmentation-3Bands",
    "model_name": "HDNet",
    "description": "HDNet model for 3-band building segmentation with W&B logging.",
    "tags": ["HDNet", "building-segmentation", "RGB", "high-resolution"],

    # --- Data Paths & Settings ---
    "data_root": "/mnt/data1/rove/asset/GF7_Building/3Bands",
    "image_size": 512,
    "input_channels": 3,
    "use_nir": False,
    "num_classes": 2,

    # --- Training Hyperparameters ---
    "batch_size": 6,
    "num_workers": 4,
    "num_epochs": 300,
    "seed": 42,
    
    # --- Optimizer & Scheduler Params (Upgraded) ---
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 1,
    "min_lr": 1e-6,
    
    # --- Model Specific ---
    "base_channel": 48,
    
    # --- System ---
    "save_dir": "./runs",
}
