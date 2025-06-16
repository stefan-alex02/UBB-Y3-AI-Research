standard_cnn_fixed_params = {
    # --- Skorch/Training Loop Parameters ---
    'max_epochs': 60,
    'lr': 1e-4,
    'batch_size': 32,

    # --- Optimizer Configuration ---
    'optimizer': 'AdamW',
    'optimizer__weight_decay': 0.05,

    # --- LR Scheduler ---
    'callbacks__default_lr_scheduler__policy': 'ReduceLROnPlateau',
    'callbacks__default_lr_scheduler__monitor': 'valid_loss',
    'callbacks__default_lr_scheduler__patience': 5,
    'callbacks__default_lr_scheduler__factor': 0.2,
    'callbacks__default_lr_scheduler__min_lr': 1e-7,
    'callbacks__default_lr_scheduler__mode': 'min',

    # CosineAnnealingLR
    # 'callbacks__default_lr_scheduler__policy': 'CosineAnnealingLR',
    # 'callbacks__default_lr_scheduler__T_max': 50, # Match max_epochs
    # 'callbacks__default_lr_scheduler__eta_min': 1e-7,

    # --- Early Stopping ---
    'callbacks__default_early_stopping__patience': 15,

    # --- CutMix Parameters ---
    'cutmix_alpha': 1.0,
    'cutmix_probability': 0.9,  # for CCSN
    # 'cutmix_probability': 0.3, # for Swimcat
    # 'cutmix_probability': 0.5, # for GCD

    # --- Label Smoothing ---
    'criterion__label_smoothing': 0.1,

    # --- Module Parameters for StandardCNNFeatureExtractor ---
    'module__model_name': "efficientnet_b0",
    'module__pretrained': True,
    'module__output_channels_target': None,

    'module__freeze_extractor': False,
    'module__num_frozen_stages': 3,
}
