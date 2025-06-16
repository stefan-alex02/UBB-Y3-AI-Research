shufflenet_cloud_fixed_params = {
    # --- Skorch/Training Loop Parameters ---
    'max_epochs': 50,
    'lr': 1e-4,
    'batch_size': 64,

    # --- Optimizer Configuration ---
    'optimizer': 'AdamW',
    'optimizer__weight_decay': 1e-2,

    # --- LR Scheduler ---
    'callbacks__default_lr_scheduler__policy': 'ReduceLROnPlateau',
    'callbacks__default_lr_scheduler__monitor': 'valid_loss',
    'callbacks__default_lr_scheduler__patience': 5,
    'callbacks__default_lr_scheduler__factor': 0.2,
    'callbacks__default_lr_scheduler__min_lr': 1e-7,
    'callbacks__default_lr_scheduler__mode': 'min',

    # --- Early Stopping ---
    'callbacks__default_early_stopping__patience': 10,

    # --- Module Parameters for ShuffleNetCloud ---
    'module__pretrained': True,
}