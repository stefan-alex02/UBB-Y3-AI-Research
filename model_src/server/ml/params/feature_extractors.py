paper_cnn_standalone_fixed_params = {
    # --- Skorch/Training Loop Parameters ---
    'max_epochs': 60,

    'lr': 1e-3,

    'batch_size': 16,


    # --- Optimizer Configuration ---
    'optimizer': 'AdamW',
    'optimizer__weight_decay': 1e-2,

    'optimizer__betas': (0.9, 0.999),

    # --- LR Scheduler ---
    'callbacks__default_lr_scheduler__policy': 'CosineAnnealingLR',
    'callbacks__default_lr_scheduler__T_max': 60,
    'callbacks__default_lr_scheduler__eta_min': 1e-6,

    # 'callbacks__default_lr_scheduler__policy': 'ReduceLROnPlateau',
    # 'callbacks__default_lr_scheduler__monitor': 'valid_loss',
    # 'callbacks__default_lr_scheduler__patience': 5,
    # 'callbacks__default_lr_scheduler__factor': 0.2,
    # 'callbacks__default_lr_scheduler__min_lr': 1e-6,

    'callbacks__default_early_stopping__patience': 15,
}
