cloudnet_fixed_params = {
    # --- Skorch/Training Loop Parameters ---
    'max_epochs': 200,
    'lr': 1e-3,
    'batch_size': 64,

    # --- Optimizer Configuration ---
    'optimizer': 'SGD',
    'optimizer__momentum': 0.9,
    'optimizer__weight_decay': 5e-4,

    # 'optimizer': 'AdamW',
    # 'optimizer__weight_decay': 1e-2,


    # --- LR Scheduler ---
    'callbacks__default_lr_scheduler__policy': 'ReduceLROnPlateau',
    'callbacks__default_lr_scheduler__monitor': 'valid_loss',
    'callbacks__default_lr_scheduler__patience': 10,
    'callbacks__default_lr_scheduler__factor': 0.1,
    'callbacks__default_lr_scheduler__min_lr': 1e-6,
    'callbacks__default_lr_scheduler__mode': 'min',

    # if max_epochs = 200:
    # 'callbacks__default_lr_scheduler__policy': 'MultiStepLR',
    # 'callbacks__default_lr_scheduler__milestones': [80, 150], # Drop at ~40% and ~75% of epochs
    # 'callbacks__default_lr_scheduler__gamma': 0.1,


    # --- Early Stopping ---
    'callbacks__default_early_stopping__patience': 20,

    # --- Module Parameters for CloudNetPyTorch ---
    'module__dropout_p': 0.5,
}