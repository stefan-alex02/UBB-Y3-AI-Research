xcloud_fixed_params = {
    'max_epochs': 100,
    'lr': 1e-4,
    'batch_size': 32,

    'optimizer': 'AdamW',
    'optimizer__weight_decay': 0.2,

    'callbacks__default_lr_scheduler__policy': 'ReduceLROnPlateau',
    'callbacks__default_lr_scheduler__monitor': 'valid_loss',
    'callbacks__default_lr_scheduler__patience': 10,
    'callbacks__default_lr_scheduler__factor': 0.2,
    'callbacks__default_lr_scheduler__min_lr': 1e-8,

    'callbacks__default_early_stopping__patience': 15,

    # Module parameters for XceptionBasedCloudNet
    'module__pretrained': True,
    'module__dense_neurons1': 128,

    # 'use_amp': True,
}

mcloud_fixed_params = {
    'max_epochs': 100,
    'lr': 1e-4,
    'batch_size': 32,

    'optimizer': 'Adam',

    'callbacks__default_lr_scheduler__policy': 'ReduceLROnPlateau',
    'callbacks__default_lr_scheduler__monitor': 'valid_loss',
    'callbacks__default_lr_scheduler__patience': 10,
    'callbacks__default_lr_scheduler__factor': 0.2,
    'callbacks__default_lr_scheduler__min_lr': 1e-8,

    'callbacks__default_early_stopping__patience': 15,

    # Module parameters for MobileNetBasedCloudNet
    'module__pretrained': True,
    'module__dense_neurons1': 128,
    # 'module__l2_reg': 0.01,

    # 'use_amp': True,
}
