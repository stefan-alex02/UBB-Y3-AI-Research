resnet18_cloud_fixed_params = {
    # --- Skorch/Training Loop Parameters ---
    'max_epochs': 100,
    'lr': 0.001,
    'batch_size': 16,

    # --- Optimizer Configuration ---
    'optimizer': 'SGD',
    'optimizer__momentum': 0.9,
    'optimizer__weight_decay': 5e-4,

    # --- LR Scheduler ---
    'callbacks__default_lr_scheduler__policy': 'MultiStepLR',
    'callbacks__default_lr_scheduler__milestones': [30, 50, 70],
    'callbacks__default_lr_scheduler__gamma': 0.1,

    # --- Early Stopping ---
    'callbacks__default_early_stopping__patience': 15,
    'callbacks__default_early_stopping__monitor': 'valid_acc',
    'callbacks__default_early_stopping__lower_is_better': False,

    # --- CutMix Parameters ---
    'cutmix_alpha': 1.0,
    'cutmix_probability': 0.5,

    # --- Gradient Clipping ---
    'gradient_clip_value': 5.0,


    # --- Module Parameters for ResNet18BasedCloud ---
    'module__pretrained': True,
    'module__dropout_rate_fc': 0.3,
    'module__fc_hidden_neurons': 128,

    'iterator_train__drop_last': True, # To avoid batchnorm issues with one-sample batches
}
