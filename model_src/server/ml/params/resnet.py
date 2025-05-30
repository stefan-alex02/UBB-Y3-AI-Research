resnet18_cloud_fixed_params = {
    # --- Skorch/Training Loop Parameters ---
    'max_epochs': 100, # From script's num_epochs_train
    'lr': 0.001,       # From script's optimizer_ft
    'batch_size': 16,  # From script's config_batch_size

    # --- Optimizer Configuration ---
    'optimizer': 'SGD', # From script's optimizer_ft = optim.SGD(...)
    'optimizer__momentum': 0.9,
    'optimizer__weight_decay': 5e-4, # L2 penalty

    # --- LR Scheduler ---
    'callbacks__default_lr_scheduler__policy': 'MultiStepLR', # From script
    'callbacks__default_lr_scheduler__milestones': [30, 50, 70], # From script
    'callbacks__default_lr_scheduler__gamma': 0.1,             # From script

    # --- Early Stopping (Script doesn't use explicit early stopping based on patience, but saves best model) ---
    'callbacks__default_early_stopping__patience': 15,
    'callbacks__default_early_stopping__monitor': 'valid_acc', # Script saves based on best val_acc
    'callbacks__default_early_stopping__lower_is_better': False,

    # --- CutMix Parameters ---
    'cutmix_alpha': 1.0,  # Value for beta distribution (alpha=beta=1.0 is common)
    'cutmix_probability': 0.5,  # Apply CutMix to 50% of training batches

    # --- Gradient Clipping (already discussed) ---
    'gradient_clip_value': 5.0,  # If you want to use it


    # --- Module Parameters for ResNet18BasedCloud ---
    'module__pretrained': True,
    'module__dropout_rate_fc': 0.3,
    'module__fc_hidden_neurons': 128,
}
