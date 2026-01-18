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

resnet18_finetune_best_practice_params = {
    # --- Skorch/Training Loop Parameters ---
    'max_epochs': 60,  # A good starting point for fine-tuning. Early stopping will handle convergence.
    'lr': 1e-4,  # <<< A more standard fine-tuning LR for AdamW.
    # Start here. If it overfits too quickly, try 5e-5. If it learns too slowly, try 2e-4.
    'batch_size': 32,  # 16 is fine, but 32 or 64 can provide more stable gradients if memory allows.

    # --- Optimizer Configuration ---
    'optimizer': 'AdamW',  # <<< AdamW is often preferred over SGD for fine-tuning Transformers and modern CNNs.
    'optimizer__weight_decay': 0.01,  # Standard weight decay for AdamW.

    # --- LR Scheduler ---
    'callbacks__default_lr_scheduler__policy': 'CosineAnnealingLR',  # <<< Often outperforms MultiStepLR
    'callbacks__default_lr_scheduler__T_max': 60,  # Match max_epochs for a full cosine cycle
    'callbacks__default_lr_scheduler__eta_min': 1e-6,  # End LR

    # --- Early Stopping ---
    'callbacks__default_early_stopping__patience': 15,  # Wait 15 epochs after best result before stopping.
    'callbacks__default_early_stopping__monitor': 'valid_loss',  # Monitor validation loss for saving the best model
    'callbacks__default_early_stopping__lower_is_better': True,

    # --- Advanced Regularization (Keep these as they are very effective) ---
    'criterion__label_smoothing': 0.1,

    'cutmix_alpha': 1.0,
    'cutmix_probability': 0.5,
    # A more standard probability. 0.9 is very high and might be over-regularizing. Start with 0.5.

    # --- Gradient Clipping (Optional but good for stability) ---
    'gradient_clip_value': 1.0,

    # --- Module Parameters for ResNet18BasedCloud ---
    'module__pretrained': True,
    'module__dropout_rate_fc': 0.5,  # <<< Increased dropout in the head for more regularization
    'module__fc_hidden_neurons': 256,  # <<< Slightly larger head for more capacity before the final layer

    # --- DataLoader Parameters ---
    'iterator_train__drop_last': True,  # Keep this to prevent BatchNorm errors
}