shufflenet_cloud_fixed_params = {
    # --- Skorch/Training Loop Parameters ---
    'max_epochs': 50,       # Based on paper's Table 1 & Fig 7-9
    'lr': 1e-3,             # A common starting LR for fine-tuning CNNs like ShuffleNet with Adam/AdamW.
                            # The paper's 1e-4 was for their full X-Cloud/M-Cloud system.
                            # This might need tuning (1e-3, 5e-4, 1e-4).
    'batch_size': 64,       # As per paper's Table 1 description for ShuffleNet.

    # --- Optimizer Configuration ---
    'optimizer': 'AdamW',   # Good default for fine-tuning. Paper used Adam for overall.
    'optimizer__weight_decay': 1e-2, # (0.01) Reasonable weight decay.

    # --- LR Scheduler ---
    # Let's use ReduceLROnPlateau as it's adaptive and common for fine-tuning.
    'callbacks__default_lr_scheduler__policy': 'ReduceLROnPlateau',
    'callbacks__default_lr_scheduler__monitor': 'valid_loss',
    'callbacks__default_lr_scheduler__patience': 5,  # Reduce LR if no val_loss improvement for 5 epochs
    'callbacks__default_lr_scheduler__factor': 0.2,  # Reduce LR by a factor of 0.2
    'callbacks__default_lr_scheduler__min_lr': 1e-7,
    'callbacks__default_lr_scheduler__mode': 'min',

    # --- Early Stopping ---
    'callbacks__default_early_stopping__patience': 10, # Stop after 10 epochs of no improvement post LR drop.

    # --- Module Parameters for ShuffleNetCloud ---
    'module__pretrained': True, # Use ImageNet pre-trained weights

    # --- Optional: Label Smoothing, Gradient Clipping, AMP ---
    # 'criterion__label_smoothing': 0.1,
    # 'gradient_clip_value': 1.0,
    # 'use_amp': True,
}