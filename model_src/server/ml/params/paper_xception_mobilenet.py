# In a new file, e.g., server/ml/params/xcloud_params.py
xcloud_fixed_params = {
    'max_epochs': 100,  # For CCSN dataset as per paper. Adjust to 30 for GCD.
    'lr': 1e-4,
    'batch_size': 32, # Paper uses 32, Keras code uses 8. Let's stick to paper's for training.

    'optimizer': 'Adam', # Paper specifies Adam. Keras code confirms.
    # Adam in PyTorch doesn't have weight_decay intrinsically like AdamW.
    # If L2 reg from Keras (0.01) is desired, it's tricky.
    # Common practice: Use AdamW if weight decay is important, or Adam if not.
    # Let's use AdamW for consistency with your other models and apply weight decay.
    # 'optimizer': 'AdamW',
    # 'optimizer__weight_decay': 0.01, # To somewhat mimic Keras L2

    'callbacks__default_lr_scheduler__policy': 'ReduceLROnPlateau',
    'callbacks__default_lr_scheduler__monitor': 'valid_loss',
    'callbacks__default_lr_scheduler__patience': 10, # Paper: "decay of 0.2 after 10 epochs without any validation loss improvements"
    'callbacks__default_lr_scheduler__factor': 0.2,
    'callbacks__default_lr_scheduler__min_lr': 1e-8,

    'callbacks__default_early_stopping__patience': 15, # Example: stop if no val_loss improvement for 20 epochs after LR reduction

    # Module parameters for XceptionBasedCloudNet
    'module__pretrained': True,
    'module__dense_neurons1': 128,
    # 'module__l2_reg': 0.01, # Not directly used by PyTorch Linear layer, handled by optimizer weight_decay

    # 'use_amp': True, # Consider adding for performance
}

# In a new file, e.g., server/ml/params/mcloud_params.py
mcloud_fixed_params = {
    'max_epochs': 100,  # For CCSN. Adjust to 30 for GCD.
    'lr': 1e-4,
    'batch_size': 32,

    'optimizer': 'Adam',
    # 'optimizer': 'AdamW',
    # 'optimizer__weight_decay': 0.01,

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
