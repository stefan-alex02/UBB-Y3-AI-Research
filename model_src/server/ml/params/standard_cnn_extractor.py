# In a new params file: server/ml/params/standard_cnn_standalone_params.py

standard_cnn_fixed_params = {
    # --- Skorch/Training Loop Parameters ---
    'max_epochs': 60,  # Fine-tuning might converge faster than training from scratch.
    # Early stopping is key.
    'lr': 1e-4,  # <<< Lower LR for fine-tuning a pre-trained model.
    # Common range for fine-tuning: 1e-5 to 5e-4.
    'batch_size': 32,  # EfficientNet-B0 is efficient. 32 or 64 should be fine.

    # --- Optimizer Configuration ---
    'optimizer': 'AdamW',
    'optimizer__weight_decay': 0.05,  # Standard weight decay for fine-tuning.
    # Can go up to 0.05 if strong regularization is needed.

    # --- LR Scheduler ---
    'callbacks__default_lr_scheduler__policy': 'ReduceLROnPlateau',  # Good for fine-tuning
    'callbacks__default_lr_scheduler__monitor': 'valid_loss',
    'callbacks__default_lr_scheduler__patience': 5,  # Reduce LR if no val_loss improvement for 5 epochs
    'callbacks__default_lr_scheduler__factor': 0.2,  # Reduce LR by a factor of 0.2
    'callbacks__default_lr_scheduler__min_lr': 1e-7,
    'callbacks__default_lr_scheduler__mode': 'min',
    # Alternative: CosineAnnealingLR
    # 'callbacks__default_lr_scheduler__policy': 'CosineAnnealingLR',
    # 'callbacks__default_lr_scheduler__T_max': 50, # Match max_epochs
    # 'callbacks__default_lr_scheduler__eta_min': 1e-7,

    # --- Early Stopping ---
    'callbacks__default_early_stopping__patience': 15,  # Stop if no improvement for 10 epochs (after LR drops)

    # --- Label Smoothing ---
    'criterion__label_smoothing': 0.1,  # Can help prevent overconfidence

    # --- Module Parameters for StandardCNNFeatureExtractor ---
    'module__model_name': "efficientnet_b0",
    'module__pretrained': True,  # Crucial: Use ImageNet pre-trained weights as starting point.
    'module__output_channels_target': None,  # For standalone, we don't need to project yet.
    # The standalone_head will attach to natural output channels.
    # Or, if you want to test the projection:
    # 'module__output_channels_target': 1280 (same as natural for EffNetB0)
    # or even project down, e.g. 512, then head from 512.
    # Simplest is None or matching natural output for now.

    # Fine-tuning strategy for the EfficientNet-B0 backbone itself during this standalone training:
    'module__freeze_extractor': False,  # We want to fine-tune it.
    'module__num_frozen_stages': 3,  # Example: Freeze first 2 "stages" (stem + first block group).
    # Fine-tune the rest. Adjust this (0 to ~4).
    # 0 means fine-tune all of EffNet-B0.

    # 'module__num_classes_for_standalone': Will be set by Skorch/Pipeline from dataset_handler.num_classes
}
