# In main.py or a new params file (e.g., server/ml/params/paper_cnn_standalone.py)

paper_cnn_standalone_fixed_params = {
    # --- Skorch/Training Loop Parameters ---
    'max_epochs': 60,          # Train for a decent number of epochs; EarlyStopping will help.
                              # Could be 50-100 depending on dataset size and convergence.
    'lr': 1e-3,                # AdamW often starts well here for training from scratch.
                              # Could also try 5e-4.
    'batch_size': 16,          # A common default. Try 64 if memory allows and it helps.
                              # For your GPU, 32 or 64 should be fine for this CNN size.

    # --- Optimizer Configuration ---
    'optimizer': 'AdamW',      # Good general-purpose optimizer.
    'optimizer__weight_decay': 1e-2, # (0.01) Common weight decay for AdamW.
                                   # Could be a bit higher (e.g., 0.05) if overfitting is observed.
    'optimizer__betas': (0.9, 0.999), # Standard AdamW betas.

    # --- LR Scheduler (Important for training from scratch) ---
    'callbacks__default_lr_scheduler__policy': 'CosineAnnealingLR', # Smooth decay, often good.
    'callbacks__default_lr_scheduler__T_max': 60, # Should match max_epochs for full cosine cycle.
    'callbacks__default_lr_scheduler__eta_min': 1e-6, # Smallest LR at the end of annealing.
    # Alternatively, ReduceLROnPlateau:
    # 'callbacks__default_lr_scheduler__policy': 'ReduceLROnPlateau',
    # 'callbacks__default_lr_scheduler__monitor': 'valid_loss',
    # 'callbacks__default_lr_scheduler__patience': 5, # Number of epochs with no improvement to wait.
    # 'callbacks__default_lr_scheduler__factor': 0.2, # Factor by which LR is reduced.
    # 'callbacks__default_lr_scheduler__min_lr': 1e-6,

    'callbacks__default_early_stopping__patience': 15, # Number of epochs with no improvement before stopping.

    # --- Module Parameters for PaperCNNFeatureExtractor ---
    # 'module__in_channels': 3, # This is usually fixed based on image data.
    # 'module__num_classes_for_standalone': will be set by Skorch via pipeline
    # (based on dataset_handler.num_classes)

    # --- Optional: Augmentations ---
    # The pipeline's `augmentation_strategy` will be used by default.
    # Ensure it's a reasonably strong one for training from scratch.
    # Your GROUND_AWARE_NO_ROTATION or SKY_ONLY_ROTATION are good.

    # --- Optional: Gradient Clipping (can help stabilize training from scratch) ---
    # 'gradient_clip_value': 1.0, # If you experience exploding gradients.

    # --- Optional: Mixed Precision Training ---
    # 'use_amp': True, # Recommended for speed and memory, especially on Turing+ GPUs.
}
