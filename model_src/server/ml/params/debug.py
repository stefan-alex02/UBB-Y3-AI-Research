# This file contains the parameters used for debugging the pipeline.

# --- Fixed Parameter Sets ---

debug_fixed_params = {
    # Skorch/Training Loop Params (Fast Execution)
    'max_epochs': 4,        # Run only for 1 epoch
    'batch_size': 8,        # Small batch size
    'lr': 1e-4,             # A reasonably small LR unlikely to explode immediately

    # Optimizer Params (Common Default)
    'optimizer__weight_decay': 0.01,

    # Callbacks (using pipeline defaults, but could potentially disable for speed if needed)
    # Note: EarlyStopping might trigger immediately if validation loss increases on epoch 1.

    # --- NO module__ specific parameters here ---
    # This ensures compatibility. FlexibleViT, SimpleCNN, DiffusionClassifier
    # will use their defaults for dropout, unfreezing, etc., as set in
    # ClassificationPipeline.__init__'s use of model_adapter_config.
}

# --- Parameter Space Definitions ---

debug_param_grid = {
    # Skorch/Training Loop Params (Single Combination)
    'max_epochs': [4],        # List with one value
    'batch_size': [8],        # List with one value
    'lr': [1e-4, 0.001],  # List with one value

    # Optimizer Params (Single Combination)
    'optimizer__weight_decay': [0.01], # List with one value

    # --- NO module__ specific parameters here ---
    # Ensures compatibility across different model types selected for the pipeline.
    # The specific module (CNN, ViT, etc.) will use its defaults for other params.
}
