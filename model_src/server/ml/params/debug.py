# This file contains the parameters used for debugging the pipeline.
import torch

# --- Fixed Parameter Sets ---

debug_fixed_params = {
    # Skorch/Training Loop Params (Fast Execution)
    'max_epochs': 4,        # Run only for 1 epoch
    'lr': 0.01,
    'optimizer': 'sgd',
    'optimizer__momentum': 0.9,
    'optimizer__weight_decay': 5e-4,
    'optimizer__nesterov': True,
    'batch_size': 32,

    # 'module__dropout_rate': 0.3 # For CNNs
    'module__head_dropout_rate': 0.3 # For ViTs
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
