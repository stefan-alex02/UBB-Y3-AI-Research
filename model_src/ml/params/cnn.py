# This file contains the parameters used for the CNN model in the ML pipeline.

# --- Fixed Parameter Sets ---

cnn_fixed_params = {
    # Skorch params
    'max_epochs': 15,
    'lr': 0.001,
    'batch_size': 32,
    'optimizer__weight_decay': 0.01
}

# --- Parameter Space Definitions ---

cnn_param_grid = {
    # Skorch parameters
    'lr': [0.005, 0.001, 0.0005],
    'batch_size': [16, 32],  # Note: Changing batch size can affect memory and convergence

    # Optimizer (AdamW) parameters
    'optimizer__weight_decay': [0.01, 0.001, 0.0001],
    # 'optimizer__betas': [(0.9, 0.999), (0.85, 0.99)], # Less common to tune

    # Module (SimpleCNN) parameters
    'module__dropout_rate': [0.3, 0.4, 0.5, 0.6],  # Tune dropout in the classifier head

    # Maybe max_epochs if not using EarlyStopping effectively? Usually fixed or high w/ early stopping.
    # 'max_epochs': [15, 25],
}
