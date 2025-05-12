# This file contains the parameters used for the Diffusion Classifier model in the ML pipeline.

# --- Fixed Parameter Sets ---

# TODO: fixed params for diffusion classifier (TBA)

# --- Parameter Space Definitions ---

# TODO: adjust parameter space for diffusion classifier (TBA)

diffusion_param_grid = { # TODO: update with actual params
    # Skorch parameters
    'lr': [0.001, 0.0005, 0.0001],  # Fine-tuning learning rate
    'batch_size': [16, 32, 64],  # ResNet might be less memory-intensive than ViT

    # Optimizer (AdamW) parameters
    'optimizer__weight_decay': [0.01, 0.001, 0.0001],

    # Module (DiffusionClassifier) parameters
    'module__dropout_rate': [0.3, 0.4, 0.5, 0.6],  # Tune dropout in the custom head

    # Training duration
    # 'max_epochs': [10, 20, 30],
}
