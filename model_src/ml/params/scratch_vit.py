# This file contains the parameters used for the Sratch ViT model in the ML pipeline.

# --- Fixed Parameter Sets ---

fixed_params_vit_scratch = {
    # Skorch/Training Loop Params
    'max_epochs': 75, # Training from scratch needs more epochs
    'lr': 1e-3,       # Starting LR for training from scratch (AdamW often uses 1e-3 or 3e-4)
                      # Needs a good LR scheduler (e.g., cosine annealing)
    'batch_size': 64, # Or 128, 256 depending on GPU. ViT-Base needs decent batch size.
    'optimizer__weight_decay': 0.05, # Weight decay is important

    # ViTFromScratch Module Params (prefixed with 'module__')
    # 'module__num_classes': YOUR_NUM_CLASSES, # Set by pipeline

    'module__img_size': 224,
    'module__patch_size': 16,
    'module__in_channels': 3,
    'module__embed_dim': 768,  # ViT-Base embed_dim
    'module__depth': 12,       # ViT-Base depth
    'module__num_heads': 12,     # ViT-Base num_heads
    'module__mlp_ratio': 4.0,
    'module__attention_dropout': 0.0, # Or 0.1
    'module__projection_dropout': 0.0, # Or 0.1 (applied after MHA and MLP's second FC)
    'module__mlp_dropout': 0.0, # Dropout within MLP hidden layers (between fc1 and fc2)

    'module__pos_embedding_type': 'learnable', # 'learnable' is standard for ViT

    'module__head_hidden_dims': None,       # Simple linear head
    'module__head_dropout_rate': 0.0,      # Can add if desired, e.g., 0.1 or 0.2
}

# --- Parameter Space Definitions ---

param_grid_vit_from_scratch = {
    # Skorch/Training Loop Params
    'lr': [1e-4, 3e-4, 1e-3],
    'batch_size': [32, 64],
    'optimizer__weight_decay': [0.01, 0.05, 0.1],
    'max_epochs': [50, 100, 150], # Long training

    # ViTFromScratch Module Params (module__)
    'module__embed_dim': [512, 768],                 # Smaller vs. Base-like dimension
    'module__depth': [6, 8, 12],                     # Number of encoder blocks
    'module__num_heads': [8, 12],                    # Must be divisor of embed_dim
    'module__mlp_ratio': [3.0, 4.0],
    'module__projection_dropout': [0.0, 0.1],        # Dropout after MHA/MLP projections
    'module__attention_dropout': [0.0, 0.1],         # Dropout in attention mechanism
    'module__mlp_dropout': [0.0, 0.1],               # Dropout within MLP layers

    # Head configuration
    'module__head_hidden_dims': [None, [256], [512, 256]],
    'module__head_dropout_rate': [0.0, 0.2, 0.5],

    # Fixed for this grid example, but could be tuned:
    'module__img_size': [224],
    'module__patch_size': [16],
    'module__pos_embedding_type': ['learnable'],
}

# For RandomizedSearchCV, you might use distributions:
# import scipy.stats as stats
# param_dist_vit_from_scratch = {
#     'lr': stats.loguniform(1e-5, 1e-3),
#     'batch_size': [32, 64, 128],
#     'optimizer__weight_decay': stats.uniform(0.0, 0.1),
#     'max_epochs': [100], # Fix for RandomizedSearch usually, or set high with early stopping
#
#     'module__embed_dim': [384, 512, 768],
#     'module__depth': stats.randint(6, 13),
#     'module__num_heads': [4, 6, 8, 12], # Ensure these are compatible with chosen embed_dim
#     'module__mlp_ratio': [3.0, 4.0],
#     'module__projection_dropout': stats.uniform(0.0, 0.3),
#     'module__attention_dropout': stats.uniform(0.0, 0.3),
#     'module__mlp_dropout': stats.uniform(0.0, 0.3),
#
#     'module__head_hidden_dims': [None, [256], [512]],
#     'module__head_dropout_rate': stats.uniform(0.0, 0.5),
# }
# NOTE: For num_heads, ensure it's a divisor of embed_dim. This is hard to enforce directly in
# RandomizedSearchCV with independent sampling. You might need to filter invalid combinations
# or run searches for fixed embed_dim/num_heads pairs.
