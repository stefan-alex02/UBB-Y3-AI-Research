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

param_dist_vit_scratch_single_dict_focused = {
    'module__img_size': [224],
    'module__patch_size': [16], # Could also try [32] if img_size is larger
    'module__embed_dim': [384, 512, 768], # Test different capacities
    'module__depth': [6, 8, 10, 12],      # Deeper models need more data/regularization
    'module__num_heads': [6, 8, 12],      # Must be compatible with embed_dim
                                        # e.g., if embed_dim=384, heads=[6,8,12]
                                        # if embed_dim=512, heads=[8]
                                        # if embed_dim=768, heads=[12]
                                        # This makes direct random sampling tricky; often fix embed_dim and then sample heads.
                                        # For a truly random search, you might need to post-filter or use a more complex sampling.
                                        # Let's assume we'll mostly test with embed_dim where these heads are valid.
    'module__mlp_ratio': [3.0, 4.0],
    'module__projection_dropout': [0.0, 0.1, 0.2],
    'module__attention_dropout': [0.0, 0.1, 0.2],
    'module__mlp_dropout': [0.0, 0.1, 0.2],
    'module__pos_embedding_type': ['learnable'],
    'module__head_hidden_dims': [None, [256], [512], [512,256]],
    'module__head_dropout_rate': [0.1, 0.25, 0.5],

    'lr': [1e-4, 3e-4, 5e-4, 1e-3], # Higher LRs are common for training from scratch with good schedulers
    'optimizer__weight_decay': [0.01, 0.05, 0.1],
    'batch_size': [32, 64, 128], # Larger batch sizes often preferred
    'max_epochs': [150],         # Set high, rely on early stopping and schedulers
    'criterion__smoothing': [0.0, 0.1, 0.15], # Label smoothing
}

param_grid_vit_from_scratch_regularized = {
    # --- Skorch Training Loop Parameters ---
    'lr': [1e-4, 3e-4, 5e-4], # Initial LR for training from scratch
                               # Often combined with a linear warmup then cosine decay scheduler.
    'batch_size': [32, 64, 128], # Larger batches can sometimes help stabilize ViT scratch training
    'max_epochs': [100, 200, 300], # Needs many epochs

    # --- Optimizer (AdamW) Parameters ---
    'optimizer__weight_decay': [0.05, 0.1, 0.2], # Strong weight decay is critical
    'optimizer__betas': [(0.9, 0.999), (0.9, 0.98)], # Some papers suggest different beta2

    # --- ViTFromScratch Module Parameters (module__) ---
    # Keep architecture relatively fixed for a given search, or this explodes.
    # Example: A smaller ViT configuration
    'module__img_size': [224],
    'module__patch_size': [16], # Or [32] if images are larger
    'module__embed_dim': [384, 512], # Smaller embedding dimensions than ViT-Base
    'module__depth': [6, 8],         # Fewer layers
    'module__num_heads': [6, 8],       # Fewer heads (must divide embed_dim)
    'module__mlp_ratio': [3.0, 4.0],

    # Dropout - these are key regularizers
    'module__attention_dropout': [0.0, 0.1, 0.2],
    'module__projection_dropout': [0.0, 0.1, 0.2], # After MHA & MLP output
    'module__mlp_dropout': [0.1, 0.2, 0.3],       # Dropout within MLP

    'module__pos_embedding_type': ['learnable'],

    'module__head_hidden_dims': [None, [256]], # Simpler head
    'module__head_dropout_rate': [0.2, 0.5],  # Higher dropout for head

    # --- Callbacks for Regularization & Training Stability ---
    'gradient_clip_value': [1.0, 0.5], # Gradient clipping more important from scratch

    # LR Scheduler (Example: CosineAnnealingLR with warmup - skorch's LRScheduler can do this
    # by setting policy='CosineAnnealingWarmRestarts' or by chaining if needed,
    # or using one that directly supports warmup with another policy.)
    # For simplicity with current LRScheduler, let's stick to ReduceLROnPlateau or StepLR
    # and assume warmup is handled by a more advanced setup if needed.
    'callbacks__default_lr_scheduler__policy': ['ReduceLROnPlateau', 'StepLR', 'CosineAnnealingLR'],
    'callbacks__default_lr_scheduler__patience': [10, 15], # For ReduceLROnPlateau (needs more patience)
    'callbacks__default_lr_scheduler__factor': [0.1, 0.5],   # For ReduceLROnPlateau
    'callbacks__default_lr_scheduler__step_size': [30, 50], # For StepLR
    'callbacks__default_lr_scheduler__gamma': [0.1],      # For StepLR
    'callbacks__default_lr_scheduler__T_max': [100, 150], # For CosineAnnealingLR (should relate to max_epochs)
    'callbacks__default_lr_scheduler__eta_min': [1e-6, 1e-5], # For CosineAnnealingLR

    # Early Stopping
    'callbacks__default_early_stopping__patience': [20, 30], # More patience if LR schedule is long
}
