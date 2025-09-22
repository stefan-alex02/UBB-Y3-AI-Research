fixed_params_vit_scratch = {
    # Skorch/Training Loop Params
    'max_epochs': 75,
    'lr': 1e-3,

    'batch_size': 64,
    'optimizer__weight_decay': 0.05,


    'module__img_size': 224,
    'module__patch_size': 16,
    'module__in_channels': 3,
    'module__embed_dim': 768,
    'module__depth': 12,
    'module__num_heads': 12,
    'module__mlp_ratio': 4.0,
    'module__attention_dropout': 0.0,
    'module__projection_dropout': 0.0,
    'module__mlp_dropout': 0.0,

    'module__pos_embedding_type': 'learnable',

    'module__head_hidden_dims': None,
    'module__head_dropout_rate': 0.0,
}

param_grid_vit_from_scratch = {
    'lr': [1e-4, 3e-4, 1e-3],
    'batch_size': [32, 64],
    'optimizer__weight_decay': [0.01, 0.05, 0.1],
    'max_epochs': [50, 100, 150],

    # ViTFromScratch Module Params (module__)
    'module__embed_dim': [512, 768],
    'module__depth': [6, 8, 12],
    'module__num_heads': [8, 12],
    'module__mlp_ratio': [3.0, 4.0],
    'module__projection_dropout': [0.0, 0.1],
    'module__attention_dropout': [0.0, 0.1],
    'module__mlp_dropout': [0.0, 0.1],

    # Head configuration
    'module__head_hidden_dims': [None, [256], [512, 256]],
    'module__head_dropout_rate': [0.0, 0.2, 0.5],

    # Fixed for this grid example, but could be tuned:
    'module__img_size': [224],
    'module__patch_size': [16],
    'module__pos_embedding_type': ['learnable'],
}

param_dist_vit_scratch_single_dict_focused = {
    'module__img_size': [224],
    'module__patch_size': [16],
    'module__embed_dim': [384, 512, 768],
    'module__depth': [6, 8, 10, 12],
    'module__num_heads': [6, 8, 12],

    'module__mlp_ratio': [3.0, 4.0],
    'module__projection_dropout': [0.0, 0.1, 0.2],
    'module__attention_dropout': [0.0, 0.1, 0.2],
    'module__mlp_dropout': [0.0, 0.1, 0.2],
    'module__pos_embedding_type': ['learnable'],
    'module__head_hidden_dims': [None, [256], [512], [512,256]],
    'module__head_dropout_rate': [0.1, 0.25, 0.5],

    'lr': [1e-4, 3e-4, 5e-4, 1e-3],
    'optimizer__weight_decay': [0.01, 0.05, 0.1],
    'batch_size': [32, 64, 128],
    'max_epochs': [150],
    'criterion__smoothing': [0.0, 0.1, 0.15],
}

param_grid_vit_from_scratch_regularized = {
    'lr': [1e-4, 3e-4, 5e-4],
    'batch_size': [32, 64, 128],
    'max_epochs': [100, 200, 300],

    # --- Optimizer (AdamW) Parameters ---
    'optimizer__weight_decay': [0.05, 0.1, 0.2],
    'optimizer__betas': [(0.9, 0.999), (0.9, 0.98)],

    # --- ViTFromScratch Module Parameters ---
    'module__img_size': [224],
    'module__patch_size': [16],
    'module__embed_dim': [384, 512],
    'module__depth': [6, 8],
    'module__num_heads': [6, 8],
    'module__mlp_ratio': [3.0, 4.0],

    # Dropout
    'module__attention_dropout': [0.0, 0.1, 0.2],
    'module__projection_dropout': [0.0, 0.1, 0.2],
    'module__mlp_dropout': [0.1, 0.2, 0.3],

    'module__pos_embedding_type': ['learnable'],

    'module__head_hidden_dims': [None, [256]],
    'module__head_dropout_rate': [0.2, 0.5],

    'gradient_clip_value': [1.0, 0.5],

    'callbacks__default_lr_scheduler__policy': ['ReduceLROnPlateau', 'StepLR', 'CosineAnnealingLR'],
    'callbacks__default_lr_scheduler__patience': [10, 15],
    'callbacks__default_lr_scheduler__factor': [0.1, 0.5],
    'callbacks__default_lr_scheduler__step_size': [30, 50],
    'callbacks__default_lr_scheduler__gamma': [0.1],
    'callbacks__default_lr_scheduler__T_max': [100, 150],
    'callbacks__default_lr_scheduler__eta_min': [1e-6, 1e-5],

    # Early Stopping
    'callbacks__default_early_stopping__patience': [20, 30],
}
