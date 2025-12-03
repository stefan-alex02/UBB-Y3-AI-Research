pretrained_vit_fixed_params = {
    'max_epochs': 70,
    'lr': 5e-5,
    'batch_size': 16,

    'optimizer': 'AdamW',
    'optimizer__weight_decay': 0.2,

    'callbacks__default_lr_scheduler__policy': 'CosineAnnealingLR',
    'callbacks__default_lr_scheduler__T_max': 70,

    # 'callbacks__default_lr_scheduler__policy': 'CosineAnnealingWarmRestarts',
    # 'callbacks__default_lr_scheduler__T_0': 15,
    # 'callbacks__default_lr_scheduler__T_mult': 1,

    'callbacks__default_lr_scheduler__eta_min': 1e-06,

    'callbacks__default_early_stopping__patience': 15,

    # --- CutMix Parameters ---
    'cutmix_alpha': 1.0,
    # 'cutmix_probability': 0.9, # for CCSN
    'cutmix_probability': 0.5, # for Swimcat
    # 'cutmix_probability': 0.5, # for GCD

    # --- Gradient Clipping ---
    # 'gradient_clip_value': 5.0,

    # --- PretrainedViT Module Parameters ---
    'module__vit_model_variant': 'vit_b_16',
    'module__pretrained': True,
    'module__unfreeze_strategy': 'encoder_tail',
    'module__num_transformer_blocks_to_unfreeze': 6,
    'module__unfreeze_cls_token': True,
    'module__unfreeze_pos_embedding': True,
    'module__unfreeze_patch_embedding': False,
    'module__unfreeze_encoder_layernorm': True,
    'module__custom_head_hidden_dims': None,
    'module__head_dropout_rate': 0.50,

    # --- Data Loader Parameters ---
    'iterator_train__shuffle': True,
}


param_dist_pretrained_vit_single_dict = {
    # --- Skorch Training Loop Parameters ---
    'lr': [1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3],
    'batch_size': [8, 16, 32],
    'max_epochs': [15, 25, 30, 40, 50],

    # --- Optimizer Parameters ---
    'optimizer__weight_decay': [0.0, 0.001, 0.005, 0.01, 0.05],

    # --- PretrainedViT Module Parameters (module__) ---
    'module__vit_model_variant': ['vit_b_16'],

    'module__pretrained': [True],

    'module__unfreeze_strategy': [
        'none',
        'encoder_tail',
        # 'full_encoder'
    ],
    'module__num_transformer_blocks_to_unfreeze': [1, 2, 3, 4, 6],

    'module__unfreeze_cls_token': [True, False],
    'module__unfreeze_pos_embedding': [True, False],
    'module__unfreeze_patch_embedding': [False],
    'module__unfreeze_encoder_layernorm': [True, False],

    'module__custom_head_hidden_dims': [
        None,
        [256],
        [512],
        [512, 256]
    ],
    'module__head_dropout_rate': [0.0, 0.1, 0.25, 0.4, 0.5],
}

param_grid_pretrained_vit_conditional = [
    # --- Scenario 1: encoder_tail ---
    {
        'module__vit_model_variant': ['vit_b_16'],
        'module__pretrained': [True],
        'module__unfreeze_strategy': ['encoder_tail'],
        'module__num_transformer_blocks_to_unfreeze': [1, 2, 4],
        'module__unfreeze_cls_token': [True],
        'module__unfreeze_pos_embedding': [True],
        'module__unfreeze_patch_embedding': [False],
        'module__unfreeze_encoder_layernorm': [True],

        'module__custom_head_hidden_dims': [None, [512]],
        'module__head_dropout_rate': [0.0, 0.25, 0.5],

        # Skorch & Optimizer parameters for this scenario
        'lr': [1e-5, 5e-5, 1e-4],
        'optimizer__weight_decay': [0.01, 0.05],
        'batch_size': [8, 16],
        'max_epochs': [20, 30, 50],
    },

    # --- Scenario 2: 'head_only' ---
    {
        'module__vit_model_variant': ['vit_b_16'],
        'module__pretrained': [True],
        'module__unfreeze_strategy': ['none'],
        # 'module__num_transformer_blocks_to_unfreeze': [0],
        'module__unfreeze_cls_token': [True],
        'module__unfreeze_pos_embedding': [True],
        'module__unfreeze_patch_embedding': [False],
        'module__unfreeze_encoder_layernorm': [False],

        'module__custom_head_hidden_dims': [None, [256], [512]],
        'module__head_dropout_rate': [0.0, 0.2, 0.4],

        # Skorch & Optimizer parameters
        'lr': [5e-5, 1e-4, 5e-4],
        'optimizer__weight_decay': [0.0, 0.001, 0.01],
        'batch_size': [16, 32],
        'max_epochs': [15, 25],
    },
]

param_grid_pretrained_vit_focused = [
    # Scenario 1
    {
        'module__vit_model_variant': ['vit_b_16'],
        'module__pretrained': [True],
        'module__unfreeze_strategy': ['none'],
        'module__unfreeze_cls_token': [True],
        'module__unfreeze_pos_embedding': [True],
        'module__unfreeze_patch_embedding': [False],
        'module__unfreeze_encoder_layernorm': [True],
        'module__custom_head_hidden_dims': [None, [512]],
        'module__head_dropout_rate': [0.25, 0.5],

        'lr': [1e-4, 3e-4],
        'optimizer__weight_decay': [0.001, 0.01],
        'batch_size': [32],
        'max_epochs': [50],
    },
    # Scenario 2
    {
        'module__vit_model_variant': ['vit_b_16'],
        'module__pretrained': [True],
        'module__unfreeze_strategy': ['encoder_tail'],
        'module__num_transformer_blocks_to_unfreeze': [2, 4],
        'module__unfreeze_cls_token': [True],
        'module__unfreeze_pos_embedding': [True],
        'module__unfreeze_patch_embedding': [False],
        'module__unfreeze_encoder_layernorm': [True],
        'module__custom_head_hidden_dims': [None],
        'module__head_dropout_rate': [0.0, 0.25],

        'lr': [3e-5, 5e-5, 1e-4],
        'optimizer__weight_decay': [0.01, 0.05],
        'batch_size': [16],
        'max_epochs': [70],
    }
]



best_config_as_grid_vit = [
    {
        # Skorch/Training Loop Params
        'max_epochs': [70],
        'lr': [5e-5],
        'batch_size': [16],

        # Optimizer Configuration
        'optimizer': ['AdamW'],
        'optimizer__weight_decay': [0.05],

        # LR Scheduler Configuration
        'callbacks__default_lr_scheduler__policy': ['ReduceLROnPlateau'],
        'callbacks__default_lr_scheduler__monitor': ['valid_loss'],
        'callbacks__default_lr_scheduler__factor': [0.1],
        'callbacks__default_lr_scheduler__patience': [5],
        'callbacks__default_lr_scheduler__min_lr': [1e-7],
        'callbacks__default_lr_scheduler__mode': ['min'],
        # 'callbacks__default_lr_scheduler__verbose': [False],

        # PretrainedViT Module Parameters
        'module__vit_model_variant': ['vit_b_16'],
        'module__pretrained': [True],
        'module__unfreeze_strategy': ['encoder_tail'],
        'module__num_transformer_blocks_to_unfreeze': [4],
        'module__unfreeze_cls_token': [True],
        'module__unfreeze_pos_embedding': [True],
        'module__unfreeze_patch_embedding': [False],
        'module__unfreeze_encoder_layernorm': [True],
        'module__custom_head_hidden_dims': [None],
        'module__head_dropout_rate': [0.0],
    }
]
