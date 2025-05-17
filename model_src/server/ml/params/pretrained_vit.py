# This file contains the parameters used for the ViT model in the ML pipeline.

# --- Fixed Parameter Sets ---

# --- Option 1: A common fine-tuning setup (similar to your old SimpleViT's intent) ---
pretrained_vit_fixed_params_option1 = {
    # Skorch/Training Loop Params
    'max_epochs': 25,
    'lr': 5e-5, # Common starting LR for ViT fine-tuning
    'batch_size': 16, # Adjust based on GPU memory and model variant
    'optimizer__weight_decay': 0.01,

    # PretrainedViT Module Params (prefixed with 'module__')
    'module__vit_model_variant': 'vit_b_16',
    'module__pretrained': True,
    'module__unfreeze_strategy': 'encoder_tail',
    'module__num_transformer_blocks_to_unfreeze': 2, # Unfreeze last 2 blocks (plus LN, CLS, PosEmb)
    'module__unfreeze_cls_token': True,
    'module__unfreeze_pos_embedding': True,
    'module__unfreeze_patch_embedding': False, # Usually False
    'module__unfreeze_encoder_layernorm': True, # Usually True when unfreezing end blocks
    'module__custom_head_hidden_dims': None,   # Simple linear head
    'module__head_dropout_rate': 0.0,          # No dropout in the simple linear head
}

# --- Option 2: Head-only fine-tuning (faster, less memory, good for similar datasets) ---
pretrained_vit_fixed_params_option2_head_only = {
    'max_epochs': 15,
    'lr': 1e-4, # Can often use a slightly higher LR for head-only
    'batch_size': 32,
    'optimizer__weight_decay': 0.001,

    'module__vit_model_variant': 'vit_b_16',
    'module__pretrained': True,
    'module__unfreeze_strategy': 'none', # No encoder blocks unfrozen by strategy
    'module__num_transformer_blocks_to_unfreeze': 0, # Not used by 'none' strategy
    'module__unfreeze_cls_token': True, # Still good to fine-tune these
    'module__unfreeze_pos_embedding': True,
    'module__unfreeze_patch_embedding': False,
    'module__unfreeze_encoder_layernorm': False, # Keep encoder LN frozen if encoder blocks are
    'module__custom_head_hidden_dims': [512], # Example: one hidden layer in head
    'module__head_dropout_rate': 0.25,
}

# --- Parameter Space Definitions ---

param_grid_pretrained_vit_conditional = [
    # --- Scenario 1: 'encoder_tail' - Fine-tune last N encoder blocks ---
    # This is a common and effective strategy.
    {
        'module__vit_model_variant': ['vit_b_16'], # Could add 'vit_l_16' if resources permit and careful with other params
        'module__pretrained': [True],
        'module__unfreeze_strategy': ['encoder_tail'],
        'module__num_transformer_blocks_to_unfreeze': [1, 2, 4], # How many final encoder blocks
        'module__unfreeze_cls_token': [True],
        'module__unfreeze_pos_embedding': [True],
        'module__unfreeze_patch_embedding': [False], # Generally False
        'module__unfreeze_encoder_layernorm': [True], # Usually True with 'encoder_tail'

        'module__custom_head_hidden_dims': [None, [512]], # Simple linear vs. one hidden layer
        'module__head_dropout_rate': [0.0, 0.25, 0.5],

        # Skorch & Optimizer parameters for this scenario
        'lr': [1e-5, 5e-5, 1e-4],       # Lower LRs often better for deeper fine-tuning
        'optimizer__weight_decay': [0.01, 0.05],
        'batch_size': [8, 16],          # Smaller batches if more layers unfrozen
        'max_epochs': [20, 30, 50],     # May need more epochs
    },

    # --- Scenario 2: 'head_only' style via 'none' strategy + specific unfreezes ---
    # (Effectively fine-tunes only explicitly flagged components like head, CLS, PosEmb)
    {
        'module__vit_model_variant': ['vit_b_16'],
        'module__pretrained': [True],
        'module__unfreeze_strategy': ['none'], # Encoder blocks remain frozen by strategy
        # 'module__num_transformer_blocks_to_unfreeze': [0], # Not used by 'none' strategy for blocks
        'module__unfreeze_cls_token': [True],       # Unfreeze these critical parts
        'module__unfreeze_pos_embedding': [True],
        'module__unfreeze_patch_embedding': [False],
        'module__unfreeze_encoder_layernorm': [False], # Keep encoder LN frozen if blocks are

        'module__custom_head_hidden_dims': [None, [256], [512]],
        'module__head_dropout_rate': [0.0, 0.2, 0.4],

        # Skorch & Optimizer parameters - can often use slightly higher LR
        'lr': [5e-5, 1e-4, 5e-4],
        'optimizer__weight_decay': [0.0, 0.001, 0.01],
        'batch_size': [16, 32],
        'max_epochs': [15, 25],
    },

    # --- (Optional) Scenario 3: 'full_encoder' - Fine-tune the entire encoder ---
    # Use with caution: computationally expensive, needs more data, prone to overfitting.
    # {
    #     'module__vit_model_variant': ['vit_b_16'],
    #     'module__pretrained': [True],
    #     'module__unfreeze_strategy': ['full_encoder'],
    #     # 'module__num_transformer_blocks_to_unfreeze': [12], # All 12 for vit_b_16 if strategy didn't handle it
    #     'module__unfreeze_cls_token': [True],
    #     'module__unfreeze_pos_embedding': [True],
    #     'module__unfreeze_patch_embedding': [False], # Still typically False
    #     'module__unfreeze_encoder_layernorm': [True], # Unfreeze with full encoder

    #     'module__custom_head_hidden_dims': [None, [512]],
    #     'module__head_dropout_rate': [0.0, 0.25],

    #     'lr': [1e-5, 3e-5], # Very low LRs
    #     'optimizer__weight_decay': [0.01, 0.05, 0.1], # Weight decay is very important here
    #     'batch_size': [8], # Often smallest batch size
    #     'max_epochs': [40, 60, 80], # May need significantly more epochs
    # },
]

# Slightly Diminished Parameter Grid for PretrainedViT (Conditional - List of Dicts)
# Suitable for a more focused GridSearchCV or a quicker RandomizedSearchCV
param_grid_pretrained_vit_diminished = [
    # --- Scenario 1: 'encoder_tail' - Fine-tune last N encoder blocks ---
    # This is a common and effective strategy.
    {
        'module__vit_model_variant': ['vit_b_16'],
        'module__pretrained': [True],
        'module__unfreeze_strategy': ['encoder_tail'],
        'module__num_transformer_blocks_to_unfreeze': [2, 4], # Test unfreezing a few vs. more
        'module__unfreeze_cls_token': [True],
        'module__unfreeze_pos_embedding': [True],
        'module__unfreeze_patch_embedding': [False],
        'module__unfreeze_encoder_layernorm': [True],

        'module__custom_head_hidden_dims': [None, [512]], # Simple linear vs. one hidden layer
        'module__head_dropout_rate': [0.0, 0.25],      # No dropout vs. some dropout

        # Skorch & Optimizer parameters
        'lr': [5e-5, 1e-4],             # Key learning rates for this strategy
        'optimizer__weight_decay': [0.01], # Often a good default
        'batch_size': [16],                 # Keep batch size fixed or pick one common value
        'max_epochs': [30],                 # Fixed epochs, rely on early stopping
    },

    # --- Scenario 2: 'head_only' style via 'none' strategy + specific unfreezes ---
    # (Effectively fine-tunes only explicitly flagged components like head, CLS, PosEmb)
    {
        'module__vit_model_variant': ['vit_b_16'],
        'module__pretrained': [True],
        'module__unfreeze_strategy': ['none'],
        'module__unfreeze_cls_token': [True],
        'module__unfreeze_pos_embedding': [True],
        'module__unfreeze_patch_embedding': [False],
        'module__unfreeze_encoder_layernorm': [False], # Keep LN frozen if blocks are

        'module__custom_head_hidden_dims': [None, [512]],
        'module__head_dropout_rate': [0.0, 0.25],

        # Skorch & Optimizer parameters
        'lr': [1e-4, 5e-4],             # Can try slightly higher LRs
        'optimizer__weight_decay': [0.001],
        'batch_size': [32],
        'max_epochs': [20],
    },

    # --- Scenario 3: (Optional but more focused) 'full_encoder' ---
    # If you want to test this, keep it very limited.
    # {
    #     'module__vit_model_variant': ['vit_b_16'],
    #     'module__pretrained': [True],
    #     'module__unfreeze_strategy': ['full_encoder'],
    #     'module__unfreeze_cls_token': [True],
    #     'module__unfreeze_pos_embedding': [True],
    #     'module__unfreeze_patch_embedding': [False],
    #     'module__unfreeze_encoder_layernorm': [True],

    #     'module__custom_head_hidden_dims': [None], # Simplest head for full fine-tune
    #     'module__head_dropout_rate': [0.0, 0.2],

    #     'lr': [1e-5, 3e-5], # Very low LRs
    #     'optimizer__weight_decay': [0.01, 0.05],
    #     'batch_size': [8], # Smallest batch size
    #     'max_epochs': [50],
    # },
]
