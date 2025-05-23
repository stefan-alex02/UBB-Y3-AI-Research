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

param_dist_pretrained_vit_single_dict = {
    # --- Skorch Training Loop Parameters ---
    'lr': [1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3], # Wider range for LR
    'batch_size': [8, 16, 32],
    'max_epochs': [15, 25, 30, 40, 50], # Rely on EarlyStopping

    # --- Optimizer (AdamW) Parameters ---
    'optimizer__weight_decay': [0.0, 0.001, 0.005, 0.01, 0.05],

    # --- PretrainedViT Module Parameters (module__) ---
    'module__vit_model_variant': ['vit_b_16'], # Keep fixed for this example, or add 'vit_l_16'
                                               # but be mindful of batch size/LR differences.
    'module__pretrained': [True],

    'module__unfreeze_strategy': [
        'none',             # Corresponds to 'head_only' style active unfreezing
        'encoder_tail',
        # 'full_encoder'    # Generally very resource intensive, include if specifically testing
    ],
    # This parameter will be sampled regardless of strategy, but only used by 'encoder_tail'.
    # Your PretrainedViT should ignore it if strategy is not 'encoder_tail'.
    'module__num_transformer_blocks_to_unfreeze': [1, 2, 3, 4, 6], # Range of blocks for 'encoder_tail'

    # Flags for specific component unfreezing (these are active regardless of block strategy,
    # unless 'none' strategy in PretrainedViT re-freezes them, which it currently does not explicitly for CLS/PosEmb if strategy is 'none')
    # For 'none' strategy (head_only style), these define what besides the head gets unfrozen.
    # For 'encoder_tail' or 'full_encoder', these are often True.
    'module__unfreeze_cls_token': [True, False], # Test impact of freezing/unfreezing
    'module__unfreeze_pos_embedding': [True, False],
    'module__unfreeze_patch_embedding': [False], # Typically always False for fine-tuning
    'module__unfreeze_encoder_layernorm': [True, False], # Test impact

    'module__custom_head_hidden_dims': [
        None,       # Simple linear head
        [256],
        [512],
        [512, 256]
    ],
    'module__head_dropout_rate': [0.0, 0.1, 0.25, 0.4, 0.5],
}

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

param_grid_pretrained_vit_focused = [
    # Scenario 1: Minimal fine-tuning (Head + CLS/PosEmb/EncoderLN)
    {
        'module__vit_model_variant': ['vit_b_16'],
        'module__pretrained': [True],
        'module__unfreeze_strategy': ['none'], # Encoder blocks frozen
        'module__unfreeze_cls_token': [True],
        'module__unfreeze_pos_embedding': [True],
        'module__unfreeze_patch_embedding': [False],
        'module__unfreeze_encoder_layernorm': [True], # Unfreeze final LN even if blocks are frozen
        'module__custom_head_hidden_dims': [None, [512]],
        'module__head_dropout_rate': [0.25, 0.5],

        'lr': [1e-4, 3e-4],
        'optimizer__weight_decay': [0.001, 0.01],
        'batch_size': [32],
        'max_epochs': [50], # Rely on early stopping
    },
    # Scenario 2: Fine-tune last few encoder blocks
    {
        'module__vit_model_variant': ['vit_b_16'],
        'module__pretrained': [True],
        'module__unfreeze_strategy': ['encoder_tail'],
        'module__num_transformer_blocks_to_unfreeze': [2, 4], # ViT-B has 12 blocks
        'module__unfreeze_cls_token': [True],
        'module__unfreeze_pos_embedding': [True],
        'module__unfreeze_patch_embedding': [False],
        'module__unfreeze_encoder_layernorm': [True],
        'module__custom_head_hidden_dims': [None], # Keep head simple when unfreezing more
        'module__head_dropout_rate': [0.0, 0.25],

        'lr': [3e-5, 5e-5, 1e-4], # Smaller LRs for deeper fine-tuning
        'optimizer__weight_decay': [0.01, 0.05],
        'batch_size': [16], # Potentially smaller batch for more unfrozen layers
        'max_epochs': [70],
    }
]





# --- Revised params with custom optimzier (SGD) ---

import torch # For torch.optim types

# --- Configuration 1 (Based on Rank 1 from your CSV) ---
params_config_1_adamw = {
    'optimizer': ['sgd'],
    'optimizer__momentum': [0.9],
    'optimizer__weight_decay': [5e-4], # Common weight decay for SGD
    'optimizer__nesterov': [True],     # Often helps

    'lr': [5e-5],
    'batch_size': [16],
    'max_epochs': [70], # Or a smaller value if using effective early stopping

    'module__vit_model_variant': ['vit_b_16'],
    'module__pretrained': [True],
    'module__unfreeze_strategy': ['encoder_tail'],
    'module__num_transformer_blocks_to_unfreeze': [4], # From CSV (was 4.0)
    'module__unfreeze_cls_token': [True],
    'module__unfreeze_pos_embedding': [True],
    'module__unfreeze_patch_embedding': [False],
    'module__unfreeze_encoder_layernorm': [True],
    'module__custom_head_hidden_dims': [None],
    'module__head_dropout_rate': [0.0],
}

# --- Configuration 2 (Based on Rank 2 from your CSV, different LR) ---
params_config_2_adamw = {
    'optimizer': ['sgd'],
    'optimizer__momentum': [0.9],
    'optimizer__weight_decay': [5e-4], # Common weight decay for SGD
    'optimizer__nesterov': [True],     # Often helps

    'lr': [3e-5], # Different LR
    'batch_size': [16],
    'max_epochs': [70],

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

# --- Configuration 3 (Based on Rank 4 from your CSV, different unfreeze blocks & LR) ---
# Rank 3 is similar to Rank 1 but with different weight_decay, let's pick Rank 4 for more variety.
params_config_3_adamw = {
    'optimizer': ['sgd'],
    'optimizer__momentum': [0.9],
    'optimizer__weight_decay': [5e-4], # Common weight decay for SGD
    'optimizer__nesterov': [True],     # Often helps

    'lr': [3e-5],
    'batch_size': [16],
    'max_epochs': [70],

    'module__vit_model_variant': ['vit_b_16'],
    'module__pretrained': [True],
    'module__unfreeze_strategy': ['encoder_tail'],
    'module__num_transformer_blocks_to_unfreeze': [2], # Different number of blocks
    'module__unfreeze_cls_token': [True],
    'module__unfreeze_pos_embedding': [True],
    'module__unfreeze_patch_embedding': [False],
    'module__unfreeze_encoder_layernorm': [True],
    'module__custom_head_hidden_dims': [None],
    'module__head_dropout_rate': [0.0],
}

# --- Configuration 4: SGD (New Strategy) ---
# SGD often requires different LR, more epochs, and momentum.
# These are example starting points for SGD with ViT fine-tuning.
params_config_4_sgd = {
    'optimizer': ['sgd'],
    'optimizer__momentum': [0.9],
    'optimizer__weight_decay': [5e-4], # Common weight decay for SGD
    'optimizer__nesterov': [True],     # Often helps

    'lr': [0.001], # SGD typically needs a larger LR than AdamW for fine-tuning, but schedule is key
    'batch_size': [16], # Keep consistent with others for this grid
    'max_epochs': [70], # SGD might need more, but for a fixed grid comparison...

    'module__vit_model_variant': ['vit_b_16'],
    'module__pretrained': [True],
    'module__unfreeze_strategy': ['encoder_tail'],      # A common strategy
    'module__num_transformer_blocks_to_unfreeze': [2],  # Fine-tune a few blocks
    'module__unfreeze_cls_token': [True],
    'module__unfreeze_pos_embedding': [True],
    'module__unfreeze_patch_embedding': [False],
    'module__unfreeze_encoder_layernorm': [True],
    'module__custom_head_hidden_dims': [None],
    'module__head_dropout_rate': [0.0], # Start simple
}

# --- Combine into the list of dictionaries for GridSearchCV ---
# This grid will result in exactly 4 configurations being tested by GridSearchCV.
fixed_grid_for_vit_top_plus_sgd = [
    params_config_1_adamw,
    params_config_2_adamw,
    params_config_3_adamw,
    params_config_4_sgd,
]
