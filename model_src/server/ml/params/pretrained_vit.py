# This file contains the parameters used for the ViT model in the ML pipeline.
import torch
from skorch.callbacks import LRScheduler

# --- Fixed Parameter Sets ---

# --- Option 1: A common fine-tuning setup (similar to your old SimpleViT's intent) ---
pretrained_vit_fixed_params = {
    'max_epochs': 70, # Allow more room for early stopping if LR changes are slower
    # 'lr': 3e-5,
    'lr': 3e-7,
    'batch_size': 16,
    'optimizer': 'AdamW',
    'optimizer__weight_decay': 0.05,

    'callbacks__default_lr_scheduler__policy': 'CosineAnnealingLR',
    'callbacks__default_lr_scheduler__T_max': 50, # Should match max_epochs
    'callbacks__default_lr_scheduler__eta_min': 1e-7,

    # 'module__vit_model_variant': 'vit_b_16',
    'module__vit_model_variant': 'vit_l_16',
    'module__pretrained': True,
    'module__unfreeze_strategy': 'encoder_tail',
    # 'module__num_transformer_blocks_to_unfreeze': 1, # <<< SIGNIFICANTLY REDUCED
    'module__num_transformer_blocks_to_unfreeze': 2,
    'module__unfreeze_cls_token': True,
    'module__unfreeze_pos_embedding': True,
    'module__unfreeze_patch_embedding': False,
    'module__unfreeze_encoder_layernorm': True,     # Unfreeze if unfreezing end blocks
    'module__custom_head_hidden_dims': None,        # Simplest head
    # 'module__head_dropout_rate': 0.25,              # <<< ADDED SOME DROPOUT
    'module__head_dropout_rate': 0.55,
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



best_config_as_grid_vit = [ # Outer list for GridSearchCV's param_grid
    {
        # Skorch/Training Loop Params
        'max_epochs': [70],
        'lr': [5e-5],
        'batch_size': [16],

        # Optimizer Configuration (string name for the expander)
        'optimizer': ['AdamW'], # Expander will convert "AdamW" to torch.optim.AdamW
        'optimizer__weight_decay': [0.05],

        # LR Scheduler Configuration (using strings and individual params for the expander)
        # Assuming ReduceLROnPlateau was used or is a good choice for these best params.
        'callbacks__default_lr_scheduler__policy': ['ReduceLROnPlateau'],
        'callbacks__default_lr_scheduler__monitor': ['valid_loss'],
        'callbacks__default_lr_scheduler__factor': [0.1],
        'callbacks__default_lr_scheduler__patience': [5],
        'callbacks__default_lr_scheduler__min_lr': [1e-7],
        'callbacks__default_lr_scheduler__mode': ['min'],
        # 'callbacks__default_lr_scheduler__verbose': [False], # Optional

        # PretrainedViT Module Parameters
        'module__vit_model_variant': ['vit_b_16'],
        'module__pretrained': [True],
        'module__unfreeze_strategy': ['encoder_tail'],
        'module__num_transformer_blocks_to_unfreeze': [4], # From CSV
        'module__unfreeze_cls_token': [True],
        'module__unfreeze_pos_embedding': [True],
        'module__unfreeze_patch_embedding': [False],
        'module__unfreeze_encoder_layernorm': [True],
        'module__custom_head_hidden_dims': [None], # [None] represents a single choice: None
        'module__head_dropout_rate': [0.0],
    }
]


