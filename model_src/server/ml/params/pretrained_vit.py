# This file contains the parameters used for the ViT model in the ML pipeline.
import torch
from skorch.callbacks import LRScheduler

# --- Fixed Parameter Sets ---

# --- Option 1: A common fine-tuning setup (similar to your old SimpleViT's intent) ---
pretrained_vit_fixed_params = {
    # Skorch/Training Loop General Params
    'max_epochs': 4,
    'batch_size': 16,
    'lr': 3e-5, # Initial learning rate

    # Optimizer Configuration (string name)
    'optimizer': 'AdamW', # Will be resolved to torch.optim.AdamW
    'optimizer__weight_decay': 0.05,
    # 'optimizer__betas': (0.9, 0.999), # AdamW/Adam specific, usually default is fine

    # LR Scheduler Configuration (using strings and individual params)
    # This example uses ReduceLROnPlateau
    'callbacks__default_lr_scheduler__policy': 'ReduceLROnPlateau',
    'callbacks__default_lr_scheduler__monitor': 'valid_loss', # Specific to ReduceLROnPlateau
    'callbacks__default_lr_scheduler__factor': 0.2,         # Specific to ReduceLROnPlateau
    'callbacks__default_lr_scheduler__patience': 7,         # Specific to ReduceLROnPlateau
    'callbacks__default_lr_scheduler__min_lr': 1e-7,        # Specific to ReduceLROnPlateau
    'callbacks__default_lr_scheduler__mode': 'min',         # Specific to ReduceLROnPlateau
    # 'callbacks__default_lr_scheduler__verbose': False, # Passed to PyTorch scheduler

    # Example of configuring EarlyStopping patience for this specific run
    # 'callbacks__default_early_stopping__patience': 15,

    # PretrainedViT Module Parameters
    'module__vit_model_variant': 'vit_b_16',
    'module__pretrained': True,
    'module__unfreeze_strategy': 'encoder_tail',
    'module__num_transformer_blocks_to_unfreeze': 3,
    'module__unfreeze_cls_token': True,
    'module__unfreeze_pos_embedding': True,
    'module__unfreeze_patch_embedding': False,
    'module__unfreeze_encoder_layernorm': True,
    'module__custom_head_hidden_dims': [512], # One hidden layer in the head
    'module__head_dropout_rate': 0.25,

    # Other Skorch parameters (if needed to override pipeline defaults)
    # 'iterator_train__num_workers': 2,
    # 'show_first_batch_augmentation': True, # If you want to override pipeline default
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
# Configuration 1: SGD with a fairly standard StepLR
comprehensive_param_grid_list_vit = [
    # --- Scenario 1: Focus on AdamW with ReduceLROnPlateau, varying unfreezing ---
    # {
    #     'optimizer': ['adamw'], # String, expander will resolve to type
    #     'lr': [3e-5, 5e-5],
    #     'optimizer__weight_decay': [0.01, 0.05],
    #     'batch_size': [16],
    #     'max_epochs': [6],
    #
    #     'callbacks__default_lr_scheduler__policy': ['ReduceLROnPlateau'], # Must be single policy for this sub-grid
    #     'callbacks__default_lr_scheduler__monitor': ['valid_loss'],
    #     'callbacks__default_lr_scheduler__factor': [0.1, 0.2], # Tuned
    #     'callbacks__default_lr_scheduler__patience': [5, 8],    # Tuned
    #     'callbacks__default_lr_scheduler__min_lr': [1e-7],
    #     'callbacks__default_lr_scheduler__mode': ['min'],
    #
    #     'module__vit_model_variant': ['vit_b_16'],
    #     'module__pretrained': [True],
    #     'module__unfreeze_strategy': ['encoder_tail'],
    #     'module__num_transformer_blocks_to_unfreeze': [2, 4, 6], # Tune number of blocks
    #     'module__unfreeze_cls_token': [True],
    #     'module__unfreeze_pos_embedding': [True],
    #     'module__unfreeze_patch_embedding': [False],
    #     'module__unfreeze_encoder_layernorm': [True],
    #     'module__custom_head_hidden_dims': [None, [512]],
    #     'module__head_dropout_rate': [0.0, 0.25],
    # },
    #
    # # --- Scenario 2: Focus on AdamW with CosineAnnealingLR, head variations ---
    # {
    #     'optimizer': ['adamw'],
    #     'lr': [1e-5, 3e-5], # Different LRs for Cosine
    #     'optimizer__weight_decay': [0.01],
    #     'batch_size': [16, 32], # Try different batch size
    #     'max_epochs': [7],     # T_max will be based on this
    #
    #     'callbacks__default_lr_scheduler__policy': ['CosineAnnealingLR'],
    #     # T_max for CosineAnnealingLR is often set to max_epochs.
    #     # If max_epochs is also in the grid, expand_hyperparameter_grid would need to handle this,
    #     # or you fix T_max here if max_epochs is fixed for this scenario.
    #     # For simplicity, assume expand_hyperparameter_grid sets T_max based on the 'max_epochs' value for that combo.
    #     # Alternatively, list T_max values matching max_epochs values:
    #     'callbacks__default_lr_scheduler__T_max': [70], # Match max_epochs
    #     'callbacks__default_lr_scheduler__eta_min': [0, 1e-7],
    #
    #     'module__vit_model_variant': ['vit_b_16'],
    #     'module__pretrained': [True],
    #     'module__unfreeze_strategy': ['encoder_tail'],
    #     'module__num_transformer_blocks_to_unfreeze': [3], # Fixed unfreeze depth for this scenario
    #     'module__unfreeze_cls_token': [True],
    #     'module__unfreeze_pos_embedding': [True],
    #     'module__unfreeze_patch_embedding': [False],
    #     'module__unfreeze_encoder_layernorm': [True],
    #     'module__custom_head_hidden_dims': [None, [256], [512, 256]], # Tune head structure
    #     'module__head_dropout_rate': [0.1, 0.3],
    # },

    # --- Scenario 3: Focus on SGD with StepLR, different unfreeze amounts ---
    {
        'optimizer': ['sgd'],
        'lr': [0.01, 0.005], # Typical SGD LRs
        'optimizer__momentum': [0.9],
        'optimizer__weight_decay': [1e-4, 5e-4],
        'optimizer__nesterov': [True],
        'batch_size': [32],
        'max_epochs': [8], # SGD might need more

        'callbacks__default_lr_scheduler__policy': ['StepLR'],
        'callbacks__default_lr_scheduler__step_size': [2, 4],
        'callbacks__default_lr_scheduler__gamma': [0.1, 0.2],

        'module__vit_model_variant': ['vit_b_16'],
        'module__pretrained': [True],
        'module__unfreeze_strategy': ['encoder_tail', 'none'], # Try head-only style too
        'module__num_transformer_blocks_to_unfreeze': [1, 2], # For 'encoder_tail'
        'module__unfreeze_cls_token': [True],
        'module__unfreeze_pos_embedding': [True],
        'module__unfreeze_patch_embedding': [False],
        'module__unfreeze_encoder_layernorm': [True], # If 'encoder_tail'
        'module__custom_head_hidden_dims': [None],
        'module__head_dropout_rate': [0.0, 0.1],
    }
]

