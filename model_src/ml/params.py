# --- Fixed Parameter Sets (for Single Configuration Methods) ---

cnn_fixed_params = {
    # Skorch params
    'max_epochs': 15,
    'lr': 0.001,
    'batch_size': 32,
    'optimizer__weight_decay': 0.01
}

flexible_vit_fixed_params = { # These params are equivalent to the SimpleViT params
    # Skorch params
    'max_epochs': 25,
    'lr': 1e-4,
    'batch_size': 32,
    'optimizer__weight_decay': 0.01,
    # Module params for FlexibleViT to mimic SimpleViT
    # 'module__num_classes': your_actual_num_classes,  # This is passed by pipeline automatically
    # but if you were overriding module directly.
    # SkorchModelAdapter gets module__num_classes
    # from model_adapter_config first.
    'module__vit_model_variant': 'vit_b_16',
    'module__pretrained': True,
    'module__unfreeze_strategy': 'simple_vit_compat',
    'module__custom_head_hidden_dims': None,
    'module__head_dropout_rate': 0.0,
}

debug_fixed_params = {
    # Skorch/Training Loop Params (Fast Execution)
    'max_epochs': 4,        # Run only for 1 epoch
    'batch_size': 8,        # Small batch size
    'lr': 1e-4,             # A reasonably small LR unlikely to explode immediately

    # Optimizer Params (Common Default)
    'optimizer__weight_decay': 0.01,

    # Callbacks (using pipeline defaults, but could potentially disable for speed if needed)
    # Note: EarlyStopping might trigger immediately if validation loss increases on epoch 1.

    # --- NO module__ specific parameters here ---
    # This ensures compatibility. FlexibleViT, SimpleCNN, DiffusionClassifier
    # will use their defaults for dropout, unfreezing, etc., as set in
    # ClassificationPipeline.__init__'s use of model_adapter_config.
}


# --- Parameter Space Definitions (for Search Methods) ---

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

# param_grid_simple_vit = {
#     # Skorch parameters (especially LR for fine-tuning)
#     'lr': [0.001, 0.0005, 0.0001, 0.00005],  # Often lower LRs for fine-tuning
#     'batch_size': [16, 32],  # Memory constraints often tighter with ViT
#
#     # Optimizer (AdamW) parameters
#     'optimizer__weight_decay': [0.01, 0.001, 0.0],  # Weight decay is important
#
#     # Module (SimpleViT) parameters
#     # Since we only replaced the head and froze most layers, there are fewer
#     # *direct* module hyperparameters to tune via __init__.
#     # If you added dropout to the new head, you could tune 'module__dropout_rate'.
#     # You *could* potentially tune which layers are frozen, but that's complex via grid search.
#
#     # Training duration / EarlyStopping focus
#     'max_epochs': [5, 10, 15],  # If fine-tuning quickly
# }
simple_vit_param_grid = None # TODO: remove this

flexible_vit_param_grid = [
    # --- Scenario 1: 'simple_vit_compat' - Explicitly test SimpleViT-equivalent behavior ---
    {
        'module__vit_model_variant': ['vit_b_16'],  # Fixed to vit_b_16
        'module__pretrained': [True],  # Fixed to True
        'module__unfreeze_strategy': ['simple_vit_compat'],  # Fixed to this strategy
        # 'module__num_end_encoder_layers_to_unfreeze': [2], # Effectively fixed by 'simple_vit_compat' logic internally
        # No need to list if not directly used by this strategy's __init__ path,
        # but skorch is robust to extra params.
        'module__custom_head_hidden_dims': [None],  # Fixed to simple linear head
        'module__head_dropout_rate': [0.0],  # Fixed to no dropout in head

        # Skorch & Optimizer parameters for this scenario (can be tuned or fixed)
        # Let's include a small range for LR to see if the compat strategy benefits from slight LR tuning
        'lr': [5e-5, 1e-4, 2e-4],
        'optimizer__weight_decay': [0.01, 0.05],  # SimpleViT might have had a default or tuned one
        'batch_size': [16, 32],  # Common batch sizes
        'max_epochs': [25, 40],  # Typical epochs for this kind of fine-tuning
    },

    # --- Scenario 2: 'head_only' - Fine-tune only the classification head ---
    {
        'module__vit_model_variant': ['vit_b_16'],
        'module__pretrained': [True],
        'module__unfreeze_strategy': ['head_only'],
        'module__custom_head_hidden_dims': [None, [512]], # Try simple linear vs one hidden layer
        # 'module__custom_head_hidden_dims': [None, [256], [512]], # Longer Option
        'module__head_dropout_rate': [0.0, 0.25],           # Try no dropout vs some dropout
        # 'module__head_dropout_rate': [0.0, 0.25, 0.5], # Longer Option

        # Skorch & Optimizer parameters - maybe tune LR slightly for head only
        'lr': [5e-4, 1e-3],
        # 'lr': [1e-4, 5e-4, 1e-3], # Longer Option
        'optimizer__weight_decay': [0.001],
        # 'optimizer__weight_decay': [0.0, 0.001, 0.01], # Longer Option
        'batch_size': [32],
        # 'batch_size': [16, 32, 64], # Longer Option
        'max_epochs': [20],
        # 'max_epochs': [15, 25], # Longer Option
    },

    # --- Scenario 3: 'end_encoder_layers' - Fine-tune last N encoder layers ---
    {
        'module__vit_model_variant': ['vit_b_16'],
        'module__pretrained': [True],
        'module__unfreeze_strategy': ['end_encoder_layers'],
        'module__num_end_encoder_layers_to_unfreeze': [2, 4], # Try unfreezing 2 vs 4 layers
        # 'module__num_end_encoder_layers_to_unfreeze': [1, 2, 4], # Longer Option
        'module__custom_head_hidden_dims': [None],        # Keep head simple initially for this strategy
        # 'module__custom_head_hidden_dims': [None, [512]], # Longer Option
        'module__head_dropout_rate': [0.0, 0.2],            # Try dropout vs none
        # 'module__head_dropout_rate': [0.0, 0.2, 0.4], # Longer Option

        # Skorch & Optimizer parameters - lower LR often needed
        'lr': [5e-5, 1e-4],
        # 'lr': [1e-5, 5e-5, 1e-4], # Longer Option
        'optimizer__weight_decay': [0.01],
        # 'optimizer__weight_decay': [0.01, 0.05], # Longer Option
        'batch_size': [16], # Usually smaller batch size needed for more unfrozen layers
        # 'batch_size': [8, 16], # Longer Option
        'max_epochs': [30],
        # 'max_epochs': [20, 30, 50], # Longer Option
    },

    # --- (Optional) Scenario 4: 'all_encoder_layers' - Fine-tune all encoder layers ---
    # (Use cautiously - computationally expensive and prone to overfitting)
    # {
    #     'module__vit_model_variant': ['vit_b_16'],
    #     'module__pretrained': [True],
    #     'module__unfreeze_strategy': ['all_encoder_layers'],
    #     'module__custom_head_hidden_dims': [None], # Keep head simple when unfreezing so much
    #     # 'module__custom_head_hidden_dims': [None, [512]], # Longer Option
    #     'module__head_dropout_rate': [0.0],       # Minimal head dropout initially
    #     # 'module__head_dropout_rate': [0.0, 0.25], # Longer Option
    #
    #     # Skorch & Optimizer parameters - typically lowest LR, small batches
    #     'lr': [1e-5, 5e-5], # Very low learning rates
    #     # 'lr': [1e-5, 3e-5, 5e-5], # Longer Option
    #     'optimizer__weight_decay': [0.01],
    #     # 'optimizer__weight_decay': [0.01, 0.05], # Longer Option
    #     'batch_size': [8],   # Often requires the smallest batch size due to memory
    #     # 'batch_size': [8, 16], # Longer Option
    #     'max_epochs': [50], # Often needs more epochs to fine-tune fully
    #     # 'max_epochs': [30, 50, 75], # Longer Option
    # },
]

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

debug_param_grid = {
    # Skorch/Training Loop Params (Single Combination)
    'max_epochs': [4],        # List with one value
    'batch_size': [8],        # List with one value
    'lr': [1e-4, 0.001],  # List with one value

    # Optimizer Params (Single Combination)
    'optimizer__weight_decay': [0.01], # List with one value

    # --- NO module__ specific parameters here ---
    # Ensures compatibility across different model types selected for the pipeline.
    # The specific module (CNN, ViT, etc.) will use its defaults for other params.
}
