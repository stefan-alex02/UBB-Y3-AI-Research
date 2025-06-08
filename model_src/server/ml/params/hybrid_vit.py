hybrid_vit_fixed_params = {
    # --- Skorch/Training Loop Parameters ---
    'max_epochs': 60,
    'lr': 5e-5,
    'batch_size': 32,

    # --- Optimizer Configuration ---
    'optimizer': 'AdamW',
    'optimizer__weight_decay': 0.1, # Good default for AdamW, helps with regularization.

    # --- LR Scheduler ---
    'callbacks__default_lr_scheduler__policy': 'CosineAnnealingLR',
    'callbacks__default_lr_scheduler__T_max': 60, # Match max_epochs
    'callbacks__default_lr_scheduler__eta_min': 1e-6,

    # --- Early Stopping ---
    'callbacks__default_early_stopping__patience': 15,

    'criterion__label_smoothing': 0.1, # Common value for label smoothing

    # --- CutMix Parameters ---
    'cutmix_alpha': 1.0,
    # 'cutmix_probability': 0.9,  # for CCSN
    'cutmix_probability': 0.3, # for Swimcat
    # 'cutmix_probability': 0.5, # for GCD

    # --- Gradient Clipping---
    'gradient_clip_value': 5.0,  # If you want to use it

    # --- HybridViT Module Parameters ---
    'module__cnn_extractor_type': "standard_cnn",
    'module__cnn_model_name': "efficientnet_b0",
    'module__cnn_pretrained_imagenet': True,
    'module__cnn_output_channels_target': 192,

    'module__cnn_freeze_extractor': False,
    'module__cnn_num_frozen_stages': 2,
    'module__cnn_fine_tuned_weights_path': None,
    # 'module__cnn_fine_tuned_weights_path': 'experiments/CCSN/stfeat/20250606_053320_seed42/single_train_053320/stfeat_sngl_ep13_val_loss1p54_053320.pt',

    # --- ViT Backend Parameters (passed to PretrainedViT within HybridViT) ---
    'module__vit_model_variant': 'vit_b_16',
    'module__vit_pretrained_imagenet': True, # ViT backend uses its ImageNet pretraining
    'module__unfreeze_strategy': 'encoder_tail',
    'module__num_transformer_blocks_to_unfreeze': 2, # Fine-tune last 2 ViT blocks. Could try 1 to 4.
    'module__unfreeze_cls_token': True,
    'module__unfreeze_pos_embedding': True,
    'module__unfreeze_patch_embedding': False,       # For ViT's own, not used in hybrid mode by ViT
    'module__unfreeze_encoder_layernorm': True,
    'module__custom_head_hidden_dims': None,         # Keep head simple initially
    'module__head_dropout_rate': 0.2,

    # --- Parameters for HybridViT constructor (related to image/feature sizes) ---
    'module__pipeline_img_h': 224,
    'module__pipeline_img_w': 224,

    # --- Data Loader Parameters ---
    'iterator_train__shuffle': True,
}

hybrid_vit_fixed_params_no_prev_finetune = {
    # --- Skorch/Training Loop Parameters ---
    'max_epochs': 70,
    'lr': 5e-5,
    'batch_size': 32,

    # --- Optimizer Configuration ---
    'optimizer': 'AdamW',
    'optimizer__weight_decay': 0.2, # Good default for AdamW, helps with regularization.

    # --- LR Scheduler ---
    'callbacks__default_lr_scheduler__policy': 'CosineAnnealingLR',
    'callbacks__default_lr_scheduler__T_max': 70, # Match max_epochs
    'callbacks__default_lr_scheduler__eta_min': 1e-6,

    # --- Early Stopping ---
    'callbacks__default_early_stopping__patience': 15,

    'criterion__label_smoothing': 0.1, # Common value for label smoothing

    # --- HybridViT Module Parameters ---
    'module__cnn_extractor_type': "standard_cnn",
    'module__cnn_model_name': "efficientnet_b0",
    'module__cnn_pretrained_imagenet': True,
    'module__cnn_output_channels_target': 192,


    'module__cnn_freeze_extractor': True,
    'module__cnn_num_frozen_stages': 0,

    'module__cnn_fine_tuned_weights_path': None,

    # --- ViT Backend Parameters (passed to PretrainedViT within HybridViT) ---
    'module__vit_model_variant': 'vit_b_16',
    'module__vit_pretrained_imagenet': True, # ViT backend uses its ImageNet pretraining
    'module__unfreeze_strategy': 'encoder_tail',
    'module__num_transformer_blocks_to_unfreeze': 2, # Fine-tune last 2 ViT blocks. Could try 1 to 4.
    'module__unfreeze_cls_token': True,
    'module__unfreeze_pos_embedding': True,
    'module__unfreeze_patch_embedding': True,       # For ViT's own, not used in hybrid mode by ViT
    'module__unfreeze_encoder_layernorm': True,
    'module__custom_head_hidden_dims': None,         # Keep head simple initially
    'module__head_dropout_rate': 0.5,

    # --- Parameters for HybridViT constructor (related to image/feature sizes) ---
    'module__pipeline_img_h': 224,
    'module__pipeline_img_w': 224,
}

hybrid_vit_param_grid = {
    # --- Skorch/Training Loop Parameters ---
    'max_epochs': [1],
    'lr': [5e-5],
    'batch_size': [32],

    # --- Optimizer Configuration ---
    'optimizer': ['AdamW'],
    'optimizer__weight_decay': [0.2], # Good default for AdamW, helps with regularization.

    # --- LR Scheduler ---
    'callbacks__default_lr_scheduler__policy': ['CosineAnnealingLR'],
    'callbacks__default_lr_scheduler__T_max': [70], # Match max_epochs
    'callbacks__default_lr_scheduler__eta_min': [1e-6],

    # --- Early Stopping ---
    'callbacks__default_early_stopping__patience': [15],

    'criterion__label_smoothing': [0.1], # Common value for label smoothing

    # --- HybridViT Module Parameters ---
    'module__cnn_extractor_type': ["standard_cnn"],
    'module__cnn_model_name': ["efficientnet_b0"],
    'module__cnn_pretrained_imagenet': [True],
    'module__cnn_output_channels_target': [192],

    'module__cnn_freeze_extractor': [True],
    'module__cnn_num_frozen_stages': [0],
    'module__cnn_fine_tuned_weights_path': [None],
    # 'module__cnn_fine_tuned_weights_path': 'experiments/CCSN/stfeat/20250530_150001_seed42/single_train_150001/stfeat_epoch6_val_valid-loss1.5795.pt',

    # --- ViT Backend Parameters (passed to PretrainedViT within HybridViT) ---
    'module__vit_model_variant': ['vit_b_16'],
    'module__vit_pretrained_imagenet': [True], # ViT backend uses its ImageNet pretraining
    'module__unfreeze_strategy': ['encoder_tail'],
    'module__num_transformer_blocks_to_unfreeze': [2], # Fine-tune last 2 ViT blocks. Could try 1 to 4.
    'module__unfreeze_cls_token': [True],
    'module__unfreeze_pos_embedding': [True],
    'module__unfreeze_patch_embedding': [True],       # For ViT's own, not used in hybrid mode by ViT
    'module__unfreeze_encoder_layernorm': [True],
    'module__custom_head_hidden_dims': [None],         # Keep head simple initially
    'module__head_dropout_rate': [0.5],

    # --- Parameters for HybridViT constructor (related to image/feature sizes) ---
    'module__pipeline_img_h': [224],
    'module__pipeline_img_w': [224],
}