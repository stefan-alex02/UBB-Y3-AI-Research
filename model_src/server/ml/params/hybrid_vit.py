hybrid_vit_fixed_params = {
    # --- Skorch/Training Loop Parameters ---
    'max_epochs': 50,  # EfficientNet is already pretrained, might not need as many epochs as CNN from scratch.
                       # Early stopping is key. 50-80 is a good range to explore.
    'lr': 5e-5,        # <<< SIGNIFICANTLY LOWER LR. Start here for fine-tuning both pretrained EffNet and ViT.
                       # Common range: 1e-5 to 1e-4.
    'batch_size': 16,  # Keep this for now, EfficientNet-B0 + ViT-B/16 should fit.
                       # Could try 32 if memory allows and if smaller batches are too noisy.

    # --- Optimizer Configuration ---
    'optimizer': 'AdamW',
    'optimizer__weight_decay': 0.05, # Good default for AdamW, helps with regularization.

    # --- LR Scheduler ---
    'callbacks__default_lr_scheduler__policy': 'CosineAnnealingLR',
    'callbacks__default_lr_scheduler__T_max': 50, # Match max_epochs
    'callbacks__default_lr_scheduler__eta_min': 1e-7, # Reduced from 1e-6 for finer control at end.
    # Alternative: ReduceLROnPlateau
    # 'callbacks__default_lr_scheduler__policy': 'ReduceLROnPlateau',
    # 'callbacks__default_lr_scheduler__monitor': 'valid_loss',
    # 'callbacks__default_lr_scheduler__patience': 5, # e.g., 3-5 for fine-tuning
    # 'callbacks__default_lr_scheduler__factor': 0.2,
    # 'callbacks__default_lr_scheduler__min_lr': 1e-7,

    # --- Early Stopping ---
    'callbacks__default_early_stopping__patience': 10, # (e.g., 7-15 is reasonable)

    # --- HybridViT Module Parameters ---
    'module__cnn_extractor_type': "standard_cnn",    # <<< CHANGED
    'module__cnn_model_name': "efficientnet_b0",    # <<< CHANGED
    'module__cnn_pretrained_imagenet': True,         # <<< CHANGED: Use ImageNet weights for EffNet
    'module__cnn_output_channels_target': 192,       # <<< NEW/ADJUSTED: Project EffNet's 1280 channels down.
                                                     # 128, 192, or 256 are good values to try.
                                                     # This becomes `hybrid_in_channels` for PretrainedViT.
    'module__cnn_freeze_extractor': False,           # <<< Fine-tune the EfficientNet
    'module__cnn_num_frozen_stages': 2,              # <<< Example: Freeze first 2 "stages" of EfficientNet-B0.
                                                     # EfficientNet features are typically grouped.
                                                     # Stage 0 (stem), Stage 1 (first MBConv block(s)), etc.
                                                     # For EffNet-B0, self.features is nn.Sequential of 8 blocks (0 to 7)
                                                     # features[0] is stem.
                                                     # features[1] is first set of MBConv blocks.
                                                     # features[2] is second set, etc.
                                                     # Freezing 2 stages means freezing features[0] and features[1].
                                                     # You can experiment: 0 (fine-tune all), 1, 2, 3, or True for all.

    'module__cnn_fine_tuned_weights_path': None, # Not using cloud-fine-tuned CNN weights for this config.

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
    'module__head_dropout_rate': 0.3,                # <<< REDUCED a bit from 0.55, can tune.
                                                     # 0.2-0.5 is a common range.

    # --- Parameters for HybridViT constructor (related to image/feature sizes) ---
    'module__pipeline_img_h': 224,
    'module__pipeline_img_w': 224,

    # --- Mixed Precision ---
    # 'use_amp': True, # Recommended
}