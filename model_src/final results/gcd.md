GCD 

- Pretrained ViT: Single train

        Acc=0.7707, Macro F1=0.7504

    AugmentationStrategy.SKY_ONLY_ROTATION

```python
pretrained_vit_fixed_params = {
    'max_epochs': 70,
    'lr': 5e-5,
    'batch_size': 16,

    'optimizer': 'AdamW',
    'optimizer__weight_decay': 0.2, # Start with original, can reduce later

    'callbacks__default_lr_scheduler__policy': 'CosineAnnealingLR',
    'callbacks__default_lr_scheduler__T_max': 70,

    # 'callbacks__default_lr_scheduler__policy': 'CosineAnnealingWarmRestarts',
    # 'callbacks__default_lr_scheduler__T_0': 15,      # Epochs for the first cycle
    # 'callbacks__default_lr_scheduler__T_mult': 1,     # Subsequent cycles are same length as T_0

    'callbacks__default_lr_scheduler__eta_min': 1e-06,

    'callbacks__default_early_stopping__patience': 15,

    # --- CutMix Parameters ---
    # 'cutmix_alpha': 1.0,
    # 'cutmix_probability': 0.9, # for CCSN
    # 'cutmix_probability': 0.5, # for Swimcat
    # 'cutmix_probability': 0.5, # for GCD

    # --- Gradient Clipping (already discussed) ---
    # 'gradient_clip_value': 5.0,  # If you want to use it

    'module__vit_model_variant': 'vit_b_16',
    'module__pretrained': True,
    'module__unfreeze_strategy': 'encoder_tail',
    'module__num_transformer_blocks_to_unfreeze': 1,
    'module__unfreeze_cls_token': True,
    'module__unfreeze_pos_embedding': True,
    'module__unfreeze_patch_embedding': False,
    'module__unfreeze_encoder_layernorm': True,
    'module__custom_head_hidden_dims': None,
    'module__head_dropout_rate': 0.50,

    'iterator_train__shuffle': True,
}
  ```

- HyViT - Single train

    Computed Metrics: Acc=0.7686, Macro F1=0.7514

```python
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
    # 'cutmix_alpha': 1.0,
    # 'cutmix_probability': 0.9,  # for CCSN
    # 'cutmix_probability': 0.3, # for Swimcat
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
    'module__num_transformer_blocks_to_unfreeze': 1, # Fine-tune last 2 ViT blocks. Could try 1 to 4.
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
```


- Pretrained ViT: Single train (Offline Augmentation)

    Computed Metrics: Acc=0.7540, Macro F1=0.7194

```python
pretrained_vit_fixed_params = {
    'max_epochs': 70,
    'lr': 5e-5,
    'batch_size': 16,

    'optimizer': 'AdamW',
    'optimizer__weight_decay': 0.2, # Start with original, can reduce later

    'callbacks__default_lr_scheduler__policy': 'CosineAnnealingLR',
    'callbacks__default_lr_scheduler__T_max': 70,

    # 'callbacks__default_lr_scheduler__policy': 'CosineAnnealingWarmRestarts',
    # 'callbacks__default_lr_scheduler__T_0': 15,      # Epochs for the first cycle
    # 'callbacks__default_lr_scheduler__T_mult': 1,     # Subsequent cycles are same length as T_0

    'callbacks__default_lr_scheduler__eta_min': 1e-06,

    'callbacks__default_early_stopping__patience': 15,

    # --- CutMix Parameters ---
    # 'cutmix_alpha': 1.0,
    # 'cutmix_probability': 0.9, # for CCSN
    # 'cutmix_probability': 0.5, # for Swimcat
    # 'cutmix_probability': 0.5, # for GCD

    # --- Gradient Clipping ---
    # 'gradient_clip_value': 5.0,  # If you want to use it

    # --- PretrainedViT Module Parameters ---
    'module__vit_model_variant': 'vit_b_16',
    'module__pretrained': True,
    'module__unfreeze_strategy': 'encoder_tail',
    'module__num_transformer_blocks_to_unfreeze': 1,
    'module__unfreeze_cls_token': True,
    'module__unfreeze_pos_embedding': True,
    'module__unfreeze_patch_embedding': False,
    'module__unfreeze_encoder_layernorm': True,
    'module__custom_head_hidden_dims': None,
    'module__head_dropout_rate': 0.50,

    # --- Data Loader Parameters ---
    'iterator_train__shuffle': True,
}
```

- Simple Vit (5-fold CV on full dataset)

      CV Evaluation Summary (on full data, 5 folds, 95% CI):
        Accuracy            : 0.8887 +/- 0.0060
        F1 Macro            : 0.8904 +/- 0.0059
        Precision Macro     : 0.8914 +/- 0.0051
        Recall Macro        : 0.8905 +/- 0.0075
        Specificity Macro   : 0.9798 +/- 0.0011
        Roc Auc Macro       : 0.9892 +/- 0.0008
        Pr Auc Macro        : 0.9508 +/- 0.0056

```python
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
    # 'cutmix_alpha': 1.0,
    # 'cutmix_probability': 0.9,  # for CCSN
    # 'cutmix_probability': 0.3, # for Swimcat
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
    'module__num_transformer_blocks_to_unfreeze': 1, # Fine-tune last 2 ViT blocks. Could try 1 to 4.
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
```