Swimcat

- Pretrained ViT: 5-Fold CV

      CV Evaluation Summary (on full data, 5 folds, 95% CI):
        Accuracy            : 0.9929 +/- 0.0055
        F1 Macro            : 0.9929 +/- 0.0055
        Precision Macro     : 0.9931 +/- 0.0053
        Recall Macro        : 0.9929 +/- 0.0055
        Specificity Macro   : 0.9986 +/- 0.0011
        Roc Auc Macro       : 0.9997 +/- 0.0003
        Pr Auc Macro        : 0.9930 +/- 0.0012

    AugmentationStrategy.PAPER_CCSN

```python
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
  
      # --- Gradient Clipping (already discussed) ---
      # 'gradient_clip_value': 5.0, 
  
      'module__vit_model_variant': 'vit_b_16',
      'module__pretrained': True,
      'module__unfreeze_strategy': 'encoder_tail',
      'module__num_transformer_blocks_to_unfreeze': 1,
      'module__unfreeze_cls_token': True,
      'module__unfreeze_pos_embedding': True,
      'module__unfreeze_patch_embedding': False,
      'module__unfreeze_encoder_layernorm': True,
      'module__custom_head_hidden_dims': None,
      'module__head_dropout_rate': 0.50
  }
 ```




- Hybrid-ViT: (5-Fold Cross-Validation)

        CV Evaluation Summary (on full data, 5 folds, 95% CI):
        Accuracy            : 0.9862 +/- 0.0105
        F1 Macro            : 0.9862 +/- 0.0106
        Precision Macro     : 0.9871 +/- 0.0093
        Recall Macro        : 0.9862 +/- 0.0105
        Specificity Macro   : 0.9972 +/- 0.0021
        Roc Auc Macro       : 0.9994 +/- 0.0007
        Pr Auc Macro        : 0.9917 +/- 0.0029

```python
hybrid_vit_fixed_params = {
    # --- Skorch/Training Loop Parameters ---
    'max_epochs': 60,
    'lr': 5e-5,
    'batch_size': 32,

    # --- Optimizer Configuration ---
    'optimizer': 'AdamW',
    'optimizer__weight_decay': 0.2,

    # --- LR Scheduler ---
    'callbacks__default_lr_scheduler__policy': 'CosineAnnealingLR',
    'callbacks__default_lr_scheduler__T_max': 60,
    'callbacks__default_lr_scheduler__eta_min': 1e-6,

    # --- Early Stopping ---
    'callbacks__default_early_stopping__patience': 15,
  
    'cutmix_alpha': 1.0, 
    'cutmix_probability': 0.3, 

    'criterion__label_smoothing': 0.1,

    # --- HybridViT Module Parameters ---
    'module__cnn_extractor_type': "standard_cnn",
    'module__cnn_model_name': "efficientnet_b0",
    'module__cnn_pretrained_imagenet': True,
    'module__cnn_output_channels_target': 192,


    'module__cnn_freeze_extractor': False,
    'module__cnn_num_frozen_stages': 4,

    'module__cnn_fine_tuned_weights_path': None,

    # --- ViT Backend Parameters (passed to PretrainedViT within HybridViT) ---
    'module__vit_model_variant': 'vit_b_16',
    'module__vit_pretrained_imagenet': True,
    'module__unfreeze_strategy': 'encoder_tail',
    'module__num_transformer_blocks_to_unfreeze': 2,
    'module__unfreeze_cls_token': True,
    'module__unfreeze_pos_embedding': True,
    'module__unfreeze_patch_embedding': False,
    'module__unfreeze_encoder_layernorm': True,
    'module__custom_head_hidden_dims': None, 
    'module__head_dropout_rate': 0.2,

    # --- Parameters for HybridViT constructor (related to image/feature sizes) ---
    'module__pipeline_img_h': 224,
    'module__pipeline_img_w': 224,
}
```




- Pretrained ViT: 5-Fold CV (Offline Augmentation)
  
      CV Evaluation Summary (on full data, 5 folds, 95% CI):
        Accuracy            : 0.9919 +/- 0.0077
        F1 Macro            : 0.9919 +/- 0.0077
        Precision Macro     : 0.9921 +/- 0.0074
        Recall Macro        : 0.9919 +/- 0.0077
        Specificity Macro   : 0.9984 +/- 0.0015
        Roc Auc Macro       : 0.9995 +/- 0.0006
        Pr Auc Macro        : 0.9925 +/- 0.0021

  AugmentationStrategy.PAPER_CCSN

  Params:
