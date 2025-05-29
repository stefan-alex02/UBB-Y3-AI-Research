hybrid_vit_fixed_params_paper_cnn_scratch = {
    'max_epochs': 80,
    'lr': 0.001, # Higher LR if CNN is from scratch
    'batch_size': 16,

    'optimizer': 'AdamW',
    'optimizer__weight_decay': 0.05,

    'callbacks__default_lr_scheduler__policy': 'CosineAnnealingLR',
    'callbacks__default_lr_scheduler__T_max': 80,
    'callbacks__default_lr_scheduler__eta_min': 1e-06,

    'callbacks__default_early_stopping__patience': 15,

    'module__cnn_extractor_name': 'paper_cnn',
    'module__cnn_out_channels': 48, # Matches PaperCNNFeatureExtractor
    # 'module__pretrained_cnn_path': None, # Train CNN from scratch
    'module__pretrained_cnn_path': "experiments/CCSN/cnn_feat/20250529_154435_seed42/single_train_154435/cnn_feat_epoch30_val_valid-loss1.6400.pt",

    # Then params are:
    'module__vit_model_variant': 'vit_b_16',
    'module__pretrained': True,
    'module__unfreeze_strategy': 'encoder_tail',
    'module__num_transformer_blocks_to_unfreeze': 2,
    'module__unfreeze_cls_token': True,
    'module__unfreeze_pos_embedding': True,
    'module__unfreeze_patch_embedding': False,
    'module__unfreeze_encoder_layernorm': True,
    'module__custom_head_hidden_dims': None,
    'module__head_dropout_rate': 0.55,

    'module__pipeline_img_h': 224,  # Or whatever your main.py img_size is
    'module__pipeline_img_w': 224,
}