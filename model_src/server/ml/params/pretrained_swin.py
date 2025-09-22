# --- Fixed Parameter Sets ---

pretrained_swin_fixed_params = {
    'max_epochs': 70,
    'lr': 2e-5,
    'batch_size': 16,
    'optimizer': 'AdamW',
    'optimizer__weight_decay': 0.05,

    'callbacks__default_lr_scheduler__policy': 'CosineAnnealingLR',
    'callbacks__default_lr_scheduler__T_max': 70,
    'callbacks__default_lr_scheduler__eta_min': 1e-7,

    'module__swin_model_variant': 'swin_t',
    'module__pretrained': True,
    'module__num_stages_to_unfreeze': 1,
    'module__head_dropout_rate': 0.2,
}

# --- Parameter Space Definitions ---
