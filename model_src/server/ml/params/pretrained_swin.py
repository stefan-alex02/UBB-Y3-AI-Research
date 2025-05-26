# --- Fixed Parameter Sets ---

pretrained_swin_fixed_params = {
    'max_epochs': 70,
    'lr': 2e-5, # Swin often uses smaller LRs for fine-tuning
    'batch_size': 16, # Swin can be memory intensive
    'optimizer': 'AdamW', # or torch.optim.AdamW
    'optimizer__weight_decay': 0.05,

    'callbacks__default_lr_scheduler__policy': 'CosineAnnealingLR',
    'callbacks__default_lr_scheduler__T_max': 70, # Match max_epochs
    'callbacks__default_lr_scheduler__eta_min': 1e-7,

    'module__swin_model_variant': 'swin_t',
    'module__pretrained': True,
    'module__num_stages_to_unfreeze': 1, # Unfreeze last stage + norm + head
    'module__head_dropout_rate': 0.2, # Swin head often has dropout
}

# --- Parameter Space Definitions ---
