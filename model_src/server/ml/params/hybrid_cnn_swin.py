# --- Fixed Parameter Sets ---

hybrid_cnn_swin_fixed_params_paper = {
    # Skorch / Training Loop
    'max_epochs': 100,  # Paper doesn't state, this is a guess, might need more
    'lr': 0.0016,  # From paper
    'batch_size': 16,  # Common for transformers, paper doesn't state
    'optimizer': 'AdamW',
    'optimizer__weight_decay': 0.01,  # Typical AdamW default, paper doesn't state AdamW's WD

    # Callbacks - LR Scheduler
    'callbacks__default_lr_scheduler__policy': 'CosineAnnealingLR',  # From paper
    'callbacks__default_lr_scheduler__T_max': 100,  # Should match max_epochs if not restarting
    'callbacks__default_lr_scheduler__eta_min': 1e-6,  # Common small value

    # Callbacks - Early Stopping (good practice, paper doesn't mention)
    'callbacks__default_early_stopping__patience': 15,
    'callbacks__default_early_stopping__monitor': 'valid_loss',

    # HybridCNNRSModel specific parameters (prefixed with 'module__')
    # num_classes is set by pipeline_v1
    # cnn_in_channels is default 3

    # Swin Transformer part of the hybrid model
    'module__swin_model_variant': 'swin_t',  # From paper
    'module__swin_pretrained': True,  # From paper (for the Swin part)
    'module__swin_num_stages_to_unfreeze': 2,  # Paper doesn't state, 1 or 2 is reasonable for fine-tuning
    'module__swin_head_dropout_rate': 0.0,
    # Paper's head is Linear, Swin's internal head dropout controlled by PretrainedSwin
    # If PretrainedSwin's internal dropout is 0, this won't have effect unless MLP head in Swin
    # Let's assume simple linear head for Swin from torchvision initially.
}