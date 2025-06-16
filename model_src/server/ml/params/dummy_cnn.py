cnn_fixed_params = {
    # Skorch params
    'max_epochs': 10,
    'lr': 0.001,
    'batch_size': 32,
    'optimizer__weight_decay': 0.01
}

cnn_param_grid = {
    'lr': [0.005, 0.001, 0.0005],
    'batch_size': [16, 32],

    # Optimizer (AdamW) parameters
    'optimizer__weight_decay': [0.01, 0.001, 0.0001],
    # 'optimizer__betas': [(0.9, 0.999), (0.85, 0.99)],

    # Module (SimpleCNN) parameters
    'module__dropout_rate': [0.3, 0.4, 0.5, 0.6],

    # 'max_epochs': [15, 25],
}
