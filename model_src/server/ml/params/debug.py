debug_fixed_params = {
    'max_epochs': 4,
    'lr': 0.01,
    'optimizer': 'sgd',
    'optimizer__momentum': 0.9,
    'optimizer__weight_decay': 5e-4,
    'optimizer__nesterov': True,
    'batch_size': 32,

    # 'module__dropout_rate': 0.3
    'module__head_dropout_rate': 0.3
}

debug_param_grid = {
    'max_epochs': [4],
    'batch_size': [8],
    'lr': [1e-4, 0.001],

    'optimizer__weight_decay': [0.01],

}
