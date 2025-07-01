export const MODEL_TYPES = [
    { value: 'cnn', label: 'Simple CNN' },
    { value: 'pvit', label: 'Pretrained ViT (Vision Transformer)' },
    { value: 'svit', label: 'Scratch ViT (Vision Transformer)' },
    { value: 'diff', label: 'Diffusion Classifier (ResNet50 based)' },
    { value: 'hyvit', label: 'Hybrid ViT (CNN + ViT Backend)' },
    { value: 'cnn_feat', label: 'Paper CNN Feature Extractor (Standalone)' },
    { value: 'stfeat', label: 'Standard CNN Feature Extractor (Standalone)' },
    { value: 'xcloud', label: 'Xception CloudNet' },
    { value: 'mcloud', label: 'MobileNet CloudNet' },
    { value: 'resnet', label: 'ResNet18 Cloud (Custom Head)' },
    { value: 'swin', label: 'Pretrained Swin Transformer' },
];

export const DATASET_NAMES = [
    'mGCD',
    'mGCDf',
    'GCD',
    'swimcat',
    'ccsn',
];

export const AVAILABLE_AUG_STRATEGIES = [
    { value: "default_standard", label: "Default Standard Augmentations" },
    { value: "sky_only_rotation", label: "Sky/Cloud Optimized (Full Rotation)" },
    { value: "ground_aware_no_rotation", label: "Ground Aware (No Vertical Flips/Major Rotations)" },
    { value: "no_augmentation", label: "No Augmentation (Resize & Normalize Only)" },
    { value: "paper_replication_gcd", label: "Paper Replication (GCD Specific)" },
    { value: "paper_replication_ccsn", label: "Paper Replication (CCSN Specific)" },
];

export const PIPELINE_METHODS = [
    { value: 'single_train', label: 'Single Train' },
    { value: 'single_eval', label: 'Single Evaluate (on Test Set)' },
    { value: 'non_nested_grid_search', label: 'Tune Hyperparameters (Non-Nested CV)' },
    { value: 'nested_grid_search', label: 'Estimate Generalization (Nested CV)' },
    { value: 'cv_model_evaluation', label: 'Evaluate Stability (K-Fold CV with Fixed Params)' },
];

export const EXPERIMENT_MODES = [
    { value: 'single_train_eval', label: '1. Single Train + Evaluate' },
    { value: 'nn_cv_eval', label: '2. Tune Hyperparams + Evaluate Best' },
    { value: 'nested_cv', label: '3. Estimate Generalization (Nested CV)' },
    { value: 'cv_eval_fixed', label: '4. Evaluate Model Stability (K-Fold CV)' },
    { value: 'custom', label: 'Custom Sequence' },
];

export const METHOD_DEFAULTS = {
    single_train: {
        method_name: 'single_train',
        params: { lr: 0.001, batch_size: 32, max_epochs: 2 },
        save_model: true,
        plot_level: 1,
        results_detail_level: 2,
        val_split_ratio: 0.2,
    },
    single_eval: {
        method_name: 'single_eval',
        plot_level: 1,
        results_detail_level: 2,
    },
    non_nested_grid_search: {
        method_name: 'non_nested_grid_search',
        params: {
            lr: [0.0001, 0.0005],
        },
        cv: 3,
        scoring: 'accuracy',
        method_search_type: 'grid',
        n_iter: 10,
        save_best_model: true,
        plot_level: 1,
        results_detail_level: 2,
        val_split_ratio: 0.15,
    },
    nested_grid_search: {
        method_name: 'nested_grid_search',
        params: {
            lr: [0.001],
        },
        outer_cv: 3,
        inner_cv: 2,
        scoring: 'accuracy',
        method_search_type: 'grid',
        n_iter: 5,
        plot_level: 1,
        results_detail_level: 2,
        val_split_ratio: 0.15,
    },
    cv_model_evaluation: {
        method_name: 'cv_model_evaluation',
        params: { lr: 0.001, max_epochs: 15 },
        cv: 5,
        evaluate_on: 'full',
        plot_level: 1,
        results_detail_level: 3,
        val_split_ratio: 0.1,
        use_best_params_from_step: undefined,
    },
};


export const PRESET_SEQUENCES = {
    single_train_eval: [
        { ...METHOD_DEFAULTS.single_train },
        { ...METHOD_DEFAULTS.single_eval },
    ],
    nn_cv_eval: [
        { ...METHOD_DEFAULTS.non_nested_grid_search },
        {
            ...METHOD_DEFAULTS.single_eval,
            plot_level: 1,
        },
    ],
    nested_cv: [
        { ...METHOD_DEFAULTS.nested_grid_search },
    ],
    cv_eval_fixed: [
        { ...METHOD_DEFAULTS.cv_model_evaluation },
    ],
    custom: [
        { ...METHOD_DEFAULTS.single_train }
    ]
};

export const PARAM_INFO = {
    common_skorch: [
        { key: 'lr', type: 'float', example: '0.001 or [0.001, 0.0001] (for grid search)', description: 'Learning rate for the optimizer.' },
        { key: 'batch_size', type: 'int', example: '32 or [16, 32, 64]', description: 'Number of samples per training iteration.' },
        { key: 'max_epochs', type: 'int', example: '50', description: 'Maximum number of training epochs.' },
        { key: 'optimizer__weight_decay', type: 'float', example: '0.01 or [0.01, 0.05]', description: 'Weight decay (L2 penalty) for AdamW/SGD with momentum.' },
    ],
    module_cnn: [
        { key: 'module__dropout_rate', type: 'float', example: '0.5 or [0.3, 0.5]', description: 'Dropout rate in the CNN classifier head.' },
    ],
    module_vit: [
        { key: 'module__head_dropout_rate', type: 'float', example: '0.25 or [0.0, 0.25, 0.5]', description: 'Dropout rate in the ViT classification head.' },
        { key: 'module__num_transformer_blocks_to_unfreeze', type: 'int', example: '2 or [1, 2, 4]', description: '(PretrainedViT) Number of final transformer blocks to unfreeze.' },
    ],
    grid_search_specific: [
        { key: 'param_grid', type: 'object', example: '{"lr": [1e-4, 5e-5], "batch_size": [16, 32]}', description: 'Dictionary of hyperparameters to search for Skorch model.' },
        { key: 'cv', type: 'int', example: '3', description: 'Number of cross-validation folds for the hyperparameter search itself.' },
        { key: 'scoring', type: 'string', example: '"accuracy" or "f1_macro"', description: 'Metric to evaluate parameter settings.' },
        { key: 'method', type: 'string', example: '"grid" or "random"', description: 'Search strategy (GridSearchCV or RandomizedSearchCV).' },
        { key: 'n_iter', type: 'int', example: '10', description: '(RandomizedSearch) Number of parameter settings sampled.' },
    ],
};

export const initializeMethodStep = (methodConfig) => {
    const baseDefaults = METHOD_DEFAULTS[methodConfig.method_name] || METHOD_DEFAULTS.single_train;
    return {
        method_name: methodConfig.method_name,
        paramsEditorMode: 'json',
        params: JSON.stringify(methodConfig.params || baseDefaults.params || {}, null, 2),
        save_model: methodConfig.save_model !== undefined ? methodConfig.save_model : baseDefaults.save_model,
        save_best_model: methodConfig.save_best_model !== undefined ? methodConfig.save_best_model : baseDefaults.save_best_model,
        plot_level: methodConfig.plot_level !== undefined ? methodConfig.plot_level : baseDefaults.plot_level,
        results_detail_level: methodConfig.results_detail_level !== undefined ? methodConfig.results_detail_level : baseDefaults.results_detail_level,
        cv: methodConfig.cv !== undefined ? methodConfig.cv : baseDefaults.cv,
        outer_cv: methodConfig.outer_cv !== undefined ? methodConfig.outer_cv : baseDefaults.outer_cv,
        inner_cv: methodConfig.inner_cv !== undefined ? methodConfig.inner_cv : baseDefaults.inner_cv,
        scoring: methodConfig.scoring !== undefined ? methodConfig.scoring : baseDefaults.scoring,
        method_search_type: methodConfig.method_search_type !== undefined ? methodConfig.method_search_type : baseDefaults.method_search_type,
        n_iter: methodConfig.n_iter !== undefined ? methodConfig.n_iter : baseDefaults.n_iter,
        evaluate_on: methodConfig.evaluate_on !== undefined ? methodConfig.evaluate_on : baseDefaults.evaluate_on,
        val_split_ratio: methodConfig.val_split_ratio !== undefined ? methodConfig.val_split_ratio : baseDefaults.val_split_ratio,
        use_best_params_from_step: methodConfig.use_best_params_from_step !== undefined ? methodConfig.use_best_params_from_step : baseDefaults.use_best_params_from_step,
        use_best_params_from_step_checkbox: methodConfig.use_best_params_from_step !== undefined,
    };
};

export const PRESET_SEQUENCES_INITIALIZED = {
    single_train_eval: PRESET_SEQUENCES.single_train_eval.map(initializeMethodStep),
    nn_cv_eval: PRESET_SEQUENCES.nn_cv_eval.map(initializeMethodStep),
    nested_cv: PRESET_SEQUENCES.nested_cv.map(initializeMethodStep),
    cv_eval_fixed: PRESET_SEQUENCES.cv_eval_fixed.map(initializeMethodStep),
    custom: PRESET_SEQUENCES.custom.map(initializeMethodStep),
};