import traceback
from pathlib import Path

import numpy as np

from model_src.server.ml.params.hybrid_cnn_swin import hybrid_cnn_swin_fixed_params_paper
from model_src.server.ml.params.pretrained_swin import pretrained_swin_fixed_params
from server.ml.logger_utils import logger
from server.ml.params.pretrained_vit import param_grid_pretrained_vit_focused, best_config_as_grid_vit
from server.ml.params.scratch_vit import fixed_params_vit_scratch, param_grid_vit_from_scratch
from server.ml import ModelType
from server.ml import PipelineExecutor
from server.ml.config import DATASET_DICT, AugmentationStrategy
from server.ml.params import (debug_fixed_params, cnn_fixed_params, pretrained_vit_fixed_params,
                              diffusion_param_grid, param_grid_pretrained_vit_conditional)
from server.ml.params import debug_param_grid, cnn_param_grid
from server.persistence import load_file_repository, load_minio_repository

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    # --- Repository Configuration ---
    minio_bucket_name = "ml-experiment-artifacts"
    local_repo_base_path = str(script_dir)

    # --- Load Artifact Repository ---
    repo_option = "local"  # "local", "minio", or "none"

    # --- Base Prefix/Directory for this set of experiments ---
    # This will be further structured by dataset/model/timestamp by the PipelineExecutor
    experiment_base_prefix_for_repo = "experiments"  # For S3 prefix or local subfolder

    # --- Set up results directory ---
    results_base_dir = script_dir / 'results'
    # results_base_dir = None

    # --- Configuration ---
    # Select Dataset:
    selected_dataset = "ccsn"  # 'GCD', 'mGCD', 'mGCDf', 'swimcat', 'ccsn'

    # Select Model:
    model_type = "hswin"  # 'cnn', 'pvit', 'swin', 'hswin', 'svit', 'diff'

    # Chosen sequence index: (1-7)
    # 1: Single Train and Eval
    # 2: Non-Nested Grid Search + Eval best model
    # 3: Nested Grid Search (Requires FLAT or FIXED with force_flat=True)
    # 4: Simple CV Evaluation (Requires FLAT or FIXED with force_flat=True)
    # 5: Non-Nested Grid Search + CV Evaluation (Requires FLAT or FIXED with force_flat=True)
    # 6: Load Pre-trained and Evaluate
    # 7: Load Pre-trained and Predict on New Images
    chosen_sequence_idx = 1  # Change this to select the sequence you want to run

    # Image size for the model
    img_size = (224, 224)  # Common size for CNNs and ViTs
    # img_size = (448, 448)  # Common size for CNNs and ViTs

    # Flag for CV methods on FIXED datasets:
    # Set to True to allow nested_grid_search and cv_model_evaluation on FIXED datasets
    # by treating train+test as one pool (USE WITH CAUTION - not standard evaluation).
    force_flat = False

    # Flag for overriding parameters:
    enable_debug_params = False # Set to True to use the override params for any model type

    # Trained model path for loading
    # saved_model_path = "./results/mini-GCD/cnn/20250509_021630_seed42/single_train_20250509_021630_121786/cnn_epoch4_val_valid-loss0.9061.pt"
    # saved_model_path = "./results/Swimcat-extend/cnn/20250515_160130_seed42/single_train_20250515_160130_450999/cnn_epoch4_val_valid-loss0.3059.pt"
    saved_model_path = "./experiments/CCSN/pvit/20250518_193300_seed42/non_nested_random_193300/pvit_best_batch_size=16_lr=3e-05_max_epochs=70_custom_head_h_cv_score0p4671.pt"

    # New image paths for prediction
    # existing_prediction_paths = [
    #     "C:/Users/Stefan/Downloads/cumulonimbus-clouds-1024x641.jpeg",
    #     "C:/Users/Stefan/Downloads/heavy-downpours-cumulonimbus.webp",
    #     "C:/Users/Stefan/Downloads/cumulonimbus-at-night-cschoeps.jpg",
    # ]
    # iterate over the images in the directory (first 10)
    existing_prediction_paths = [
        str(p) for p in Path("./data/Swimcat-extend/E-Thick Dark Clouds").glob("*.png")
    ][:2]

    # --- Check if the dataset path exists ---
    dataset_path = script_dir / DATASET_DICT[selected_dataset]  # Path to the dataset
    if not Path(dataset_path).exists():
         logger.error(f"Dataset path not found: {dataset_path}")
         logger.error("Please create the dataset or modify the 'dataset_path' variable.")
         exit()

    # --- Load the repository ---
    if repo_option == "local":
        repo = load_file_repository(logger, repo_base_path=local_repo_base_path)
    elif repo_option == "minio":
        repo = load_minio_repository(logger, bucket_name=minio_bucket_name)
    elif repo_option == "none":
        repo = None

    # --- Define Hyperparameter Grid / Fixed Params based on Model Type ---
    if enable_debug_params:
        # Override the chosen_param_grid with the override_params
        chosen_fixed_params = debug_fixed_params
        chosen_param_grid = debug_param_grid

    elif model_type == ModelType.CNN:
        chosen_fixed_params = cnn_fixed_params
        chosen_param_grid = cnn_param_grid

    elif model_type == ModelType.PRETRAINED_VIT:
        chosen_fixed_params = pretrained_vit_fixed_params
        chosen_param_grid = best_config_as_grid_vit

    elif model_type == ModelType.PRETRAINED_SWIN:
        chosen_fixed_params = pretrained_swin_fixed_params
        chosen_param_grid = best_config_as_grid_vit # TODO: update

    elif model_type == ModelType.HYBRID_SWIN:
        chosen_fixed_params = hybrid_cnn_swin_fixed_params_paper
        chosen_param_grid = best_config_as_grid_vit # TODO: update

    elif model_type == ModelType.SCRATCH_VIT:
        chosen_fixed_params = fixed_params_vit_scratch
        chosen_param_grid = param_grid_vit_from_scratch

    elif model_type == ModelType.DIFFUSION:
        chosen_fixed_params = debug_fixed_params # Using debug fixed params for the moment (TODO: update)
        chosen_param_grid = diffusion_param_grid

    else:
        logger.error(f"Model type '{model_type}' not recognized. Supported: {[m.value for m in ModelType]}")
        exit()

    # --- Define Augmentation Strategy ---
    # Choose the augmentation strategy based on the dataset
    if selected_dataset in ['mGCD', 'mGCDf', 'GCD', 'swimcat']:
        augmentation_strategy = AugmentationStrategy.SKY_ONLY_ROTATION
    elif selected_dataset == 'ccsn':
        augmentation_strategy = AugmentationStrategy.GROUND_AWARE_NO_ROTATION
        # augmentation_strategy = AugmentationStrategy.DEFAULT_STANDARD
    else:
        augmentation_strategy = AugmentationStrategy.DEFAULT_STANDARD

    # --- Define Method Sequence ---
    # Example 1: Single Train and Eval
    methods_seq_1 = [
        ('single_train', {
            'params': chosen_fixed_params, # Fixed hyperparams
            'save_model': False,
            'val_split_ratio': 0.2, # Explicit val split
            'results_detail_level': 2,
        }),
        ('single_eval', {
            'plot_level': 2
        }),
    ]

    # Example 2: Non-Nested Grid Search + Eval best model
    methods_seq_2 = [
        ('non_nested_grid_search', {
            'param_grid': chosen_param_grid,
            'cv': 5,
            'method': 'grid',
            # 'method': 'random',
            # 'n_iter': 4,
            'internal_val_split_ratio': 0.2,
            'scoring': 'accuracy',
            'save_best_model': False,
            'results_detail_level': 2,
        }),
        # The best model is refit and stored in pipeline.model_adapter after search
        ('single_eval', {}), # Evaluate the refit best model
    ]

    # Example 3: Nested Grid Search (Requires FLAT or FIXED with force_flat=True)
    methods_seq_3 = [
         ('nested_grid_search', {
             'param_grid': chosen_param_grid,
             'outer_cv': 2,
             'inner_cv': 2,
             'method': 'grid',
             'scoring': 'accuracy'
         })
    ]

    # Example 4: Simple CV Evaluation (Requires FLAT or FIXED with force_flat=True)
    methods_seq_4 = [
         ('cv_model_evaluation', {
             'params': chosen_fixed_params, # Pass fixed hyperparams
             'cv': 2,
             'evaluate_on': 'full', # Explicitly state (or rely on default)
             'results_detail_level': 3,
        })
    ]

    # Example 5: Non-Nested Grid Search + CV Evaluation (Requires FLAT or FIXED with force_flat=True)
    methods_seq_5 = [
        ('non_nested_grid_search', {
            'param_grid': chosen_param_grid,
            'cv': 2,
            'method': 'grid',
            'scoring': 'accuracy',
            'save_best_model': True
        }),
         ('cv_model_evaluation', {
             'cv': 2,
             'use_best_params_from_step': 0, # Special key indicates using best_params from previous step (index 0)
             # Optionally provide specific params for cv_eval to override defaults if needed
             # 'params': {'max_epochs': 15}, # e.g., override max_epochs just for CV eval
             'evaluate_on': 'test',
             'results_detail_level': 2,
        })
    ]

    # Example 6: Load Pre-trained and Evaluate
    methods_seq_6 = [
        ('load_model', {'model_path_or_key': saved_model_path}),
        ('single_eval', {
            'plot_level': 2  # Save AND show plots after single_eval
        }),
    ]

    # Example 7: Load Pre-trained and Predict on New Images
    methods_seq_7 = [
        ('load_model', {
            'model_path_or_key': saved_model_path
        }),
        ('predict_images', {
            'image_sources': existing_prediction_paths,  # Use the list of image paths
            'generate_lime_explanations': True,
            'results_detail_level': 3,  # Save a basic JSON of predictions
            'plot_level': 2  # Save and show prediction plots
        })
    ]

    # Example 8: Load Pre-trained and Fine-tune + Evaluate (Non-functional for now)
    # pretrained_model_path = "./results/mini-GCD-flat/cnn/20250508_155404_seed42/single_train_20250508_155404_816633/cnn_epoch5_val_valid-loss3.1052.pt" # Replace with actual path
    # methods_seq_8 = [
    #     ('load_model', {'model_path_or_key': pretrained_model_path}),
    #     ('single_train', {
    #         'save_model': True,
    #         'val_split_ratio': 0.2,
    #     }),
    #     ('single_eval', {}),
    # ]

    # --- Select the chosen sequence based on index (1-6) ---
    chosen_sequence = globals()[f"methods_seq_{chosen_sequence_idx}"]

    logger.info(f"Executing sequence: {[m[0] for m in chosen_sequence]}")

    # --- Create and Run Executor ---
    try:
        executor = PipelineExecutor(
            dataset_path=dataset_path,
            model_type=model_type,
            artifact_repository=repo,
            experiment_base_key_prefix=experiment_base_prefix_for_repo,
            methods=chosen_sequence,
            force_flat_for_fixed_cv=force_flat, # Pass the flag
            augmentation_strategy=augmentation_strategy,
            show_first_batch_augmentation_default=True,

            # Pipeline default parameters (can be overridden by methods)
            img_size=img_size,
            batch_size=16,
            max_epochs=10,
            patience=10,
            lr=0.001,
            optimizer__weight_decay=0.01,
            test_split_ratio_if_flat=0.2, # For flat datasets
            # module__dropout_rate=0.5 # If applicable to model
            results_detail_level=3,
            plot_level=2,
        )
        final_results = executor.run()

        # Print final results summary
        logger.info("--- Final Execution Results Summary ---")
        for method_id, result_data in final_results.items():
            if isinstance(result_data, dict) and 'error' in result_data:
                 logger.error(f"Method {method_id}: FAILED - {result_data['error']}")
            elif isinstance(result_data, dict):
                 # Try to extract relevant metrics for summary log
                 acc = result_data.get('accuracy', result_data.get('mean_test_accuracy', np.nan))
                 f1 = result_data.get('macro_avg',{}).get('f1', result_data.get('mean_test_f1_macro', np.nan))
                 best_s = result_data.get('best_score', result_data.get('best_tuning_score', np.nan))
                 logger.info(f"Method {method_id}: Completed. "
                             f"(Acc: {acc:.4f}, F1: {f1:.4f}, BestScore: {best_s:.4f})")
            else:
                 logger.info(f"Method {method_id}: Completed. Result: {result_data}")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
         logger.error(f"Pipeline initialization or execution failed: {e}", exc_info=True)
    except Exception as e:
         stack_trace = traceback.format_exc()
         logger.critical(f"An unexpected error occurred: {e}")
         logger.critical(f"Stack trace: {stack_trace}")
