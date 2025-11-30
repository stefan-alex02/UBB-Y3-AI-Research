import traceback
from pathlib import Path, PurePath
from typing import List

import numpy as np

from model_src.server.api.utils import ImagePredictionTask
from model_src.server.ml.params.cloudnet import cloudnet_fixed_params
from model_src.server.ml.params.feature_extractors import paper_cnn_standalone_fixed_params
from model_src.server.ml.params.hybrid_vit import hybrid_vit_fixed_params, hybrid_vit_param_grid
from model_src.server.ml.params.paper_xception_mobilenet import xcloud_fixed_params, mcloud_fixed_params
from model_src.server.ml.params.pretrained_swin import pretrained_swin_fixed_params
from model_src.server.ml.params.resnet import resnet18_cloud_fixed_params, resnet18_finetune_best_practice_params
from model_src.server.ml.params.shufflenet import shufflenet_cloud_fixed_params
from model_src.server.ml.params.standard_cnn_extractor import standard_cnn_fixed_params
from server.ml import ModelType
from server.ml import PipelineExecutor
from server.ml.config import DATASET_DICT, AugmentationStrategy
from server.ml.logger_utils import logger
from server.ml.params import (debug_fixed_params, cnn_fixed_params, pretrained_vit_fixed_params)
from server.ml.params import debug_param_grid, cnn_param_grid
from server.ml.params.pretrained_vit import best_config_as_grid_vit
from server.ml.params.scratch_vit import fixed_params_vit_scratch, param_grid_vit_from_scratch
from server.persistence import load_file_repository, load_minio_repository

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    # --- Repository Configuration ---
    minio_bucket_name = "clouds"
    local_repo_base_path = str(script_dir)

    # --- Load Artifact Repository ---
    repo_option = "local"  # "local", "minio", or "none"

    # --- Base Prefix/Directory for experiments ---
    experiment_base_prefix_for_repo = "experiments"

    # --- Set up results directory ---
    results_base_dir = script_dir / 'results'
    # results_base_dir = None

    # --- Configuration ---
    # Select Dataset:
    selected_dataset = "ccsn"  # 'GCD', 'GCDf', 'mGCD', 'mGCDf', 'swimcat', 'ccsn', 'eurosat'
    selected_dataset = selected_dataset.lower()

    # Select Model:
    model_type = "pvit"
    # 'cnn', 'pvit', 'swin', 'svit', 'hyvit', 'cnn_feat', 'stfeat', 'xcloud', 'mcloud', 'resnet', 'shufflenet', 'cloudnet'

    # Weights for class imbalance
    use_weighted_loss_for_run = False


    # Offline Augmentation:
    offline_augmentation = False

    # Chosen sequence index: (1-7)
    # 1: Single Train and Eval
    # 2: Non-Nested Grid Search + Eval best model
    # 3: Nested Grid Search (Requires FLAT or FIXED with force_flat=True)
    # 4: Simple CV Evaluation (Requires FLAT or FIXED with force_flat=True)
    # 5: Non-Nested Grid Search + CV Evaluation (Requires FLAT or FIXED with force_flat=True)
    # 6: Load Pre-trained and Evaluate
    # 7: Load Pre-trained and Predict on New Images
    chosen_sequence_idx = 4

    # Image size for the model
    img_size = (224, 224)
    # img_size = (64, 64)
    # img_size = (448, 448)

    # Flag for CV methods on FIXED datasets:
    force_flat = False

    save_model = False  # Whether to save the model after training

    data_augmentation_mode_override = None
    # data_augmentation_mode_override = AugmentationStrategy.CCSN_RESNET

    # Flag for overriding parameters:
    enable_debug_params = False

    # Save run log file:
    save_run_log_file = True  # Whether to save the main log file

    # --- MODEL TO LOAD ---
    saved_model_dataset = "experiments\\GCD-flat"
    saved_model_type_folder = "pvit"
    saved_model_experiment_run_id = "20250612_234618_seed42"
    saved_model_relative_path = "single_train_234618\\pvit_sngl_ep32_val_lossp27_234618.pt" # Actual .pt filename

    # Construct the path
    model_path = str(
        PurePath(saved_model_dataset) / saved_model_type_folder / saved_model_experiment_run_id / saved_model_relative_path
    )

    username: str = 'eugen2'

    images_to_predict_info: List[any] = [
        # ImagePredictionTask(image_id='TG Mures_25-05-2025_square', image_format='jpg', prediction_id='p1'),
        # ImagePredictionTask(image_id='Brasov_18-05-2025_square', image_format='png', prediction_id='p2'),
        # ImagePredictionTask(image_id='Brasov_19-05-2025_square', image_format='png', prediction_id='p3'),
        # ImagePredictionTask(image_id='Brasov_21-05-2025_square', image_format='jpg', prediction_id='p4'),
        # ImagePredictionTask(image_id='Tarnaveni_05-05-2025', image_format='jpg', prediction_id='p5'),
        # ImagePredictionTask(image_id='Tarnaveni_09-05-2025_square1', image_format='jpg', prediction_id='p6'),
        # ImagePredictionTask(image_id='Tarnaveni_09-05-2025_square2', image_format='jpg', prediction_id='p7'),
        ImagePredictionTask(image_id='tarnaveni 11-08', image_format='png', prediction_id='p8'),
        ImagePredictionTask(image_id='tarnaveni 17-08', image_format='png', prediction_id='p9'),
        ImagePredictionTask(image_id='tarnaveni 21-08-2025', image_format='png', prediction_id='p10'),
    ]

    # --- Check if the dataset path exists ---
    dataset_path = script_dir / 'data' / DATASET_DICT[selected_dataset]
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

    # --- Define Params based on Model Type ---
    if enable_debug_params:
        chosen_fixed_params = debug_fixed_params
        chosen_param_grid = debug_param_grid

    elif model_type == ModelType.CNN:
        chosen_fixed_params = cnn_fixed_params
        chosen_param_grid = cnn_param_grid

    elif model_type == ModelType.PRETRAINED_VIT:
        chosen_fixed_params = pretrained_vit_fixed_params
        chosen_param_grid = best_config_as_grid_vit

    elif model_type == ModelType.HYBRID_VIT:
        chosen_fixed_params = hybrid_vit_fixed_params
        chosen_param_grid = hybrid_vit_param_grid

    elif model_type == ModelType.CNN_FEAT:
        chosen_fixed_params = paper_cnn_standalone_fixed_params
        chosen_param_grid = None

    elif model_type == ModelType.STANDARD_FEAT:
        chosen_fixed_params = standard_cnn_fixed_params
        chosen_param_grid = None

    elif model_type == ModelType.SCRATCH_VIT:
        chosen_fixed_params = fixed_params_vit_scratch
        chosen_param_grid = param_grid_vit_from_scratch

    elif model_type == ModelType.XCLOUD:
        chosen_fixed_params = xcloud_fixed_params
        chosen_param_grid = None

    elif model_type == ModelType.MCLOUD:
        chosen_fixed_params = mcloud_fixed_params
        chosen_param_grid = None

    elif model_type == ModelType.RESNET18_CLOUD:
        # chosen_fixed_params = resnet18_cloud_fixed_params
        chosen_fixed_params = resnet18_finetune_best_practice_params
        chosen_param_grid = None

    elif model_type == ModelType.PRETRAINED_SWIN:
        chosen_fixed_params = pretrained_swin_fixed_params
        chosen_param_grid = None

    elif model_type == ModelType.SHUFFLE_CLOUD:
        chosen_fixed_params = shufflenet_cloud_fixed_params
        chosen_param_grid = None

    elif model_type == ModelType.CLOUD_NET:
        chosen_fixed_params = cloudnet_fixed_params
        chosen_param_grid = None

    else:
        logger.error(f"Model type '{model_type}' not recognized. Supported: {[m.value for m in ModelType]}")
        exit()

    if selected_dataset == 'ccsn':
        effective_test_split_ratio_if_flat = 0.1
        effective_val_split_ratio = 0.1 / (1.0 - effective_test_split_ratio_if_flat)
        cv_folds = 10
        augmentation_strategy = AugmentationStrategy.CCSN_MODERATE
    elif selected_dataset in ['gcd', 'gcdf', 'mgcd', 'mgcdf']:
        if selected_dataset == 'gcdf':
            effective_test_split_ratio_if_flat = 0.2
        else:
            effective_test_split_ratio_if_flat = 9000 / 19000
        effective_val_split_ratio = 0.1
        cv_folds = 5
        augmentation_strategy = AugmentationStrategy.SKY_ONLY_ROTATION
        # augmentation_strategy = AugmentationStrategy.CCSN_MODERATE
    elif selected_dataset == 'swimcat':
        effective_test_split_ratio_if_flat = 0.2
        effective_val_split_ratio = 0.1 / (1.0 - effective_test_split_ratio_if_flat)
        cv_folds = 5
        augmentation_strategy = AugmentationStrategy.SWIMCAT_MILD
    else:
        effective_test_split_ratio_if_flat = 0.2
        effective_val_split_ratio = 0.1
        cv_folds = 5
        augmentation_strategy = AugmentationStrategy.DEFAULT_STANDARD

    # --- Define Augmentation Strategy ---
    if data_augmentation_mode_override is not None:
        augmentation_strategy = data_augmentation_mode_override

    # Example 1: Single Train and Eval
    methods_seq_1 = [
        ('single_train', {
            'params': chosen_fixed_params,
            'save_model': save_model,
            'val_split_ratio': effective_val_split_ratio,
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
            'cv': cv_folds,
            'method': 'grid',
            # 'method': 'random',
            # 'n_iter': 4,
            'val_split_ratio': effective_val_split_ratio,
            'scoring': 'accuracy',
            'save_best_model': save_model,
            'results_detail_level': 2,
        }),
        ('single_eval', {}),
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
             'params': chosen_fixed_params,
             'val_split_ratio': effective_val_split_ratio,
             'cv': cv_folds,
             'evaluate_on': 'full',
             'results_detail_level': 2,
        })
    ]

    # Example 5: Non-Nested Grid Search + CV Evaluation (Requires FLAT or FIXED with force_flat=True)
    methods_seq_5 = [
        ('non_nested_grid_search', {
            'param_grid': chosen_param_grid,
            'cv': 2,
            'method': 'grid',
            'scoring': 'accuracy',
            'save_best_model': save_model
        }),
         ('cv_model_evaluation', {
             'cv': 2,
             'use_best_params_from_step': 0,
             # 'params': {'max_epochs': 15},
             'evaluate_on': 'test',
             'results_detail_level': 2,
        })
    ]

    # Example 6: Load Pre-trained and Evaluate
    methods_seq_6 = [
        ('load_model', {'model_path_or_key': model_path}),
        ('single_eval', {
            'plot_level': 2
        }),
    ]

    # Example 7: Load Pre-trained and Predict on New Images
    methods_seq_7 = [
        ('load_model', {
            'model_path_or_key': model_path
        }),
        ('predict_images', {
            'username': username,
            'image_tasks_for_pipeline': images_to_predict_info,
            'experiment_run_id_of_model': saved_model_experiment_run_id,
            'generate_lime_explanations': True,
            'lime_num_features_to_show_plot': 20,
            'lime_num_samples_for_explainer': 200,
            'results_detail_level': 3,
            'plot_level': 2
        })
    ]

    # --- Select the chosen sequence based on index (1-7) ---
    chosen_sequence = globals()[f"methods_seq_{chosen_sequence_idx}"]

    logger.info(f"Executing sequence: {[m[0] for m in chosen_sequence]}")

    # --- Create and Run Executor ---
    try:
        executor = PipelineExecutor(
            dataset_path=dataset_path,
            model_type=model_type,
            artifact_repository=repo,
            save_main_log_file=save_run_log_file,
            experiment_base_key_prefix=experiment_base_prefix_for_repo,
            methods=chosen_sequence,
            force_flat_for_fixed_cv=force_flat,
            augmentation_strategy=augmentation_strategy,
            show_first_batch_augmentation_default=True,
            use_offline_augmented_data=offline_augmentation,
            use_weighted_loss=use_weighted_loss_for_run,

            # Pipeline default parameters
            img_size=img_size,
            batch_size=16,
            max_epochs=10,
            patience=10,
            lr=0.001,
            optimizer__weight_decay=0.01,
            test_split_ratio_if_flat=effective_test_split_ratio_if_flat,
            # module__dropout_rate=0.5
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
