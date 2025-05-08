# --- START OF FILE main.py ---
from pathlib import Path

import numpy as np

from lib import PipelineExecutor
from lib.logger_utils import logger

# --- Example Usage ---
if __name__ == "__main__":
    script_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    results_base_dir = script_dir / 'results'

    # --- Configuration ---
    # Select Dataset:
    dataset_path = script_dir / "data/mini-GCD-flat" # FLAT example
    # dataset_path = script_dir / "data/Swimcat-extend" # FIXED example
    # dataset_path = Path("PATH_TO_YOUR_DATASET") # Use your actual path

    # if not Path(dataset_path).exists():
    #      logger.error(f"Dataset path not found: {dataset_path}")
    #      logger.error("Please create the dataset or modify the 'dataset_path' variable.")
    #      exit()

    # Select Model:
    model_type = "cnn"  # 'cnn', 'vit', 'diffusion'

    # Image size for the model
    img_size = (224, 224)  # Common size for CNNs and ViTs

    # Flag for CV methods on FIXED datasets:
    # Set to True to allow nested_grid_search and cv_model_evaluation on FIXED datasets
    # by treating train+test as one pool (USE WITH CAUTION - not standard evaluation).
    force_flat = False

    param_grid_cnn = {
        # Skorch parameters
        'lr': [0.005, 0.001, 0.0005],
        'batch_size': [16, 32],  # Note: Changing batch size can affect memory and convergence

        # Optimizer (AdamW) parameters
        'optimizer__weight_decay': [0.01, 0.001, 0.0001],
        # 'optimizer__betas': [(0.9, 0.999), (0.85, 0.99)], # Less common to tune

        # Module (SimpleCNN) parameters
        'module__dropout_rate': [0.3, 0.4, 0.5, 0.6],  # Tune dropout in the classifier head

        # Maybe max_epochs if not using EarlyStopping effectively? Usually fixed or high w/ early stopping.
        # 'max_epochs': [15, 25],
    }

    param_grid_vit = {
        # Skorch parameters (especially LR for fine-tuning)
        'lr': [0.001, 0.0005, 0.0001, 0.00005],  # Often lower LRs for fine-tuning
        'batch_size': [16, 32],  # Memory constraints often tighter with ViT

        # Optimizer (AdamW) parameters
        'optimizer__weight_decay': [0.01, 0.001, 0.0],  # Weight decay is important

        # Module (SimpleViT) parameters
        # Since we only replaced the head and froze most layers, there are fewer
        # *direct* module hyperparameters to tune via __init__.
        # If you added dropout to the new head, you could tune 'module__dropout_rate'.
        # You *could* potentially tune which layers are frozen, but that's complex via grid search.

        # Training duration / EarlyStopping focus
        'max_epochs': [5, 10, 15], # If fine-tuning quickly
    }

    param_grid_diffusion = {
        # Skorch parameters
        'lr': [0.001, 0.0005, 0.0001],  # Fine-tuning learning rate
        'batch_size': [16, 32, 64],  # ResNet might be less memory-intensive than ViT

        # Optimizer (AdamW) parameters
        'optimizer__weight_decay': [0.01, 0.001, 0.0001],

        # Module (DiffusionClassifier) parameters
        'module__dropout_rate': [0.3, 0.4, 0.5, 0.6],  # Tune dropout in the custom head

        # Training duration
        # 'max_epochs': [10, 20, 30],
    }

    # --- Define Hyperparameter Grid / Fixed Params based on Model Type ---
    if model_type == 'cnn':
        chosen_param_grid = param_grid_cnn

    elif model_type == 'vit':
        chosen_param_grid = param_grid_vit

    elif model_type == 'diffusion':
        chosen_param_grid = param_grid_diffusion
    else:
        logger.error(f"Model type '{model_type}' not recognized. Supported: 'cnn', 'vit', 'diffusion'.")
        exit()

    # Temporarily set param grid
    chosen_param_grid = {
        # Skorch parameters
        'lr': [0.001, 0.0005],  # Fine-tuning learning rate

        'max_epochs': [5], # If fine-tuning quickly

        # Module (SimpleCNN) parameters
        # 'module__dropout_rate': [0.3, 0.6],  # Tune dropout in the classifier head
    }

    # chosen_param_grid = {
    #     # Skorch parameters (especially LR for fine-tuning)
    #     'lr': [0.001, 0.0005, 0.0001],  # Often lower LRs for fine-tuning
    #     'batch_size': [16],  # Memory constraints often tighter with ViT
    #
    #     # Optimizer (AdamW) parameters
    #     'optimizer__weight_decay': [0.01, 0.001],  # Weight decay is important
    #
    #     # Module (SimpleViT) parameters
    #     # Since we only replaced the head and froze most layers, there are fewer
    #     # *direct* module hyperparameters to tune via __init__.
    #     # If you added dropout to the new head, you could tune 'module__dropout_rate'.
    #     # You *could* potentially tune which layers are frozen, but that's complex via grid search.
    #
    #     # Training duration / EarlyStopping focus
    #     'max_epochs': [10, 15], # If fine-tuning quickly
    # }

    fixed_params_for_eval = {
        'lr': 0.001,
        'optimizer__weight_decay': 0.01,
        # 'module__dropout_rate': 0.4
    }


    # --- Define Method Sequence ---
    # Example 1: Single Train (using val split) and Eval
    methods_seq_1 = [
        ('single_train', {
            'max_epochs': 5,
            'save_model': True,
            'val_split_ratio': 0.2,
        }), # Explicit val split
        ('single_eval', {}),
    ]

    # Example 2: Non-Nested Grid Search + Eval best model
    methods_seq_2 = [
        ('non_nested_grid_search', {
            'param_grid': chosen_param_grid, 'cv': 3, 'method': 'grid',
            'scoring': 'accuracy', 'save_best_model': True
        }),
        # The best model is refit and stored in pipeline_v1.model_adapter after search
        ('single_eval', {}), # Evaluate the refit best model
    ]

    # Example 3: Nested Grid Search (Requires FLAT or FIXED with force_flat=True)
    methods_seq_3 = [
         ('nested_grid_search', {
             'param_grid': chosen_param_grid, 'outer_cv': 3, 'inner_cv': 2,
             'method': 'grid', 'scoring': 'accuracy'
         })
    ]

    # Example 4: Simple CV Evaluation (Requires FLAT or FIXED with force_flat=True)
    methods_seq_4 = [
         ('cv_model_evaluation', {
             'cv': 3,
             'params': fixed_params_for_eval, # Pass fixed hyperparams
             # 'params': { 'max_epochs': 5 }, # Pass fixed hyperparams
             'evaluate_on': 'full', # Explicitly state (or rely on default)
             # 'results_detail_level': 2,
        })
    ]

    # Example 5: Non-Nested Grid Search + CV Evaluation (Requires FLAT or FIXED with force_flat=True)
    methods_seq_5 = [
        ('non_nested_grid_search', {
            'param_grid': chosen_param_grid,
            'cv': 4,
            'method': 'grid',
            'scoring': 'accuracy',
            'save_best_model': True
        }),
         ('cv_model_evaluation', {
             'cv': 4,
             # Special key indicates using best_params from previous step (index 0)
             'use_best_params_from_step': 0,
             # Optionally provide specific params for cv_eval to override defaults if needed
             # 'params': {'max_epochs': 15}, # e.g., override max_epochs just for CV eval
             'evaluate_on': 'test'
        })
    ]

    # Example 6: Load Pre-trained and Evaluate
    pretrained_model_path = "results/SOME_DATASET_cnn_TIMESTAMP/cnn_epochX_val....pt" # Replace with actual path
    methods_seq_6 = [
        ('load_model', {'model_path': pretrained_model_path}),
        ('single_eval', {}),
    ]


    # --- Choose Sequence and Execute ---
    chosen_sequence = methods_seq_4 # <--- SELECT SEQUENCE TO RUN

    logger.info(f"Executing sequence: {[m[0] for m in chosen_sequence]}")

    # --- Create and Run Executor ---
    try:
        executor = PipelineExecutor(
            dataset_path=dataset_path,
            model_type=model_type,
            results_dir=results_base_dir,
            methods=chosen_sequence,
            force_flat_for_fixed_cv=force_flat, # Pass the flag
            # Pipeline default parameters (can be overridden by methods)
            img_size=img_size, # Smaller size for faster demo
            batch_size=16,     # Smaller batch size for demo
            max_epochs=10,     # Fewer epochs for demo
            patience=3,        # Reduced patience for demo
            lr=0.001,
            optimizer__weight_decay=0.01,
            test_split_ratio_if_flat=0.4, # For flat datasets
            # module__dropout_rate=0.5 # If applicable to model
            results_detail_level=3,
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
         logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
