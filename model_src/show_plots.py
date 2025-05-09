# Example in a separate script or notebook
from pathlib import Path
from lib.plotter import ResultsPlotter
# Assuming config and dataset_utils are needed for setup or class names
# from model_src.config import ModelType
from lib.dataset_utils import ImageDatasetHandler
from lib.logger_utils import logger # Use your logger if needed

# --- Configuration ---
# IMPORTANT: Replace these paths with your actual paths
# json_file_path_str = "./results/Swimcat-extend/vit/20250507_054532_seed42/cv_model_evaluation_test_064434_189526/cv_model_evaluation_results_20250507_064434.json"
json_file_path_str = "./results/Swimcat-extend/vit/20250507_054532_seed42/non_nested_grid_054532_610581/non_nested_grid_search_results_20250507_064104.json"

# dataset_root_for_classes_str = "data/mini-GCD" # Path to the dataset root to get classes
dataset_root_for_classes_str = "data/Swimcat-extend" # Path to the dataset root to get classes

json_file_path = Path(json_file_path_str)
dataset_root_for_classes = Path(dataset_root_for_classes_str)

# --- Load class names (if needed for the plot type) ---
class_names = None
# Check if the plot method requires class names (eval methods usually do)
needs_classes = False
if "single_eval" in json_file_path.stem or "cv_model_evaluation" in json_file_path.stem:
    needs_classes = True

if needs_classes:
    try:
        # Make sure the dataset path exists before trying to load from it
        if not dataset_root_for_classes.is_dir():
             raise FileNotFoundError(f"Dataset root path for classes not found: {dataset_root_for_classes}")
        # Instantiate handler minimally just to get classes
        handler = ImageDatasetHandler(root_path=dataset_root_for_classes)
        class_names = handler.classes
        if not class_names:
            raise ValueError("Could not load class names from dataset handler.")
        logger.info(f"Loaded class names: {class_names}")
    except Exception as e:
        logger.error(f"Failed to load class names from {dataset_root_for_classes}: {e}")
        logger.error("Plotting methods requiring class names might fail.")
        # Decide if you want to exit or continue without class names
        # exit() # Or just let it potentially fail later

# --- Call the plotter method ---
try:
    # Extract the method name from the JSON filename stem
    # Assumes filenames like "method_name_results.json"
    stem = json_file_path.stem
    method_name_in_file = stem.replace('_results', '')

    logger.info(f"Attempting to plot results for method '{method_name_in_file}' from: {json_file_path}")

    if method_name_in_file == "single_eval":
        if class_names:
            ResultsPlotter.plot_single_eval_results(json_file_path, class_names=class_names, show_plots=True)
        else:
            logger.error("Cannot plot single_eval results: Class names are required but could not be loaded.")
    elif method_name_in_file.startswith("single_train"): # Handle potential random suffix
        ResultsPlotter.plot_single_train_results(json_file_path, show_plots=True)
    elif method_name_in_file.startswith("non_nested"): # Handle potential grid/random suffix
        ResultsPlotter.plot_non_nested_cv_results(json_file_path, show_plots=True)
    elif method_name_in_file.startswith("nested"): # Handle potential grid/random suffix
        ResultsPlotter.plot_nested_cv_results(json_file_path, show_plots=True)
    elif method_name_in_file.startswith("cv_model_evaluation"): # Handle potential random suffix
        if class_names:
             ResultsPlotter.plot_cv_model_evaluation_results(json_file_path, class_names=class_names, show_plots=True)
        else:
            logger.error("Cannot plot cv_model_evaluation results: Class names are required but could not be loaded.")
    else:
        logger.warning(f"No specific plotting method found for results type inferred from filename: '{method_name_in_file}'. File: {json_file_path}")

except FileNotFoundError:
    logger.error(f"Error: Results file not found at {json_file_path}")
except ImportError as ie:
     logger.error(f"Plotting failed due to missing libraries: {ie}. Please install required plotting libraries.")
except Exception as e:
    logger.error(f"An unexpected error occurred during plotting: {e}", exc_info=True)
