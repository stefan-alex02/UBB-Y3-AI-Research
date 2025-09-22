import os
from pathlib import Path
import shutil
import torch

# Import necessary components from our modules
from pipeline import ClassificationPipeline
from utils import logger  # Use the configured logger

# --- Configuration ---
SEED = 42
IMAGE_SIZE = (128, 128)  # Smaller size for faster demo
OUTPUT_DIR = "results_pipeline"
DATASET_NAME = "mini-GCD"  # We'll create a dummy dataset

# Choose Model: 'dummycnn', 'vit', 'diffusion_placeholder'
MODEL_NAME = 'dummycnn'

# --- Create Dummy Dataset ---
# You should replace this with the actual path to your dataset
DUMMY_DATASET_PATH = Path(DATASET_NAME)


def create_dummy_dataset(base_path: Path, structure: str = "FLAT", num_classes: int = 2, img_per_class: int = 20):
    """Creates a dummy image dataset for testing."""
    if base_path.exists():
        logger.warning(f"⚠️ Dummy dataset path '{base_path}' already exists. Removing and recreating.")
        shutil.rmtree(base_path)

    logger.info(f"🛠️ Creating dummy dataset '{base_path.name}' with structure: {structure}")

    img_size = (IMAGE_SIZE[1], IMAGE_SIZE[0])  # PIL uses (width, height)

    if structure == "FLAT":
        base_path.mkdir(parents=True)
        for i in range(num_classes):
            class_path = base_path / f"class_{i}"
            class_path.mkdir()
            for j in range(img_per_class):
                img = torch.randint(0, 256, (IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=torch.uint8)
                img_pil = transforms.ToPILImage()(img.permute(2, 0, 1))  # Convert to PIL
                img_pil.save(class_path / f"img_{j}.png")
    elif structure == "FIXED":
        train_path = base_path / "train"
        test_path = base_path / "test"
        train_path.mkdir(parents=True)
        test_path.mkdir(parents=True)

        for phase_path, num_img in [(train_path, img_per_class // 2), (test_path, img_per_class // 2)]:
            if num_img == 0: continue  # Skip if 0 images for phase
            for i in range(num_classes):
                class_path = phase_path / f"class_{i}"
                class_path.mkdir()
                for j in range(num_img):
                    img = torch.randint(0, 256, (IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=torch.uint8)
                    img_pil = transforms.ToPILImage()(img.permute(2, 0, 1))
                    img_pil.save(class_path / f"img_{j}.png")
    else:
        raise ValueError(f"Unsupported dummy structure: {structure}")
    logger.info(f"✅ Dummy dataset created at '{base_path}'")


# --- Choose Dataset Structure and Create ---
# DATASET_STRUCTURE = "FLAT"
DATASET_STRUCTURE = "FIXED"
from torchvision import transforms  # Needed for dummy dataset creation

create_dummy_dataset(DUMMY_DATASET_PATH, structure=DATASET_STRUCTURE, num_classes=3,
                     img_per_class=50)  # Increase samples for meaningful CV/split

# --- Pipeline Definition ---
if __name__ == "__main__":

    # --- Base Skorch Model Parameters ---
    # These are defaults, can be overridden by search methods
    base_model_params = {
        'lr': 0.001,
        'batch_size': 16,
        'max_epochs': 10,  # Low epochs for demo speed
        'patience': 3,  # Early stopping patience
        'pretrained': False,  # Use pretrained weights for ViT? (Only applies if model_path_load is not set)
    }

    # --- Instantiate the Pipeline ---
    pipeline = ClassificationPipeline(
        dataset_path=str(DUMMY_DATASET_PATH),
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR,
        image_size=IMAGE_SIZE,
        base_model_params=base_model_params,
        # model_path_load="path/to/your/saved_skorch_model.pt", # Optional: Load existing model
        model_path_save=os.path.join(OUTPUT_DIR, DUMMY_DATASET_PATH.name, MODEL_NAME, "final_pipeline_model.pt"),
        # Optional: Save final model
        seed=SEED,
        use_gpu=True  # Set to False if no GPU or to force CPU
    )

    # --- Add Methods to the Pipeline ---

    # Example 1: Single Train followed by Single Eval
    pipeline.add_method('single_train', config={'val_size': 0.25, 'save_results': True})
    pipeline.add_method('single_eval', config={'save_results': True})

    # # Example 2: Non-nested CV for Hyperparameter Search
    # param_grid_search = {
    #     'lr': [0.001, 0.0005],
    #     'module__fc1.out_features': [64, 128] # Example: Search over a layer size in DummyCNN
    #     # Add other hyperparameters relevant to the model_adapter (prefix with 'module__' for torch module params)
    # }
    # pipeline_v1.add_method('non_nested_cv', config={
    #     'search_type': 'GridSearchCV', # or 'RandomizedSearchCV'
    #     'param_grid': param_grid_search,
    #     'cv_folds': 3, # Inner folds for search
    #     'scoring': 'accuracy',
    #     'save_results': True
    # })
    # # Follow up with evaluation using the best model found (if needed)
    # # Note: non_nested_cv returns the best estimator, which updates the pipeline_v1's model adapter
    # pipeline_v1.add_method('single_eval', config={'save_results': True})

    # # Example 3: Nested CV for Performance Estimation
    # param_grid_nested = {
    #     'lr': [0.01, 0.001, 0.0005],
    #     # 'optimizer__weight_decay': [0, 0.01]
    # }
    # pipeline_v1.add_method('nested_cv', config={
    #      'search_type': 'RandomizedSearchCV', # Or GridSearchCV
    #      'param_grid': param_grid_nested,
    #      'outer_cv_folds': 4, # Folds for outer performance estimation
    #      'inner_cv_folds': 2, # Folds for inner hyperparameter tuning
    #      'scoring': 'accuracy', # Can use multiple scorers ['accuracy', 'f1_macro'] etc. check sklearn/skorch docs
    #      'n_iter': 4, # Number of random combinations to try
    #      'save_results': True
    # })

    # # Example 4: CV Evaluation (Only for FLAT dataset)
    # if DATASET_STRUCTURE == "FLAT":
    #     pipeline_v1.add_method('cv_evaluation', config={
    #         'cv_folds': 5,
    #         'save_results': True
    #     })
    # else:
    #      logger.warning("⚠️ Skipping 'cv_evaluation' method example as dataset structure is FIXED.")

    # --- Run the Pipeline ---
    try:
        results = pipeline.run()
        logger.info("\n--- Pipeline Completed ---")
        # print("Pipeline Results Summary:")
        # import json
        # print(json.dumps(results, indent=2))
    except Exception as e:
        logger.critical(f"💥 Pipeline execution failed critically: {e}", exc_info=True)

    # --- Cleanup Dummy Dataset (Optional) ---
    # logger.info(f"🧹 Cleaning up dummy dataset '{DUMMY_DATASET_PATH}'...")
    # shutil.rmtree(DUMMY_DATASET_PATH)
    # logger.info("✅ Cleanup complete.")
