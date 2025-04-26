import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional, Any, Dict
import os
import logging

# Scikit-learn and Skorch
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau # Example scheduler

# TIMM for Vision Transformer
try:
    import timm
except ImportError:
    print("Please install timm: pip install timm")
    timm = None

# --- Logging Setup (Optional but Recommended) ---
LOG_EMOJIS = {'info': 'ℹ️', 'warning': '⚠️', 'error': '❌', 'success': '✅'}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Custom Dataset Implementation (Provided by User) ---
class CustomImageDataset(Dataset):
    """A custom dataset to load images from paths."""
    def __init__(self, image_paths: List[str], labels: List[int], transform: Optional[Any] = None):
        self.image_paths = image_paths
        self.labels = np.array(labels) # Use numpy array for easier slicing/indexing if needed
        self.transform = transform
        if len(image_paths) != len(labels):
             raise ValueError("image_paths and labels must have the same length.")
        logger.info(f"{LOG_EMOJIS['info']} Dataset initialized with {len(image_paths)} samples.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        # Use self.labels directly as it's now a numpy array or list
        label = self.labels[idx].item() if isinstance(self.labels[idx], np.integer) else self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"{LOG_EMOJIS['error']} Error loading image {img_path}: {e}")
            # Return a dummy tensor and label, or raise error depending on desired handling
            # Raising error is safer during development
            raise IOError(f"Could not load image: {img_path}") from e

        if self.transform:
            image = self.transform(image)

        # Ensure label is compatible type for loss function (e.g., int or long tensor)
        # Skorch generally expects primitive types or numpy types for y
        return image, label # Return label as int/long

# --- 2. Model Definitions ---

# --- Model Type 1: Simple CNN ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5, num_filters1=32, num_filters2=64, input_size=32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, num_filters1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters1)
        self.pool1 = nn.MaxPool2d(2) # input_size / 2
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters2)
        self.pool2 = nn.MaxPool2d(2) # input_size / 4

        # Calculate flattened size dynamically
        # Use a dummy forward pass to get the size
        with torch.no_grad():
             dummy_input = torch.zeros(1, 3, input_size, input_size)
             dummy_output = self.pool2(self.bn2(self.conv2(self.pool1(self.bn1(self.conv1(dummy_input))))))
             self.fc1_input_dim = dummy_output.numel()


        self.fc1 = nn.Linear(self.fc1_input_dim, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # No softmax here, CrossEntropyLoss expects logits
        return x

# --- Model Type 2: Vision Transformer (using TIMM) ---
class VisionTransformerModel(nn.Module):
    def __init__(self, num_classes=10, model_name='vit_tiny_patch16_224', pretrained=True, dropout_rate=0.1):
        super().__init__()
        if timm is None:
            raise ImportError("timm library is required for VisionTransformerModel.")
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0) # Load without final classifier
        # Get the feature dimension of the ViT model
        vit_feature_dim = self.vit.head.in_features if hasattr(self.vit, 'head') else self.vit.num_features
        # Add custom head
        self.head = nn.Sequential(
            nn.LayerNorm(vit_feature_dim), # Often beneficial before final layer
            nn.Dropout(dropout_rate),
            nn.Linear(vit_feature_dim, num_classes)
        )

    def forward(self, x):
        features = self.vit(x)
        logits = self.head(features)
        return logits

# --- Model Type 3: Diffusion Classifier (Placeholder) ---
# NOTE: This is a *highly* simplified placeholder. A real diffusion classifier
# might involve conditioning on timesteps, extracting features from a pre-trained
# diffusion model, or using a U-Net style architecture differently.
# Replace this with your actual diffusion-based classifier architecture.
class DiffusionClassifierPlaceholder(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=256, input_size=32, dropout_rate=0.2):
        super().__init__()
        # Example: A simple MLP acting on flattened image data
        # A real implementation would be vastly different based on the diffusion approach
        self.input_dim = 3 * input_size * input_size
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        logger.warning(f"{LOG_EMOJIS['warning']} Using a placeholder for DiffusionClassifier. Replace with actual implementation.")

    def forward(self, x):
        return self.model(x)

# --- 3. Data Preparation ---

# --- Create Dummy Data (Replace with your actual data loading) ---
NUM_SAMPLES = 200
NUM_CLASSES = 5 # Make sure this matches model outputs
IMG_SIZE_CNN_DIFF = 32 # Input size for CNN and Diffusion Placeholder
IMG_SIZE_VIT = 224   # Input size typically required by ViT

# Create dummy image files (optional, only needed if loading from files)
DUMMY_DIR = "dummy_images"
if not os.path.exists(DUMMY_DIR):
    os.makedirs(DUMMY_DIR)
    logger.info(f"{LOG_EMOJIS['info']} Creating dummy image files in {DUMMY_DIR}...")
    for i in range(NUM_SAMPLES):
        img = Image.new('RGB', (IMG_SIZE_VIT, IMG_SIZE_VIT), color = (i % 255, (i*5)%255 , (i*10)%255))
        img.save(os.path.join(DUMMY_DIR, f"img_{i}.png"))

dummy_paths = [os.path.join(DUMMY_DIR, f"img_{i}.png") for i in range(NUM_SAMPLES)]
dummy_labels = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES).tolist()

# --- Define Transforms ---
# Note: Different models might require different input sizes and normalization!
# We define separate transforms here.

transform_cnn_diff = transforms.Compose([
    transforms.Resize((IMG_SIZE_CNN_DIFF, IMG_SIZE_CNN_DIFF)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Example: ImageNet stats
])

transform_vit = transforms.Compose([
    transforms.Resize((IMG_SIZE_VIT, IMG_SIZE_VIT)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Example: ImageNet stats
])

# --- Create Datasets ---
# It's often best practice to create the dataset once and then potentially
# create subsets or different views if needed. However, if transforms differ
# significantly (like size), creating separate dataset instances might be easier.

# We'll create the dataset *inside* the loop for simplicity here,
# ensuring the correct transform is used for each model type.
# Alternatively, load data once and apply transforms dynamically if possible.

# --- 4. Skorch Wrappers and Hyperparameter Search Setup ---

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"{LOG_EMOJIS['info']} Using device: {device}")

# --- Common Skorch Settings ---
# Callbacks can be shared or customized per model
lr_scheduler_callback = LRScheduler(policy=ReduceLROnPlateau, monitor='valid_loss', patience=5, factor=0.1)
early_stopping_callback = EarlyStopping(monitor='valid_loss', patience=10, threshold=0.001)

common_skorch_params = {
    'criterion': nn.CrossEntropyLoss,
    'optimizer': optim.AdamW, # AdamW often works well
    'device': device,
    'batch_size': 32, # Can be tuned
    'max_epochs': 2, # Low for demo; increase for real training (e.g., 50, 100)
    'train_split': skorch.dataset.ValidSplit(cv=0.2, stratified=True), # Use 20% of training data for validation
    'callbacks': [
        # ('lr_scheduler', lr_scheduler_callback), # Uncomment if desired
        ('early_stopping', early_stopping_callback)
    ],
    'iterator_train__shuffle': True, # Shuffle training data each epoch
    'verbose': 1, # 0 = silent, 1 = progress bar, 2 = one line per epoch
}

# --- Model-Specific Configurations ---
models_to_tune = {}

# Configuration for SimpleCNN
models_to_tune['SimpleCNN'] = {
    'model_class': SimpleCNN,
    'transform': transform_cnn_diff,
    'skorch_params': { # Specific params to override/add to common_skorch_params
        'module__num_classes': NUM_CLASSES,
        'module__input_size': IMG_SIZE_CNN_DIFF,
    },
    'param_dist': { # Parameter distribution for RandomizedSearchCV
        'lr': [1e-4, 5e-4, 1e-3, 5e-3], # Learning rate
        'optimizer__weight_decay': [0, 1e-5, 1e-4],
        'module__dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'module__num_filters1': [16, 32],
        'module__num_filters2': [32, 64],
        'batch_size': [16, 32, 64],
        'max_epochs': [10, 20, 30] # Tune epochs too
    }
}

# Configuration for VisionTransformer (if timm is available)
if timm:
    models_to_tune['VisionTransformer'] = {
        'model_class': VisionTransformerModel,
        'transform': transform_vit,
        'skorch_params': {
            'module__num_classes': NUM_CLASSES,
            # ViT models often benefit from smaller batch sizes due to memory
            # 'batch_size': 16, # Example override
        },
        'param_dist': {
            'lr': [1e-5, 5e-5, 1e-4], # ViTs often need smaller LRs
            'optimizer__weight_decay': [0, 1e-4, 1e-3],
            'module__model_name': ['vit_tiny_patch16_224'], # Can try different ViT sizes
            'module__pretrained': [True], # Usually start with pretrained
            'module__dropout_rate': [0.1, 0.2],
            'batch_size': [8, 16, 32], # Adjust based on GPU memory
            'max_epochs': [10, 20, 30]
        }
    }

# Configuration for DiffusionClassifierPlaceholder
models_to_tune['DiffusionClassifierPlaceholder'] = {
    'model_class': DiffusionClassifierPlaceholder,
    'transform': transform_cnn_diff, # Assuming same input size as CNN
    'skorch_params': {
        'module__num_classes': NUM_CLASSES,
        'module__input_size': IMG_SIZE_CNN_DIFF,
    },
    'param_dist': {
        'lr': [1e-4, 1e-3, 1e-2],
        'optimizer__weight_decay': [0, 1e-5],
        'module__hidden_dim': [128, 256, 512],
        'module__dropout_rate': [0.1, 0.2, 0.3, 0.5],
        'batch_size': [32, 64, 128],
        'max_epochs': [10, 20, 30]
    }
}


# --- 5. Execute Hyperparameter Search for Each Model ---

N_ITER_SEARCH = 5 # Number of parameter settings that are sampled. Low for demo. Increase (e.g., 20, 50)
CV_FOLDS = 3      # Number of cross-validation folds.

results = {}

for model_name, config in models_to_tune.items():
    logger.info(f"\n{LOG_EMOJIS['info']} Starting Hyperparameter Search for: {model_name}")

    # --- Create dataset with the correct transform for this model ---
    logger.info(f"Using image size: {config['transform'].transforms[0].size}")
    full_dataset = CustomImageDataset(dummy_paths, dummy_labels, transform=config['transform'])

    # Convert dataset to X, y format suitable for scikit-learn/skorch
    # This loads all data into memory - for large data, consider skorch's Dataset handling directly
    # Or use indices with Subset in CV. For simplicity here, we load all.
    # Be careful with memory usage on large data!
    X_list = []
    y_list = []
    logger.info("Loading data into memory for X, y format...")
    try:
        for i in range(len(full_dataset)):
            img, label = full_dataset[i]
            X_list.append(img)
            y_list.append(label)
        X = torch.stack(X_list)
        y = np.array(y_list) # Skorch generally prefers numpy arrays for y
        logger.info(f"Data loaded: X shape={X.shape}, y shape={y.shape}")
    except IOError as e:
        logger.error(f"{LOG_EMOJIS['error']} Failed to load data for {model_name}: {e}. Skipping this model.")
        continue
    except Exception as e:
         logger.error(f"{LOG_EMOJIS['error']} An unexpected error occurred during data loading for {model_name}: {e}. Skipping this model.")
         continue


    # --- Instantiate Skorch Wrapper ---
    # Combine common params with model-specific params
    current_skorch_params = {**common_skorch_params, **config['skorch_params']}
    net = NeuralNetClassifier(
        module=config['model_class'],
        **current_skorch_params
    )

    # --- Setup Randomized Search ---
    # Use Stratified K-Folds for classification tasks to preserve class proportions
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    # Define scoring (use accuracy for classification)
    scorer = make_scorer(accuracy_score)

    random_search = RandomizedSearchCV(
        net,
        config['param_dist'],
        n_iter=N_ITER_SEARCH,
        cv=cv,
        scoring=scorer,
        refit=True, # Refit the best model on the whole data subset used for search
        verbose=2,
        random_state=42, # for reproducibility of search sampling
        n_jobs=1 # IMPORTANT: Set n_jobs=1 if using GPU to avoid memory conflicts, unless you have multiple GPUs managed carefully.
    )

    # --- Run the search ---
    logger.info(f"Running RandomizedSearchCV for {model_name}...")
    try:
        # Fit requires X, y format
        random_search.fit(X, y) # Pass Tensors for X, numpy array for y

        # --- Store results ---
        results[model_name] = {
            'best_score': random_search.best_score_,
            'best_params': random_search.best_params_,
            'best_estimator': random_search.best_estimator_,
            'cv_results': random_search.cv_results_,
        }
        logger.info(f"{LOG_EMOJIS['success']} Search complete for {model_name}.")
        logger.info(f"Best score: {random_search.best_score_:.4f}")
        logger.info(f"Best parameters: {random_search.best_params_}")

    except Exception as e:
        logger.error(f"{LOG_EMOJIS['error']} RandomizedSearchCV failed for {model_name}: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback

# --- 6. Analyze Results ---
logger.info("\n--- Final Results ---")
for model_name, result_data in results.items():
    print(f"\nModel: {model_name}")
    print(f"  Best Accuracy (CV): {result_data['best_score']:.4f}")
    print(f"  Best Hyperparameters:")
    for param, value in result_data['best_params'].items():
        print(f"    {param}: {value}")

# You can now access the best refitted model for each type, e.g.:
# best_cnn_model = results['SimpleCNN']['best_estimator']
# best_vit_model = results['VisionTransformer']['best_estimator'] # if it ran

# Example prediction with the best CNN model (if it exists)
if 'SimpleCNN' in results:
    best_cnn = results['SimpleCNN']['best_estimator']
    # Make sure to apply the *same* transform used during training
    # Example: Load one image, transform it, and predict
    # single_image_path = dummy_paths[0]
    # single_image_pil = Image.open(single_image_path).convert('RGB')
    # single_image_tensor = transform_cnn_diff(single_image_pil).unsqueeze(0) # Add batch dimension
    # prediction = best_cnn.predict(single_image_tensor.to(device))
    # print(f"Prediction for first image using best CNN: {prediction}")

# Clean up dummy files (optional)
# import shutil
# if os.path.exists(DUMMY_DIR):
#     logger.info(f"Removing dummy image directory: {DUMMY_DIR}")
#     shutil.rmtree(DUMMY_DIR)