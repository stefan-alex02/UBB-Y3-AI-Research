But won't the augmented data be used in the validation subset as well this way? I only wanted it to be used in the train subset, in order not to affect the validation results (or test results as well).


You are absolutely right to question that! My previous simplification for `ImageDatasetHandler` has a flaw if you want to *strictly* keep offline augmented data out of the validation sets created by `single_train` or the internal validation splits of Skorch during `GridSearchCV` (which operates on `_train_val_paths`).

If you add offline augmented data directly into `self._train_val_paths` *before* any train/validation split occurs on this pool, then yes, those augmented samples **will** have a chance to end up in the validation part of those splits. This is not ideal, as you want your validation set to reflect the original, unaugmented data distribution.

**Corrected Strategy to Keep Validation Sets Clean from Offline Augmented Data:**

The offline augmented data should only ever be part of a "training" partition, never a "validation" or "test" partition derived from the original data.

This requires a more careful handling of data splits, especially within `SkorchModelAdapter` or how `ImageDatasetHandler` provides data for training.

**Option 1: Modify `SkorchModelAdapter.get_split_datasets` (More Encapsulated)**

This is probably the cleanest way as it centralizes the logic within the adapter, which is responsible for preparing data for `fit`.

1.  **`ImageDatasetHandler` Role:**
    *   It loads the original dataset (`_train_val_paths_orig`, `_test_paths_orig`).
    *   It loads the offline augmented dataset (`_offline_aug_paths`, `_offline_aug_labels`) *separately* if `use_offline_augmented_data` is true. It does *not* mix them into `_train_val_paths` directly.
    *   It needs new getter methods like `get_offline_augmented_paths_labels()`.

2.  **`SkorchModelAdapter.get_split_datasets` Role:**
    *   It receives `X` (which are paths from `_train_val_paths_orig`) and `y`.
    *   It performs the train/validation split on these *original* `X` and `y` to get `train_indices_orig` and `valid_indices_orig`.
    *   The validation dataset (`ds_valid`) is created using `X[valid_indices_orig]` with the `valid_transform`.
    *   **For the training dataset (`ds_train`):**
        *   Start with `X_train_fold_orig = X[train_indices_orig]`.
        *   If `self.use_offline_augmented_data` (a new param for the adapter, passed from pipeline config) is true, fetch the offline augmented data from `ImageDatasetHandler`.
        *   **Crucially, filter/select only those augmented samples whose original counterparts would have belonged to `X_train_fold_orig`.** This is the tricky part if your augmented dataset was created from the *entire* original train+val pool. A simpler approach if you can guarantee your offline augmented data was *only* generated from the true original *training* portion (e.g. the 70% part of CCSN, not the 20% val or 10% test).
        *   For now, let's assume a simpler case: the offline augmented dataset was generated *only from what would constitute the initial, largest possible training pool before any splits*.
        *   Combine `X_train_fold_orig` with the relevant (or all, if generated only from training data) offline augmented samples.
        *   Create `ds_train` using this combined set and `train_transform`.

**Revised `ImageDatasetHandler`:**

```python
# server/ml/dataset_utils.py
class ImageDatasetHandler:
    def __init__(self,
                 root_path: Union[str, Path],
                 # ...
                 use_offline_augmented_data: bool = False,
                 # ...
                 ):
        # ...
        self.use_offline_augmented_data = use_offline_augmented_data
        
        # Initialize new attributes for separate storage
        self._train_val_paths_orig: List[Path] = []
        self._train_val_labels_orig: List[int] = []
        self._offline_aug_paths: List[Path] = []
        self._offline_aug_labels: List[int] = []
        # _test_paths, _test_labels remain for original test data
        # _all_paths, _all_labels remain for original full data

        self._load_paths_and_labels() # Modified logic below
        # ...

    def _load_paths_and_labels(self) -> None:
        # 1. Load ORIGINAL dataset structure (FLAT or FIXED)
        if self.structure == DatasetStructure.FLAT:
            all_paths_orig_ds, all_labels_orig_ds_list, classes, class_to_idx = self._scan_dir_for_paths_labels(self.root_path)
            # ... (split all_paths_orig_ds into self._train_val_paths_orig, self._train_val_labels_orig
            #      and self._test_paths, self._test_labels as before) ...
            # ... self.classes, self.class_to_idx are set ...
            # self._all_paths and self._all_labels are set from all_paths_orig_ds
            # For FLAT:
            if self.test_split_ratio_if_flat > 0 and len(all_paths_orig_ds) >=2:
                # ... split logic ...
                self._train_val_paths_orig = # paths from train_val split of all_paths_orig_ds
                self._train_val_labels_orig = # labels for these
                self._test_paths = # paths from test split
                self._test_labels = # labels for these
            else:
                self._train_val_paths_orig = all_paths_orig_ds[:]
                self._train_val_labels_orig = all_labels_orig_ds_list[:]
                self._test_paths = []
                self._test_labels = []
            self._all_paths = all_paths_orig_ds # Full original dataset
            self._all_labels = all_labels_orig_ds_list

        elif self.structure == DatasetStructure.FIXED:
            train_path = self.root_path / 'train'
            test_path = self.root_path / 'test'
            # ... (load train_paths_orig, train_labels_orig from train_path)
            # ... (load self._test_paths, self._test_labels from test_path)
            # ... (set self.classes, self.class_to_idx based on train_path)
            self._train_val_paths_orig = # from train_path
            self._train_val_labels_orig = # from train_path
            self._test_paths = # from test_path
            self._test_labels = # from test_path

            if self.force_flat_for_fixed_cv:
                self._all_paths = self._train_val_paths_orig + self._test_paths
                self._all_labels = self._train_val_labels_orig + self._test_labels
            else:
                self._all_paths = self._train_val_paths_orig + self._test_paths # Still represents all original data
                self._all_labels = self._train_val_labels_orig + self._test_labels


        # 2. Load OFFLINE AUGMENTED data (if requested)
        if self.use_offline_augmented_data:
            original_dataset_name = self.root_path.name
            augmented_dataset_name = f"{original_dataset_name}_augmented"
            augmented_dataset_path = self.root_path.parent / augmented_dataset_name

            if augmented_dataset_path.is_dir():
                if not self.class_to_idx: # Should have been set by now
                    logger.error("Cannot load augmented data: Main dataset classes not determined.")
                else:
                    aug_paths, aug_labels_mapped, _, _ = \
                        self._scan_dir_for_paths_labels_using_existing_map(
                            augmented_dataset_path, self.classes, self.class_to_idx
                        )
                    if aug_paths:
                        self._offline_aug_paths = aug_paths
                        self._offline_aug_labels = aug_labels_mapped
                        logger.info(f"Loaded {len(self._offline_aug_paths)} offline augmented samples separately.")
                    else:
                        logger.warning(f"Augmented dataset directory {augmented_dataset_path} is empty.")
            else:
                logger.warning(f"Offline augmented dataset directory not found: {augmented_dataset_path}.")
        
        logger.info(f"Final Original Dataset sizes: "
                    f"{len(self._train_val_paths_orig)} original train+val, "
                    f"{len(self._test_paths)} original test. "
                    f"Offline augmented samples loaded: {len(self._offline_aug_paths)}.")


    def get_train_val_paths_labels_orig(self) -> Tuple[List[Path], List[int]]:
        """Returns paths and labels for the ORIGINAL training + validation set."""
        return self._train_val_paths_orig, self._train_val_labels_orig

    def get_offline_augmented_paths_labels(self) -> Tuple[List[Path], List[int]]:
        """Returns paths and labels for the OFFLINE AUGMENTED dataset."""
        return self._offline_aug_paths, self._offline_aug_labels

    # get_test_paths_labels() remains the same, returns original test data
    # get_full_paths_labels_for_cv() remains the same, returns original full data
```

**Revised `SkorchModelAdapter.__init__` and `get_split_datasets`:**

```python
# server/ml/skorch_utils/adapter.py
class SkorchModelAdapter(NeuralNetClassifier):
    def __init__(self, *args,
                 # ... existing params ...
                 use_offline_augmented_data: bool = False, # <<< NEW from pipeline config
                 dataset_handler_ref: Optional[ImageDatasetHandler] = None, # <<< NEW: Pass a reference
                 **kwargs):
        # ...
        self.use_offline_augmented_data = use_offline_augmented_data
        self.dataset_handler_ref = dataset_handler_ref # Store reference
        if self.use_offline_augmented_data and self.dataset_handler_ref is None:
            raise ValueError("dataset_handler_ref must be provided if use_offline_augmented_data is True")
        super().__init__(*args, **kwargs)
        # ...

    def get_split_datasets(self, X, y=None, **fit_params):
        # X here are paths from dataset_handler.get_train_val_paths_labels_orig()
        # y are corresponding original labels
        if y is None: raise ValueError("y must be provided.")
        y_arr = to_numpy(y)
        X_paths_np_orig = np.asarray(X) # These are from the original train_val pool

        ds_train_final = None
        ds_valid_final = None

        if self.train_split: # e.g., ValidSplit for train/val or PredefinedSplit for single_train
            indices_orig_pool = np.arange(len(X_paths_np_orig))
            # This split gives indices relative to X_paths_np_orig (the original train+val pool)
            ds_train_indices_wrapper, ds_valid_indices_wrapper = self.train_split(indices_orig_pool, y=y_arr, **fit_params)
            
            train_indices_from_orig_pool = np.asarray(ds_train_indices_wrapper.indices)
            
            # Create validation dataset from ORIGINAL data ONLY
            if ds_valid_indices_wrapper is not None and len(ds_valid_indices_wrapper) > 0:
                valid_indices_from_orig_pool = np.asarray(ds_valid_indices_wrapper.indices)
                valid_paths = X_paths_np_orig[valid_indices_from_orig_pool].tolist()
                valid_labels = y_arr[valid_indices_from_orig_pool].tolist()
                ds_valid_final = PathImageDataset(paths=valid_paths, labels=valid_labels, transform=self.valid_transform)
                logger.debug(f"Validation split created with {len(ds_valid_final)} original samples.")
            else:
                logger.debug("No validation split created by train_split strategy.")

            # Create training dataset: original training part of the fold + offline augmented data
            current_fold_train_paths_orig = X_paths_np_orig[train_indices_from_orig_pool].tolist()
            current_fold_train_labels_orig = y_arr[train_indices_from_orig_pool].tolist()
            
            combined_train_paths = current_fold_train_paths_orig[:]
            combined_train_labels = current_fold_train_labels_orig[:]

            if self.use_offline_augmented_data and self.dataset_handler_ref:
                aug_paths, aug_labels = self.dataset_handler_ref.get_offline_augmented_paths_labels()
                if aug_paths:
                    logger.debug(f"Adding {len(aug_paths)} offline augmented samples to the current training fold.")
                    combined_train_paths.extend(aug_paths)
                    combined_train_labels.extend(aug_labels)
                    # Optional: shuffle combined_train_paths and combined_train_labels together here
                    # This might not be strictly necessary if iterator_train__shuffle=True for DataLoader

            ds_train_final = PathImageDataset(paths=combined_train_paths, labels=combined_train_labels, transform=self.train_transform)
            logger.debug(f"Training split for fold created with {len(ds_train_final)} samples (orig + offline aug).")

        else: # No train_split (e.g., training on full X, y without validation)
            logger.debug("No train_split defined. Using all provided X,y for training.")
            combined_train_paths = X_paths_np_orig.tolist()
            combined_train_labels = y_arr.tolist()
            if self.use_offline_augmented_data and self.dataset_handler_ref:
                aug_paths, aug_labels = self.dataset_handler_ref.get_offline_augmented_paths_labels()
                if aug_paths:
                    combined_train_paths.extend(aug_paths)
                    combined_train_labels.extend(aug_labels)
            ds_train_final = PathImageDataset(paths=combined_train_paths, labels=combined_train_labels, transform=self.train_transform)
        
        return ds_train_final, ds_valid_final
```

**Changes to `ClassificationPipeline.__init__`:**

```python
# server/ml/pipeline/pipeline.py
class ClassificationPipeline:
    def __init__(self,
                 # ...
                 use_offline_augmented_data: bool = False, # Existing
                 # ...
                ):
        # ...
        self.dataset_handler = ImageDatasetHandler(
            # ...
            use_offline_augmented_data=use_offline_augmented_data, # Pass to handler
            # ...
        )
        # ...
        # When creating self.model_adapter_config:
        self.model_adapter_config = {
            # ...
            'use_offline_augmented_data': use_offline_augmented_data, # Pass to adapter
            'dataset_handler_ref': self.dataset_handler,            # Pass reference to adapter
            # ...
        }
        # ...
        # And when instantiating self.model_adapter:
        init_config_for_adapter = self.model_adapter_config.copy()
        # ... (pop other non-init keys)
        self.model_adapter = SkorchModelAdapter(**init_config_for_adapter)
```

**Changes to `single_train` and `non_nested_grid_search` in `ClassificationPipeline`:**

*   These methods currently call `self.dataset_handler.get_train_val_paths_labels()`. This needs to change to `self.dataset_handler.get_train_val_paths_labels_orig()` so that `SkorchModelAdapter.get_split_datasets` receives only the original data pool to split for validation.

    ```python
    # In ClassificationPipeline.single_train
    # X_trainval, y_trainval = self.dataset_handler.get_train_val_paths_labels() # OLD
    X_trainval_orig, y_trainval_orig_list = self.dataset_handler.get_train_val_paths_labels_orig() # NEW
    if not X_trainval_orig: raise RuntimeError("Original train+validation data is empty.")
    y_trainval_orig_np = np.array(y_trainval_orig_list)
    # ... then proceed to use X_trainval_orig and y_trainval_orig_np for splits and fitting ...
    # The adapter's get_split_datasets will handle adding offline aug to its training part.

    # Similar change in non_nested_grid_search:
    # X_trainval, y_trainval = self.dataset_handler.get_train_val_paths_labels() # OLD
    X_trainval_orig, y_trainval_orig_list = self.dataset_handler.get_train_val_paths_labels_orig() # NEW
    # ... use X_trainval_orig, np.array(y_trainval_orig_list) with search.fit()
    ```

**How this addresses CV methods (`nested_grid_search`, `cv_model_evaluation`):**

*   These methods use `self.dataset_handler.get_full_paths_labels_for_cv()`, which **correctly returns only original data**.
*   When `cross_validate` calls `fit` on the `SkorchModelAdapter` for each outer training fold:
    *   The `X` and `y` passed to `fit` will be from the *original* data (a training portion of an outer fold).
    *   `SkorchModelAdapter.get_split_datasets` will receive these original data paths.
    *   It will then correctly create its own internal validation set *from these original paths*.
    *   It will then augment its internal training set (which is also from these original paths) with *all* the loaded offline augmented data (from `self.dataset_handler_ref.get_offline_augmented_paths_labels()`).

This ensures:
1.  **Validation sets (internal to `single_train`, `non_nested_grid_search`, or outer test folds in `nested_cv`/`cv_model_eval`) are always from the original, unaugmented data distribution.**
2.  **Training sets (for `single_train` or any training part of a CV fold) are supplemented with the *entire* offline augmented dataset.**

**Simplification/Assumption:**
This approach assumes that your offline augmented dataset (`[dataset_name]_augmented`) was generated from a pool of images that is representative of what you'd consider "training data" more generally (e.g., from the `train` split of a FIXED dataset, or from the non-test portion of a FLAT dataset before you even make specific val splits). It adds the *entire* offline augmented set to *any* training operation. This is simpler than trying to only add augmented versions of specific images that fall into a particular CV training split.

If your offline augmented set contains augmentations of images that might end up in a validation/test set of an outer CV fold, then this could still be a source of subtle data leakage for those CV methods. The "most pure" way would be to generate offline augmentations *only* from the specific training indices of *each fold* during the CV process, but that's much more complex to integrate.

The current revised strategy is a good balance of correctness and simplicity for most use cases.




For the CV methods, should all the offline augmented data be appended in the skorch adapter, or should a separate CV be done just for the offline data, to "mimic" the k-1 behaviour of the original train set, in order not to train on augmented variations of original images which could be in the validation or even test subsets? This is also the case for single train, so actually for all training methods, not necessarily CV-related. During a fit, the rest of the augmented images which would correspond to validation or test subsets would just be ignored (either during single train or CV iterations). How to tackle this the easiest way? Another mention, each generated image is actually mapped to the original, i.e. it is named [original_image_name]_augmented.[png/jpg/etc].



You've hit on a very important and nuanced point about data leakage when using offline augmented data with any kind of splitting (including train/validation splits and cross-validation).

Your concern is valid: if the offline augmented dataset was created from the *entire original dataset* (including images that will eventually end up in validation or test sets for a specific run/fold), then simply appending all offline augmented data to every training split can lead to the model effectively "seeing" augmented versions of its validation/test data during training. This inflates validation/test performance and doesn't reflect true generalization.

The fact that your augmented images are named `[original_image_name]_augmented...` is **extremely helpful** as it allows us to map augmented samples back to their originals!

**The Goal:**
For any given training split (whether it's the training part of a `single_train`'s val split, or the training part of a CV fold), we only want to add offline augmented samples whose *original source images* are also part of that *current training split*.

**Revised Strategy (More Robust Against Leakage):**

1.  **`ImageDatasetHandler`:**
    *   Still loads original data (`_train_val_paths_orig`, `_test_paths_orig`) and offline augmented data (`_offline_aug_paths`, `_offline_aug_labels`) separately.
    *   Crucially, when loading `_offline_aug_paths`, it also needs to parse and store the *original base name* for each augmented image.

2.  **`SkorchModelAdapter.get_split_datasets` (Main Logic Change):**
    *   Receives `X_orig_current_pool` (e.g., paths from `_train_val_paths_orig`) and `y_orig_current_pool`.
    *   Performs the train/validation split on `X_orig_current_pool` to get `train_indices_in_current_pool` and `valid_indices_in_current_pool`.
    *   **Validation Dataset:** Created using `X_orig_current_pool[valid_indices_in_current_pool]` (original images only) with `valid_transform`. This remains clean.
    *   **Training Dataset:**
        *   Start with the original images for this training fold: `current_fold_train_paths_orig = X_orig_current_pool[train_indices_in_current_pool]`.
        *   Get the set of *base names* of these `current_fold_train_paths_orig`.
        *   Iterate through `self.dataset_handler_ref.get_offline_augmented_paths_labels_with_originals()`.
        *   For each augmented sample, if its *original base name* is in the set of base names for `current_fold_train_paths_orig`, then add this augmented sample to the training data for this specific fold/split.
        *   Create `ds_train` using this combined set (original training fold images + their corresponding offline augmentations) and `train_transform`.

**Implementation Details:**

**1. Modify `ImageDatasetHandler`:**

```python
# server/ml/dataset_utils.py
from pathlib import Path
import re # For parsing filenames
# ...

class ImageDatasetHandler:
    def __init__(self, # ... as before ...
                ):
        # ...
        self._offline_aug_paths: List[Path] = []
        self._offline_aug_labels: List[int] = []
        self._offline_aug_original_basenames: List[str] = [] # <<< NEW: Store original basenames
        # ...
        self._load_paths_and_labels()
        # ...

    @staticmethod
    def _get_original_basename(augmented_path: Path) -> Optional[str]:
        """
        Parses an augmented filename like 'original_image_name_augmented.png'
        to extract 'original_image_name'.
        Returns None if parsing fails.
        """
        # Define the suffix pattern more robustly
        # It looks for "_augmented" followed by an optional dot and extension.
        match = re.match(r"^(.*?)_augmented(\..*)?$", augmented_path.name)
        if match:
            return match.group(1) # The part before "_augmented"
        # Fallback or stricter parsing if needed
        # logger.warning(f"Could not parse original basename from augmented file: {augmented_path.name}")
        return None


    # Modify _scan_dir_for_paths_labels_using_existing_map OR create a new one for augmented data
    # Let's create a new one that also extracts original basenames
    @staticmethod
    def _scan_augmented_dir_for_paths_labels_and_originals(
            target_dir: Path,
            master_class_names: List[str],
            master_class_to_idx: Dict[str, int]
    ) -> Tuple[List[Path], List[int], List[str]]: # Returns paths, labels, original_basenames
        aug_paths = []
        aug_labels = []
        aug_original_basenames = []
        
        target_dir = Path(target_dir)
        if not target_dir.is_dir(): return [], [], []

        found_class_subdirs = sorted([d.name for d in target_dir.iterdir() if d.is_dir()])
        
        for class_name_in_aug in found_class_subdirs:
            if class_name_in_aug not in master_class_to_idx:
                logger.warning(f"Class '{class_name_in_aug}' in augmented set not in master. Skipping.")
                continue
            
            class_idx = master_class_to_idx[class_name_in_aug]
            class_dir = target_dir / class_name_in_aug
            for img_path in class_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    original_basename = ImageDatasetHandler._get_original_basename(img_path)
                    if original_basename:
                        aug_paths.append(img_path)
                        aug_labels.append(class_idx)
                        aug_original_basenames.append(original_basename)
                    else:
                        logger.warning(f"Could not determine original basename for augmented file {img_path}. Skipping.")
        return aug_paths, aug_labels, aug_original_basenames


    def _load_paths_and_labels(self) -> None:
        # 1. Load ORIGINAL dataset structure (FLAT or FIXED)
        # ... (This part remains mostly the same, populating:
        #      self.classes, self.class_to_idx,
        #      self._train_val_paths_orig, self._train_val_labels_orig,
        #      self._test_paths, self._test_labels,
        #      self._all_paths, self._all_labels) ...
        # Example for FLAT:
        if self.structure == DatasetStructure.FLAT:
            all_paths_orig_ds, all_labels_orig_ds_list, classes, class_to_idx = self._scan_dir_for_paths_labels(self.root_path)
            if not all_paths_orig_ds: raise ValueError(f"No images found in FLAT dataset at {self.root_path}")
            self.classes = classes
            self.class_to_idx = class_to_idx
            
            if self.test_split_ratio_if_flat > 0 and len(all_paths_orig_ds) >=2:
                indices = np.arange(len(all_paths_orig_ds))
                train_val_indices_orig, test_indices_orig = train_test_split(
                    indices, test_size=self.test_split_ratio_if_flat, 
                    stratify=np.array(all_labels_orig_ds_list), random_state=RANDOM_SEED
                ) # Ensure stratify uses numpy array
                self._train_val_paths_orig = [all_paths_orig_ds[i] for i in train_val_indices_orig]
                self._train_val_labels_orig = [all_labels_orig_ds_list[i] for i in train_val_indices_orig]
                self._test_paths = [all_paths_orig_ds[i] for i in test_indices_orig]
                self._test_labels = [all_labels_orig_ds_list[i] for i in test_indices_orig]
            else:
                self._train_val_paths_orig = all_paths_orig_ds[:]
                self._train_val_labels_orig = all_labels_orig_ds_list[:]
                self._test_paths = []
                self._test_labels = []
            self._all_paths = all_paths_orig_ds 
            self._all_labels = all_labels_orig_ds_list
        # ... (elif self.structure == DatasetStructure.FIXED: similar logic for orig paths/labels)


        # 2. Load OFFLINE AUGMENTED data (if requested)
        if self.use_offline_augmented_data:
            original_dataset_name = self.root_path.name
            augmented_dataset_name = f"{original_dataset_name}_augmented"
            augmented_dataset_path = self.root_path.parent / augmented_dataset_name

            if augmented_dataset_path.is_dir():
                if not self.class_to_idx: 
                    logger.error("Cannot load augmented data: Main dataset classes not determined.")
                else:
                    # Use the new scanning function
                    self._offline_aug_paths, self._offline_aug_labels, self._offline_aug_original_basenames = \
                        self._scan_augmented_dir_for_paths_labels_and_originals(
                            augmented_dataset_path, self.classes, self.class_to_idx
                        )
                    if self._offline_aug_paths:
                        logger.info(f"Loaded {len(self._offline_aug_paths)} offline augmented samples with original name mapping.")
                    # ... (warnings if empty as before) ...
            # ... (warnings if dir not found as before) ...
        
        logger.info(f"Final Original Dataset sizes: "
                    f"{len(self._train_val_paths_orig)} original train+val, "
                    f"{len(self._test_paths)} original test. "
                    f"Offline augmented samples loaded: {len(self._offline_aug_paths)}.")


    def get_offline_augmented_paths_labels_with_originals(self) -> Tuple[List[Path], List[int], List[str]]:
        """
        Returns paths, labels, and corresponding original basenames for the OFFLINE AUGMENTED dataset.
        """
        return self._offline_aug_paths, self._offline_aug_labels, self._offline_aug_original_basenames

    # ... other getters remain the same (_orig for train_val, normal for test/full)
```

**2. Modify `SkorchModelAdapter.get_split_datasets`:**

```python
# server/ml/skorch_utils/adapter.py
# ...
class SkorchModelAdapter(NeuralNetClassifier):
    # ... (__init__ remains the same, ensure dataset_handler_ref is passed) ...

    def get_split_datasets(self, X, y=None, **fit_params):
        # X here are paths from dataset_handler.get_train_val_paths_labels_orig()
        # y are corresponding original labels
        if y is None: raise ValueError("y must be provided.")
        y_arr_orig_pool = to_numpy(y) # Labels of the current original pool (e.g., full train+val)
        X_paths_np_orig_pool = np.asarray(X)

        ds_train_final = None
        ds_valid_final = None

        if self.train_split:
            indices_in_orig_pool = np.arange(len(X_paths_np_orig_pool))
            ds_train_indices_wrapper, ds_valid_indices_wrapper = \
                self.train_split(indices_in_orig_pool, y=y_arr_orig_pool, **fit_params)
            
            train_indices_for_this_fold = np.asarray(ds_train_indices_wrapper.indices)
            
            # 1. Create VALIDATION dataset from ORIGINAL data ONLY
            if ds_valid_indices_wrapper is not None and len(ds_valid_indices_wrapper) > 0:
                valid_indices_for_this_fold = np.asarray(ds_valid_indices_wrapper.indices)
                valid_paths = X_paths_np_orig_pool[valid_indices_for_this_fold].tolist()
                valid_labels = y_arr_orig_pool[valid_indices_for_this_fold].tolist()
                ds_valid_final = PathImageDataset(paths=valid_paths, labels=valid_labels, transform=self.valid_transform)
                logger.debug(f"Validation split for fold created with {len(ds_valid_final)} original samples.")

            # 2. Create TRAINING dataset
            # Start with original images for this training fold
            current_fold_train_paths_orig = X_paths_np_orig_pool[train_indices_for_this_fold].tolist()
            current_fold_train_labels_orig = y_arr_orig_pool[train_indices_for_this_fold].tolist()
            
            # Get basenames of these original training images for mapping to augmentations
            current_fold_train_original_basenames = set()
            for p_orig in current_fold_train_paths_orig:
                # Assuming Path.stem gives the name without final extension, which should match original_basename
                current_fold_train_original_basenames.add(Path(p_orig).stem) 

            combined_train_paths_for_fold = current_fold_train_paths_orig[:]
            combined_train_labels_for_fold = current_fold_train_labels_orig[:]

            if self.use_offline_augmented_data and self.dataset_handler_ref:
                all_aug_paths, all_aug_labels, all_aug_original_basenames = \
                    self.dataset_handler_ref.get_offline_augmented_paths_labels_with_originals()
                
                added_aug_count = 0
                if all_aug_paths:
                    for aug_path, aug_label, aug_orig_basename in zip(all_aug_paths, all_aug_labels, all_aug_original_basenames):
                        if aug_orig_basename in current_fold_train_original_basenames:
                            combined_train_paths_for_fold.append(aug_path)
                            combined_train_labels_for_fold.append(aug_label)
                            added_aug_count +=1
                    logger.debug(f"Added {added_aug_count} relevant offline augmented samples to current training fold.")
            
            ds_train_final = PathImageDataset(
                paths=combined_train_paths_for_fold, 
                labels=combined_train_labels_for_fold, 
                transform=self.train_transform
            )
            logger.debug(f"Training split for fold created with {len(ds_train_final)} samples (orig fold + relevant offline aug).")

        else: # No train_split (e.g., training on full X, y without validation, like during final refit of GridSearchCV)
            logger.debug("No train_split defined by skorch. Using all provided X,y for training, plus all offline augmentations.")
            combined_train_paths = X_paths_np_orig_pool.tolist()
            combined_train_labels = y_arr_orig_pool.tolist()
            if self.use_offline_augmented_data and self.dataset_handler_ref:
                aug_paths, aug_labels, _ = self.dataset_handler_ref.get_offline_augmented_paths_labels_with_originals()
                if aug_paths: # Add ALL offline augmentations here because there's no val set to protect
                    combined_train_paths.extend(aug_paths)
                    combined_train_labels.extend(aug_labels)
                    logger.debug(f"Added {len(aug_paths)} offline augmented samples to full training set (no validation split).")
            ds_train_final = PathImageDataset(paths=combined_train_paths, labels=combined_train_labels, transform=self.train_transform)
        
        return ds_train_final, ds_valid_final
```

**Explanation of Key Logic in `SkorchModelAdapter.get_split_datasets`:**

1.  **Input `X`, `y`:** These are assumed to be from the *original, unaugmented* pool that `SkorchModelAdapter` is asked to fit on (e.g., `_train_val_paths_orig` for `single_train` or `non_nested_grid_search`, or a training fold from `_all_paths` in `nested_cv`).
2.  **Split Original Data:** `self.train_split` (e.g., `ValidSplit`) divides `X` and `y` into training indices and validation indices *for the current original pool*.
3.  **Clean Validation Set:** `ds_valid_final` is created *only* from the original validation indices and paths.
4.  **Augment Training Fold:**
    *   The original training paths for the current fold/split are identified (`current_fold_train_paths_orig`).
    *   Their basenames are extracted (e.g., "image001" from "path/to/image001.jpg").
    *   The code iterates through all available offline augmented samples.
    *   If an augmented sample's *original basename* matches one of the basenames in the *current original training fold*, then that augmented sample is added to this fold's training data.
5.  **Final Training Dataset:** `ds_train_final` consists of the original training images for that fold *plus only their corresponding offline augmentations*.

**How this Solves the Problem:**

*   **`single_train`:**
    *   `X` passed to `fit` is `_train_val_paths_orig`.
    *   `get_split_datasets` splits this into an original train part and an original validation part.
    *   Only the original train part gets its corresponding offline augmentations added. The validation part remains clean.
*   **`non_nested_grid_search` (Inner CV of GridSearchCV):**
    *   `X` passed to `fit` is `_train_val_paths_orig`.
    *   `GridSearchCV`'s internal `StratifiedKFold` will split this into multiple folds. For each fold:
        *   The (K-1)/K part becomes the "training data" for that inner Skorch fit.
        *   `get_split_datasets` receives this training part as `X`.
        *   It further splits this `X` into its own train/val for Skorch's internal monitoring.
        *   The offline augmentations corresponding *only* to the innermost training portion are added. Validation remains clean original data.
*   **`nested_grid_search` / `cv_model_evaluation(evaluate_on='full')` (Outer CV):**
    *   `X` passed to `cross_validate` is `_all_paths` (original full data).
    *   The outer `StratifiedKFold` splits `_all_paths` into outer_train_orig and outer_test_orig.
    *   When `SkorchModelAdapter` (inside `GridSearchCV` for nested) is `fit` on `outer_train_orig`:
        *   Its `get_split_datasets` works as described for `non_nested_grid_search`, ensuring inner validation sets are clean original data, and inner training sets get appropriate augmentations.
        *   The `outer_test_orig` used for final scoring of that outer fold is guaranteed to be original data.

This approach correctly isolates offline augmented data to only supplement the actual training samples for any given split, preventing leakage into validation or test sets. The key is the reliable mapping provided by your `[original_image_name]_augmented` naming convention.
