# Research Pipeline Description

The given dataset (by path) may be structured in the file system in 2 variants:

  - root folder with multiple folders with images, each folder representing a class.
  - root folder with train and test folders, each one having multiple subfolders (the same number) with images, each subfolder representing a class. This means that the test dataset is already fixed.

### Important Notes: 

  - All operations/functions that use randomness of any kind (e.g. shuffling, splitting, etc.) are set to a fixed random seed (e.g. 42) in order to ensure reproducibility of the results. The random seed can be changed in the `config.py` file.
  - The purpose of (somewhat complex) logic is to ensure decoupling of the dataset loading and the model training. This allows for easy switching between different datasets, different loading methods, and different models. The pipeline is designed to be flexible and extensible, so that new datasets, loading methods, and models can be added easily. The pipeline is also designed to be easy to use, so that users can easily switch between different datasets, loading methods, and models without having to modify the code.

### In general, the pipeline provides four possible methods of training:

 - fixed train-test split: the dataset is split into training, validation and test subsets (e.g. 70% train, 20% validation, 10% test)
 - cross-validation: the dataset will be split into k folds, where k is a hyperparameter. The test set will have 1 fold, and for the rest of the dataset a split between training and validation will be done (e.g. 70% train, 30% validation). Traniing is performed, then the model is evaluated on the test subset. The process is repeated k times, each time using a different fold as test set. All results are centralized and averaged.
 - grid/random hyperparam search: search in the hyperparameter space to find the combination that yields best results. Exploration is done by taking every possible combination (grid) or randomly sampling a few. The dataset will have a fixed test set (kept for later testing), and the rest of the dataset is split into k folds, where k is a hyperparameter. The training and validation set will have k-1 folds, and the search-specific test subset will have 1 fold. Similarly to CV, the process is repeated k times, and at the end of each iteration the model is evaluated on the k test fold. Results are averaged, and the same process goes for the other hyperparameter combinations. At the end, the best hyperparameter combination is selected. A final training may also be performed on the whole designated training set and evaluated on the initial test set.
 - nested grid/random-search: (experimental) CV is performed as the outer loop, and for each fold a hyperparameter search is performed as the inner loop. This assesses the stability of the hyperparameter search process, by checking if the same hyperparameter combination is selected for each outer fold. This is a very expensive process.

For testing, a previously model can be loaded back and given new unseen instances. It can also perform LIME explainability on the predictions.

# Using the Pipeline:

In `model_src/main.py` there are multiple variables that can be set to control the pipeline. The main ones are:

 - `selected_dataset`: the name of the dataset to be used. It should match one of the datasets defined in `model_src/data/`. A custom dataset can also be added (make sure that a valid online augmentation strategy is also assigned)
 - `model_type`: the name of the model to be used. It should match one of the models defined in `model_src/ml/architectures/model_types.py`
 - `use_weighted_loss_for_run`: whether to use weighted loss for training. This is useful for imbalanced datasets. The weights are computed based on the class distribution in the training set.
 - `offline_augmentation`: (experimental) whether to use offline augmentation (additional synthesized images). If True, the dataset is required to have an additional folder with augmented images. If False, only online augmentation is used.
 - `chosen_sequence_idx`: the index of the pipeline example sequence to be used. A sequence is made of steps (presented above). Custom sequences can be made, but corectness is not guaranteed.
 - `img_size`: the size of the images to be used. The images will be resized to this size before being fed to the model.
 - `force_flat`: if dataset has a fixed test set, this forces the pipeline to integrate test within the entire dataset. Useful for CV-based sequences.
 - `save_model`: whether to save the trained model after training. The model is saved in the `experiments/` folder, with a timestamp and a unique identifier.
 - `data_augmentation_mode_override`: if set, this overrides the default online augmentation strategy defined for the dataset.

For testing a previously trained model, consider updating the following variables:

 - `username`: the username folder under which the images are stored. The folder should be in the `images/` folder.
 - `images_to_predict_info`: a list of tuples, each tuple containing the image filename, extension and a manually assigned prediction id (for tracking purposes).
 - `saved_model_dataset`: the name of the dataset used for training the saved model, along with the parent folder for experiments.
 - `saved_model_type_folder`: the name of the model type used for training the saved model.
 - `saved_model_experiment_run_id`: the unique identifier of the experiment run used for training the saved model. This can be found in the `experiments/` folder.
 - `saved_model_relative_path`: the relative path to the saved model file within the experiment run folder.

Other variables can be modified for more fine-grained control of the pipeline, but the above ones should be sufficient for most use cases.

This python project is configured to also work as a FastAPI app for the frontend React app, which demonstrates the model training and testing process, as well as image and prediction management and visualization.

### See Also 

Sub-folder for ML research source code: [model_src](model_src/server/ml/)