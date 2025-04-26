
# Pipeline description

The given dataset (by path) may be structured in the file system in 2 variants:
  - root folder with multiple folders with images, each folder representing a class. In this case any combination of steps (1), (2) or alternatives is possible.
  - root folder with train and test folders, each one having multiple subfolders (the same number) with images, each subfolder representing a class. This means that the test dataset is already fixed. As mentioned above, this means that step (2) cannot be performed, only a fixed split with a single training and eval process.

Disclaimer: 

  - All operations/functions that use randomness of any kind (e.g. shuffling, splitting, etc.) are set to a fixed random seed (e.g. 42) in order to ensure reproducibility of the results. The random seed can be changed in the `config.json` file.
  - The purpose of (somewhat complex) logic is to ensure decoupling of the dataset loading and the model training. This allows for easy switching between different datasets, different loading methods, and different models. The pipeline is designed to be flexible and extensible, so that new datasets, loading methods, and models can be added easily. The pipeline is also designed to be easy to use, so that users can easily switch between different datasets, loading methods, and models without having to modify the code.

In general, the pipeline provides three possible methods of loading a dataset:

 - fixed train-test split: the dataset is split into training, validation and test subsets (e.g. 70% train, 20% validation, 10% test)
 - cross-validation on the training set: the dataset will have a fixed test set, and the rest of the dataset will be split into k folds, where k is a hyper-parameter. The training set will have k-1 folds, and the validation set will have 1 fold. The process is repeated k times, each time using a different fold as validation set.
 - cross-validation for evaluation: the dataset will be split into k folds, where k is a hyper-parameter. The test set will have 1 fold, and for the rest of the dataset a split between training and validation will be done (e.g. 70% train, 30% validation). The process is repeated k times, each time using a different fold as test set.

Loading operations/steps: When loading, a dataset may be processed in these steps:

  - 0: A function that loads the dataset from a given path, and returns it as a custom dataset object, while also detecting the type of dataset (e.g. folder-based or split-based).
  - 1: A function that takes a dataset (or a subset), and splits into two subsets: training and validation, or training and test (depending on the situation). The split is done in a stratified way, so that the distribution of classes is preserved in both subsets. The split is done using the `train_test_split` function from `sklearn.model_selection`, with a fixed ratio.
  - 2: A function that takes a dataset (or a subset), and performs a cross-validation split. The split is done in a stratified way, so that the distribution of classes is preserved in both subsets. The split is done using the `StratifiedKFold` function from `sklearn.model_selection`, with a fixed number of folds.

For a folder-based dataset, the three methods of loading a dataset can be viewed as a pipeline of the above-mentioned steps:
  - fixed train-test split: 0 -> 1 -> 1
  - cross-validation on the training set: 0 -> 1 -> 2
  - cross-validation for evaluation: 0 -> 2 -> 1

For a split-based dataset, the three methods of loading a dataset can be viewed as a pipeline of the above-mentioned steps:
  - fixed train-test split: 0 -> 1
  - cross-validation on the training set: 0 -> 2
  - cross-validation for evaluation: not possible

Pipeline methodologies: The rest of the pipeline will have the following operations:

  - it can take a train 


Depending on the mode the dataset is loaded in, the pipeline will behave as follows:

  - fixed train-test split: the dataset is loaded as a whole, and then split into training and validation sets. The training set is used to train the model, and the validation set is used to evaluate the model.


## Setup instructions
When creating a new .venv for the project, make sure to install the following dependencies using the following command (do not install pytorch before running this command):

- For CUDA 12.x:

        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

- For CUDA 11.8:

        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118