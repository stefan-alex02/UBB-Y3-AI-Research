import os
import shutil

directory = 'GCD'
# directory = 'mini-GCD'
modified_directory = directory + '-modified/'

# Load the data from train and test folders into a common class dictionary
def load_data(only_include_classes=None):
    data = {}
    for folder in ['train', 'test']:
        for class_name in os.listdir(directory + '/' + folder):
            if only_include_classes and class_name not in only_include_classes:
                continue
            if class_name not in data:
                data[class_name] = []
            for file in os.listdir(directory + '/' + folder + '/' + class_name):
                data[class_name].append(directory + '/' + folder + '/' + class_name + '/' + file)

    return data

# Split the data into train, test and validation sets
def split_data(data, train_size=0.8, test_size=0.1, limit_ratio=None):
    train_data = {}
    test_data = {}
    val_data = {}

    if limit_ratio:
        for class_name, files in data.items():
            data[class_name] = files[:int(len(files) * limit_ratio)]

    for class_name, files in data.items():
        train_data[class_name] = files[:int(len(files) * train_size)]
        test_data[class_name] = files[int(len(files) * train_size):int(len(files) * (train_size + test_size))]
        val_data[class_name] = files[int(len(files) * (train_size + test_size)):]

    return train_data, test_data, val_data

def delete_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)

# Save the data into train, test and validation folders
def save_data(data: dict[str, list[str]], folder: str):
    os.makedirs(modified_directory + folder, exist_ok=True)
    for class_name, files in data.items():
        os.makedirs(modified_directory + folder + '/' + class_name, exist_ok=True)
        for file in files:
            shutil.copy(file, modified_directory + folder + '/' + class_name + '/')

# mini_dataset_classes = ['1_cumulus', '4_clearsky', '6_cumulonimbus']
# mini_dataset_classes = ['1_cumulus', '4_clearsky']
# mini_dataset_classes = ['3_cirrus', '4_clearsky', '6_cumulonimbus']

# Load the data
# data = load_data(only_include_classes=mini_dataset_classes)
data = load_data()

# Split the data
train_data, test_data, val_data = split_data(data, limit_ratio=0.1)
# train_data, test_data, val_data = split_data(data)

# Delete the existing modified directory
delete_directory(modified_directory)

# Save the data
os.makedirs(modified_directory, exist_ok=True)
save_data(train_data, 'train')
save_data(test_data, 'test')
save_data(val_data, 'val')
