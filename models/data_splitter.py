import os
import shutil

directory = 'mini-GCD/'
modified_directory = directory[:-1] + '-modified/'

# Load the data from train and test folders into a common class dictionary
def load_data():
    data = {}
    for folder in ['train', 'test']:
        for class_name in os.listdir(directory + folder):
            if class_name not in data:
                data[class_name] = []
            for file in os.listdir(directory + folder + '/' + class_name):
                data[class_name].append(directory + folder + '/' + class_name + '/' + file)

    return data

# Split the data into train, test and validation sets
def split_data(data, train_size=0.8, test_size=0.1):
    train_data = {}
    test_data = {}
    val_data = {}

    for class_name, files in data.items():
        train_data[class_name] = files[:int(len(files) * train_size)]
        test_data[class_name] = files[int(len(files) * train_size):int(len(files) * (train_size + test_size))]
        val_data[class_name] = files[int(len(files) * (train_size + test_size)):]

    return train_data, test_data, val_data

# Save the data into train, test and validation folders
def save_data(data: dict[str, list[str]], folder: str):
    os.makedirs(modified_directory + folder, exist_ok=True)
    for class_name, files in data.items():
        os.makedirs(modified_directory + folder + '/' + class_name, exist_ok=True)
        for file in files:
            shutil.copy(file, modified_directory + folder + '/' + class_name + '/')

# Load the data
data = load_data()

# Split the data
train_data, test_data, val_data = split_data(data)

# Save the data
os.makedirs(modified_directory, exist_ok=True)
save_data(train_data, 'train')
save_data(test_data, 'test')
save_data(val_data, 'val')
