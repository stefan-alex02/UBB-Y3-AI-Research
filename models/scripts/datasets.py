import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim import AdamW
from transformers import ViTModel, ViTForImageClassification, ViTFeatureExtractor
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import logging
import sys
from datetime import datetime
from pathlib import Path

# Dataset directories
selected_data_dir_idx = 0 # Index of the selected dataset directory
root_data_dir = "../datasets"
data_dir = [ "mini-GCD", "Swimcat-extend"] # Dataset directories

# Model directory and filename
model_folder = "models" # Model folder
model_filename = os.path.join(model_folder, "vit_model.pth") # Model filename

# Training hyperparameters
num_epochs = 30 # Number of epochs to train
batch_size = 8 # Batch size
learning_rate = 3e-5 # Learning rate
patience = 10 # Number of epochs with no improvement before early stopping

# Results directory
scenario = "v0" # Scenario name
results_dir = "results/vit/" + data_dir[selected_data_dir_idx] + "/" + scenario + "/" # Results directory

# Define train dataloader with online data augmentation
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Define validation and test dataloaders
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Specify the dataset directories
train_dir = os.path.join(root_data_dir, data_dir[selected_data_dir_idx], "train")
val_dir = os.path.join(root_data_dir, data_dir[selected_data_dir_idx], "val")
test_dir = os.path.join(root_data_dir, data_dir[selected_data_dir_idx], "test")

# Load the datasets
train_dataset = ImageFolder(root=train_dir, transform=transform_train)
val_dataset = ImageFolder(root=val_dir, transform=transform_val)
test_dataset = ImageFolder(root=test_dir, transform=transform_test)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Class Names
class_names = train_dataset.classes
print(f"Classes: {class_names}")

# Show number of training samples per class
train_class_counts = {class_names[i]: 0 for i in range(len(class_names))}
for _, label in train_dataset:
    train_class_counts[class_names[label]] += 1
print("Train Class Counts:", train_class_counts)

# Display the number of training samples
print(f"Total number of training samples: {len(train_dataset)}")