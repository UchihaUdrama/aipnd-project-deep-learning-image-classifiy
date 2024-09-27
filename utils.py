import argparse
import requests
import tarfile
import os
import shutil
import torch                        # Using tensor and its operation for deep learning
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  # Image processing and pre-training the models
import time
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
from torchvision import datasets, transforms, models
from torchvision.models import VGG16_Weights, ResNet34_Weights
from torch.optim.lr_scheduler import StepLR

# some contrains
INPUT_SIZE = 25088
PATH_TO_CHECKPOINT = os.path.join('.')
LEARNING_RATE = 0.001
HIDDEN_UNITS = 4069
EPOCHES = 5
MY_FAV_CLASS = [1, 2, 10, 15, 20, 22, 63, 64, 90, 92]
BATCH_SIZE = 64 # Choose same value as 5.10 lesson
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

cat_to_name_file = os.path.join('.', 'cat_to_name.json')

with open(cat_to_name_file, 'r') as f:
    cat_to_name = json.load(f)

if(len(cat_to_name) != len(MY_FAV_CLASS)):
    print("Number of classes before change:", len(cat_to_name)) # It should be 102 at the original
    # I only want to keep my_fav_class class only
    cat_to_name = {key: value for key, value in cat_to_name.items() if int(key) in MY_FAV_CLASS}
    print("Number of classes after change:", len(cat_to_name))  # Should be 10 now
    with open(cat_to_name_file, 'w') as json_file:
        json.dump(cat_to_name, json_file, indent=4)

OUTPUT_SIZE = len(cat_to_name)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array(MEAN)
    std = np.array(STD)
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    # Hide the x and y axes
    ax.axis('off')
    ax.set_title(title)
    ax.imshow(image)
    return

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    preprocess = transforms.Compose([
        transforms.Resize(256),             # Resize the image to 256x256
        transforms.CenterCrop(224),         # Crop the center of the image to 224x224
        transforms.ToTensor(),              # Convert the image to a tensor
        transforms.Normalize(mean=MEAN, std=STD)  # Normalize the image
    ])
    # Load the image
    image = Image.open(image_path)

     # Apply the transformations
    image_tensor = preprocess(image)
    return image_tensor

def set_device(gpu_enable=True):
    """
    Enable gpu if supported
    """
    if gpu_enable and torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    return device

def create_pre_train_model(arch, hidden_units, num_classes):
    """
    Create pre-train model based on architecture, only support for vgg or resnet
    """
    if arch == "vgg":
        # Load the pre-trained model
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[0].in_features
        # Make sure parameters are frozen (during training, the weights of the pretrained layers will not be updated.)
        for param in model.parameters():
            param.requires_grad = False
        # Create the feedforward classifier using nn.Sequential
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, hidden_units),          # First fully connected layer
            nn.ReLU(),                                  # Activation function
            nn.Dropout(p=0.5),                          # Dropout layer for regularization
            nn.Linear(hidden_units, hidden_units),      # Second fully connected layer
            nn.ReLU(),                                  # Activation function
            nn.Dropout(p=0.5),                          # Dropout layer for regularization
            nn.Linear(hidden_units, num_classes)        # Output layer
    )
    else:
        # Load the pre-trained ResNet34 model
        model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        # Make sure parameters are frozen (during training, the weights of the pretrained layers will not be updated.)
        for param in model.parameters():
            param.requires_grad = False
            
        # Modify the final layer to match the number of classes in your dataset
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(HIDDEN_UNITS, int(HIDDEN_UNITS/2)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(int(HIDDEN_UNITS/2), num_classes)  # Output layer
        )
    
    return model