import argparse
import requests
import tarfile
import os
import shutil
import torch                        # Using tensor and its operation for deep learning
import torch.optim as optim
import torch.nn.functional as F
import torchvision                  # Image processing and pre-training the models
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
from torchvision import datasets, transforms, models

# some contrains
input_size = 25088
path_to_checkpoint = os.path.join('.','checkpoint.pth')
learning_rate = 0.001
my_fav_class = [1, 2, 10, 15, 20, 22, 63, 64, 90, 92]
batch_size = 64 # Choose same value as 5.10 lesson

data_dir = os.path.join('.', 'flowers')
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

cat_to_name_file = os.path.join('.', 'cat_to_name.json')

with open(cat_to_name_file, 'r') as f:
    cat_to_name = json.load(f)
print(len(cat_to_name)) # It should be 102 at the original

if(len(cat_to_name) != len(my_fav_class)):
    # I only want to keep my_fav_class class only
    cat_to_name = {key: value for key, value in cat_to_name.items() if int(key) in my_fav_class}
    print(len(cat_to_name))  # Should be 10 now
    with open(cat_to_name_file, 'w') as json_file:
        json.dump(cat_to_name, json_file, indent=4)

output_size = len(cat_to_name)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])
    # Load the image
    image = Image.open(image_path)

     # Apply the transformations
    image_tensor = preprocess(image)
    return image_tensor