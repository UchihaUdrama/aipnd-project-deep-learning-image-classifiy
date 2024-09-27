# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

[Github](https://github.com/udacity/aipnd-project)

# Struct

flowers: contains images group into predict, train, valid, test purpose
|--- predict: with my 2 images using for predict
|--- test: contain images labled in classId
|--- train: contain images labled in classId
|--- valid: contain images labled in classId
cat_to_name.json: which map the classId to the actual flowers name
checkpoint.pth: checkpoint (my last test using resnet with 50 epochs, accuracy rate is 93%) ~120 MB
CODEOWNERS: default provide by Udacity
Image Classifier Project.html: html export version of my notebook
Image Classifier Project.ipynb: my notebook
LICENSE: default provide by Udacity
predict.py: use to predict the image
requirements.txt: my conda environmnet using conda list command
train.py: use to train the model
utils.py: common methods, contrains that I think should be shared between train.py and preict.py

# Installation
```
conda activate ai_env
```
Below are my computer information
```
(ai_env) PS C:\Users\<UserName>> nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_May__3_19:00:59_Pacific_Daylight_Time_2022
Cuda compilation tools, release 11.7, V11.7.64
Build cuda_11.7.r11.7/compiler.31294372_0
```

Use conda to install the pytorch which suite to my gpu

```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Also this repository include my requirements.txt which all packages I have installed in my conda environment

# Note
I only support resnet (resnet34) and vgg (vgg16)
I found that vgg with 5 epochs is good enough (96.55% accuracy), 
38 epochs is good for Resnet (93.10% accuracy) (And stop increasing the accuracy even I increasing the epochs)
But the checkpoint given from the resnet took lesser memory (~170 MB comparing to 1.5 GB of vgg)

# How to use

Make sure you are on the root folder using cd command

## For train.py
- General information
```
python train.py -h
```

- Simple run:
```
python train.py flowers --arch resnet --epochs 50
```

## Example of use:
- Load default 'checkpoint.pth' file at same directory with the train.py file.
- Using the default cat_to_name.json file which map the classId to the actual flowers name

```
python train.py flowers --arch resnet --epochs 50
```
Or using other architecture (Make sure you run that same architure first with the train.py)

```
python train.py flowers --arch vgg --epochs 5
```