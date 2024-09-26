# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

[Github](https://github.com/udacity/aipnd-project)

# Installation

conda activate ai_env
(ai_env) PS C:\Users\<UserName>> nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_May__3_19:00:59_Pacific_Daylight_Time_2022
Cuda compilation tools, release 11.7, V11.7.64
Build cuda_11.7.r11.7/compiler.31294372_0

```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

I only support resnet (resnet34) and vgg (vgg16)
I found that vgg with 5 epochs is good enough (96.55% accuracy), 
20 epochs is good for Resnet (93.10% accuracy)
But the checkpoint given from the resnet took lesser memory (~170 MB comparing to 1.5 GB of vgg)