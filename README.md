# SDAC
##  You Need Glimpse Before Segmentation: Stochastic Detector-Actor-Critic for Medical Image Segmentation

The code will be released soon!

## Requirements

We recommend Anaconda as the environment

* Linux Platform
* NVIDIA GPU + CUDA CuDNN
* Torch == 1.8.0
* torchvision == 0.9.0
* Python3.8.0
* numpy1.19.2
* opencv-python
* visdom

## Training
1. Modify data path in src/main.py <br/>
datapath/img/\*\*\*<br/>
datapath/anno/\*\*\*
2. Select different dataloader.py according to different dataset
3. Begining training:
```
$ cd ./src/
$ python main.py 
```


## Test

## Datasets

Brain: [Link](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

Hecktor:  [Link](https://www.aicrowd.com/challenges/miccai-2020-hecktor)
