import os
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler

from skimage import measure

from torch.autograd import Variable
# from roialign.roi_align.crop_and_resize import CropAndResizeFunction
import random

def PathList(path):


    train_img_list = os.listdir(path + '/Train/img')
    train_img_list.sort()
    train_label_list = os.listdir(path + '/Train/anno')
    train_label_list.sort()
    test_img_list = os.listdir(path + '/Test/img')
    test_img_list.sort()
    test_label_list = os.listdir(path + '/Test/anno')
    test_label_list.sort()




    # print(train_jpg_list[4],train_png_list[4])
    return train_img_list, train_label_list, test_img_list, test_label_list,


class MakeDataset(Dataset):
    def __init__(self, baseroot, imglist, annolist, train, img_size=None):
        self.baseroot = baseroot
        self.imglist = imglist
        self.annolist = annolist
        self.img_size = img_size
        self.train = train
        # self.transform1 = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5,), (0.5,))
        # ])
        # self.transform = transforms.Normalize((0.5,), (0.5,))
        # self.transform2 = transforms.Compose([
        #     transforms.RandomResizedCrop((128, 128)),
        #     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

        # transforms.Compose([transforms.RandomResizedCrop((224, 224)),
        #                 transforms.RandomHorizontalFlip(),
        #                 transforms.ToTensor()])

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):  
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        img_ilist = self.img_ilist
        img_alist = self.img_alist
        img_ipath = self.img_ilist[index]    
        imgi =np.load(img_ipath).astype(np.float32)

        # if self.img_size is not None:
        #     imgi = np.resize(imgi,(self.img_size,self.img_size))
        ##  Do pic resize or others here     ##
        #imgi = np.load(img_ipath)
        #print(img_ipath)
        #print(imgi.shape)
        #print(imgi)
        #npy = np.zeros((240,240,1))
        #npy[:,:,0] = imgi
        #print(npy.shape)
        #######################################

        img_apath = self.img_alist[index]    
        # img_apath = img_apath.split('.')[0] + '_mask.tif'
        imga = np.load(img_apath).astype(np.float32)
        # imga = imga.resize((240,240))        
        # if self.img_size is not None:
        #     imga = np.resize(imga, (self.img_size,self.img_size))        
        #######################################
        ##  Do pic resize or others here     ##
        #imga = np.load(img_apath)
        #print(imga)
        #print(imga.size)
        #
        # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 8))
        # ax0.imshow(imgi,cmap='gray')
        # ax1.imshow(imga,cmap='gray')
        # plt.show()
        #######################################
        # print(imga.shape)
        # print(imgi)
        file_name = os.path.splitext(img_ipath)[0].split('/')[-1]
        # imagei = imgi[np.newaxis,:,:]
        # imagea = imga[np.newaxis,:,:]
        if self.transform is not None:
             imgi = self.transform(imgi)   
             imga = self.transform(imga)
             imagei = imgi.type(torch.FloatTensor)
             imagea = imga.type(torch.FloatTensor)
        return imagei,imagea,file_name



