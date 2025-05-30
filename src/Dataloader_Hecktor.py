import numpy as np
import torch
import torch.utils.data
import os
import torchvision
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
class med(torch.utils.data.Dataset):      #数据集的处理
    def __init__(self,img_dir,anno_dir, img_size=None, transform = None):
        # TODO
        # 1. Initialize file paths or a list of file names.
        # 初始化文件路径或文件名列表
        img_ilist = []   #图像的列表
        img_alist = []   #label的列表
        temp = []
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        for subdir in os.listdir(img_dir):  #遍历文件夹中的所有文件及文件夹
                                         # 所以需要两个for循环。
            img_ilist.append(os.path.join(img_dir,subdir))
            img_alist.append(os.path.join(anno_dir,subdir))
        self.img_ilist = img_ilist
        self.img_alist = img_alist
        self.transform = transform
        self.img_size = img_size
    # 需要先对数据集进行处理，即先读取image和label并进行预处理（比如设置图像大小等）
    def __getitem__(self, index):  #获取数据集的序列。数据的预处理就在这个部分编写。
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        img_ilist = self.img_ilist
        img_alist = self.img_alist
        img_ipath = self.img_ilist[index]    #获取img_ilist的序列
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

        img_apath = self.img_alist[index]    #获取img_alist的序列
        # img_apath = img_apath.split('.')[0] + '_mask.tif'
        imga = np.load(img_apath).astype(np.float32)
        # imga = imga.resize((240,240))        #根据自己的需要改变图像大小
        # if self.img_size is not None:
        #     imga = np.resize(imga, (self.img_size,self.img_size))        #根据自己的需要改变图像大小
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
             imgi = self.transform(imgi)   # 在这里做transform，转为tensor等等
             imga = self.transform(imga)
             imagei = imgi.type(torch.FloatTensor)
             imagea = imga.type(torch.FloatTensor)
        return imagei,imagea,file_name

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.img_ilist)

