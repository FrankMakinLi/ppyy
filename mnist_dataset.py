# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:24:05 2019

@author: lisong
"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
#实例化一个dataset来装载本地的CIFAR10
import pandas as pd

def load_mnist(path,kind = 'train'):  #设置kind的原因：方便我们之后打开测试集数据，扩展程序
    """Load MNIST data from path"""
    """os.path.join为合并括号里面的所有路径"""
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte' % kind)
 
    images_path = os.path.join(path,'%s-images.idx3-ubyte' % kind)
 
    with open(labels_path, 'rb') as lbpath:
        # 'I'表示一个无符号整数，大小为四个字节
        # '>II'表示读取两个无符号整数，即8个字节
        #将文件中指针定位到数据集开头处，file.read(8)就是把文件的读取指针放到第九个字节开头处
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype = np.uint8)
        print(magic, n)   #便于读者知道这些对应是文件中的那些内容
 
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath, dtype = np.uint8).reshape(len(labels), 784)
        print(magic, num, rows, cols)  #便于读者知道这些对应是文件中的那些内容
 
 
    return images, labels

path=r'C:\Users\LISONG\Desktop\数据分析档案\MNIST'

class My_MNIST_Dataset(Dataset):
    def __init__(self,data):
        self.images=data[0]
        self.labels=data[1]
        
    def __getitem__(self,idx):
        return self.images[idx]
    
    def __len__(self):
        if len(self.images)==len(self.labels):
            return len(self.images)
        else:
            return len(self.labels)
            print('Labels are differ than Images')
#这里是仿照例子用的路径来读取，但实际上应该是从内存中或者从流中读取数据并保存成Dataset
#才对啊
train = load_mnist(path)
test = load_mnist(path,'t10k')
train_dataset = My_MNIST_Dataset(train)
#这里dataloaer自动将ndarray转为了torch.tensor。这里是个坑，需要注意。
#今天练习了如何构造一个Dataset子类，根据要求设定构造函数，getitem和len方法，并且依据这个dataset
#构造dataloader。
dataloader=torch.utils.data.DataLoader(train_dataset.images,
                                       batch_size=10,shuffle=True,num_workers=0)

X_train, y_train = load_mnist(path,kind = 'train')
X_test, y_test = load_mnist(path,kind = 't10k')
 
fig, ax = plt.subplots(nrows = 2,ncols = 5,sharex = True,sharey=True)
 
ax = ax.flatten()  #将2X5矩阵拉伸成元组形式，以便之后迭代
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    #imshow:cmap 代表绘图的样式；  interpolation：代表插值的方法
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
 
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
 
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 9][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
 
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
