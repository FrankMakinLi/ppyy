# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:22:53 2019

@author: lisong
"""
#这是用来跟随官方样例定义和使用CNN的copy。
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
#实例化一个dataset来装载本地的CIFAR10
import pandas as pd

class My_Dataset(Dataset):
    def __init__(self,csvfile):
        self.df=pd.read_csv(csvfile)
        
    def __getitem__(self,idx):
        return self.df.iloc[idx,]
    
    def __len__(self):
        return len(self.df)
#这里是仿照例子用的路径来读取，但实际上应该是从内存中或者从流中读取数据并保存成Dataset
#才对啊
csvfile=r'C:\Users\LISONG\python_script\train.csv'

        
#获取数据集CIFAR10,在线下载并保存到本地。这些都被封装在torchvision中，所以具体实现细节不知。
#test数据集也需要下载。
#但是为什么标记是手工定义的呢。
#可以看到test数据集并没有shuffle，batch_size都是4，也说明torchvision实现了数据装载时的batch设置。
#所以如何加载本地数据呢，使用torchvision？目前都是在内存中操作。
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=r'C:\Users\LISONG\python_script', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=r'C:\Users\LISONG\python_script', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#对于获得的数据，随机选取并绘制。
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

#开始定义网络。如何定义网络此前已经有介绍了，可以看到有2个卷积层，1个池层，3个全连接层。
#Conv2d以及MaxPool2d这些类的构造函数需要了解。
#以及按照别人的解释，nn.functional里实现的function和Module里实现的算法是一致的，但是F
#中的功能是定值，就像公式。而Module里的是会根据数据做调整，貌似是这么说的。
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

#定义损失函数使用交叉熵和SGD，SGD的参数可以看一下。
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#开始训练网络，从这个for循环可以看出，训练2轮。
#预设一个running_loss为0，便于后续观察他的变化。
#optimizer需要将其原始的梯度重置为0.为啥呢，难道定义了优化函数，原数据集就已经有gradient的属性了吗
#这里调用获得输出，并没有显示的调用forward函数，很显然是内置的方法。调用封装在了其他的方法中。
#这使得训练模型很简便。
#从循环来看，其mini-batches的size是多少没有明说，2000次mini-batches的训练之后，才返回一次平均loss
#从后面的返回结果来看，i累计到了12000，那么50000的总量的话，mini-batches是4，和获取数据时的
#batch-size=4符合。
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

#训练后需要测试精度。这个里面不太了解的点是torch.no_grad()，以及torch.max()
#predicted和labels都是tensor，所以可以使用其定义之上的一些方法。
#tensor.item()用于将一个标量的tensor转为python float
#最后测试精度是53%，作者认为其比随机挑选的精度即10%要高，说明还是学到了点什么。
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

#让程序告诉我们在每一类上的精度分别都是什么
#这个循环没怎么看得懂。。。
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

#最后这个例子建议我们尝试把cpu tensor转为GPU tensor，观察速度的不同。比如把单个卷积层包含的
#神经元数量提高。在将tensor迁移的过程中，要想使用GPU加速，那么所涉及到的所有tensor都需要迁移。
