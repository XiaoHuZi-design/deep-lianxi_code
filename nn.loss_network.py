#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/11 12:51
# @Author  : LingJiaXiaoHu
# @File    : nn.loss_network.py
# @Software: win11 pytorch(GPU版本） python3.9.16
import torchvision.datasets
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./CF_dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                       download = False)
dataloader = DataLoader(dataset,batch_size=1)
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()

        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self,x):
        x = self.model1(x)          #使用Squential()简洁很多
        return x

# tudui = Tudui()
# for data in dataloader:
#     imgs,targets = data
#     outputs = tudui(imgs)
#     print(outputs)
#     print(targets)

loss = nn.CrossEntropyLoss()
tudui = Tudui()
for data in dataloader:
    imgs,targets = data
    outputs = tudui(imgs)
    result_loss = loss(outputs,targets)
    print(result_loss)
    result_loss.backward()    #反向传播
    print("ok")