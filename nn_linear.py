#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/10 22:04
# @Author  : LingJiaXiaoHu
# @File    : nn_linear.py
# @Software: win11 pytorch(GPU版本） python3.9.16
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

#1、加载数据集
dataset =torchvision.datasets.CIFAR10("./CF_dataset",train=False,download=False,
                                      transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

#2、定义神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.linear1 = Linear(196608,10)   #线性变换最后一个维度 torch.Size([1, 1, 1, 196608])  -> torch.Size([1, 1, 1, 10])

    def forward(self,input):
        output = self.linear1(input)
        return output

tudui = Tudui()


for data in dataloader:
    imgs,targets = data
    print(imgs.shape)
    #output = torch.reshape(imgs,(1,1,1,-1))  #torch.Size([1, 1, 1, 196608])  -> torch.Size([1, 1, 1, 10])
    output = torch.flatten(imgs)              #将矩阵展成一行 torch.Size([196608])  -> torch.Size([10])
    print(output.shape)
    output = tudui(output)
    print(output.shape)