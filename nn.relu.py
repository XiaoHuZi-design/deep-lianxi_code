#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/10 21:05
# @Author  : LingJiaXiaoHu
# @File    : nn.relu.py
# @Software: win11 pytorch(GPU版本） python3.9.16
import torch
import torchvision
from torch import nn

from torch.nn import ReLU, Sigmoid  # 非线性变换
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1,-0.5],
                      [-1,3]])

input = torch.reshape(input,(-1,1,2,2))
print(input.shape)

#1、加载数据集
dataset =torchvision.datasets.CIFAR10("./CF_dataset",train=False,download=False,
                                      transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

#2、定义神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self,input):
        #output = self.relu1(input)
        output = self.sigmoid1(input)
        return output

tudui = Tudui()
# output = tudui(input)
# print(output)

#3、输出到Tensorboard
writer = SummaryWriter("./logs_relu")
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,global_step=step)
    output = tudui(imgs)
    writer.add_images("output",output,step)
    step += 1

writer.close()