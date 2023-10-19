#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/10 16:59
# @Author  : LingJiaXiaoHu
# @File    : nn_maxpool.py
# @Software: win11 pytorch(GPU版本） python3.9.16
import torch
import torchvision.datasets
from torch import  nn
from torch.nn import MaxPool2d

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]],dtype=torch.float32)   #最大池化无法对 “long" 进行定义

input = torch.reshape(input,(-1,1,5,5))
print(input.shape)

#1、加载数据集
dataset =torchvision.datasets.CIFAR10("./CF_dataset",train=False,download=False,
                                      transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

#2、定义神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self,input):
        output = self.maxpool1(input)
        return  output

tudui = Tudui()
output = tudui(input)
print(output)

#3、输出到Tensorboard
writer = SummaryWriter("./logs_maxpool")
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    output = tudui(imgs)
    writer.add_images("output",output,step)
    step = step + 1

writer.close()


# (DL) PS D:\studyFiles\Py-project1\lianxi_code> Tensorboard --logdir=logs_maxpool
# TensorFlow installation not found - running with reduced feature set.
# Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
# TensorBoard 2.13.0 at http://localhost:6006/ (Press CTRL+C to quit)
