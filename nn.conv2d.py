#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/9 17:14
# @Author  : LingJiaXiaoHu
# @File    : nn.conv2d.py
# @Software: win11 pytorch(GPU版本） python3.9.16
import torch
import torchvision

from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./CF_dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=False)
# 加载数据集
dataloader = DataLoader(dataset,batch_size=64)

#定义一个神经网络
class Tudui(nn.Module):
    # 初始化
    def __init__(self):
        super(Tudui,self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
    # 定义一个forward()函数 进行卷积操作
    def forward(self,x):
        x = self.conv1(x)
        return  x
# 初始化网络
tudui = Tudui()
#print(tudui)

writer = SummaryWriter("./logs2d")  #输出到Tensorboard

step = 0
# 图像img经过ToTensor()的转换可以直接送到网络中
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)

    # torch.Size([64, 3, 32, 32])   输入大小
    writer.add_images("input",imgs,step)

    # torch.Size([64, 6, 30, 30])   输出大小  6个channel无法显示 会报错 需要改变形状 ->[xxx,3,30,30]
    output = torch.reshape(output,(-1,3,30,30))   #第一个数不知道多少就写-1，它会根据后面数字计算
    writer.add_images("output",output,step)

    step = step + 1

writer.close()

# (DL) PS D:\studyFiles\Py-project1\lianxi_code> Tensorboard --logdir=logs2d
# TensorFlow installation not found - running with reduced feature set.
# Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
# TensorBoard 2.13.0 at http://localhost:6006/ (Press CTRL+C to quit)

