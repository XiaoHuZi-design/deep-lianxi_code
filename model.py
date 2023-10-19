#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/11 15:15
# @Author  : LingJiaXiaoHu
# @File    : model.py
# @Software: win11 pytorch(GPU版本） python3.9.16
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, Flatten


#搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        # 输出大小N = (输入大小W - 卷积核大小F + 2倍填充值大小P)/步长大小S + 1
        # 池化层的输出尺寸计算公式与卷积层相同，但因为池化操作不会改变通道数，所以输出尺寸的通道数与输入尺寸相同。
        # pytorch中tensor（也就是输入输出层）的通道排序为：[batch, channel, height, width]
        self.model1 = Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),    # 展平 64 X 4 X 4 = 1024
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self,x):
        x = self.model1(x)          #使用Squential()简洁很多
        return x

if __name__ == '__main__':       # main就是其缩写
    tudui = Tudui()
    input = torch.ones(64,3,32,32)
    output = tudui(input)
    print(output.shape)
