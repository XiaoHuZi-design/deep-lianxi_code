#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/11 13:38
# @Author  : LingJiaXiaoHu
# @File    : nn.optim.py
# @Software: win11 pytorch(GPU版本） python3.9.16
import torch.optim
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

loss = nn.CrossEntropyLoss()
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(),lr=0.01)   #定义一个优化器
#循环20轮
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs,targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs,targets)
        optim.zero_grad()        #优化器梯度参数清零
        result_loss.backward()   #调用损失函数反向传播求出每个节点的梯度
        optim.step()             #使用step对模型每个参数进行调优
        running_loss = running_loss + result_loss
    print(running_loss)

#print(result_loss)   #tensor(0.8064, grad_fn=<NllLossBackward0>)

