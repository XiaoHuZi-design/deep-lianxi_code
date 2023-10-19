#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/11 14:49
# @Author  : LingJiaXiaoHu
# @File    : train.py
# @Software: win11 pytorch(GPU版本） python3.9.16
import time

import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Tudui

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./CF_dataset",train=True,transform=torchvision.transforms.ToTensor(),
                                          download=False)
test_data = torchvision.datasets.CIFAR10(root="./CF_dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                         download=False)
# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size = 10，训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format((test_data_size)))

# 利用 DataLoader来加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

#搭建神经网络  可以单独放在一个文件叫 model
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui,self).__init__()
#
#         self.model1 = nn.Sequential(
#             nn.Conv2d(3, 32, 5, stride=1, padding=2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 32, 5, stride=1, padding=2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 5, stride=1, padding=2),
#             nn.MaxPool2d(2),
#             nn.Flatten(),    # 展平 64 X 4 X 4 = 1024
#             nn.Linear(1024, 64),
#             nn.Linear(64, 10)
#         )
#
#     def forward(self,x):
#         x = self.model1(x)          #使用Squential()简洁很多
#         return x

# 创建网络模型
tudui = Tudui()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01     #学习速率 1e-2
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)    #随机梯度下降

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# *添加Tensorboard
writer = SummaryWriter(".logs_train")
start_time = time.time()  #记录这个时候的时间赋值给开始时间
for i in range(epoch):
    print("------第{}轮训练开始------".format(i+1))

    # 训练步骤开始
    tudui.train()  #不用也可以
    for data in train_dataloader:
        imgs,targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs,targets)

        # 优化器优化模型
        optimizer.zero_grad()    #优化器梯度清零
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:  # 使输出更好看  逢百记录
            end_time = time.time()  # 每一次结束时间
            print(end_time - start_time)
            print("训练次数:{}，Loss:{}".format(total_train_step,loss.item()))     # item()将tensor数据类型转换成数字
            writer.add_scalar("train_loss",loss.item(),total_train_step)         # *

    # 测试步骤开始
    tudui.eval()   #不用也可以
    total_accuracy = 0
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss + loss.item()      # *
            accuracy = (outputs.argmax(1) == targets).sum()   #argmax(), 1 取矩阵横向最大值 ， 0 取矩阵纵向最大值
            toatl_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss:{}".format(total_test_step))
    print("整体测试集上的正确率:{}".format(toatl_accuracy / test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)   # *
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step = total_test_step + 1   # *

    # 保存模型
    torch.save(tudui,"tudui_{}.pth".format(i))     #后缀可以改，通常pth
    #官方 torch.save(tudui.state_dict(),"dudui_{}.pth".format(i))
    print("模型已保存")

writer.close()  # *
