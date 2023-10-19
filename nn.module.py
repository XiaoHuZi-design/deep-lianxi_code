#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/4 20:09
# @Author  : LingJiaXiaoHu
# @File    : nn.module.py
# @Software: win11 pytorch(GPU版本） python3.9.16
#定义一个神经网络   神经网络模版
# import torch.nn as nn
# import torch.nn.functional as F
#
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5)
#         self.conv2 = nn.Conv2d(20, 20, 5)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         return F.relu(self.conv2(x))
import torch
from torch import nn

#定义神经网络模版
class Tudui(nn.Module):
    def __init__(self):           #初始化
        super().__init__()

    def forward(self,input):
        output = input + 1
        return output

#用模版创建神经网络
tudui =Tudui()            #此处断点  先debug 然后点终端小箭头 step into my code 步进代码 一步一步运行
x = torch.tensor(1.0)
output = tudui(x)
print(output)

#1、首先为了调用tudui =Tudui() 进行初始化super().__init__()
#2、然后   x = torch.tensor(1.0)   创建一个输入x
#3、接着   output = tudui(x)    把x放在神经网络当中  forward中input变为  -> tensor(1.)
#4、再是   output = input + 1      output变为-> tensor(2.)
#5、然后   return output   返回output
#6、最后打印output   print(output)   -> tensor(2.)