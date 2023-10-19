#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/4 21:29
# @Author  : LingJiaXiaoHu
# @File    : nn.conv.py
# @Software: win11 pytorch(GPU版本） python3.9.16

import torch

#torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
import torch.nn.functional as F    #用conv2d 一般 写 F.conv2d()

# 输入头像（5X5)
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])
# 卷积核（3X3)
kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

input = torch.reshape(input,(1,1,5,5))       #1batch_size, 1通道, 5X5大小   reshape()将数据重新组织，改变数组或矩阵的形状
kernel = torch.reshape(kernel,(1,1,3,3))

print(input.shape)
print(kernel.shape)    #输出矩阵的形状

output = F.conv2d(input, kernel, stride=1)  #每次滑移1步
print(output)   #卷积后的输出

output2 = F.conv2d(input, kernel, stride=2)  #每次滑移2步
print(output2)

output3 = F.conv2d(input, kernel, stride=1, padding=1)  #stride每次滑移1步  padding输入图像上下左右填充一行和一列（默认为0）
print(output3)

#终端输出的结果
# torch.Size([1, 1, 5, 5])
# torch.Size([1, 1, 3, 3])

# tensor([[[[10, 12, 12],
#           [18, 16, 16],
#           [13,  9,  3]]]])

# tensor([[[[10, 12],
#           [13,  3]]]])

# tensor([[[[ 1,  3,  4, 10,  8],
#           [ 5, 10, 12, 12,  6],
#           [ 7, 18, 16, 16,  8],
#           [11, 13,  9,  3,  4],
#           [14, 13,  9,  7,  4]]]])