#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/2 20:54
# @Author  : LingJiaXiaoHu
# @File    : dataloader.py
# @Software: win11 pytorch(GPU版本） python3.9.16
import torchvision
#准备的测试数据集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("CF_dataset",train=False,transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
# 例如batch_size=4 每次从数据集取4张图片打包   这里数据集下好了就可以不用 download了
#测试数据集中第一张图片及target
img,target = test_data[0]  #单个图片 torch.Size([3, 32, 32])  3通道,像素 3 X 3
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")   #日志logs文件夹名
step = 0
for data in test_loader:
    imgs,targets=data
    # print(imgs.shape)   #读取像素
    # print(targets)
#torch.Size([batch_size大小，图像通道数，图像高度，图像宽度])
    writer.add_images("test_data",imgs,step)
    step = step + 1
#最后一步 图片不足64个，可以drop_last=True，自动舍去
writer.close()

