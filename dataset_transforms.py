#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/2 18:49
# @Author  : LingJiaXiaoHu
# @File    : dataset_transforms.py
# @Software: win11 pytorch(GPU版本） python3.9.16
import torchvision
from torch.utils.tensorboard import SummaryWriter

# 转换成 tonsor数据类型 可用tensorboard窗口显示
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# CIFAR10 数据集 下载
# torchvision.datasets.CIFAR10(root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
# root (string)-数据集的根目录，目录cifar-10-batch -py存在，或者如果download设置为True将保存到该目录。
# train (bool，可选)-如果为True，则从训练集创建数据集，否则从测试集创建数据集。
# transform(可调用的，可选的)—―一个接收PIL图像并返回转换后的版本的函数/转换。例如，变换。
# target_transform(可调用的，可选的)—―一个接受目标并对其进行转换的函数/转换。
# download(bool，可选)-如果为真，从互联网下载数据集并将其放在根目录。如果数据集已经下载，则不会再次下载。
# 先下载压缩包再解压，所以下载文件夹里为一个压缩包cifar-10-python.tar.gz和一个解压文件夹cifar-10-baches-py及里面baches文件
train_set = torchvision.datasets.CIFAR10(root="./CF_dataset",train=True,transform=dataset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root="./CF_dataset",train=False,transform=dataset_transform,download=True)

# print(test_set[0]) #测试集中的第一个
# print(test_set.classes)   #classes={list:10}['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#
# img,target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()   #一只猫

# print(test_set[0]) #输出tensor数据类型，所以可以用tensorboard显示
writer = SummaryWriter("p10")  #日志logs文件夹名   输出前10张图片
for i in range(10):       #0-9
    img, target = test_set[i]
    writer.add_image("test_set",img,i)

# (DL) PS D:\studyFiles\Py-project1\lianxi_code> tensorboard --logdir="p10"
# TensorFlow installation not found - running with reduced feature set.
# Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
# TensorBoard 2.13.0 at http://localhost:6006/ (Press CTRL+C to quit)
