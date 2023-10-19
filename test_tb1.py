#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/1 10:36
# @Author  : LingJiaXiaoHu
# @File    : test_tb1.py.py
# @Software: win11 pytorch(GPU版本） python3.9.16

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "data/train/bees_image/16838648_415acd9e3f.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

#添加图像和数量   Pillow 10.0报错 PIL.Image‘ has no attribute ‘ANTIALIAS‘  卸载安装9.5.0
writer.add_image("test",img_array,2,dataformats='HWC')
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x",3*i,i)

writer.close()

#Tensorboardd网页刷新Image没有图片！！！！！！！！！！！！！！！！！！！！靠

#logdir = 事件文件所在文件夹名称  按ctrl+c取消重新指定端口port

# (DL) PS D:\Py-project1> tensorboard --logdir=logs
# TensorFlow installation not found - running with reduced feature set.
# Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
# TensorBoard 2.13.0 at http://localhost:6006/ (Press CTRL+C to quit)

# (DL) PS D:\Py-project1> tensorboard --logdir=logs --port=6007
# TensorFlow installation not found - running with reduced feature set.
# Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
# TensorBoard 2.13.0 at http://localhost:6007/ (Press CTRL+C to quit)