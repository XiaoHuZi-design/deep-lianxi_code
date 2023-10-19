#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/4 17:06
# @Author  : LingJiaXiaoHu
# @File    : test_tb.py
# @Software: win11 pytorch(GPU版本） python3.9.16

from torch.utils.tensorboard import SummaryWriter
#ctrl + 点击summaryWriter 获取函数解析
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "dataset/train/ants/0013035.jpg"     #绝对路径要双反斜杠，相对路径超过4个目录级就失效了
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)
#添加图像和数量     Pillow 10.0报错 PIL.Image‘ has no attribute ‘ANTIALIAS‘  卸载安装9.5.0
writer.add_image("test",img_array,1,dataformats='HWC')

# y = 2x
for i in range(100):
    writer.add_scalar("y=2x",3*i,i)

writer.close()


#logdir = 事件文件所在文件夹名称  按ctrl+c取消重新指定端口port

# (DL) PS D:\Py-project1> tensorboard --logdir=logs
# TensorFlow installation not found - running with reduced feature set.
# Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
# TensorBoard 2.13.0 at http://localhost:6006/ (Press CTRL+C to quit)

# (DL) PS D:\Py-project1> tensorboard --logdir=logs --port=6007
# TensorFlow installation not found - running with reduced feature set.
# Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
# TensorBoard 2.13.0 at http://localhost:6007/ (Press CTRL+C to quit)

