#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/1 14:53
# @Author  : LingJiaXiaoHu
# @File    : Transforms.py.py
# @Software: win11 pytorch(GPU版本） python3.9.16
#常见的transforms
#*输入  *PIL      *Image.open()
#*输出  *tensor   *Totensor()
#*作用  *narrays  *cv.imread()
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

#python的用法 -》tensor数据类型
#通过 transforms.ToTensor去看两个问题
#1、transforms该如何使用（python)
#2、为什么我们需要Tensor数据类型

#绝对路径 D:\\studyFiles\\Py-project1\\lianxi_code\\dataset\\train\\ants\\0013035.jpg
#相对路径 dataset/train/ants/0013035.jpg    不知道为啥有时总是出错
#鼠标移到红色下划线 ALT + import the name           鼠标移到括号里 CTRL + P 显示参数类型
#opencv-python和numpy的版本不匹配，import cv2报错，需要升高或降低其版本
#解决方法：
#安装环境：python3.9
#opencv :4.5.2.54
#numpy :1.25.2
img_path = "D:\\studyFiles\\Py-project1\\lianxi_code\\dataset\\train\\ants\\0013035.jpg"     #图片路径
img = Image.open(img_path)      #读取路径打开图片
#print(img)   #打印出来图片格式类型

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()     #准换成ToTensor类型的图片
tensor_img = tensor_trans(img)
#print(tensor_img)

writer.add_image("Tensor_img",tensor_img)
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

#logs文件相对路径无法找到，使用绝对路径（难怪窗口没有图片，之前改了项目目录的，估计项目文件夹里面就是主函数相对路径才有效）
# (DL) PS D:\studyFiles\Py-project1> tensorboard --logdir=D:\studyFiles\Py-project1\lianxi_code\logs --port=6007
# TensorFlow installation not found - running with reduced feature set.
# Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
# TensorBoard 2.13.0 at http://localhost:6007/ (Press CTRL+C to quit)
