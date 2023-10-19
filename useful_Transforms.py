#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/2 15:00
# @Author  : LingJiaXiaoHu
# @File    : useful_Transforms.py
# @Software: win11 pytorch(GPU版本） python3.9.16
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 注意大小写
writer = SummaryWriter("logs")
img = Image.open("images/youdao.png")
print(img)

#ToTensor()
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor",img_tensor)

#Normalize()
#公式  output[channel] = (input[channel] - mean[channel]) / std[channel]
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm,2)  #不写默认step0  可依次修改1、2、3....tensorboard窗口出现多张图片位于一栏

#Resize()
print(img.size)
trans_resize = transforms.Resize((512,512))  #序列 512 X 512大小
#img PIL -> totensor -> img_resize tensor
img_resize = trans_resize(img)
#img_resize PIL -> totensor ->img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize",img_resize,0)   #压缩小成正方形图片
print(img_resize)

#Compose - resize - 2
trans_resize_2 = transforms.Resize(512)   #transforms.Resize(x) 将图片短边缩放至x,长宽比保持不变
#PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize",img_resize_2,1)   #长方形图片

#RandomCrop() 随机裁剪
# trans_random = transforms.RandomCrop(512)
# trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
# for i in range(10):
#     img_crop = trans_compose_2(img)
#     writer.add_image("RandomCrop",img_crop,i)

trans_random = transforms.RandomCrop((500,1000))  #宽width 500 高height 1000
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHW",img_crop,i)

writer.close()
