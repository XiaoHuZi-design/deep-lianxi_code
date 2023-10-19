#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/11 19:51
# @Author  : LingJiaXiaoHu
# @File    : test.py
# @Software: win11 pytorch(GPU版本） python3.9.16
import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "./images/dog.png"
image = Image.open(image_path)
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

#定义神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),    # 展平 64 X 4 X 4 = 1024
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self,x):
        x = self.model1(x)          #使用Squential()简洁很多
        return x

model = torch.load("tudui_2.pth",map_location=torch.device("cpu"))   # 从gpu映射到cpu上  map映射
print(model)
image = torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))

