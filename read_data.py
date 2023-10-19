#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/3 10:34
# @Author  : LingJiaXiaoHu
# @File    : read_data.py
# @Software: win11 pytorch(GPU版本） python3.9.16
from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self,idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "dataset/train"
#蚂蚁数据集
ants_label_dir = "ants"
#蜜蜂数据集
bees_label_dir = "bees"

ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

train_dataset =ants_dataset + bees_dataset

#首先全选粘贴到控制台Python console，然后使用以下命令
#train_dataset =ants_dataset + bees_dataset
# len(train_dataset)
# Out[16]: 245
# len(ants_dataset)
# Out[17]: 124
# len(bees_dataset)
# Out[18]: 121
# img, label =  ants_dataset[123]
# img.show()

# img, label =train_dataset[123]
# img.show()
# img, label = train_dataset[124]
# img.show()