#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description: 分析数据集数据分布特点
@Date:2021/11/29 15:28:53
@Author: ljt
'''

from numpy.lib.function_base import _copy_dispatcher, copy
import piq
import torch
from tqdm import tqdm
from tqdm import tqdm
from PIL import Image
# from utils.datasets import ImageDataset
from datasets import ImageDataset
from mean_tools import *
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import os
from shutil import copyfile
from ana_tools import *

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

path = r"/mnt/4t/ljt/datasets/madic_pre/oral/data"

num_workers = 12
batch_size = 1


transforms_ = [
    transforms.Resize((256, 256), InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    # transforms.Normalize(0.5, 0.5),
]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
dataloader = DataLoader(
    ImageDataset(path, transforms_=transforms_, mode='train'),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

# val_dataloader = DataLoader(
#     ImageDataset(path, transforms_=transforms_, mode="val"),
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_workers,
#     drop_last=True,
# )



# ----------------------
#  查看数据范围
# ----------------------
# dataiter = dataloader.__iter__()
# for i in range(200):
#     batch = dataiter.__next__()
#     real_A = batch["A"]
#     real_B = batch["B"]
#     mmm_value(real_B, print_=True)
#     print(real_A.shape, real_B.shape, type(real_A))


# ----------------------
#  遍历文件夹满足某个条件的复制到指定的文件夹，查看特点
# ----------------------
# num = 0
# for i, batch in enumerate(tqdm(dataloader)):
#     if batch_size != 1:
#         print("The batch size must be 0! ")
#         os._exit(0)
#     copy_path = "/mnt/4t/ljt/project/pet2ct/copy_images" # 指定复制到的文件夹
#     os.makedirs(copy_path, exist_ok=True)
#     real_A = batch["A"]
#     real_B = batch["B"]
#     img_path = batch['path'][0]
#     max_value, min_value, mean_value = mmm_value(real_B, print_=False)
#     if mean_value <= 0.05:
#         num += 1
#         filename = os.path.basename(img_path)
#         copyfile(img_path, copy_path + "/" + filename)

# print("共发现了满足要求的{}张图片！".format(num))

# ----------------------
#  将满足要求的图片删除
#  20748 -> 17007
# ----------------------
num = 0
for i, batch in enumerate(dataloader):
    if batch_size != 1:
        print("The batch size must be 0! ")
        os._exit(0)
    real_A = batch["A"]
    real_B = batch["B"]
    img_path = batch['path'][0]
    max_value, min_value, mean_value = mmm_value(real_B, print_=False)
    if mean_value <= 0.05:
        num += 1
        filename = os.path.basename(img_path)
        os.remove(img_path)
        print("DELETE: {}".format(img_path))
print("共删除了满足要求的{}张图片！".format(num))

# 开始数据集划分
# *********************************train*************************************
# train类按照0.8：0.1：0.1的比例划分完成，一共17007张图片
# 训练集/mnt/4t/ljt/datasets/madic_pre/out/train/train：13606张
# 验证集/mnt/4t/ljt/datasets/madic_pre/out/val/train：1701张
# 测试集/mnt/4t/ljt/datasets/madic_pre/out/test/train：1700张



# ----------------------
#  确认采用目前的训练方式，每个epoch的数据都是不同的
# ----------------------