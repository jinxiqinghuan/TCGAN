#!usr/bin/env python
# -*- coding:utf-8 _*

"""
@File : test.py 
@Author : ljt
@Description: xx
@Time : 2021/7/7 15:54 
"""

import argparse
import os
import math
from threading import stack_size

import matplotlib.pyplot
from numpy.lib.type_check import real
from tqdm import tqdm
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import time
import datetime
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchvision.transforms import InterpolationMode

# 自定义模块
from models.model import *
# from models.atten_unet import AttU_Net, R2AttU_Net, main
# from models.ResNetUnet import *
# from models.trans_unet.unet_transformer.unet import TransUnet
from utils.datasets import *
# from models.transGANv3 import TransGeneratorUNetV3, 
from models.transGANv4 import TransGeneratorUNetV4
# from models.pix2pix import *
import torch
import piq
from utils.eval_tools import *
# from models.nnn import ResUNet_LRes, ResUNet_LRes_new
# from models.SwinIR import SwinIR
# from models.swin_unet_one import SwinTransformerSys
# from models.atten_unet import AttU_Net
# from models.swin_unet.vision_transformer import SwinUnet
from matplotlib import pyplot as plt
import matplotlib.image as mplimg
# from matplotlib import cm
import cv2 as cv

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

path = r"D:\ljt\Study\datasts\new_madic_data"
# path = "/root/cloud_hard_drive/datasets/new_madic_data"
# path = '/root/cloud_hard_drive/project/pet2ct/brats2020'

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
print(cuda)
# cuda =True
# cuda = False

# generator = TransUnet(in_channels=1, img_dim=256, vit_blocks=1, vit_dim_linear_mhsa_block=512, classes=1)

# generator = R2AttU_Net()
# generator = GeneratorUNet()
# generator = ResUNet_LRes(3,3)
# generator = TransUnet(in_channels=1, img_dim=256, vit_blocks=4, vit_dim_linear_mhsa_block=512, classes=1)

# generator = ResUNet_LRes_new(3, 3)
# generator = RTGAN(in_channels=1, out_channels=1, depth1=5, depth2=4, depth3=2, initial_size=8, dim=1024, heads=4, mlp_ratio=4, drop_rate=0.2)#,device = device)
# generator = TransGeneratorUNetV3(in_channels=1, out_channels=1, depth1=5, depth2=4, depth3=2, initial_size=8, dim=1024, heads=4, mlp_ratio=4, drop_rate=0.2)#,device = device)

# generator = SwinIR(upscale=1, img_size=(256, 256),
#                    window_size=64, img_range=1., depths=[2, 2, 2, 2],
#                    embed_dim=20, num_heads=[2, 2, 2, 2], mlp_ratio=2, upsampler='pixelshuffledirect')


# generator = SwinTransformerSys(img_size=256, patch_size=4, in_chans=1, num_classes=1,
#                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
#                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
#                 use_checkpoint=False, final_upsample="expand_first")

# generator = AttU_Net(3, 3)
# generator =  SwinUnet(img_size=256, num_classes=1)
generator = TransGeneratorUNetV4(in_channels=1, out_channels=1, depth1=5, depth2=4, depth3=2, initial_size=8, dim=1024, heads=4, mlp_ratio=4, drop_rate=0.2)#,device = device)


                                                                                                                                                                                                                                                                                                                                   
if cuda:
    generator = generator.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor




# 设置数据加载器
# transforms_ = [
#     transforms.Resize((opt.img_height, opt.img_width), InterpolationMode.BICUBIC),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ]

transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
]


test_dataloader = DataLoader(
    ImageDataset(path, transforms_=transforms_, mode="test"),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
    drop_last=True,
)

# ----------
#  test_model
# ----------
def test_model_main():
    if opt.batch_size != 1:
        print("The batch size is not '1', please be careful! ")
        os._exit(0)
    # model_path = r"/mnt/4t/ljt/project/pet2ct/generator_2000.pth"
    # model_path = r"/mnt/4t/ljt/project/pet2ct/exp_data/2021-12-03/saved_models/use_new_data/generator_700.pth"
    model_path = r"/root/cloud_hard_drive/project/pet2ct/saved_models/exp22/generator_300.pth"

    generator.load_state_dict(torch.load(model_path))
    ssim, psnr, vif, mse = test_model(test_dataloader, generator, cuda=cuda)
    # print(ssim)

    print("测试模型: {} \n 结果为: ssim: {}, psnr: {}, vif: {}, mse: {}".format(model_path, ssim, psnr, vif, mse))


# ----------
#  gen_test_images
# ----------
def gen_test_images_main():
    model_path = r"/mnt/4t/ljt/project/pet2ct/saved_models/atten_unet/generator_300.pth"
    # model_path = r"/mnt/4t/ljt/project/pet2ct/saved_models/atten_unet/generator_40.pth"
    generator.load_state_dict(torch.load(model_path), state_dict=False)
    output_path = "output/test_images"
    os.makedirs(output_path, exist_ok=True)
    gen_test_images(output_path=output_path, test_dataloader=test_dataloader, generator=generator, cuda=True)

def norm_(a):
    a = (a + 1) / 2
    a = a * 256
    a = a.int()
    return a

def gen_exp_images():
    model_path = r"D:\ljt\project\pet2ct328\tmp_saved_models\generator_300.pth"
    generator.load_state_dict(torch.load(model_path))
    output_path = r"output\test_images"
    os.makedirs(output_path, exist_ok=True)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    generator.eval()
    for i, test_batch in enumerate(tqdm(test_dataloader)):
        real_A = Variable(test_batch["A"].type(Tensor))
        real_B = Variable(test_batch["B"].type(Tensor))
        fake_B = generator(real_A)
        # real_A = norm_(real_A)
        # real_B = norm_(real_B)
        # fake_B = norm_(fake_B)
        # sub = norm_(real_B - fake_B)
        sub = fake_B - real_B
        # sub = np.int8(sub.data.cpu().numpy())

        sub = sub.data.cpu().numpy()
        sub = (sub + 1) * 128 - 128

        sub = abs(sub)
        # sub = 255 - sub
        # sub = norm_(sub)
        # sub.dtype = np.int8
        print(sub.min(), sub.max(), sub.shape)
        # print(real_B.min(), fake_B.min(), real_B.max(), fake_B.max())



        mplimg.imsave(output_path + '\s{}_0realA.png'.format(i), real_A.data.cpu().reshape(256, 256), cmap=matplotlib.pyplot.gray())
        mplimg.imsave(output_path + '\s{}_1realB.png'.format(i), real_B.data.cpu().reshape(256, 256), cmap=matplotlib.pyplot.gray())
        mplimg.imsave(output_path + '\s{}_2fakeB.png'.format(i), fake_B.data.cpu().reshape(256, 256), cmap=matplotlib.pyplot.gray())
        # mplimg.imsave(output_path + '\s{}hotpic.png'.format(i), sub.reshape(256, 256), cmap=matplotlib.pyplot.gray())
        mplimg.imsave(output_path + '\s{}_3hotpic.png'.format(i), sub.reshape(256, 256), cmap=matplotlib.pyplot.jet())

        # plt.imsave(output_path + '\s{}hotpic.png'.format(i), (real_B.data.cpu() - fake_B.data.cpu()).reshape(256, 256))


        # img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        # save_image(img_sample, "%s/%s.png" % (output_path, i), nrow=4, normalize=True)





# ----------------------
#  view the gen data value
# ----------------------
def view_gen_data_value():
    model_path = "/mnt/4t/ljt/project/pet2ct/generator_2000.pth"
    generator.load_state_dict(torch.load(model_path))
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    generator.eval()
    for i, test_batch in enumerate(test_dataloader):
        real_A = Variable(test_batch["A"].type(Tensor))
        real_B = Variable(test_batch["B"].type(Tensor))
        fake_B = generator(real_A)
        print("max: {}     min: {}".format(torch.max(fake_B), torch.min(fake_B)))


# ----------
#  gen_patchGAN images
# ----------

if __name__ == '__main__':
    gen_exp_images()