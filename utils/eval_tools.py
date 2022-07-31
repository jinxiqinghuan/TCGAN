#!usr/bin/env python
# -*- coding:utf-8 _*

"""
@File : eval_tools.py 
@Author : ljt
@Description: 测试相关的工具函数
@Time : 2021/7/7 15:54 
"""

import os
from torch.utils import data
from tqdm import tqdm
import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
from torchvision.transforms import functional as F
import torch
import piq

# 自定义模块
from utils.mean_tools import *


# ----------------------
#  验证集表现
# ----------------------
def test_model(test_dataloader, generator, cuda=True):
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # os.makedirs("exp_log/text.txt", exist_ok=True)
    generator.eval()
    test_all_ssim = []
    test_all_psnr = []
    test_all_vif = []
    test_all_mse = []
    # 进度条显示
    
    
    for i, test_batch in enumerate(tqdm(test_dataloader)):
    # for _, test_batch in enumerate(test_dataloader):
        test_real_A = Variable(test_batch["A"].type(Tensor), requires_grad=False)
        test_real_B = Variable(test_batch["B"].type(Tensor), requires_grad=False)
        # with torch.no_grad():
        test_fake_B = generator(test_real_A)
        
        
        # test_tmp_real_B = batch_norm_Zero2One(test_real_B, stratic=True)
        # test_tmp_fake_B = batch_norm_Zero2One(test_fake_B, stratic=True)

        test_tmp_real_B = (test_real_B + 1) / 2
        test_tmp_fake_B = limit_zero2one(test_fake_B)

        test_ssim = piq.ssim(test_tmp_real_B, test_tmp_fake_B, data_range=1., reduction='none')
        test_psnr = piq.psnr(test_tmp_real_B, test_tmp_fake_B, data_range=1., reduction='none')
        test_vif = piq.vif_p(test_tmp_real_B, test_tmp_fake_B, data_range=1., reduction='none')
        test_mse = (torch.abs(test_tmp_real_B - test_tmp_fake_B) ** 2)
        # print(test_ssim.mean().item(), test_psnr.mean().item(), test_vif.mean().item(), test_mse.mean().item())
        test_all_ssim.append(test_ssim.mean().item())
        test_all_psnr.append(test_psnr.mean().item())
        test_all_vif.append(test_vif.mean().item())
        test_all_mse.append(test_mse.mean().item())
    return np.mean(test_all_ssim), np.mean(test_all_psnr), np.mean(test_all_vif), np.mean(test_all_mse)



def test_cyclegan(test_dataloader, generator, cuda=True):
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # os.makedirs("exp_log/text.txt", exist_ok=True)
    generator.eval()
    test_all_ssim = []
    test_all_psnr = []
    test_all_vif = []
    test_all_mse = []
    # 进度条显示
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # 彩色图像转灰度图像num_output_channels默认1
        transforms.ToTensor()
    ])
    
    for i, test_batch in enumerate(tqdm(test_dataloader)):
    # for _, test_batch in enumerate(test_dataloader):
        test_real_A = Variable(test_batch["A"].type(Tensor), requires_grad=False)
        test_real_B = Variable(test_batch["B"].type(Tensor), requires_grad=False)
        # with torch.no_grad():
        test_fake_B = generator(test_real_A)
        test_fake_B = transform(test_fake_B)
        print(test_fake_B.shape)
        
        # test_tmp_real_B = batch_norm_Zero2One(test_real_B, stratic=True)
        # test_tmp_fake_B = batch_norm_Zero2One(test_fake_B, stratic=True)

        test_tmp_real_B = (test_real_B + 1) / 2
        test_tmp_fake_B = limit_zero2one(test_fake_B)

        test_ssim = piq.ssim(test_tmp_real_B, test_tmp_fake_B, data_range=1., reduction='none')
        test_psnr = piq.psnr(test_tmp_real_B, test_tmp_fake_B, data_range=1., reduction='none')
        test_vif = piq.vif_p(test_tmp_real_B, test_tmp_fake_B, data_range=1., reduction='none')
        test_mse = (torch.abs(test_tmp_real_B - test_tmp_fake_B) ** 2)
        # print(test_ssim.mean().item(), test_psnr.mean().item(), test_vif.mean().item(), test_mse.mean().item())
        test_all_ssim.append(test_ssim.mean().item())
        test_all_psnr.append(test_psnr.mean().item())
        test_all_vif.append(test_vif.mean().item())
        test_all_mse.append(test_mse.mean().item())
    return np.mean(test_all_ssim), np.mean(test_all_psnr), np.mean(test_all_vif), np.mean(test_all_mse)



def gen_test_images(output_path, test_dataloader, generator, cuda=True):
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    generator.eval()
    for i, test_batch in enumerate(tqdm(test_dataloader)):
        real_A = Variable(test_batch["A"].type(Tensor))
        real_B = Variable(test_batch["B"].type(Tensor))
        fake_B = generator(real_A)
        # real_A = (real_A + 1) / 2
        # real_B = (real_B + 1) / 2
        # fake_B = limit_zero2one(fake_B)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        save_image(img_sample, "%s/%s.png" % (output_path, i), nrow=4, normalize=True)

