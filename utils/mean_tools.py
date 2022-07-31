#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description: 
@Date:2021/11/28 21:55:48
@Author: ljt
'''

import torch
from torch.autograd import Function



# ----------------------
#  将传入的tensor归一化到[0,1]
# ----------------------
# 该函数的安全性问题还有待验证
# 需要将batch数据取出来分别做normlization, 因为如果batch大小不一样，可能会导致训练结果也不同。

def batch_norm_Zero2One(x, stratic=False):
    min_v = torch.min(x)
    max_v = torch.max(x)
    norm_x = torch.zeros_like(x)
    if min_v == max_v:
        norm_x[:] = torch.abs(max_v)
    else:
        if torch.min(x) >= 0:
            norm_x = x / (max_v - min_v)
        elif torch.min(x) < 0:
            norm_x = (x - min_v) / (max_v - min_v)
    return norm_x

def norm_Zero2One(x, stratic=False):
    norm_x = torch.zeros_like(x)

    for i in range(x.shape[0]):
        tmp = x[i]
        min_v = torch.min(tmp)
        max_v = torch.max(tmp)
        norm_tmp = torch.zeros_like(tmp)
        if min_v == max_v:
            norm_tmp[:] = torch.abs(max_v)
        else:
            if min_v >= 0:
                norm_tmp = tmp / (max_v - min_v)
            elif min_v < 0:
                norm_tmp = (tmp - min_v) / (max_v - min_v)
        norm_x[i] = norm_tmp
    if stratic:
        pass
    return norm_x




# ----------------------
#  限制数据范围到[0,1],超出置0/1
# ----------------------

def limit_zero2one(data_):
    data = data_.clone()
    data[data>=1] = 1
    data[data<=-1] = -1
    data = (data + 1) / 2
    return data


if __name__ == '__main__':
    data = torch.randn(5, 5)
    # data = data + 0.2
    limit_zero2one(data)
    print(data.min(), data.max())


# ----------------------
#  学习率改变函数
# ----------------------
def adjust_learning_rate(epoch, lr, optimizer_G, optimizer_D, writer):

    decay =  (lr / 2000) * epoch
    lr = lr - decay
    if lr <= 0:
        lr = 0
    
    writer.add_scalar('learning_rate', lr, epoch)

    for G_param_group in optimizer_G.param_groups:
        G_param_group["lr"] = lr
    for D_param_group in optimizer_D.param_groups:
        D_param_group["lr"] = lr



"""
图像边缘检测
"""
# def edge_dect(im):
#     # im = im.reshape((opt.batch_size, 1, im.shape[0], im.shape[1]))
#     conv1 = nn.Conv2d(1, 1, 3, bias=False)
#     sobel_kernel = torch.cuda.FloatTensor([[-1,	-1,	-1],	[-1,	8,	-1],	[-1,	-1,	-1]])
#     sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
#     conv1.weight.data = sobel_kernel
#     edge = conv1(Variable(im))
#     return edge

# def get_edge_loss(tmp_real_B, tmp_fake_B):
#     edge_real_B = edge_dect(tmp_real_B)
#     edge_fake_B = edge_dect(tmp_fake_B)
#     loss_edge = criterion_pixelwise(edge_real_B, edge_fake_B)
#     return loss_edge