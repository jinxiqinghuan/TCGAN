import torch
import torch.nn as nn 
from torch.nn import Conv2d,LeakyReLU,BatchNorm2d, ConvTranspose2d,ReLU
# import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
# from dataset import get_transforms
def tensor2img(one_tensor):# [b,c,h,w] [-1,1]
    tensor = one_tensor.squeeze(0) #[c,h,w] [0,1]
    tensor = (tensor*0.5 + 0.5)*255 # [c,h,w] [0,255]
    tensor_cpu = tensor.cpu()
    img = np.array(tensor_cpu,dtype=np.uint8)
    img = np.transpose(img,(1,2,0))
    return img
# def img2tensor(np_img):# [h,w,c]
#     tensor = get_transforms()(np_img).cuda() # [c,h,w] [-1,1]
#     tensor = tensor.unsqueeze(0) # [b,c,h,w] [-1,1]
#     return tensor


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') !=-1:
        nn.init.normal_(module.weight.data,0.0,0.02)

def loss_gradient_difference(real_image,generated): # b x c x h x w
    true_x_shifted_right = real_image[:,:,1:,:]# 32 x 3 x 255 x 256
    true_x_shifted_left = real_image[:,:,:-1,:]
    true_x_gradient = torch.abs(true_x_shifted_left - true_x_shifted_right)

    generated_x_shift_right = generated[:,:,1:,:]# 32 x 3 x 255 x 256
    generated_x_shift_left = generated[:,:,:-1,:]
    generated_x_griednt = torch.abs(generated_x_shift_left - generated_x_shift_right)

    difference_x = true_x_gradient - generated_x_griednt

    loss_x_gradient = (torch.sum(difference_x)**2)/2 # tf.nn.l2_loss(true_x_gradient - generated_x_gradient)

    true_y_shifted_right = real_image[:,:,:,1:]
    true_y_shifted_left = real_image[:,:,:,:-1]
    true_y_gradient = torch.abs(true_y_shifted_left - true_y_shifted_right)

    generated_y_shift_right = generated[:,:,:,1:]
    generated_y_shift_left = generated[:,:,:,:-1]
    generated_y_griednt = torch.abs(generated_y_shift_left - generated_y_shift_right)

    difference_y = true_y_gradient - generated_y_griednt
    loss_y_gradient = (torch.sum(difference_y)**2)/2 # tf.nn.l2_loss(true_y_gradient - generated_y_gradient)

    igdl = loss_x_gradient + loss_y_gradient
    return igdl

def loss_gradient_map(image): # b x c x h x w
    x_shifted_right = image[:,:,1:,:]# 32 x 3 x 255 x 256
    x_shifted_left = image[:,:,:-1,:]
    x_gradient = x_shifted_left - x_shifted_right

    y_shifted_right = image[:,:,:,1:]
    y_shifted_left = image[:,:,:,:-1]
    true_y_gradient = y_shifted_left - y_shifted_right

    return x_gradient, true_y_gradient


def calculate_x_gradient(images):
    x_gradient_filter = torch.Tensor(
        [
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],

        ]
    ).cuda()
    # filters代表卷积核的大小(out_channels，in_channe/groups，H，W)，是一个四维tensor
    x_gradient_filter = x_gradient_filter.view(1, 1, 3, 3)
    result = torch.functional.F.conv2d(
        images, x_gradient_filter, groups=1, padding=(1, 1)
    )
    return result


def calculate_y_gradient(images):
    y_gradient_filter = torch.Tensor(
        [
            [[0, 1, 0], [0, 0, 0], [0, -1, 0]]

        ]
    ).cuda()
    y_gradient_filter = y_gradient_filter.view(1, 1, 3, 3)
    result = torch.functional.F.conv2d(
        images, y_gradient_filter, groups=1, padding=(1, 1)
    )
    return result

# GDL loss的实现，该代码来源于网络
def loss_igdl(correct_images, generated_images): # taken from https://github.com/Arquestro/ugan-pytorch/blob/master/ops/loss_modules.py
    correct_images_gradient_x = calculate_x_gradient(correct_images)
    generated_images_gradient_x = calculate_x_gradient(generated_images)
    correct_images_gradient_y = calculate_y_gradient(correct_images)
    generated_images_gradient_y = calculate_y_gradient(generated_images)
    pairwise_p_distance = torch.nn.PairwiseDistance(p=1)
    distances_x_gradient = pairwise_p_distance(
        correct_images_gradient_x, generated_images_gradient_x
    )
    distances_y_gradient = pairwise_p_distance(
        correct_images_gradient_y, generated_images_gradient_y
    )
    loss_x_gradient = torch.mean(distances_x_gradient)
    loss_y_gradient = torch.mean(distances_y_gradient)
    loss = 0.5 * (loss_x_gradient + loss_y_gradient)
    return loss


def grad_L1_2_correct(correct_images, generated_images):
    L1 = torch.nn.L1Loss(reduction="sum")
    L2 = torch.nn.MSELoss(reduction="sum")
    correct_images = correct_images.expand(correct_images.shape[0], 1, 256, 256)
    generated_images = generated_images.expand(generated_images.shape[0], 1, 256, 256)
    # 根据真实图像的梯度信息来条件使用L1 L2 loss
    # 梯度的计算直接使用GDL的代码, 分别对X Y方向计算梯度
    correct_images_gradient_x = calculate_x_gradient(correct_images)
    correct_images_gradient_y = calculate_y_gradient(correct_images)
    # 直接将x y方向梯度图相加
    correct_images_gradient = torch.abs(correct_images_gradient_x) + torch.abs(correct_images_gradient_y)
    #限制性阈值设置为梯度图的最大值的一半，一半以上L1 loss计算，反之L2 loss 计算
    half_mean = correct_images_gradient.mean() * 2
    # half_mean = 0
    # 计算思思路: 将原始图像拷贝
    # L1 loss: 将梯度小于half_max的部分赋值为0, correct和gen images都是相同的操作
    # L2 loss: 将梯度大于half_max的部分赋值为0, correct和gen images都是相同的操作
    l1_correct = correct_images.clone()
    l1_generated = generated_images.clone()
    l2_correct = correct_images.clone()
    l2_generated = generated_images.clone()
    l1_correct[correct_images_gradient<=half_mean] = 0
    l1_generated[correct_images_gradient<=half_mean] = 0
    l2_correct[correct_images_gradient>half_mean] = 0
    l2_generated[correct_images_gradient>half_mean] = 0
    l1_pixel_num = torch.where(correct_images_gradient<=half_mean)[0].shape[0]
    l2_pixel_num = torch.where(correct_images_gradient>half_mean)[0].shape[0]
    # assert (l1_pixel_num + l2_pixel_num) == 256 * 256
    # print(l1_pixel_num + l2_pixel_num)
    loss = L1(l1_correct, l1_generated) / l1_pixel_num + L2(l2_correct, l2_generated) / l2_pixel_num
    return loss

def grad_L1_2_generated(correct_images, generated_images):
    L1 = torch.nn.L1Loss()
    L2 = torch.nn.MSELoss()

    # 根据真实图像的梯度信息来条件使用L1 L2 loss
    # 梯度的计算直接使用GDL的代码, 分别对X Y方向计算梯度
    generated_images_gradient_x = calculate_x_gradient(generated_images)
    generated_images_gradient_y = calculate_y_gradient(generated_images)
    
    # 直接将x y方向梯度图相加
    generated_images_gradient = torch.abs(generated_images_gradient_x) + torch.abs(generated_images_gradient_y)

    #限制性阈值设置为梯度图的最大值的一半，一半以上L1 loss计算，反之L2 loss 计算
    half_max = generated_images_gradient.max() / 2

    # 计算思思路: 将原始图像拷贝
    # L1 loss: 将梯度小于half_max的部分赋值为0, correct和gen images都是相同的操作
    # L2 loss: 将梯度大于half_max的部分赋值为0, correct和gen images都是相同的操作
    l1_correct = correct_images.clone()
    l1_generated = generated_images.clone()
    l2_correct = correct_images.clone()
    l2_generated = generated_images.clone()
    l1_correct[generated_images_gradient<half_max] = 0
    l1_generated[generated_images_gradient<half_max] = 0
    l2_correct[generated_images_gradient>=half_max] = 0
    l2_generated[generated_images_gradient>=half_max] = 0

    loss = L1(l1_correct, l1_generated) + L2(l2_correct, l2_generated)
    return loss

def grad_L1_2_sub(correct_images, generated_images):
    L1 = torch.nn.L1Loss()
    L2 = torch.nn.MSELoss()

    # 根据真实图像的梯度信息来条件使用L1 L2 loss
    # 梯度的计算直接使用GDL的代码, 分别对X Y方向计算梯度
    correct_images_gradient_x = calculate_x_gradient(correct_images)
    correct_images_gradient_y = calculate_y_gradient(correct_images)
    # 直接将x y方向梯度图相加
    correct_images_gradient = torch.abs(correct_images_gradient_x) + torch.abs(correct_images_gradient_y)


    generated_images_gradient_x = calculate_x_gradient(generated_images)
    generated_images_gradient_y = calculate_y_gradient(generated_images)
    generated_images_gradient = torch.abs(generated_images_gradient_x) + torch.abs(generated_images_gradient_y)


    sub_images_gradient = torch.abs(correct_images_gradient - generated_images_gradient)


    #限制性阈值设置为梯度图的最大值的一半，一半以上L1 loss计算，反之L2 loss 计算
    half_max = sub_images_gradient.max() / 2

    # 计算思思路: 将原始图像拷贝
    # L1 loss: 将梯度小于half_max的部分赋值为0, correct和gen images都是相同的操作
    # L2 loss: 将梯度大于half_max的部分赋值为0, correct和gen images都是相同的操作
    l1_correct = correct_images.clone()
    l1_generated = generated_images.clone()
    l2_correct = correct_images.clone()
    l2_generated = generated_images.clone()
    l1_correct[sub_images_gradient<half_max] = 0
    l1_generated[sub_images_gradient<half_max] = 0
    l2_correct[sub_images_gradient>=half_max] = 0
    l2_generated[sub_images_gradient>=half_max] = 0

    loss = L1(l1_correct, l1_generated) + L2(l2_correct, l2_generated)
    return loss


def main():
    x = torch.randn((1, 1, 256, 256)).cuda()#.expand(1, 3, 256, 256).cuda()
    y = torch.randn((1, 1, 256, 256)).cuda()#.expand(1, 3, 256, 256).cuda()
    # print(img.shape)
    loss = grad_L1_2_correct(x, y)
    # out = calculate_x_gradient(x)
    # plt.imsave("dd.png", out)
    print(loss)


if __name__ == '__main__':
    main()

