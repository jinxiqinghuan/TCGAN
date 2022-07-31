from asyncore import read
from email.mime import base
import readline
from unicodedata import name
import torch
import glob
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
# import skimage.io as io
from skimage import io as io
import imageio
from tqdm import tqdm
import os

def read_img(img_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))

reader = sitk.ImageSeriesReader()



# -----------------------
#  Madic pet 极值处理
# -----------------------

# root = 'D:\ljt\Study\Madic\Project\MedGAN\datasets\madic_original'


# for i in range(6):
#     path = root + "\{}".format(i+1)
#     pet_path = path + r"\reg.nii.gz"
#     pet = read_img(pet_path)
#     print(pet.mean(), pet.min(), pet.max())
#     pet[pet>0.04] = 0.04
#     # for j in range(522):
#     #     for k in range(480):
#     #         for l in range(480):
#     #             pet[j][k][l] = min(pet.mean(), pet[j][k][l])
#     plt.imshow(pet[200])
#     plt.show()

# for j in len(pet):
#     for k in j:
#         for l in k:


# -----------------------
#  IXI数据集二维切片预处理
# -----------------------

# 训练集
t1_list = glob.glob(r"/root/cloud_hard_drive/datasets/ixi_bet/IXI-T1-bet-reg/*.nii.gz")
t2_list = glob.glob(r"/root/cloud_hard_drive/datasets/ixi_bet/IXI-T2-bet/*.nii.gz")
sum_rows = 256
sum_cols = 512
# new image
final_matrix = np.zeros((sum_rows, sum_cols), np.float64)

print(len(t1_list), len(t2_list))


for i in range(len(t1_list)):
    # flag = 0
    base_name1 = os.path.basename(t1_list[i])
    name1 = base_name1[:-14]
    for j in range(len(t2_list)):
        base_name2 = os.path.basename((t2_list[j]))
        name2 = base_name2[:-10]
        # print(name1, name2)
        if name1 == name2:
            t1 = read_img(t1_list[i])
            t2 = read_img(t2_list[j])
            t1 = t1 / 13198
            t2 = t2 / 13198
            # print(name1, name2)
            # print(t1.shape, t2.shape)
            for k in range(t1.shape[0]):
                print(k)
                img_t1 = t1[k, :, :]
                img_t2 = t2[k, :, :]
                # print(img_t1.shape, img_t2.shape)
                final_matrix[0:sum_rows, 0:sum_cols // 2] = img_t1
                final_matrix[0:sum_rows, sum_cols // 2:sum_cols] = img_t2
                # print(final_matrix.shape)
                # final_matrix
                # imageio.imwrite("/root/cloud_hard_drive/project/pet2ct/ixi_dataset/all/s{}_{}t1.png".format(i, k), img_t1)
                # imageio.imwrite("/root/cloud_hard_drive/project/pet2ct/ixi_dataset/all/s{}_{}t2.png".format(i, k), img_t2)
                imageio.imwrite("/root/cloud_hard_drive/project/pet2ct/ixi_dataset/norm_all/t1_t2_{}_{}.png".format(i, k), final_matrix)        
    print("第{}个影像已处理完成！".format(i+1))
print("数据集已全部处理完毕！")




"""
求数据集中的极值, 并根据该极值进行归一化.
"""

# t1_list = glob.glob(r"/root/cloud_hard_drive/datasets/ixi_bet/IXI-T1-bet-reg/*.nii.gz")
# t2_list = glob.glob(r"/root/cloud_hard_drive/datasets/ixi_bet/IXI-T2-bet/*.nii.gz")

# t1_max = 0
# t2_max = 0

# for i in range(len(t1_list)):

#     t1 = read_img(t1_list[i])
#     t2 = read_img(t2_list[i])
#     if t1.max() > t1_max:
#         t1_max = t1.max()
#     if t2.max() > t2_max:
#         t2_max = t2.max()
#     print(t1_max, t2_max)
# print(t1_max, t2_max)
# 13197.867 4310.0

# sum_rows = 256
# sum_cols = 512
# # new image
# final_matrix = np.zeros((sum_rows, sum_cols), np.float64)


# t1 = read_img("/root/cloud_hard_drive/datasets/ixi_bet/reg.nii.gz")
# t2 = read_img("/root/cloud_hard_drive/datasets/ixi_bet/IXI-T2-bet/IXI002-Guys-0828-T2.nii.gz")
# for i in range(100):
#     print(t1.shape, t2.shape)
#     img_t1 = t1[i, :, :]
#     img_t2 = t2[i, :, :]
#     print(img_t1.shape, img_t2.shape)
#     final_matrix[0:sum_rows, 0:sum_cols // 2] = img_t1
#     final_matrix[0:sum_rows, sum_cols // 2:sum_cols] = img_t2
#     # print(final_matrix.shape)
#     # final_matrix
#     # imageio.imwrite("/root/cloud_hard_drive/project/pet2ct/ixi_dataset/all/s{}_{}t1.png".format(i, 0), img_t1)
#     # imageio.imwrite("/root/cloud_hard_drive/project/pet2ct/ixi_dataset/all/s{}_{}t2.png".format(i, 0), img_t2)
#     imageio.imwrite("/root/cloud_hard_drive/project/pet2ct/ixi_dataset/all/t1_t2_{}_{}.png".format(i, i), final_matrix)
                
#     print("第{}个影像已处理完成！".format(i+1))
# print("数据集已全部处理完毕！")




## 验证集
# root = '/root/cloud_hard_drive/datasets/Brats2020/MICCAI_BraTS2020_ValidationData'
# sum_rows = 240
# sum_cols = 480
# # new image
# final_matrix = np.zeros((sum_rows, sum_cols), np.float64)

# for i in tqdm(range(62)):
#     path = root + "/BraTS20_Validation_%03d" % (i+1)
#     # print(path)
#     t1_path = path + r"/BraTS20_Validation_%03d_t1.nii.gz" % (i+1)
#     t2_path = path + r"/BraTS20_Validation_%03d_t2.nii.gz" % (i+1)

#     t1 = read_img(t1_path)
#     t2 = read_img(t2_path)

#     # print(t1.shape) # 522 480 480
#     # print(t1.mean(), t1.min(), t1.max())

#     for j in range(0, 155, 1):
#         img_t1 = t1[j,:,:]
#         img_t2 = t2[j,:,:]
#         final_matrix[0:sum_rows, 0:sum_cols // 2] = img_t1
#         final_matrix[0:sum_rows, sum_cols // 2:sum_cols] = img_t2
#         # final_matrix
#         imageio.imwrite("/home/gxzy/ljt/project/pet2ct/brats2020/val/t1_t2_{}_{}.png".format(i, j), final_matrix)
#     # print("第{}个影像已处理完成！".format(i+1))
# print("数据集已全部处理完毕！")



# ### 测试集
# root = '/root/cloud_hard_drive/datasets/Brats2020/MICCAI_BraTS2020_ValidationData'
# sum_rows = 240
# sum_cols = 480
# # new image
# final_matrix = np.zeros((sum_rows, sum_cols), np.float64)

# for i in tqdm(range(62, 125, 1)):
#     path = root + "/BraTS20_Validation_%03d" % (i+1)
#     # print(path)
#     t1_path = path + r"/BraTS20_Validation_%03d_t1.nii.gz" % (i+1)
#     t2_path = path + r"/BraTS20_Validation_%03d_t2.nii.gz" % (i+1)

#     t1 = read_img(t1_path)
#     t2 = read_img(t2_path)

#     # print(t1.shape) # 522 480 480
#     # print(t1.mean(), t1.min(), t1.max())

#     for j in range(0, 155, 1):
#         img_t1 = t1[j,:,:]
#         img_t2 = t2[j,:,:]
#         final_matrix[0:sum_rows, 0:sum_cols // 2] = img_t1
#         final_matrix[0:sum_rows, sum_cols // 2:sum_cols] = img_t2
#         # final_matrix
#         imageio.imwrite("/home/gxzy/ljt/project/pet2ct/brats2020/test/t1_t2_{}_{}.png".format(i, j), final_matrix)
#     # print("第{}个影像已处理完成！".format(i+1))
# print("数据集已全部处理完毕！")





# -------------------
#  数据集划分
# -------------------

# 工具类
# import os
# import random
# import shutil
# from shutil import copy2


# def data_set_split(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0.1, test_scale=0.1):
#     '''
#     读取源数据文件夹,生成划分好的文件夹,分为trian、val、test三个文件夹进行
#     :param src_data_folder: 源文件夹 E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/src_data
#     :param target_data_folder: 目标文件夹 E:/biye/gogogo/note_book/torch_note/data/utils_test/data_split/target_data
#     :param train_scale: 训练集比例
#     :param val_scale: 验证集比例
#     :param test_scale: 测试集比例
#     :return:
#     '''
#     print("开始数据集划分")
#     class_names = os.listdir(src_data_folder)
#     # 在目标目录下创建文件夹
#     split_names = ['train', 'val', 'test']
#     for split_name in split_names:
#         split_path = os.path.join(target_data_folder, split_name)
#         if os.path.isdir(split_path):
#             pass
#         else:
#             os.mkdir(split_path)
#         # 然后在split_path的目录下创建类别文件夹
#         for class_name in class_names:
#             class_split_path = os.path.join(split_path, class_name)
#             if os.path.isdir(class_split_path):
#                 pass
#             else:
#                 os.mkdir(class_split_path)

#     # 按照比例划分数据集，并进行数据图片的复制
#     # 首先进行分类遍历
#     for class_name in class_names:
#         current_class_data_path = os.path.join(src_data_folder, class_name)
#         current_all_data = os.listdir(current_class_data_path)
#         current_data_length = len(current_all_data)
#         current_data_index_list = list(range(current_data_length))
#         random.shuffle(current_data_index_list)  # shuffle

#         train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
#         val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
#         test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
#         train_stop_flag = current_data_length * train_scale
#         val_stop_flag = current_data_length * (train_scale + val_scale)
#         current_idx = 0
#         train_num = 0
#         val_num = 0
#         test_num = 0
#         for i in current_data_index_list:
#             src_img_path = os.path.join(current_class_data_path, current_all_data[i])
#             if current_idx <= train_stop_flag:
#                 copy2(src_img_path, train_folder)
#                 # print("{}复制到了{}".format(src_img_path, train_folder))
#                 train_num = train_num + 1
#             elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
#                 copy2(src_img_path, val_folder)
#                 # print("{}复制到了{}".format(src_img_path, val_folder))
#                 val_num = val_num + 1
#             else:
#                 copy2(src_img_path, test_folder)
#                 # print("{}复制到了{}".format(src_img_path, test_folder))
#                 test_num = test_num + 1

#             current_idx = current_idx + 1

#         print("*********************************{}*************************************".format(class_name))
#         print(
#             "{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, val_scale, test_scale, current_data_length))
#         print("训练集{}：{}张".format(train_folder, train_num))
#         print("验证集{}：{}张".format(val_folder, val_num))
#         print("测试集{}：{}张".format(test_folder, test_num))


# if __name__ == '__main__':
#     src_data_folder = r"/root/cloud_hard_drive/project/pet2ct/ixi_dataset"
#     target_data_folder = r"/root/cloud_hard_drive/project/pet2ct/ixi_out_data"
#     data_set_split(src_data_folder, target_data_folder)



# -------------------
#  二维图像提取与处理 
# -------------------


# img_t1 = (read_img(t1[0]))
# img_t2 = (read_img(t2[53]))
# img_t2 = img_t2[5:155,45:195,45:195]
# origin = img_t2.GetOrigin()
# print(origin)

# mr = sitk.ReadImage(seg[100])
# mr_array = sitk.GetArrayFromImage(mr)

# origin = mr.GetOrigin()
# print("origin", origin)
# direction = mr.GetDirection()
# print("direction", direction)
# sapce = mr.GetSpacing()
# print("space", sapce)

# cv.imwrite("test.tiff", img_t2[50])
# cv.waitKey(0)
# plt.imshow(img_t1, cmap='gray')
# plt.show()
# plt.imshow(img_t2, cmap='gray')
# plt.show()

# sum_rows = 240
# sum_cols = 480
# # new image
# final_matrix = np.ones((sum_rows, sum_cols), np.float64)


# for i in range(len(t1)):
#     for j in range(50, 120, 10):
#         img_t1 = (read_img(t1[i])[j])
#         img_t2 = (read_img(t2[i])[j])
#         img_flair = (read_img(flair[i])[j])
#         img_t1ce = (read_img(t1ce[i])[j])
#         img_seg = (read_img(seg[i])[j])
#         imageio.imwrite("new_data/t1/t1_{}_{}.png".format(i, j), img_t1)
#         imageio.imwrite("new_data/t2/t2_{}_{}.png".format(i, j), img_t2)
#         imageio.imwrite("new_data/flair/flair_{}_{}.png".format(i, j), img_flair)
#         imageio.imwrite("new_data/t1ce/t1ce_{}_{}.png".format(i, j), img_t1ce)
#         imageio.imwrite("new_data/seg/seg_{}_{}.png".format(i, j), img_seg)
#         print("正在保存第{}张图片的位置{}的5个模态的照片".format(i, j))
#         # change
#         final_matrix[0:sum_rows, 0:sum_cols // 2] = img_t1
#         final_matrix[0:sum_rows, sum_cols // 2:sum_cols] = img_t2

#         imageio.imwrite("new_data/t1_t2/t1_t2_{}_{}.png".format(i, j), final_matrix)
#     print("第{}张照片已保存完成！".format(i))
# print("数据集已全部处理完毕！")

# ## 照片拼接
# path1 = "/home/sd/lijitao/project/MedGAN/new_data/t1"


# import os
# from PIL.Image import Image

# img_t1 = (read_img(t1[0])[100]).astype(np.uint8)
# img_t2 = (read_img(t2[0])[100]).astype(np.uint8)


# img_t1 = (read_img(t1[0])[100]).astype(np.uint8)
# img_t1ce = (read_img(t1ce[0])[100]).astype(np.uint8)
# plt.imshow(img_t1, cmap='gray')
# plt.show()
# plt.imshow(img_t1ce, cmap='gray')
# plt.show()
# i = 1
# plt.imsave("new_data/t1/t1_{}.png".format(i), img_t1, cmap='gray')


# -------------------
#  三维图像读取与处理
# -------------------


# img_t1 = (read_img(t1[0])).astype(np.uint8)
# img_t1ce = (read_img(t2[0])).astype(np.uint8)
# print(img_t1)

# import glob
# import random
# import os
# import numpy as np

# from torch.utils.data import Dataset
# from PIL import Image
# import torchvision.transforms as transforms


# class Image_3D_Dataset(Dataset):
#     def __init__(self, root, transforms_=None, mode="train"):
#         self.transform = transforms.Compose(transforms_)

#         self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
#         print(self.files)
#         if mode == "train":
#             self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))


#     def __getitem__(self, index):

#         img = Image.open(self.files[index % len(self.files)]).convert('RGB')
#         w, h = img.size
#         img_A = img.crop((0, 0, w / 2, h))
#         img_B = img.crop((w / 2, 0, w, h))

#         if np.random.random() < 0.5:
#             img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
#             img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

#         img_A = self.transform(img_A)
#         img_B = self.transform(img_B)

#         return {"A": img_A, "B": img_B}

#     def __len__(self):
#         return len(self.files)


# def main():
#     path = r"/home/sd/lijitao/datasets/2-MICCAI_BraTS_2018/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_10_1"
#     data = Image_3D_Dataset(path)

# if __name__ == '__main__':
#     main()


# -------------------
#  A4数据集处理
# -------------------

#
# path = glob.glob(r"/home/sd/lijitao/datasets/ttt/A4/*")
# t1 = glob.glob(r"/home/sd/lijitao/datasets/ttt/A4/*/T1*/*/*/*.nii")
# pet = glob.glob(r"/home/sd/lijitao/datasets/ttt/A4/*/Florbetapir/*/*/*.nii")
# t1_reg = glob.glob(r"/home/sd/lijitao/datasets/ttt/A4/*/reg/t1_reg.nii.gz")
# pet_reg = glob.glob(r"/home/sd/lijitao/datasets/ttt/A4/*/reg/pet_reg.nii.gz")
#
#
# sum_rows = 256
# sum_cols = 512
# # new image
# final_matrix = np.ones((sum_rows, sum_cols), np.float64)
#
#
# for i in range(len(t1)):
#     for j in range(50, 120, 10):
#         img_t1 = (read_img(t1[i])[:,:,j])
#         img_pet_reg = (read_img(pet_reg[i])[:,:,j])
#         print(img_t1.shape)
#         # 形状统一
#         print(img_t1.shape)
#         if img_t1.shape != (256, 256):
#             print("ok")


#         imageio.imwrite("new_data/A4/t1/t1_{}_{}.png".format(i, j), img_t1)
#         imageio.imwrite("new_data/A4/pet_reg/_{}_{}.png".format(i, j), img_pet_reg)
#         print("正在保存A4数据集第{}张图片的位置{}的2个模态的照片".format(i, j))
#         # change
#         final_matrix[0:sum_rows, 0:sum_cols // 2] = img_t1
#         final_matrix[0:sum_rows, sum_cols // 2:sum_cols] = img_t2

#         imageio.imwrite("new_data/A4/t1_pet/t1_pet_{}_{}.png".format(i, j), final_matrix)
#     print("第{}张照片已保存完成！".format(i))
# print("数据集已全部处理完毕！")
