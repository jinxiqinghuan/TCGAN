import sys
import datetime
import time
import numpy as np
import argparse
import os
import piq
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable
# from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
# 自定义模型模块
from models.model import *
# from models.feature import *

# 自定义工具模块
from utils.datasets import *
from utils.mean_tools import *

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# path = r"/mnt/4t/ljt/datasets/madic"
# path = r"/mnt/4t/ljt/datasets/new_madic_data"
# path = r"/root/cloud_hard_drive/datasets/tmp/moire_train_dataset_patchers"
path = "/root/cloud_hard_drive/datasets/new_madic_data"

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0,
                    help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=2001,
                    help="number of epochs of training")
parser.add_argument("--ever_num", type=int, default=5000,
                    help="number of step of training and validation")
parser.add_argument("--dataset_name", type=str,
                    default="exp7", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=16,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100,
                    help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256,
                    help="size of image height")
parser.add_argument("--img_width", type=int, default=256,
                    help="size of image width")
parser.add_argument("--channels", type=int, default=3,
                    help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int,
                    default=100, help="interval between model checkpoints")
parser.add_argument("--lambda_pixel", type=int, default=100,
                    help="the value of labmda pixel")
parser.add_argument("--use_PixelGAN", type=bool,
                    default=False, help="the value of labmda pixel")
opt = parser.parse_args()
# print(opt)
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)
writer = SummaryWriter("runs/%s" % opt.dataset_name)


# # 释放闲置内存
# if hasattr(torch.cuda, 'empty_cache'):
# 	torch.cuda.empty_cache()


step = int(opt.ever_num / opt.batch_size)

cuda = True if torch.cuda.is_available() else False
# cuda = False

# 损失函数
criterion_GAN = torch.nn.MSELoss()
'''L1Loss: 平均绝对误差(MAE)'''
# criterion_pixelwise = torch.nn.L1Loss()
criterion_pixelwise = torch.nn.SmoothL1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
# 翻译图像和真实图像
lambda_pixel = opt.lambda_pixel

# Calculate output of image discriminator (PatchGAN)

patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)
# if opt.use_PixelGAN:
#     patch = (1, 256, 256)

# Initialize generator and discriminator

discriminator = Discriminator(3)
generator = GeneratorUNet(3, 3)
# generator = TransGeneratorV2(depth1=5, depth2=4, depth3=2, initial_size=8, dim=384, heads=4, mlp_ratio=4, drop_rate=0.5)#,device = device)
# generator = TransGeneratorUNetV3(in_channels=1, out_channels=1, depth1=5, depth2=4, depth3=2, initial_size=8, dim=1024, heads=4, mlp_ratio=4, drop_rate=0.2)#,device = device)

# sobel loss
# sobel = sobel_model()


if cuda:
    generator = generator.cuda()
    # generator = nn.DataParallel(generator)
    discriminator = discriminator.cuda()
    # sobel = sobel.cuda()
    # discriminator = nn.DataParallel(discriminator)
    criterion_GAN.cuda()
    # criterion_GAN = nn.DataParallel(criterion_GAN)
    criterion_pixelwise.cuda()
    # criterion_pixelwise = nn.DataParallel(criterion_pixelwise)

if opt.epoch != 0:
    # 加载已经训练好的模型
    generator.load_state_dict(torch.load(
        "saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load(
        "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # 初始化权重
    # generator.apply(atten_unet_init_weights)
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)


# 优化器
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 设置数据加载器
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# transforms_ = [
#     transforms.Resize((opt.img_height, opt.img_width),
#                       InterpolationMode.BICUBIC),
#     transforms.ToTensor(),
#     transforms.Normalize(0.5, 0.5),
# ]

dataloader = DataLoader(
    ImageDataset_3c(path, transforms_=transforms_, mode="train"),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    drop_last=True,
)

val_dataloader = DataLoader(
    ImageDataset_3c(path, transforms_=transforms_, mode="val"),
    batch_size=10,
    shuffle=True,
    num_workers=opt.n_cpu,
    drop_last=True,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------------------
#  图片取样
# ----------------------


def sample_images(batches_done, norm=False):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs["A"].type(Tensor))
    real_B = Variable(imgs["B"].type(Tensor))
    fake_B = generator(real_A)
    # if norm:
    #     real_A = norm_Zero2One(real_A)
    #     real_B = norm_Zero2One(real_B)
    #     fake_B = norm_Zero2One(fake_B)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, "images/%s/%s.png" %
               (opt.dataset_name, batches_done), nrow=4, normalize=True)

# ----------------------
#  验证集表现
# ----------------------
def val(epoch):
    generator.eval()
    val_all_ms_ssim = 0.0
    val_all_ssim = 0.0
    val_all_psnr = 0.0
    val_all_vif = 0.0
    val_all_mse = 0.0
    val_dataiter = iter(val_dataloader)
    # for val_i, val_batch in enumerate(val_dataloader):
    for val_i in range(50):
        val_batch = val_dataiter.next()
        val_real_A = Variable(val_batch["A"].type(Tensor), requires_grad=False)
        val_real_B = Variable(val_batch["B"].type(Tensor), requires_grad=False)
        # with torch.no_grad():
        val_fake_B = generator(val_real_A)
        val_tmp_real_B = (val_real_B + 1) / 2
        val_tmp_fake_B = limit_zero2one(val_fake_B)

        val_ms_ssim = piq.multi_scale_ssim(
            val_tmp_fake_B, val_tmp_real_B, data_range=1.)
        val_ssim = piq.ssim(val_tmp_fake_B, val_tmp_real_B, data_range=1.)
        val_psnr = piq.psnr(val_tmp_fake_B, val_tmp_real_B,
                            data_range=1., reduction='none')
        val_vif = piq.vif_p(val_tmp_real_B, val_tmp_fake_B, data_range=1.)
        val_mse = (torch.abs(val_tmp_fake_B - val_tmp_real_B) ** 2)

        val_all_ms_ssim += val_ms_ssim.item()
        val_all_ssim += val_ssim.item()
        val_all_psnr += val_psnr.mean().item()
        val_all_vif += val_vif.item()
        val_all_mse += val_mse.mean().item()
        # torch.cuda.empty_cache()

    val_num = val_i + 1
    writer.add_scalar('val_ssim', val_all_ssim / val_num, epoch)
    writer.add_scalar('val_ms_ssim', val_all_ms_ssim / val_num, epoch)
    writer.add_scalar('val_psnr', val_all_psnr / val_num, epoch)
    writer.add_scalar('val_vif', val_all_vif / val_num, epoch)
    writer.add_scalar('val_mse', val_all_mse / val_num, epoch)

epoch = opt.n_epochs
# lambda1 = lambda epoch: epoch // 30
def lambda1(epoch): return 0.998 ** epoch

# scheduler_1 = StepLR(optimizer_1, step_size=3, gamma=0.1)
scheduler_G = LambdaLR(optimizer_G, lr_lambda=lambda1)
scheduler_D = LambdaLR(optimizer_D, lr_lambda=lambda1)


def train():
    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        all_loss_G = 0.0
        all_loss_D = 0.0
        all_loss_GAN = 0.0
        all_loss_pixel = 0.0
        all_loss_vif = 0.0
        all_ssim = 0.0
        all_ms_ssim = 0.0
        all_psnr = 0.0
        all_vif = 0.0
        all_mse = 0.0
        all_sobel_loss = 0.0
        dataiter = dataloader.__iter__()
        # for i, batch in enumerate(dataloader):
        for i in range(step):
            batch = dataiter.__next__()
            # Model inputs
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))),
                             requires_grad=False)  # [1, 1, 16, 16]
            fake = Variable(
                Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

            # ------------------
            #  训练生成器
            # ------------------

            optimizer_G.zero_grad()
            

            # GAN loss
            # L1 loss

            fake_B = generator(real_A)

            pred_fake = discriminator(fake_B, real_B)

            # print(pred_fake.shape, valid.shape)
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            
            
            # so loss
            # loss_sobel = torch.abs(sobel(real_B) - sobel(fake_B)).mean()

            # canny loss

            # 特征提取器实现风格迁移损失

            # loss_style, loss_content = Feature(fake_B, real_A)
            # Style Loss
            # loss_style = feature_loss(fake_B, real_B, batch_size=opt.batch_size)

            # tmp_real_B = norm_Zero2One(real_B, stratic=True)
            # tmp_fake_B = norm_Zero2One(fake_B, stratic=True)

            # ssim_loss: torch.Tensor = piq.SSIMLoss(data_range=1.)(tmp_real_B, tmp_fake_B)
            # vif_loss: torch.Tensor = piq.VIFLoss(sigma_n_sq=2.0, data_range=1.)(tmp_real_B, tmp_fake_B)

            # Total Loss
            # print("loss_GAN: {}, loss_pixel(with lambda): {}, loss_style: {}".format(loss_GAN.item(), lambda_pixel * loss_pixel.item(), loss_style.item()))
            # + loss_style + edge_loss # + vif_loss * 0.05  # + ssim_loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel # + 10 * loss_sobel

            loss_G.backward()

            optimizer_G.step()

            # ---------------------
            #  训练判别器
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            # pred_real = discriminator(real_B, real_A)
            pred_real = discriminator(fake_B.detach(), real_B)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            # pred_fake = discriminator(fake_B.detach(), real_B)
            loss_fake = criterion_GAN(pred_fake, fake)
            
            # Total loss
            loss_D = loss_real + loss_fake

            loss_D.backward()
            # loss_D.backward(retain_graph=True)
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------
            # tmp_real_B = norm_Zero2One(real_B, stratic=True)
            # tmp_fake_B = norm_Zero2One(fake_B, stratic=True)

            # print(real_B.min(), real_B.max(), "    ||    ", fake_B.min(), fake_B.max())
            limit_fake_B = limit_zero2one(fake_B)
            real_B = (real_B + 1) / 2
            ssim = piq.ssim(limit_fake_B, real_B, data_range=1.)
            ms_ssim = piq.multi_scale_ssim(
                limit_fake_B, real_B, data_range=1.)
            psnr = piq.psnr(limit_fake_B, real_B,
                            data_range=1., reduction='none')
            vif = piq.vif_p(real_B, limit_fake_B, data_range=1.)
            mse = (torch.abs(limit_fake_B - real_B) ** 2)


            all_loss_G += loss_G.item()
            all_loss_D += loss_D.item()
            all_loss_GAN += loss_GAN.item()
            all_loss_pixel += loss_pixel.item()
            # all_loss_vif += vif_loss.item()
            all_ms_ssim += ms_ssim.item()
            all_ssim += ssim.item()
            all_psnr += psnr.mean().item()
            all_vif += vif.item()
            all_mse += mse.mean().item()
            # all_sobel_loss += loss_sobel

            # 输出训练进度
            # batches_done = epoch * len(dataloader) + i
            # print(batches_done)
            batches_done = epoch * step + i
            # batches_left = opt.n_epochs * len(dataloader) - batches_done
            batches_left = opt.n_epochs * step - batches_done

            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f, sobel_loss: no_use, vif_loss: no] [ssim: %f, ms_ssim: %f, psnr: %f, vif: %f, mse: %f]ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    # len(dataloader),
                    step,
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    # loss_sobel.item(),
                    # vif_loss,
                    ssim.item(),
                    ms_ssim.item(),
                    psnr.mean().item(),
                    vif.item(),
                    mse.mean().item(),
                    time_left,
                )
            )
        # TensorboardX  可视化训练过程
        train_num = i + 1
        writer.add_scalar('Loss_G', all_loss_G / train_num, epoch)
        writer.add_scalar('Loss_D', all_loss_D / train_num, epoch)
        # writer.add_scalar('ssim_loss', ssim_loss, epoch)
        writer.add_scalar('vif_loss', all_loss_vif / train_num, epoch)
        writer.add_scalar('ssim', all_ssim / train_num, epoch)
        writer.add_scalar('ms_ssim', all_ms_ssim / train_num, epoch)
        writer.add_scalar('mse', all_mse / train_num, epoch)
        writer.add_scalar('psnr', all_psnr / train_num, epoch)
        writer.add_scalar('vif', all_vif / train_num, epoch)
        # writer.add_scalar('edge_loss', all_sobel_loss / train_num, epoch)

        if epoch % 10 == 0:
            with torch.no_grad():
                val(epoch)
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(
            ), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
            torch.save(discriminator.state_dict(
            ), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))

        scheduler_G.step()
        scheduler_D.step()
        writer.add_scalar('learning_rate', scheduler_G.get_last_lr(), epoch)

        # adjust_learning_rate(epoch)
        torch.cuda.empty_cache()
    writer.close()
    
# ----------
#  Training
# ----------
if __name__ == '__main__':
    train()
