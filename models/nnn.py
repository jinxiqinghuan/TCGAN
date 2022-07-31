import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import warnings
warnings.filterwarnings('ignore')



'''
    Ordinary UNet Conv Block
'''
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.activation = activation


        init.xavier_uniform(self.conv.weight, gain = np.sqrt(2.0))
        init.constant(self.conv.bias,0)
        init.xavier_uniform(self.conv2.weight, gain = np.sqrt(2.0))
        init.constant(self.conv2.bias,0)
    def forward(self, x):
        out = self.activation(self.bn(self.conv(x)))
        out = self.activation(self.bn2(self.conv2(out)))

        return out


'''    
 two-layer residual unit: two conv with BN/relu and identity mapping
'''
class residualUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,stride=1, padding=1, activation=F.relu):
        super(residualUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=1)
        init.xavier_uniform(self.conv1.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant(self.conv1.bias, 0)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        init.xavier_uniform(self.conv2.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant(self.conv2.bias, 0)
        self.activation = activation
        self.bn1 = nn.BatchNorm2d(out_size)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.in_size = in_size
        self.out_size = out_size
        if in_size != out_size:
            self.convX = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0)
            self.bnX = nn.BatchNorm2d(out_size)
        else:
            self.convX = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0)
            self.bnX = nn.BatchNorm2d(out_size)


    def forward(self, x):
        out1 = self.activation(self.bn1(self.conv1(x)))
        out2 = self.activation(self.bn2(self.conv2(out1)))
        if self.in_size!=self.out_size:
            bridge = self.activation(self.bnX(self.convX(x)))
        else:
            bridge = self.activation(self.bnX(self.convX(x)))
        output = torch.add(out2, bridge)

        return output


'''
    Ordinary UNet-Up Conv Block
'''
class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.bnup = nn.BatchNorm2d(out_size)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.activation = activation
        init.xavier_uniform(self.up.weight, gain = np.sqrt(2.0))
        init.constant(self.up.bias,0)
        init.xavier_uniform(self.conv.weight, gain = np.sqrt(2.0))
        init.constant(self.conv.bias,0)
        init.xavier_uniform(self.conv2.weight, gain = np.sqrt(2.0))
        init.constant(self.conv2.bias,0)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.activation(self.bnup(up))
        crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, crop1], 1)

        out = self.activation(self.bn(self.conv(out)))
        out = self.activation(self.bn2(self.conv2(out)))

        return out



'''
    Ordinary Residual UNet-Up Conv Block
'''
class UNetUpResBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpResBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.bnup = nn.BatchNorm2d(out_size)

        init.xavier_uniform(self.up.weight, gain = np.sqrt(2.0))
        init.constant(self.up.bias,0)

        self.activation = activation

        self.resUnit = residualUnit(in_size, out_size, kernel_size = kernel_size)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        up = self.activation(self.bnup(self.up(x)))
        crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, crop1], 1)

        out = self.resUnit(out)
        # out = self.activation(self.bn2(self.conv2(out)))

        return out


'''
    Ordinary UNet
'''
class UNet(nn.Module):
    def __init__(self, in_channel = 1, n_classes = 4):
        super(UNet, self).__init__()
#         self.imsize = imsize

        self.activation = F.relu
        
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_64 = UNetConvBlock(in_channel, 64)
        self.conv_block64_128 = UNetConvBlock(64, 128)
        self.conv_block128_256 = UNetConvBlock(128, 256)
        self.conv_block256_512 = UNetConvBlock(256, 512)
        self.conv_block512_1024 = UNetConvBlock(512, 1024)
        # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
        self.up_block1024_512 = UNetUpBlock(1024, 512)
        self.up_block512_256 = UNetUpBlock(512, 256)
        self.up_block256_128 = UNetUpBlock(256, 128)
        self.up_block128_64 = UNetUpBlock(128, 64)

        self.last = nn.Conv2d(64, n_classes, 1, stride=1)


    def forward(self, x):
#         print 'line 70 ',x.size()
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block256_512(pool3)
        pool4 = self.pool4(block4)

        block5 = self.conv_block512_1024(pool4)

        up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(up1, block3)

        up3 = self.up_block256_128(up2, block2)

        up4 = self.up_block128_64(up3, block1)

        return self.last(up4)


'''
    Ordinary ResUNet
'''


class ResUNet(nn.Module):
    def __init__(self, in_channel=1, n_classes=4):
        super(ResUNet, self).__init__()
        #         self.imsize = imsize

        self.activation = F.relu

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_64 = UNetConvBlock(in_channel, 64)
        self.conv_block64_128 = residualUnit(64, 128)
        self.conv_block128_256 = residualUnit(128, 256)
        self.conv_block256_512 = residualUnit(256, 512)
        self.conv_block512_1024 = residualUnit(512, 1024)
        # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
        self.up_block1024_512 = UNetUpResBlock(1024, 512)
        self.up_block512_256 = UNetUpResBlock(512, 256)
        self.up_block256_128 = UNetUpResBlock(256, 128)
        self.up_block128_64 = UNetUpResBlock(128, 64)

        self.last = nn.Conv2d(64, n_classes, 1, stride=1)

    def forward(self, x):
        #         print 'line 70 ',x.size()
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block256_512(pool3)
        pool4 = self.pool4(block4)

        block5 = self.conv_block512_1024(pool4)

        up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(up1, block3)

        up3 = self.up_block256_128(up2, block2)

        up4 = self.up_block128_64(up3, block1)

        return self.last(up4)


'''
    UNet (lateral connection) with long-skip residual connection (from 1st to last layer)
'''
class UNet_LRes(nn.Module):
    def __init__(self, in_channel = 1, n_classes = 4):
        super(UNet_LRes, self).__init__()
#         self.imsize = imsize   

        self.activation = F.relu
        
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_64 = UNetConvBlock(in_channel, 64)
        self.conv_block64_128 = UNetConvBlock(64, 128)
        self.conv_block128_256 = UNetConvBlock(128, 256)
        self.conv_block256_512 = UNetConvBlock(256, 512)
        self.conv_block512_1024 = UNetConvBlock(512, 1024)
        # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
        self.up_block1024_512 = UNetUpBlock(1024, 512)
        self.up_block512_256 = UNetUpBlock(512, 256)
        self.up_block256_128 = UNetUpBlock(256, 128)
        self.up_block128_64 = UNetUpBlock(128, 64)

        self.last = nn.Conv2d(64, n_classes, 1, stride=1)


    def forward(self, x, res_x):
#         print 'line 70 ',x.size()
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block256_512(pool3)
        pool4 = self.pool4(block4)

        block5 = self.conv_block512_1024(pool4)

        up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(up1, block3)

        up3 = self.up_block256_128(up2, block2)

        up4 = self.up_block128_64(up3, block1)
        
        last = self.last(up4)
        #print 'res_x.shape is ',res_x.shape,' and last.shape is ',last.shape
        if len(res_x.shape)==3:
            res_x = res_x.unsqueeze(1) 
        out = torch.add(last,res_x)
        
        #print 'out.shape is ',out.shape
        return out


'''
    ResUNet (lateral connection) with long-skip residual connection (from 1st to last layer)
'''


class ResUNet_LRes(nn.Module):
    def __init__(self, in_channel=1, n_classes=1, dp_prob=0):
        super(ResUNet_LRes, self).__init__()
        #         self.imsize = imsize

        self.activation = F.relu

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_64 = UNetConvBlock(in_channel, 64)
        self.conv_block64_128 = residualUnit(64, 128)
        self.conv_block128_256 = residualUnit(128, 256)
        self.conv_block256_512 = residualUnit(256, 512)
        self.conv_block512_1024 = residualUnit(512, 1024)
        # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
        self.up_block1024_512 = UNetUpResBlock(1024, 512)
        self.up_block512_256 = UNetUpResBlock(512, 256)
        self.up_block256_128 = UNetUpResBlock(256, 128)
        self.up_block128_64 = UNetUpResBlock(128, 64)
        self.Dropout = nn.Dropout2d(p=dp_prob)
        self.last = nn.Conv2d(64, n_classes, 1, stride=1)

    def forward(self, x):
        res_x = x
        #         print 'line 70 ',x.size()
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)

        pool1_dp = self.Dropout(pool1)

        block2 = self.conv_block64_128(pool1_dp)
        pool2 = self.pool2(block2)

        pool2_dp = self.Dropout(pool2)

        block3 = self.conv_block128_256(pool2_dp)
        pool3 = self.pool3(block3)

        pool3_dp = self.Dropout(pool3)

        block4 = self.conv_block256_512(pool3_dp)
        pool4 = self.pool4(block4)

        pool4_dp = self.Dropout(pool4)

        block5 = self.conv_block512_1024(pool4_dp)

        up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(up1, block3)

        up3 = self.up_block256_128(up2, block2)

        up4 = self.up_block128_64(up3, block1)

        last = self.last(up4)
        # print 'res_x.shape is ',res_x.shape,' and last.shape is ',last.shape
        if len(res_x.shape) == 3:
            res_x = res_x.unsqueeze(1)
        out = torch.add(last, res_x)

        # print 'out.shape is ',out.shape
        return out




class ResUNet_LRes_Deep(nn.Module):
    def __init__(self, in_channel=1, n_classes=1, dp_prob=0):
        super(ResUNet_LRes_Deep, self).__init__()
        #         self.imsize = imsize

        self.activation = F.relu

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.pool4_3 = nn.MaxPool2d(2)




        self.conv_block1_64 = UNetConvBlock(in_channel, 64)
        self.conv_block64_128 = residualUnit(64, 128)
        self.conv_block128_256 = residualUnit(128, 256)
        self.conv_block256_512 = residualUnit(256, 512)

        self.conv_block512_512_1 = residualUnit(512, 512)
        self.conv_block512_512_2 = residualUnit(512, 512)
        self.conv_block512_512_3 = residualUnit(512, 512)

        self.conv_block512_1024 = residualUnit(512, 1024)




        # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
        self.up_block1024_512 = UNetUpResBlock(1024, 512)

        self.up_block512_512_1 = UNetUpResBlock(512, 512)
        self.up_block512_512_2 = UNetUpResBlock(512, 512)
        self.up_block512_512_3 = UNetUpResBlock(512, 512)
        
        self.up_block512_256 = UNetUpResBlock(512, 256)
        self.up_block256_128 = UNetUpResBlock(256, 128)
        self.up_block128_64 = UNetUpResBlock(128, 64)
        self.Dropout = nn.Dropout2d(p=dp_prob)
        self.last = nn.Conv2d(64, n_classes, 1, stride=1)

    def forward(self, x):
        res_x = x
        #         print 'line 70 ',x.size()
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)

        pool1_dp = self.Dropout(pool1)

        block2 = self.conv_block64_128(pool1_dp)
        pool2 = self.pool2(block2)

        pool2_dp = self.Dropout(pool2)

        block3 = self.conv_block128_256(pool2_dp)
        pool3 = self.pool3(block3)

        pool3_dp = self.Dropout(pool3)

        block4 = self.conv_block256_512(pool3_dp)
        # pool4 = self.pool4(block4)

        pool4_dp = self.Dropout(block4)

        block4_1 = self.conv_block512_512_1(pool4_dp)
        pool4_1_dp = self.Dropout(block4_1)

        block4_2 = self.conv_block512_512_2(pool4_1_dp)
        pool4_2_dp = self.Dropout(block4_2)

        block4_3 = self.conv_block512_512_3(pool4_2_dp)
        pool4_3 = self.pool4_3(block4_3)
        pool4_3_dp = self.Dropout(pool4_3)

        block5 = self.conv_block512_1024(pool4_3_dp)


        up1 = self.up_block1024_512(block5, block4_3)
        up1_1 = self.up_block512_512_1(up1, block4_2)
        up1_2 = self.up_block512_512_2(up1_1, block4_1)
        up1_3 = self.up_block512_512_3(up1_2, block4)


        up2 = self.up_block512_256(up1_3, block3)

        up3 = self.up_block256_128(up2, block2)

        up4 = self.up_block128_64(up3, block1)

        last = self.last(up4)
        # print 'res_x.shape is ',res_x.shape,' and last.shape is ',last.shape
        if len(res_x.shape) == 3:
            res_x = res_x.unsqueeze(1)
        out = torch.add(last, res_x)

        # print 'out.shape is ',out.shape
        return out



class ResUNet_LRes_new(nn.Module):
    def __init__(self, in_channel=3, n_classes=1, dp_prob=0):
        super(ResUNet_LRes_new, self).__init__()
        #         self.imsize = imsize

        self.activation = F.relu
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.pool5 = nn.MaxPool2d(2)
        self.pool6 = nn.MaxPool2d(2)
        self.pool7 = nn.MaxPool2d(2)
        self.pool8 = nn.MaxPool2d(2)


        self.down1 = UNetConvBlock(in_channel, 8)
        self.down2 = residualUnit(8, 16)
        self.down3 = residualUnit(16, 32)
        self.down4 = residualUnit(32, 64)
        self.down5 = residualUnit(64, 128)
        self.down6 = residualUnit(128, 256)
        self.down7 = residualUnit(256, 512)
        self.down8 = residualUnit(512, 1024)


        self.up1 = UNetUpResBlock(1024, 512)
        self.up2 = UNetUpResBlock(512, 256)
        self.up3 = UNetUpResBlock(256, 128)
        self.up4 = UNetUpResBlock(128, 64)
        self.up5 = UNetUpResBlock(64, 32)
        self.up6 = UNetUpResBlock(32, 16)
        self.up7 = UNetUpResBlock(16, 8)
        self.up8 = UNetUpResBlock(8, 3)

        self.Dropout = nn.Dropout2d(p=dp_prob)
        self.last = nn.Conv2d(8, n_classes, 1, stride=1)

    def forward(self, x):
        res_x = x
        #         print 'line 70 ',x.size()
        b1 = self.down1(x)
        p1 = self.pool1(b1)
        p1_dp = self.Dropout(p1)

        b2 = self.down2(p1_dp)
        p2 = self.pool2(b2)
        p2_dp = self.Dropout(p2)

        b3 = self.down3(p2_dp)
        p3 = self.pool3(b3)
        p3_dp = self.Dropout(p3)

        b4 = self.down4(p3_dp)
        p4 = self.pool4(b4)
        p4_dp = self.Dropout(p4)

        b5 = self.down5(p4_dp)
        p5 = self.pool5(b5)
        p5_dp = self.Dropout(p5)

        b6 = self.down6(p5_dp)
        p6 = self.pool6(b6)
        p6_dp = self.Dropout(p6)

        b7 = self.down7(p6_dp)
        p7 = self.pool7(b7)
        p7_dp = self.Dropout(p7)

        b8 = self.down8(p7_dp)
        # p8 = self.pool8(b8)
        # p8_dp = self.Dropout(p8)


        up1 = self.up1(b8, b7)
        up2 = self.up2(up1, b6)
        up3 = self.up3(up2, b5)
        up4 = self.up4(up3, b4)
        up5 = self.up5(up4, b3)
        up6 = self.up6(up5, b2)
        up7 = self.up7(up6, b1)
        # up8 = self.up8(up7, res_x)

        last = self.last(up7)
        # print 'res_x.shape is ',res_x.shape,' and last.shape is ',last.shape
        if len(res_x.shape) == 3:
            res_x = res_x.unsqueeze(1)
        out = torch.add(last, res_x)

        # print 'out.shape is ',out.shape
        return out


'''
    Discriminator for the reconstruction project
'''
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        #you can make abbreviations for conv and fc, this is not necessary
        #class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(1,32,(9,9))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,(5,5))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,(5,5))
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*4*4,512)
        #self.bn3= nn.BatchNorm1d(6)
        self.fc2 = nn.Linear(512,64)
        self.fc3 = nn.Linear(64,1)
        
        
    def forward(self,x):
#         print 'line 114: x shape: ',x.size()
        #x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))),(2,2))#conv->relu->pool
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))#conv->relu->pool

        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))#conv->relu->pool
        
        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))#conv->relu->pool
        
        #reshape them into Vector, review ruturned tensor shares the same data but have different shape, same as reshape in matlab
        x = x.view(-1,self.num_of_flat_features(x))
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        
        #x = F.sigmoid(x)
        #print 'min,max,mean of x in 0st layer',x.min(),x.max(),x.mean()
        return x
    
    def num_of_flat_features(self,x):
        size=x.size()[1:]#we donot consider the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features
    


def main():
    input = torch.ones((1, 3, 256, 256))
    model = ResUNet_LRes_new()
    # model = ResUNet_LRes()
    for i in range(100):
        out = model(input)
        print(out.shape)    


if __name__ == '__main__':
    main()