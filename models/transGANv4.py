from hashlib import new
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary

try:
    from models.transGAN import TransformerEncoder, UpSampling
except:
    print("Change import path...")
    from transGAN import TransformerEncoder, UpSampling

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# 一个U-net测试 + 判别器

##############################
#           U-NET
##############################

class UNetDown(nn.Module):
    # in_size:3         out_size:64         normalize:False
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        # layers = [nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)]
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):

    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class L_Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L_Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(GeneratorUNet, self).__init__()

        # x: [1, 4, 256, 256]
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.up8 = UNetUp(128, 32)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self,x):
        # U-Net generator with skip connections from encoder to decoder
        # x [1， 3， 256， 256]
        # d1 [1， 64， 128， 128]
        d1 = self.down1(x)
        # d2 [1， 128， 64， 64]
        d2 = self.down2(d1)
        # d3 [1， 256， 32， 32]
        d3 = self.down3(d2)
        # d4 [1， 512， 16， 16]
        d4 = self.down4(d3)
        # d5 [1， 512， 8， 8]
        d5 = self.down5(d4)
        # d6 [1， 512， 4， 4]
        d6 = self.down6(d5)
        # d7 [1， 512， 2， 2]
        d7 = self.down7(d6)
        # d8 [1， 512， 1， 1]
        d8 = self.down8(d7)
        # u1 [1， 1024， 2， 2]
        u1 = self.up1(d8, d7)
        # u2 [1， 1024， 4， 4]
        u2 = self.up2(u1, d6)
        # u3 [1， 1024， 8， 8]
        u3 = self.up3(u2, d5)
        # u4 [1， 1024，16， 16]
        u4 = self.up4(u3, d4)
        # u5 [1， 512， 32， 32]
        u5 = self.up5(u4, d3)
        # u6 [1， 256， 64， 64]
        u6 = self.up6(u5, d2)
        # u7 [1， 128， 128， 128]
        u7 = self.up7(u6, d1)

        # [1, 3, 256, 256]
        return self.final(u7)

class TransGeneratorUNetV4(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth1=5, depth2=4, depth3=2, initial_size=8, dim=384, heads=4, mlp_ratio=4, drop_rate=0.2):
        super(TransGeneratorUNetV4, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # x: [1, 4, 256, 256]
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.up8 = UNetUp(128, 32)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

        # TransGAN blocks

        self.initial_size = initial_size
        self.dim = dim
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate =drop_rate

        self.trans_down1 = UNetDown(1, 16)
        self.trans_down2 = UNetDown(16, 16)
        self.trans_down3 = UNetDown(16, 3)

        self.linear1 = nn.Linear(3072, 1024)
        self.mlp = nn.Linear(1024, (self.initial_size ** 2) * self.dim)

        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, (8**2), dim))
        self.positional_embedding_2 = nn.Parameter(torch.zeros(1, (8*2)**2, dim//4))
        self.positional_embedding_3 = nn.Parameter(torch.zeros(1, (8*4)**2, dim//16))

        self.TransformerEncoder_encoder1 = TransformerEncoder(depth=self.depth1, dim=self.dim,heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder2 = TransformerEncoder(depth=self.depth2, dim=self.dim//4, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder3 = TransformerEncoder(depth=self.depth3, dim=self.dim//16, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)


        self.linear = nn.Sequential(nn.Conv2d(self.dim//16, 3, 1, 1, 0))

        self.trans_up1 = UNetUp(3, 16)
        self.trans_up2 = UNetUp(32, 16)
        self.trans_up3 = UNetUp(32, 1)

        self.out_layer = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

    def forward(self,x):
        # U-Net generator with skip connections from encoder to decoder
        # x [1， 3， 256， 256]
        # d1 [1， 64， 128， 128]
        tmp_input = x
        
 
        t1 = self.trans_down1(x)
        t2 = self.trans_down2(t1)
        t3 = self.trans_down3(t2)

        x = t3.view(x.shape[0], self.in_channels, 3*32*32)
        x = self.linear1(x)
        x = self.mlp(x).view(-1, self.initial_size ** 2, self.dim)

        x = x + self.positional_embedding_1
        H, W = self.initial_size, self.initial_size
        x = self.TransformerEncoder_encoder1(x)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_2
        x = self.TransformerEncoder_encoder2(x)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_3
        x = self.TransformerEncoder_encoder3(x)
        
        x = self.linear(x.permute(0, 2, 1).view(-1, self.dim // 16, H, W))
        x = self.trans_up1(x, t2)
        x = self.trans_up2(x, t1)
        x = self.trans_up3(x, tmp_input)
        
        trans_out = self.out_layer(x)

        new_in = (trans_out + tmp_input) / 2

        d1 = self.down1(new_in)
        # d2 [1， 128， 64， 64]
        d2 = self.down2(d1)
        # d3 [1， 256， 32， 32]
        d3 = self.down3(d2)
        # d4 [1， 512， 16， 16]
        d4 = self.down4(d3)
        # d5 [1， 512， 8， 8]
        d5 = self.down5(d4)
        # d6 [1， 512， 4， 4]
        d6 = self.down6(d5)
        # d7 [1， 512， 2， 2]
        d7 = self.down7(d6)
        # d8 [1， 512， 1， 1]
        d8 = self.down8(d7)

        # u1 [1， 1024， 2， 2]
        u1 = self.up1(d8, d7)
        # u2 [1， 1024， 4， 4]
        u2 = self.up2(u1, d6)
        # u3 [1， 1024， 8， 8]
        u3 = self.up3(u2, d5)
        # u4 [1， 1024，16， 16]
        u4 = self.up4(u3, d4)
        # u5 [1， 512， 32， 32]
        u5 = self.up5(u4, d3)
        # u6 [1， 256， 64， 64]
        u6 = self.up6(u5, d2)
        # u7 [1， 128， 128， 128]
        u7 = self.up7(u6, d1)

        out = self.final(u7)

        ### Add the Original net and Trans net
        # out = self.out_layer(torch.cat((out1, out2), 1))
        return out

class TransGeneratorUNet_ABAB(nn.Module):
    def __init__(self, block_num = 2):
        super(TransGeneratorUNet_ABAB, self).__init__()
        self.AB1 = TransGeneratorUNetV4(in_channels=1, out_channels=1, depth1=5, depth2=4, depth3=2, initial_size=8, dim=1024, heads=4, mlp_ratio=4, drop_rate=0.2)
        self.AB2 = TransGeneratorUNetV4(in_channels=1, out_channels=1, depth1=5, depth2=4, depth3=2, initial_size=8, dim=1024, heads=4, mlp_ratio=4, drop_rate=0.2)

    
    def forward(self, x):
        original_input = x
        out = self.AB1(x)
        out = self.AB2((out + original_input) / 2)
        return out
    
class TransGeneratorUNet_ABABAB(nn.Module):
    def __init__(self, block_num = 2):
        super(TransGeneratorUNet_ABABAB, self).__init__()
        self.AB1 = TransGeneratorUNetV4(in_channels=1, out_channels=1, depth1=5, depth2=4, depth3=2, initial_size=8, dim=1024, heads=4, mlp_ratio=4, drop_rate=0.2)
        self.AB2 = TransGeneratorUNetV4(in_channels=1, out_channels=1, depth1=5, depth2=4, depth3=2, initial_size=8, dim=1024, heads=4, mlp_ratio=4, drop_rate=0.2)
        self.AB3 = TransGeneratorUNetV4(in_channels=1, out_channels=1, depth1=5, depth2=4, depth3=2, initial_size=8, dim=1024, heads=4, mlp_ratio=4, drop_rate=0.2)

    
    def forward(self, x):
        original_input = x
        out1 = self.AB1(x)
        out2 = self.AB2((out1 + original_input) / 2)
        out3 = self.AB3((original_input + out1 + out2) / 3)
        return out3

class TransGeneratorUnet_AAB(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth1=5, depth2=4, depth3=2, initial_size=8, dim=384, heads=4, mlp_ratio=4, drop_rate=0.2):
        super(TransGeneratorUnet_AAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # x: [1, 4, 256, 256]
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.up8 = UNetUp(128, 32)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

        # TransGAN blocks

        self.initial_size = initial_size
        self.dim = dim
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate =drop_rate

        self.trans_down1 = UNetDown(1, 16)
        self.trans_down2 = UNetDown(16, 16)
        self.trans_down3 = UNetDown(16, 3)

        self.linear1 = nn.Linear(3072, 1024)
        self.mlp = nn.Linear(1024, (self.initial_size ** 2) * self.dim)

        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, (8**2), dim))
        self.positional_embedding_2 = nn.Parameter(torch.zeros(1, (8*2)**2, dim//4))
        self.positional_embedding_3 = nn.Parameter(torch.zeros(1, (8*4)**2, dim//16))

        self.TransformerEncoder_encoder1 = TransformerEncoder(depth=self.depth1, dim=self.dim,heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder2 = TransformerEncoder(depth=self.depth2, dim=self.dim//4, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder3 = TransformerEncoder(depth=self.depth3, dim=self.dim//16, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)


        self.linear = nn.Sequential(nn.Conv2d(self.dim//16, 3, 1, 1, 0))

        self.trans_up1 = UNetUp(3, 16)
        self.trans_up2 = UNetUp(32, 16)
        self.trans_up3 = UNetUp(32, 1)

        self.out_layer = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

    def forward(self,x):
        # U-Net generator with skip connections from encoder to decoder
        # x [1， 3， 256， 256]
        # d1 [1， 64， 128， 128]
        tmp_input = x
        
 ######## The first A
        t1 = self.trans_down1(x)
        t2 = self.trans_down2(t1)
        t3 = self.trans_down3(t2)

        x = t3.view(x.shape[0], self.in_channels, 3*32*32)
        x = self.linear1(x)
        x = self.mlp(x).view(-1, self.initial_size ** 2, self.dim)

        x = x + self.positional_embedding_1
        H, W = self.initial_size, self.initial_size
        x = self.TransformerEncoder_encoder1(x)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_2
        x = self.TransformerEncoder_encoder2(x)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_3
        x = self.TransformerEncoder_encoder3(x)
        
        x = self.linear(x.permute(0, 2, 1).view(-1, self.dim // 16, H, W))
        x = self.trans_up1(x, t2)
        x = self.trans_up2(x, t1)
        x = self.trans_up3(x, tmp_input)
        
        trans_out = self.out_layer(x)

        

 ######## The second A
        new_in = (trans_out + tmp_input) / 2

        a2t1 = self.trans_down1(new_in)
        a2t2 = self.trans_down2(a2t1)
        a2t3 = self.trans_down3(a2t2)

        x = a2t3.view(new_in.shape[0], self.in_channels, 3*32*32)
        x = self.linear1(x)
        x = self.mlp(x).view(-1, self.initial_size ** 2, self.dim)

        x = x + self.positional_embedding_1
        H, W = self.initial_size, self.initial_size
        x = self.TransformerEncoder_encoder1(x)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_2
        x = self.TransformerEncoder_encoder2(x)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_3
        x = self.TransformerEncoder_encoder3(x)
        
        x = self.linear(x.permute(0, 2, 1).view(-1, self.dim // 16, H, W))
        x = self.trans_up1(x, a2t2)
        x = self.trans_up2(x, a2t1)
        x = self.trans_up3(x, tmp_input)
        
        trans_out = self.out_layer(x)


        new_in = (trans_out + new_in) / 2


 ######## B


        d1 = self.down1(new_in)
        # d2 [1， 128， 64， 64]
        d2 = self.down2(d1)
        # d3 [1， 256， 32， 32]
        d3 = self.down3(d2)
        # d4 [1， 512， 16， 16]
        d4 = self.down4(d3)
        # d5 [1， 512， 8， 8]
        d5 = self.down5(d4)
        # d6 [1， 512， 4， 4]
        d6 = self.down6(d5)
        # d7 [1， 512， 2， 2]
        d7 = self.down7(d6)
        # d8 [1， 512， 1， 1]
        d8 = self.down8(d7)

        # u1 [1， 1024， 2， 2]
        u1 = self.up1(d8, d7)
        # u2 [1， 1024， 4， 4]
        u2 = self.up2(u1, d6)
        # u3 [1， 1024， 8， 8]
        u3 = self.up3(u2, d5)
        # u4 [1， 1024，16， 16]
        u4 = self.up4(u3, d4)
        # u5 [1， 512， 32， 32]
        u5 = self.up5(u4, d3)
        # u6 [1， 256， 64， 64]
        u6 = self.up6(u5, d2)
        # u7 [1， 128， 128， 128]
        u7 = self.up7(u6, d1)

        out = self.final(u7)

        ### Add the Original net and Trans net
        # out = self.out_layer(torch.cat((out1, out2), 1))
        return out

class TransGeneratorUnet_ABB(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth1=5, depth2=4, depth3=2, initial_size=8, dim=384, heads=4, mlp_ratio=4, drop_rate=0.2):
        super(TransGeneratorUnet_ABB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # x: [1, 4, 256, 256]
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.up8 = UNetUp(128, 32)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

        # TransGAN blocks

        self.initial_size = initial_size
        self.dim = dim
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate =drop_rate

        self.trans_down1 = UNetDown(1, 16)
        self.trans_down2 = UNetDown(16, 16)
        self.trans_down3 = UNetDown(16, 3)

        self.linear1 = nn.Linear(3072, 1024)
        self.mlp = nn.Linear(1024, (self.initial_size ** 2) * self.dim)

        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, (8**2), dim))
        self.positional_embedding_2 = nn.Parameter(torch.zeros(1, (8*2)**2, dim//4))
        self.positional_embedding_3 = nn.Parameter(torch.zeros(1, (8*4)**2, dim//16))

        self.TransformerEncoder_encoder1 = TransformerEncoder(depth=self.depth1, dim=self.dim,heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder2 = TransformerEncoder(depth=self.depth2, dim=self.dim//4, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder3 = TransformerEncoder(depth=self.depth3, dim=self.dim//16, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)


        self.linear = nn.Sequential(nn.Conv2d(self.dim//16, 3, 1, 1, 0))

        self.trans_up1 = UNetUp(3, 16)
        self.trans_up2 = UNetUp(32, 16)
        self.trans_up3 = UNetUp(32, 1)

        self.out_layer = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.B2 = GeneratorUNet()



    def forward(self,x):
        # print("hello")
        # U-Net generator with skip connections from encoder to decoder
        # x [1， 3， 256， 256]
        # d1 [1， 64， 128， 128]
        tmp_input = x
        
 ######## The first A
        t1 = self.trans_down1(x)
        t2 = self.trans_down2(t1)
        t3 = self.trans_down3(t2)

        x = t3.view(x.shape[0], self.in_channels, 3*32*32)
        x = self.linear1(x)
        x = self.mlp(x).view(-1, self.initial_size ** 2, self.dim)

        x = x + self.positional_embedding_1
        H, W = self.initial_size, self.initial_size
        x = self.TransformerEncoder_encoder1(x)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_2
        x = self.TransformerEncoder_encoder2(x)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_3
        x = self.TransformerEncoder_encoder3(x)
        
        x = self.linear(x.permute(0, 2, 1).view(-1, self.dim // 16, H, W))
        x = self.trans_up1(x, t2)
        x = self.trans_up2(x, t1)
        x = self.trans_up3(x, tmp_input)
        
        trans_out = self.out_layer(x)

        new_in = (trans_out + tmp_input) / 2

        d1 = self.down1(new_in)
        # d2 [1， 128， 64， 64]
        d2 = self.down2(d1)
        # d3 [1， 256， 32， 32]
        d3 = self.down3(d2)
        # d4 [1， 512， 16， 16]
        d4 = self.down4(d3)
        # d5 [1， 512， 8， 8]
        d5 = self.down5(d4)
        # d6 [1， 512， 4， 4]
        d6 = self.down6(d5)
        # d7 [1， 512， 2， 2]
        d7 = self.down7(d6)
        # d8 [1， 512， 1， 1]
        d8 = self.down8(d7)

        # u1 [1， 1024， 2， 2]
        u1 = self.up1(d8, d7)
        # u2 [1， 1024， 4， 4]
        u2 = self.up2(u1, d6)
        # u3 [1， 1024， 8， 8]
        u3 = self.up3(u2, d5)
        # u4 [1， 1024，16， 16]
        u4 = self.up4(u3, d4)
        # u5 [1， 512， 32， 32]
        u5 = self.up5(u4, d3)
        # u6 [1， 256， 64， 64]
        u6 = self.up6(u5, d2)
        # u7 [1， 128， 128， 128]
        u7 = self.up7(u6, d1)

        out = self.final(u7)

        new_in = (out + new_in) / 2

        out = self.B2(new_in)

        ### Add the Original net and Trans net
        # out = self.out_layer(torch.cat((out1, out2), 1))
        return out
##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

if __name__ == '__main__':
    model = TransGeneratorUnet_AAB(in_channels=1, out_channels=1, depth1=5, depth2=4, depth3=2, initial_size=8, dim=1024, heads=4, mlp_ratio=4, drop_rate=0.2)#,device = device)
    summary(model=model, input_size=(1, 256, 256), batch_size=1, device='cpu')
    x = torch.ones(2, 1, 256, 256)
    img = model(x)
    print(img.shape)