import torch
import torch.nn as nn
import os
from torchsummary import summary
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

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
        # print(x.shape)
        x = torch.cat((x, skip_input), 1)
        return x



class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None,
                 dropout=0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_feat, out_feat)
        self.droprateout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.droprateout(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = 1./dim**0.5

        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_dropout)
        )

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c//self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        dot = (q @ k.transpose(-2, -1)) * self.scale
        attn = dot.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out(x)
        return x

class ImgPatches(nn.Module):
    def __init__(self, input_channel=3, dim=768, patch_size=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(input_channel, dim,
                                     kernel_size=patch_size, stride=patch_size)

    def forward(self, img):
        patches = self.patch_embed(img).flatten(2).transpose(1, 2)
        return patches

def UpSampling(x, H, W):
        B, N, C = x.size()
        assert N == H*W
        x = x.permute(0, 2, 1)
        x = x.view(-1, C, H, W)
        x = nn.PixelShuffle(2)(x)
        B, C, H, W = x.size()
        x = x.view(-1, C, H*W)
        x = x.permute(0,2,1)
        return x, H, W

class Encoder_Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop_rate, drop_rate)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim*mlp_ratio, dropout=drop_rate)

    def forward(self, x):
        x1 = self.ln1(x)
        x = x + self.attn(x1)
        x2 = self.ln2(x)
        x = x + self.mlp(x2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.Encoder_Blocks = nn.ModuleList([
            Encoder_Block(dim, heads, mlp_ratio, drop_rate)
            for i in range(depth)])

    def forward(self, x):
        for Encoder_Block in self.Encoder_Blocks:
            x = Encoder_Block(x)
        return x

class TransGeneratorV2(nn.Module):
    """docstring for Generator"""
    def __init__(self, depth1=5, depth2=4, depth3=2, initial_size=8, dim=1024, heads=8, mlp_ratio=4, drop_rate=0.):#,device=device):
        super(TransGeneratorV2, self).__init__()

        #self.device = device
        self.initial_size = initial_size
        self.dim = dim
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate =drop_rate


        self.down1 = UNetDown(1, 32)
        self.down2 = UNetDown(32, 32)
        self.down3 = UNetDown(32, 3)
        # self.down4 = UNetDown(8, 3)


        self.linear1 = nn.Linear(3072, 1024)
        self.mlp = nn.Linear(1024, (self.initial_size ** 2) * self.dim)

        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, (8**2), 384))
        self.positional_embedding_2 = nn.Parameter(torch.zeros(1, (8*2)**2, 384//4))
        self.positional_embedding_3 = nn.Parameter(torch.zeros(1, (8*4)**2, 384//16))

        self.TransformerEncoder_encoder1 = TransformerEncoder(depth=self.depth1, dim=self.dim,heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder2 = TransformerEncoder(depth=self.depth2, dim=self.dim//4, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder3 = TransformerEncoder(depth=self.depth3, dim=self.dim//16, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)


        self.linear = nn.Sequential(nn.Conv2d(self.dim//16, 3, 1, 1, 0))

        self.up1 = UNetUp(3, 32)
        self.up2 = UNetUp(64, 32)
        self.up3 = UNetUp(64, 1)

        self.out_layer = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, x): # torch.Size([2, 1, 256, 256])
        input = x
        x1 = self.down1(x) # torch.Size([2, 8, 128, 128])
        x2 = self.down2(x1) #　torch.Size([2, 8, 64, 64])
        x3 = self.down3(x2) #  torch.Size([2, 3, 32, 32])
        x = x3

        x = x.view(x.shape[0], 1, 3 * 32 * 32)
        x = self.linear1(x)

        x = self.mlp(x).view(-1, self.initial_size ** 2, self.dim)

        x = x + self.positional_embedding_1
        H, W = self.initial_size, self.initial_size
        x = self.TransformerEncoder_encoder1(x) # x [8, 64, 384]

        x,H,W = UpSampling(x,H,W) 
        x = x + self.positional_embedding_2
        x = self.TransformerEncoder_encoder2(x) # [8, 256, 96]

        x,H,W = UpSampling(x,H,W)
        x = x + self.positional_embedding_3

        x = self.TransformerEncoder_encoder3(x)
        # print(x.shape)
        x = self.linear(x.permute(0, 2, 1).view(-1, self.dim//16, H, W)) # torch.Size([2, 3, 32, 32])

        x = self.up1(x, x2) # torch.Size([2, 8, 64, 64])
        x = self.up2(x, x1) # torch.Size([2, 8, 128, 128])
        x = self.up3(x, input) #torch.Size([2, 1, 256, 256])
        x = self.out_layer(x)
        return x


def main():
    # input = torch.ones(1, 3, 32, 32)
    model = TransGeneratorV2(depth1=5, depth2=4, depth3=2, initial_size=8, dim=384, heads=4, mlp_ratio=4, drop_rate=0.5)#,device = device)
    summary(model=model, input_size=(1, 256, 256), batch_size=2, device="cpu")
    # out = model(input)
    # print(out.shape)


if __name__ == '__main__':
    main()