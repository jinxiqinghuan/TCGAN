import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.diff_aug import DiffAugment
except:
    print("Change import path...")
    from diff_aug import DiffAugment
from torchsummary import summary
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None,
                 dropout=0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        # self.act = nn.GELU()
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

class Generator(nn.Module):
    """docstring for Generator"""
    def __init__(self, depth1=5, depth2=4, depth3=2, initial_size=8, dim=384, heads=4, mlp_ratio=4, drop_rate=0.):#,device=device):
        super(Generator, self).__init__()

        #self.device = device
        self.initial_size = initial_size
        self.dim = dim
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate =drop_rate

        self.mlp = nn.Linear(1024, (self.initial_size ** 2) * self.dim)

        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, (8**2), 384))
        self.positional_embedding_2 = nn.Parameter(torch.zeros(1, (8*2)**2, 384//4))
        self.positional_embedding_3 = nn.Parameter(torch.zeros(1, (8*4)**2, 384//16))

        self.TransformerEncoder_encoder1 = TransformerEncoder(depth=self.depth1, dim=self.dim,heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder2 = TransformerEncoder(depth=self.depth2, dim=self.dim//4, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder3 = TransformerEncoder(depth=self.depth3, dim=self.dim//16, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)


        self.linear = nn.Sequential(nn.Conv2d(self.dim//16, 3, 1, 1, 0))

    def forward(self, noise):

        x = self.mlp(noise).view(-1, self.initial_size ** 2, self.dim)

        x = x + self.positional_embedding_1
        H, W = self.initial_size, self.initial_size
        x = self.TransformerEncoder_encoder1(x) # x [8, 64, 384]

        x,H,W = UpSampling(x,H,W) 
        x = x + self.positional_embedding_2
        x = self.TransformerEncoder_encoder2(x) # [8, 256, 96]

        x,H,W = UpSampling(x,H,W)
        x = x + self.positional_embedding_3

        x = self.TransformerEncoder_encoder3(x)
        x = self.linear(x.permute(0, 2, 1).view(-1, self.dim//16, H, W))

        return x

class Down_Block(nn.Module):
    # in_size:3         out_size:64         normalize:False
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(Down_Block, self).__init__()
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

class Up_Block(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(Up_Block, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x

class TransGenerator(nn.Module):
    # Define the trans generator for pet2ct task
    def __init__(self, depth1=5, depth2=4, depth3=2, initial_size=8, dim=4096, heads=4, mlp_ratio=4, drop_rate=0):
        super(TransGenerator, self).__init__()

        # The input size 256 is too large, so we add the down sample layers. Define as below.
        # But use two down_layers make the input size down to 64, we loss many information.
        self.down_layers1 = Down_Block(1, 16, normalize=True, dropout=0.)
        self.down_layers2 = Down_Block(16, 64, normalize=True, dropout=0.)

        self.dim = dim
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate = drop_rate

        # self.mlp = nn.Linear(1024, (self.initial_size ** 2) * self.dim)
        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, 1, 64*64))
        self.positional_embedding_2 = nn.Parameter(torch.zeros(1, (8*2)**2, (64*64)//4))
        self.positional_embedding_3 = nn.Parameter(torch.zeros(1, (8*4)**2, (64*64)//16))

        self.TransformerEncoder_encoder1 = TransformerEncoder(depth=self.depth1, dim=self.dim,heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder2 = TransformerEncoder(depth=self.depth2, dim=self.dim//4, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder3 = TransformerEncoder(depth=self.depth3, dim=self.dim//16, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)


        self.linear = nn.Sequential(nn.Conv2d(self.dim // 16, 128, 1, 1, 0))

        self.up_layers1 = Up_Block(128, 64)
        self.up_layers2 = Up_Block(64, 32)
        self.up_layers3 = Up_Block(32, 1)



    def forward(self, x):
        x = self.down_layers1(x)
        x = self.down_layers2(x)
        # print(x.shape)
        x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])  # [batch_size, chanel, H*W]
        x = x + self.positional_embedding_1
        H, W = 8,8
        x = self.TransformerEncoder_encoder1(x)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_2
        x = self.TransformerEncoder_encoder2(x)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_3
        x = self.TransformerEncoder_encoder3(x)
        x = self.linear(x.permute(0, 2, 1).view(-1, self.dim//16, H, W))

        x = self.up_layers1(x)
        x = self.up_layers2(x)
        x = self.up_layers3(x)


        return x

class TransGenerator2(nn.Module):
    # Define the trans generator for pet2ct task
    def __init__(self, depth1=1, depth2=1, depth3=1, initial_size=8, dim=256, heads=1, mlp_ratio=4, drop_rate=0):
        super(TransGenerator2, self).__init__()

        # The input size 256 is too large, so we add the down sample layers. Define as below.
        # But use two down_layers make the input size down to 64, we loss many information.
        self.down_layers1 = Down_Block(1, 8, normalize=True, dropout=0.)
        self.down_layers2 = Down_Block(8, 16, normalize=True, dropout=0.)
        self.down_layers3 = Down_Block(16, 32, normalize=True, dropout=0.)
        self.down_layers4 = Down_Block(32, 64, normalize=True, dropout=0.)


        self.dim = dim
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate = drop_rate

        # self.mlp = nn.Linear(1024, (self.initial_size ** 2) * self.dim)
        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, 1, 32*32))
        self.positional_embedding_2 = nn.Parameter(torch.zeros(1, (8*2)**2, dim//4))
        self.positional_embedding_3 = nn.Parameter(torch.zeros(1, (8*4)**2, dim//16))

        self.TransformerEncoder_encoder1 = TransformerEncoder(depth=self.depth1, dim=self.dim,heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder2 = TransformerEncoder(depth=self.depth2, dim=self.dim//4, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder3 = TransformerEncoder(depth=self.depth3, dim=self.dim//16, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)


        self.linear = nn.Sequential(nn.Conv2d(self.dim // 16, 128, 1, 1, 0))

        self.up_layers1 = Up_Block(64, 32)
        self.up_layers2 = Up_Block(32, 16)
        self.up_layers3 = Up_Block(16, 8)
        self.up_layers4 = Up_Block(8, 1)




    def forward(self, x): # [1, 1, 256, 256]
        x = self.down_layers1(x) # [1, 8, 128, 128]
        x = self.down_layers2(x) # torch.Size([1, 16, 64, 64])
        x = self.down_layers3(x) # torch.Size([1, 32, 32, 32])
        x = self.down_layers4(x) # torch.Size([1, 64, 16, 16])
        # print(x.shape)
        x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])  # [batch_size, chanel, H*W] torch.Size([1, 64, 256])
        # x = x + self.positional_embedding_1
        H, W = 8,8
        x = self.TransformerEncoder_encoder1(x) # torch.Size([1, 64, 256])
        x = x.view(x.shape[0], 64, 16, 16)

        x = self.up_layers1(x)
        x = self.up_layers2(x)
        x = self.up_layers3(x)
        x = self.up_layers4(x)

        return x

class TransDiscriminator(nn.Module):
    def __init__(self, diff_aug=None, image_size=256, patch_size=4, input_channel=2, num_classes=1,
                 dim=384, depth=7, heads=4, mlp_ratio=4,
                 drop_rate=0.):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError('Image size must be divisible by patch size.')
        num_patches = (image_size//patch_size) ** 2
        self.diff_aug = diff_aug
        self.patch_size = patch_size
        self.depth = depth
        # Image patches and embedding layer
        self.patches = ImgPatches(input_channel, dim, self.patch_size)

        # Embedding for patch position and class
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches+1, dim))
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.positional_embedding, std=0.2)
        nn.init.trunc_normal_(self.class_embedding, std=0.2)

        self.droprate = nn.Dropout(p=drop_rate)
        self.TransfomerEncoder = TransformerEncoder(depth, dim, heads,
                                      mlp_ratio, drop_rate)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, img_a, img_b):
        # x = DiffAugment(x, self.diff_aug)
        x = torch.cat((img_a, img_b), 1)
        b = x.shape[0]
        cls_token = self.class_embedding.expand(b, -1, -1)

        x = self.patches(x)
        x = torch.cat((cls_token, x), dim=1)
        x += self.positional_embedding
        x = self.droprate(x)
        x = self.TransfomerEncoder(x)
        x = self.norm(x)
        x = self.out(x[:, 0])
        return x


def main():
    x = torch.ones((2, 1, 256, 256))
    model = TransDiscriminator()
    # summary(model, input_size=[(1, 256, 256)], batch_size=64, device='cuda')
    out = model(x, x)
    print(out.shape)


    # for i in range(1000):
    #     out = model(x)
    #     print(out.shape)


if __name__ == '__main__':
    main()

