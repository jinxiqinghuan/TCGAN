# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin
from xml.etree.ElementPath import prepare_parent

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
# from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

from config import get_config
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head


        self.swin_unet = SwinTransformerSys(img_size=256, patch_size=4, in_chans=1, num_classes=1,
                embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                use_checkpoint=False, final_upsample="expand_first")


    def forward(self, x):
        # 
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)


        logits = self.swin_unet(x)
        return logits

    def load_from(self, pre_path):
        pretrained_path = pre_path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


def main():
    print("tesst")

    pre_path = "/home/gxzy/ljt/project/pet2ct/models/swin_unet/swin_tiny_patch4_window7_224.pth"
    net = SwinUnet(img_size=256, num_classes=1).cuda()
    net.load_from(pre_path)
    x = torch.ones((1, 1, 256, 256)).cuda()
    out = net(x)
    print(out.shape)


if __name__ == '__main__':
    main()
