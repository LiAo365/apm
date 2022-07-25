# Windows 10 Anconda python3.8
# coding=utf-8
'''
Author       : LiAo
Date         : 2022-07-05 19:45:12
LastEditTime : 2022-07-25 11:21:53
LastAuthor   : LiAo
Description  : Please add file description
'''
import torch
import torch.nn as nn
import timm
from util import base_block


class APM(nn.Module):

    def __init__(self, in_chs: int, out_chs: int):
        super(APM, self).__init__()
        self.conv_blocks_1 = base_block.ConvBlock(in_chs, 4, kernel_size=13, stride=2, padding=1, dilation=1,
                                                  norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU, pool_layer=None)
        self.conv_blocks_2 = base_block.ConvBlock(4, 16, kernel_size=7, stride=2, padding=1, dilation=1,
                                                  norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU, pool_layer=None)
        self.conv_blocks_3 = base_block.ConvBlock(16, 32, kernel_size=5, stride=2, padding=1, dilation=1,
                                                  norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU, pool_layer=None)
        self.res_blocks_1 = base_block.Residual(32, 32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1],
                                                downsample=False)
        self.res_blocks_2 = base_block.Residual(32, 32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1],
                                                downsample=False)
        self.cbam_blocks = base_block.CBAM(32)
        self.conv_blocks_4 = base_block.ConvBlock(32, 8, kernel_size=3, stride=1, padding=1, dilation=1,
                                                  norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU, pool_layer=None)
        self.conv_blocks_5 = base_block.ConvBlock(8, out_chs, kernel_size=3, stride=1, padding=1, dilation=1,
                                                  norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU, pool_layer=None)

    def forward(self, x):
        x = self.conv_blocks_1(x)
        x = self.conv_blocks_2(x)
        x = self.conv_blocks_3(x)
        x = self.res_blocks_1(x)
        x = self.res_blocks_2(x)
        x = self.cbam_blocks(x)
        x = self.conv_blocks_4(x)
        x = self.conv_blocks_5(x)
        return x


class MultiClassification(nn.Module):
    def __init__(self, backbone='tf_efficientnetv2_b3', pretrain=True, num_classes=7, pool=True,
                 pool_size=(300, 300), pool_type='bilinear', drop_rate=0.5):
        super(MultiClassification, self).__init__()

        self.apm = APM(1, 3)
        self.backbone = timm.create_model(
            backbone, pretrained=pretrain, num_classes=num_classes)
        self.pool = nn.Identity()
        if pool and pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(pool_size)
        elif pool and pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(pool_size)
        elif pool and pool_type in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']:
            self.pool = nn.Upsample(size=pool_size, mode=pool_type)
        self.classifier = self.backbone.classifier
        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.apm(x)
        x = self.pool(x)
        x = self.backbone(x)
        x = self.dropout(self.classifier(x))
        return x


def test_apm():
    img = torch.randn((1, 1, 2148, 3692))
    apm = APM(1, 3)
    apm_img = apm(img)
    print("After APM")
    print(apm_img.shape)
    classifier = MultiClassification()
    res = classifier(img)
    print(res)
    print(res.shape)
