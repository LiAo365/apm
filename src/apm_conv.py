# Windows 10 Anconda python3.8
# coding=utf-8
'''
Author       : LiAo
Date         : 2022-07-05 19:45:12
LastEditTime : 2022-07-06 23:34:22
LastAuthor   : LiAo
Description  : Please add file description
'''
import torch
import torch.nn as nn
import timm
from src.base_module import ConvBlock


class APMConv(nn.Module):

    def __init__(self, in_chs: int, out_chs: int):
        super(APMConv, self).__init__()
        self.conv_blocks_1 = ConvBlock(in_chs, 4, kernel_size=13, stride=2, padding=1, dilation=1,
                                       norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU, pool_layer=nn.MaxPool2d(3, stride=1))
        self.conv_blocks_2 = ConvBlock(4, 16, kernel_size=5, stride=2, padding=1, dilation=1,
                                       norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU, pool_layer=nn.MaxPool2d(3, stride=1))
        self.conv_blocks_3 = ConvBlock(16, 32, kernel_size=3, stride=2, padding=1, dilation=1,
                                       norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU, pool_layer=nn.MaxPool2d(3, stride=1))
        self.res_blocks_1 = ConvBlock(32, 32, kernel_size=3, stride=1, padding=1, dilation=1,
                                      norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU, pool_layer=nn.MaxPool2d(3, stride=1))
        self.res_blocks_2 = ConvBlock(32, 32, kernel_size=3, stride=1, padding=1, dilation=1,
                                      norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU, pool_layer=nn.MaxPool2d(3, stride=1))
        self.conv_blocks_4 = ConvBlock(32, 8, kernel_size=3, stride=1, padding=1, dilation=1,
                                       norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU, pool_layer=nn.MaxPool2d(3, stride=1))
        self.conv_blocks_5 = ConvBlock(8, out_chs, kernel_size=3, stride=1, padding=1, dilation=1,
                                       norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU, pool_layer=nn.MaxPool2d(3, stride=1))

    def forward(self, x):
        x = self.conv_blocks_1(x)
        x = self.conv_blocks_2(x)
        x = self.conv_blocks_3(x)
        x = self.res_blocks_1(x)
        x = self.res_blocks_2(x)
        x = self.conv_blocks_4(x)
        x = self.conv_blocks_5(x)
        return x


class MultiClassification(nn.Module):
    def __init__(self, backbone='tf_efficientnetv2_b0', pretrain=True, num_classes=7, pool=False,
                 pool_size=(300, 300), pool_type='max'):
        super(MultiClassification, self).__init__()

        self.apm = APMConv(1, 3)
        self.backbone = timm.create_model(
            backbone, pretrained=pretrain, num_classes=num_classes)
        self.pool = nn.Identity()
        if pool and pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(pool_size)
        elif pool and pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(pool_size)
        elif pool and pool_type in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']:
            self.pool = nn.Upsample(size=pool_size, mode=pool_type)

    def forward(self, x):
        x = self.apm(x)
        x = self.pool(x)
        x = self.backbone(x)
        return x


def test_apm_conv():
    img = torch.randn((1, 1, 2148, 3692))
    apm = APMConv(1, 3)
    apm_img = apm(img)
    print("After APM")
    print(apm_img.shape)
    classifier = MultiClassification()
    res = classifier(img)
    print(res)
    print(res.shape)
