# Windows 10 Anconda python3.8
# coding=utf-8
'''
Author       : LiAo
Date         : 2022-07-10 23:36:45
LastEditTime : 2022-07-11 00:32:10
LastAuthor   : LiAo
Description  : Please add file description
'''
import torch
import torch.nn as nn
import torch.nn.functional as functional
import timm


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    """
    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
        self, in_chs, rd_ratio=0.25, rd_channels=None, act_layer=nn.SiLU,
        gate_layer=nn.Sigmoid, force_act_layer=None, rd_round_fn=None
    ):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_channel_fn = rd_round_fn or round
            rd_channels = rd_channel_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(
            in_chs, rd_channels, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(
            rd_channels, in_chs, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.gate = gate_layer()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class ConvBnAct(nn.Module):
    """ Conv + Norm Layer + Activation
        if act_layer == None, no Activation
    """

    def __init__(
            self, in_chs, out_chs, kernel_size, stride=1, padding=0,
            dilation=1, groups=1, bias=False,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.bn = norm_layer(out_chs, eps=0.0001, momentum=0.01, affine=True,
                             track_running_stats=True) if norm_layer else nn.Identity()
        self.act = act_layer(inplace=True) if act_layer else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

# Begin   Resizing_network


class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        return out


def make_block(r, n):
    residual = []
    for i in range(r):
        block = ResBlock(num_channels=n)
        residual.append(block)
    return nn.Sequential(*residual)


class ResizingNetwork(nn.Module):
    def __init__(self, r=4, n=32, target_size=(300, 300)):
        """
        Args:
            r 残差块的个数
            n channel数, 通道数
            target_size, 网络输出的size
        """
        super(ResizingNetwork, self).__init__()

        self.target_size = target_size

        self.conv1 = nn.Conv2d(1, n, kernel_size=7, stride=1, padding='same')
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(n, n, kernel_size=1, stride=1, padding='same')
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(n)

        self.resblocks = make_block(r, n)

        self.conv3 = nn.Conv2d(n, n, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(n)

        self.conv4 = nn.Conv2d(
            n, out_channels=3, kernel_size=7, stride=1, padding='same')

    def forward(self, in_tensor):
        residual = functional.interpolate(
            in_tensor, size=self.target_size, mode='bilinear')

        out = self.conv1(in_tensor)
        out = self.leakyrelu1(out)

        out = self.conv2(out)
        out = self.leakyrelu2(out)
        out = self.bn1(out)

        out_residual = functional.interpolate(
            out, size=self.target_size, mode='bilinear')

        out = self.resblocks(out_residual)

        out = self.conv3(out)
        out = self.bn2(out)
        out += out_residual

        out = self.conv4(out)
        out += residual

        return out

# End   Resizing_network


class MultiClassification(nn.Module):
    def __init__(self, backbone='tf_efficientnetv2_b0', pretrain=True, num_classes=7, pool=False,
                 pool_size=(300, 300), pool_type='max'):
        super(MultiClassification, self).__init__()
        self.resizer = ResizingNetwork(target_size=pool_size)
        self.backbone = timm.create_model(
            backbone, pretrained=pretrain, num_classes=num_classes)
        if pool and pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(pool_size)
        elif pool and pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(pool_size)
        elif pool and pool_type in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']:
            self.pool = nn.Upsample(size=pool_size, mode=pool_type)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        x = self.resizer(x)
        x = self.pool(x)
        x = self.backbone(x)
        return x


def test_resizer_net():
    img = torch.randn((1, 1, 2148, 3692))
    net = ResizingNetwork()
    res = net(img)
    print(res.shape)
