# Windows 10 Anconda python3.8
# coding=utf-8
'''
Author       : LiAo
Date         : 2022-07-05 19:45:12
LastEditTime : 2022-07-06 11:34:37
LastAuthor   : LiAo
Description  : Please add file description
'''
import torch
import torch.nn as nn
import torch.nn.functional as functional
import timm


class ConvBlock(nn.Module):
    """ Conv + Norm Layer + Activation + Pool
        if norm_layer == None no norm only support nn.BatchNorm2d
        if act_layer == None, no Activation only support Relu
        if pool_layer == None, no Pool
    """

    def __init__(
            self, in_chs, out_chs, kernel_size, stride=1, padding=0,
            dilation=1, groups=1, bias=False, norm_layer=nn.BatchNorm2d,
            act_layer=None, pool_layer=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.norm_layer = norm_layer(out_chs, eps=0.0001, momentum=0.01, affine=True, track_running_stats=True) if isinstance(
            norm_layer, nn.BatchNorm2d) else nn.Identity()
        self.act_layer = act_layer(
            inplace=True) if act_layer else nn.Identity()
        self.pool_layer = pool_layer if pool_layer else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm_layer(x)
        x = self.act_layer(x)
        x = self.pool_layer(x)
        return x


class Residual(nn.Module):
    """
    残差块
    the upsampling algorithm: one of ``'nearest'``,
    ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
    """

    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], norm_layer=[nn.BatchNorm2d, nn.BatchNorm2d],
                 act_layer=[nn.LeakyReLU, nn.LeakyReLU], pool_layer=[None, None], downsample=True, mode='bicubic'):
        super(Residual, self).__init__()
        self.mid_chans = int((in_channels + out_channels) / 2)
        self.conv_block_1 = ConvBlock(
            in_chs=in_channels, out_chs=self.mid_chans, kernel_size=kernel_size[
                0], stride=stride[0], padding=padding[0],
            norm_layer=norm_layer[0], act_layer=act_layer[0], pool_layer=pool_layer[0])
        self.conv_block_2 = ConvBlock(
            in_chs=self.mid_chans, out_chs=out_channels, kernel_size=kernel_size[
                1], stride=stride[1], padding=padding[1],
            norm_layer=norm_layer[1], act_layer=act_layer[1], pool_layer=pool_layer[1])
        self.downsample = downsample
        self.mode = mode

    def forward(self, x):
        y = self.conv_block_1(x)
        y = self.conv_block_2(y)
        # shape为NCHW, 这里获取最后的HW维度
        out_size = list(y.size())[-2:]
        self.downsample = nn.Upsample(
            size=out_size, mode=self.mode) if self.downsample else nn.Identity()
        x = self.downsample(x)
        return x + y

# begin-CBAM
# (https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)
# 即: Channel attetion + Spatial attention


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelAttention(nn.Module):
    def __init__(self, attention_channel, reduction_ratio=8, pool_types=['avg', 'max']):
        super(ChannelAttention, self).__init__()
        self.attention_channel = attention_channel
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(attention_channel, attention_channel // reduction_ratio),
            nn.ReLU(),
            nn.Linear(attention_channel // reduction_ratio, attention_channel)
        )
        self.pool_types = pool_types

    def forward(self, in_tensor):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = functional.avg_pool2d(
                    in_tensor,
                    (in_tensor.size(2), in_tensor.size(3)),
                    stride=(in_tensor.size(2), in_tensor.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = functional.max_pool2d(
                    in_tensor,
                    (in_tensor.size(2), in_tensor.size(3)),
                    stride=(in_tensor.size(2), in_tensor.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = functional.lp_pool2d(
                    in_tensor,
                    2,
                    (in_tensor.size(2), in_tensor.size(3)),
                    stride=(in_tensor.size(2), in_tensor.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(in_tensor)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = functional.sigmoid(channel_att_sum).unsqueeze(
            2).unsqueeze(3).expand_as(in_tensor)
        return in_tensor * scale


class ChannelPool(nn.Module):
    def forward(self, in_tensor):
        return torch.cat(
            (torch.max(in_tensor, 1)[0].unsqueeze(1),
             torch.mean(in_tensor, 1).unsqueeze(1)),
            dim=1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = ConvBlock(
            2, 1,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(self.kernel_size - 1) // 2,
            act_layer=None)

    def forward(self, in_tensor):
        in_tensor_compress = self.compress(in_tensor)
        in_tensor_out = self.spatial(in_tensor_compress)
        scale = functional.sigmoid(in_tensor_out)
        return in_tensor * scale


class CBAM(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=8, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(
            gate_channel, reduction_ratio=reduction_ratio, pool_types=pool_types)
        self.spatial_attention = SpatialAttention() if not no_spatial else nn.Identity()

    def forward(self, in_tensor):
        in_tensor = self.channel_attention(in_tensor)
        in_tensor = self.spatial_attention(in_tensor)
        return in_tensor

# end-CBAM


class APM(nn.Module):

    def __init__(self, in_chs: int, out_chs: int):
        super(APM, self).__init__()
        self.conv_blocks_1 = ConvBlock(in_chs, 4, kernel_size=13, stride=2, padding=1, dilation=1,
                                       norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU, pool_layer=nn.MaxPool2d(3, stride=1))
        self.conv_blocks_2 = ConvBlock(4, 16, kernel_size=5, stride=2, padding=1, dilation=1,
                                       norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU, pool_layer=nn.MaxPool2d(3, stride=1))
        self.conv_blocks_3 = ConvBlock(16, 32, kernel_size=3, stride=2, padding=1, dilation=1,
                                       norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU, pool_layer=nn.MaxPool2d(3, stride=1))
        self.res_blocks_1 = Residual(32, 32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1],
                                     downsample=False)
        self.res_blocks_2 = Residual(32, 32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1],
                                     downsample=False)
        self.cbam_blocks = CBAM(32)
        self.conv_blocks_4 = ConvBlock(32, 8, kernel_size=3, stride=1, padding=1, dilation=1,
                                       norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU, pool_layer=nn.AvgPool2d(3, stride=1))
        self.conv_blocks_5 = ConvBlock(8, out_chs, kernel_size=3, stride=1, padding=1, dilation=1,
                                       norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU, pool_layer=nn.AvgPool2d(3, stride=1))

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
    def __init__(self, backbone='tf_efficientnetv2_b0', pretrain=True, num_classes=7, pool=False,
                 pool_size=(300, 300), pool_type='max'):
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

    def forward(self, x):
        x = self.apm(x)
        x = self.pool(x)
        x = self.backbone(x)
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
