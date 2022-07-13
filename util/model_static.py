# Windows 10 Anconda python3.8
# coding=utf-8
'''
Author       : LiAo
Date         : 2022-07-13 19:05:19
LastEditTime : 2022-07-13 19:05:21
LastAuthor   : LiAo
Description  : Please add file description
'''
from thop import profile
import timm
import torch
from thop import clever_format

input = torch.randn(1, 1, 224, 224)

models = timm.list_models(pretrained=True)
# avail_pretrained_models = ['resnet50', 'vgg16', 'vit_small_patch16_224', 'repvgg_b3',
#                            'cspdarknet53', 'tf_efficientnetv2_b1', 'tf_efficientnetv2_b2', 'tf_efficientnetv2_b3', 'tf_efficientnetv2_s']
avail_pretrained_models = []
for model in models:
    if 'efficient' in model:  # or 'vgg' in model or 'vit' in model
        avail_pretrained_models.append(model)
result = []
for model in avail_pretrained_models:
    net = timm.create_model(model, pretrained=True, in_chans=1, num_classes=3)
    macs, params = profile(net, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.4f")
    result.append((model, macs, params))

print('*' * 64)
print('|{:^20}|{:^20}|{:^20}|'.format('Model', 'Params(M)', 'MACs(G)'))
print('*' * 64)
for item in result:
    print('|{:^20}|{:^20}|{:^20}|'.format(item[0], item[2], item[1]))

print('*' * 64)
