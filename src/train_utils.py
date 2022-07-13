# Windows 10 Anconda python3.8
# coding=utf-8
'''
Author       : LiAo
Date         : 2022-07-10 23:50:11
LastEditTime : 2022-07-13 23:33:19
LastAuthor   : LiAo
Description  : Please add file description
'''
import sys
from typing import Dict, Tuple
import torch
import torch.nn as nn
import numpy as np
from src import utils
import warnings
warnings.filterwarnings('ignore')
# 设置torch的随机数种子
torch.manual_seed(0)


def train_one_epoch(model, optimizer, data_loader, device, epoch, loss_weights=None) -> Tuple[float, float]:
    """训练一轮

    Args:
        model (nn.Module): 模型对象
        optimizer (_type_): 优化器
        data_loader (_type_): data loader
        device (_type_): cpu or cuda
        epoch (_type_): epoch
        loss_weights (_type_, optional): loss是否添加权重. Defaults to None.

    Returns:
        Tuple[float, float]: 返回训练的loss 和 acc
    """
    torch.cuda.empty_cache()
    model.train()
    # 设置loss function
    loss_function = nn.CrossEntropyLoss() if loss_weights is None else nn.CrossEntropyLoss(
        weight=torch.tensor(loss_weights))
    # 累计损失
    accu_loss = torch.zeros(1).to(device)
    # 累计预测正确的样本数目
    accu_num = torch.zeros(1).to(device)
    # epoch的样本个数
    sample_num = 0
    step = 0
    prefetcher = utils.DataPrefetcher(data_loader)
    images, labels = prefetcher.next()
    while images is not None:
        sample_num += images.shape[0]
        # 梯度置零
        optimizer.zero_grad()
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # forward + backward + optimize
        loss = loss_function(pred, labels.to(device))
        loss.backward()
        optimizer.step()

        accu_loss += loss.detach()
        if not torch.isfinite(loss):
            print(
                "WARNING: epoch{%d} no-finite loss {%s}, ending training ", epoch, loss)
            sys.exit(1)
        step += 1
        images, labels = prefetcher.next()
    # 返回训练的loss 和 acc
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@ torch.no_grad()
def test_model(model, data_loader, device, loss_weights=None) -> Tuple[float, float, Dict[str, np.array]]:
    """模型测试

    Args:
        model (nn.Module): 模型对象
        data_loader (_type_): data_loader
        device (_type_): cpu or cuda
        loss_weights (_type_):  loss是否添加权重

    Returns:
        Tuple[float, float, Dict[str, np.array]]: 返回测试的loss、acc、预测标签与真实标签的字典存储
    """
    torch.cuda.empty_cache()
    loss_function = nn.CrossEntropyLoss() if loss_weights is None else nn.CrossEntropyLoss(
        weight=torch.tensor(loss_weights))
    model.eval()
    # 累计损失
    accu_loss = torch.zeros(1).to(device)
    # 累计预测正确的样本数目
    accu_num = torch.zeros(1).to(device)
    sample_num = 0
    test_result = {'labels': np.array(
        [], dtype='u1'), 'preds': np.array([], dtype='u1')}
    prefetcher = utils.DataPrefetcher(data_loader)
    images, labels = prefetcher.next()
    step = 0
    while images is not None:
        test_result['labels'] = np.concatenate(
            (test_result['labels'], np.array(labels.cpu(), dtype='u1')))
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        test_result['preds'] = np.concatenate(
            (test_result['preds'], np.array(pred_classes.cpu(), dtype='u1')))

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        step += 1
        images, labels = prefetcher.next()
    torch.cuda.empty_cache()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, test_result
