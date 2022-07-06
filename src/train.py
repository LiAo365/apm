# Windows 10 Anconda python3.8
# coding=utf-8
'''
Author       : LiAo
Date         : 2022-07-05 20:08:25
LastEditTime : 2022-07-06 11:36:59
LastAuthor   : LiAo
Description  : Please add file description
'''

import os
import sys
from typing import Dict, Tuple
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchtools.optim import RangerLars
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from src import utils
from src import apm
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


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # 定义数据预处理
    data_transform = transforms.Compose([
        utils.SelfCLAHE(clip_limit=4.0, tile_grid_size=(32, 32)),
        transforms.ToTensor()
    ])
    # log是tensorboard的记录路径
    utils.path_exist(args.log_path)
    writer = SummaryWriter(log_dir=args.log_path)
    # result是指在测试集上的true label和predict label保存路径, 以及测试结果保存路径
    utils.path_exist(args.result_path)
    # 最优权重保存路径
    utils.path_exist(args.weight_path)
    # 数据加载的线程数
    num_workers = 8
    # 超参数
    batch_size = args.batch_size
    # 保存测试集上的结果
    test_result_pd = pd.DataFrame()
    allsamples = utils.AllImageFolder(root=args.dataset)
    # 类别的标签, test时保存结果需要对应各个类别
    classes = allsamples.get_classes()
    class_to_idx = allsamples.get_class_to_idx()
    train_samples, test_samples = allsamples.split()
    trainset = utils.SplitDataSet(
        classes=classes, class_to_idx=class_to_idx, samples=train_samples, transform=data_transform, loader=utils.gray_loader)
    testset = utils.SplitDataSet(
        classes=classes, class_to_idx=class_to_idx, samples=test_samples, transform=data_transform, loader=utils.gray_loader)
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True,
                             pin_memory=True, num_workers=num_workers)

    # 模型创建
    def new_module():
        """依据args参数创建模型"""
        model = apm.MultiClassification(
            backbone=args.backbone,
            pretrain=args.backbone_pretrain,
            num_classes=args.num_classes,
            pool=args.pool,
            pool_size=args.pool_size,
            pool_type=args.pool_type)
        return model
    # 如果指定weight_path则依据weight_path加载权重进行训练
    load_weight_path = args.load_weight_path
    model = new_module() if load_weight_path is None else torch.load(load_weight_path)
    model = model.to(device)

    # 定义optimizer
    optimizer = RangerLars(model.parameters(), lr=args.lr,
                           eps=1e-5, weight_decay=args.weight_decay)
    # 学习率随着训练epoch周期变化
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                               verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-08, eps=1e-06)
    best_acc = 0.0
    for epoch in range(args.epoch):
        # train
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch
        )
        # test
        test_loss, test_acc, epoch_test_result = test_model(
            model=model,
            data_loader=test_loader,
            device=device)
        # 学习率的调整
        scheduler.step(test_loss)

        # 保存测试集的测试结果
        epoch_test_result_dict = classification_report(
            epoch_test_result['labels'], epoch_test_result['preds'], target_names=classes, zero_division=0, output_dict=True, digits=6)
        epoch_test_dataframe = pd.DataFrame(epoch_test_result_dict).transpose()
        test_result_pd = test_result_pd.append(
            pd.DataFrame(epoch_test_dataframe))
        test_result_pd.to_csv(os.path.join(
            args.result_path, 'test_result.csv'), index=True)
        # 保存当前epoch测试preds与labels
        epoch_preds = pd.DataFrame(epoch_test_result)
        epoch_preds.to_csv(os.path.join(
            args.result_path, 'epoch_{:d}_test_result.csv'.format(epoch)), index=True)
        # 往TensorBoard的log文件写数据
        writer.add_scalar('loss/train_loss', train_loss, epoch)
        writer.add_scalar('accuracy/train_acc', train_acc, epoch)
        writer.add_scalar('loss/test_loss', test_loss, epoch)
        writer.add_scalar('accuracy/test_acc', test_acc, epoch)
        writer.add_scalar(
            'learning_rate', optimizer.param_groups[0]['lr'], epoch)
        for tags in ['precision', 'recall', 'f1-score']:
            for label in classes:
                writer.add_scalar(tags + '/' + label,
                                  epoch_test_result_dict[label][tags], epoch)

            # 保存训练完之后的最优模型权重
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model, os.path.join(
                args.weight_path, 'best_weight.pth'))


def test_classification_report():
    labels = [1, 2, 3, 3, 2, 1]
    preds = [0, 1, 2, 1, 2, 3]
    report = classification_report(
        labels, preds, target_names=['a', 'b', 'c', 'd'], zero_division=0, output_dict=True, digits=6)
    print(report)
    print(report['a']['precision'])
