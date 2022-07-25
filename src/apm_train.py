# Windows 10 Anconda python3.8
# coding=utf-8
'''
Author       : LiAo
Date         : 2022-07-05 20:08:25
LastEditTime : 2022-07-25 19:58:24
LastAuthor   : LiAo
Description  : Please add file description
'''

import os
import torch
import nni
import torch.nn as nn
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from torchtools.optim import RangerLars
from torch.optim import lr_scheduler
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from util import utils
from util import train_utils
from . import apm
import warnings
warnings.filterwarnings('ignore')


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # 定义数据预处理
    data_transform = {
        'train': transforms.Compose([
            utils.SelfCLAHE(clip_limit=2.0, tile_grid_size=(64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            utils.SelfCLAHE(clip_limit=2.0, tile_grid_size=(64, 64)),
            transforms.ToTensor()])}
    # 运行结果保存路径
    utils.path_exist(args.save_path)
    log_path = os.path.join(args.save_path, 'log')
    utils.path_exist(log_path)
    # log是tensorboard的记录路径
    writer = SummaryWriter(log_dir=log_path)
    # result是指在测试集上的true label和predict label保存路径, 以及测试结果保存路径
    result_path = os.path.join(args.save_path, 'result')
    utils.path_exist(result_path)
    # 最优权重保存路径
    weight_path = os.path.join(args.save_path, 'weight')
    utils.path_exist(weight_path)
    # 数据加载的线程数
    num_workers = 4
    # 超参数
    batch_size = args.batch_size
    # 保存测试集上的结果
    test_result_path = os.path.join(result_path, 'test_result.csv')
    test_result_pd = pd.read_csv(test_result_path) if os.path.exists(
        test_result_path) else pd.DataFrame()
    allsamples = utils.AllImageFolder(root=args.dataset)
    # 类别的标签, test时保存结果需要对应各个类别
    classes = allsamples.get_classes()
    class_to_idx = allsamples.get_class_to_idx()
    train_samples, test_samples = allsamples.split(ratio=0.8)
    trainset = utils.SplitDataSet(
        classes=classes, class_to_idx=class_to_idx, samples=train_samples, transform=data_transform['train'], loader=utils.gray_loader)
    testset = utils.SplitDataSet(
        classes=classes, class_to_idx=class_to_idx, samples=test_samples, transform=data_transform['test'], loader=utils.gray_loader)
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
            pool_type=args.pool_type,
            drop_rate=args.drop_rate)
        return model
    # 如果指定load_weight则依据weight_path加载权重进行训练
    model = new_module() if not args.load_weight else torch.load(args.load_weight_path)
    model = model.to(device)

    # 定义optimizer
    # total_steps = int(trainset.__len__() / batch_size) * args.epoch
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = RangerLars(model.parameters(), lr=args.lr)
    # 学习率随着训练epoch周期变化
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer=optimizer, T_0=5, T_mult=2, eta_min=1e-5)
    # scheduler = lr_scheduler.StepLR(
    #     optimizer=optimizer, step_size=20, gamma=0.5)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(1.6 * args.epoch))
    # 设置loss function
    loss_function = None
    # 交叉熵损失函数的权重-一般用于类别不平衡问题
    loss_weights = None
    if args.loss == 'focal':
        loss_function = utils.FocalLoss(gamma=args.focal_gamma)
    elif args.loss == 'cross':
        loss_function = nn.CrossEntropyLoss() if loss_weights is None else nn.CrossEntropyLoss(
            weight=torch.tensor(loss_weights))
    else:
        raise NotImplementedError(
            'For loss function [{}] not implemented.'.format(args.loss))
    best_acc = 0.0
    epoch_offset = args.epoch_offset
    for epoch in range(epoch_offset, epoch_offset + args.epoch):
        # train
        train_loss, train_acc = train_utils.train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            loss_function=loss_function
        )
        # test
        test_loss, test_acc, epoch_test_result = train_utils.test_model(
            model=model,
            data_loader=test_loader,
            device=device,
            loss_function=loss_function)
        # 学习率的调整
        # scheduler.step(test_loss)
        # scheduler.step()
        if (epoch * 1.0 - epoch_offset) / args.epoch > 0.2:
            scheduler.step()

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
        # 超惨调优中间属性可视化需要的数据
        metric = {
            'default': test_acc,  # nni要求必须是default, 其他的key-value可以用于可视化
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'learning_rate': optimizer.param_groups[0]['lr']}
        # 往TensorBoard的log文件写数据
        writer.add_scalar('loss/train_loss', train_loss, epoch)
        writer.add_scalar('accuracy/train_acc', train_acc, epoch)
        writer.add_scalar('loss/test_loss', test_loss, epoch)
        writer.add_scalar('accuracy/test_acc', test_acc, epoch)
        writer.add_scalar(
            'learning_rate', metric['learning_rate'], epoch)
        for tags in ['precision', 'recall', 'f1-score']:
            for label in classes:
                writer.add_scalar(tags + '/' + label,
                                  epoch_test_result_dict[label][tags], epoch)
        nni.report_intermediate_result(metric)
        # 保存训练完之后的最优模型权重
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model, os.path.join(
                weight_path, 'best_weight.pth'))
    nni.report_final_result(best_acc)


def test_classification_report():
    labels = [1, 2, 3, 3, 2, 1]
    preds = [0, 1, 2, 1, 2, 3]
    report = classification_report(
        labels, preds, target_names=['a', 'b', 'c', 'd'], zero_division=0, output_dict=True, digits=6)
    print(report)
    print(report['a']['precision'])


def test_convert_static_module():
    best_weight = "/data/liao/code/apm/result/apm/weight/best_weight.pth"
    static_path = "/data/liao/code/apm/result/apm/weight/static_best_weight.pth"
    model = torch.load(best_weight)
    torch.save(model.state_dict(), static_path)
