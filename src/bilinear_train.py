# Windows 10 Anconda python3.8
# coding=utf-8
'''
Author       : LiAo
Date         : 2022-07-10 23:48:30
LastEditTime : 2022-07-11 12:09:54
LastAuthor   : LiAo
Description  : Please add file description
'''
import os
import timm
import torch
import pandas as pd
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchtools.optim import RangerLars
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from src import utils
from src import train_utils
import warnings
warnings.filterwarnings('ignore')
# 设置torch的随机数种子
torch.manual_seed(123)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # 定义数据预处理
    data_transform = transforms.Compose([
        utils.SelfCLAHE(clip_limit=2.0, tile_grid_size=(32, 32)),
        transforms.Resize(
            size=(260, 260), interpolation=transforms.InterpolationMode.BILINEAR),
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
    num_workers = 4
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
    train_loader = DataLoader(
        dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(
        dataset=testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # 如果指定weight_path则依据weight_path加载权重进行训练
    model = timm.create_model(
        args.backbone, pretrained=args.backbone_pretrain, num_classes=args.num_classes, in_chans=1)
    model = model.to(device)

    # 定义optimizer
    optimizer = RangerLars(model.parameters(), lr=args.lr,
                           eps=1e-6, weight_decay=args.weight_decay)
    # 学习率随着训练epoch周期变化
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
    #                                            verbose=True, cooldown=5, min_lr=1e-04, eps=1e-06)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=5, T_mult=2, eta_min=1e-5)
    best_acc = 0.0
    for epoch in range(args.epoch):
        # train
        train_loss, train_acc = train_utils.train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch
        )
        # test
        test_loss, test_acc, epoch_test_result = train_utils.test_model(
            model=model,
            data_loader=test_loader,
            device=device)
        # 学习率的调整
        # scheduler.step(test_loss)
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
