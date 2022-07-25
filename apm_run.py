# Windows 10 Anconda python3.8
# coding=utf-8
'''
Author       : LiAo
Date         : 2022-07-05 23:42:53
LastEditTime : 2022-07-25 22:31:27
LastAuthor   : LiAo
Description  : Please add file description
'''
import argparse
import nni
from src import apm_train
from util import utils

if __name__ == '__main__':
    # 设置随机数的种子,保证结果的可复现
    utils.setup_seed(2022)
    # define for nni
    params = {
        'backbone': 'tf_efficientnetv2_b4',
        'pool': True,
        'pool_size': (320, 320),
        'pool_type': 'max',
        'lr': 0.005,
        'drop_rate': 0.5,
        'loss': 'focal'
    }
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    # unique ID of the trial that is current running.
    trial_id = nni.get_trial_id()
    parser = argparse.ArgumentParser()

    # 网络模型参数
    parser.add_argument('--backbone', type=str, default=params['backbone'])
    parser.add_argument('--backbone_pretrain', type=bool, default=True)
    parser.add_argument('--pool', type=bool, default=bool(params['pool']))
    parser.add_argument('--pool_size', type=tuple,
                        default=(params['pool_size'], params['pool_size']))
    parser.add_argument('--pool_type', type=str,
                        default=params['pool_type'], help='must one of (max, avg, nearest, linear, bilinear, bicubic, trilinear)')
    # parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--drop_rate', type=float, default=params['drop_rate'])
    parser.add_argument(
        '--loss', type=str, default=params['loss'], help='loss funcation, shuould define in code. Now only (focal, cross)')
    parser.add_argument('--focal_gamma', type=float,
                        default=4, help='gamma for FocalLoss.')
    # 分类数目
    parser.add_argument('--num_classes', type=int, default=3)

    # 训练epoch
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--epoch_offset', type=int, default=0)
    # 超参数, 依据显存设置
    parser.add_argument('--batch_size', type=int,
                        default=24, help='decide by GPU RAM')
    # 学习率参数
    parser.add_argument('--lr', type=float, default=params['lr'])

    # 数据集所在目录
    parser.add_argument('--dataset', type=str,
                        default='/data/liao/datasets/cqu_2021_abnormal/')
    # 权重加载路径
    parser.add_argument('--load_weight', type=bool, default=False,
                        help='wheather load pretrain module from file.')
    parser.add_argument('--load_weight_path', type=str,
                        default=None)  # /data/liao/code/apm/result/apm/weight/best_weight.pth
    # 数据保存的路径
    parser.add_argument('--save_path', type=str,
                        default='/data/liao/code/apm/result/{}/'.format(trial_id))

    # GPU and GPU ID
    parser.add_argument('--device', default='cuda',
                        help='device id(i.e. 0 or 0, 1 or cpu)')
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
    opt = parser.parse_args(args=[])
    # 执行训练和测试
    apm_train.main(opt)
