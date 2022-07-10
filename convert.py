# Windows 10 Anconda python3.8
# coding=utf-8
'''
Author       : LiAo
Date         : 2022-07-10 00:08:58
LastEditTime : 2022-07-10 01:25:54
LastAuthor   : LiAo
Description  : Please add file description
'''
import torch


def convert_pam():
    best_weight = "/data/liao/code/apm/result/apm/weight/best_weight.pth"
    static_path = "/data/liao/code/apm/result/apm/weight/static_best_weight.pth"
    model = torch.load(best_weight)
    torch.save(model.state_dict(), static_path)


def cpnvert_bicubic():
    best_weight = "/data/liao/code/apm/result/apm/weight/best_weight.pth"
    static_path = "/data/liao/code/apm/result/apm/weight/static_best_weight.pth"
    model = torch.load(best_weight)
    torch.save(model.state_dict(), static_path)


if __name__ == '__main__':
    pass
