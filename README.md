<!--
 * @Author       : LiAo
 * @Date         : 2022-07-06 15:17:42
 * @LastEditTime : 2022-07-14 17:45:52
 * @LastAuthor   : LiAo
 * @Description  : Please add file description
-->


# **Attention**
# **Updating, Not The Final Version**
## Currenting: 调参 & 更换数据集的鲁棒性实验

# Pavement disease detection method with adaptive perception of high-resolution image content

[![OSCS Status](https://www.oscs1024.com/platform/badge/liyuanshuo/apm.svg?size=small)](https://www.oscs1024.com/project/liyuanshuo/apm?ref=badge_small)  ![Documentation](https://img.shields.io/badge/documentation-yes-brightgreen)  

## 1. 训练代码说明

|    文件名称     |                                                                              说明                                                                              |
| :-------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   apm_run.py    |                                                                      执行apm的训练与测试                                                                       |
| apm_conv_run.py |                                                                       消融实验: apm_conv                                                                       |
| apm_res_run.py  |                                                                       消融实验: apm_res                                                                        |
| apm_cbam_run.py |                                                                       消融实验: apm_cbam                                                                       |
| upsample_run.py |                                                              对比实验代码, 传统的Upsample方法对比                                                              |
| resizer_run.py  | 对比实验代码,   [Resizer](https://openaccess.thecvf.com/content/ICCV2021/html/Talebi_Learning_To_Resize_Images_for_Computer_Vision_Tasks_ICCV_2021_paper.html) |


## 2. 模型定义代码

模型定义与训练工具类代码均在`src`路径下 


## 3. 单个方法的代码文件说明

以apm为例，首先在`src`目录中定义了`apm.py`文件(模型定义及实现文件)，对于模型的训练测试文件`apm_train.py`, 其中定义了优化器、学习率调整策略等训练参数，最终执行训练和测试时，直接运行根目录下的`apm_run.py`。

ps:
1. `apm_run.py`中定义学习率、权重衰减、训练epoch、数据集所在路径、结果和log保存路径等。每次运行需要指定使用的GPU序号
2. `str/utils.py`中定义了一些基本的工具，比如自定义数据集，自定义的data_loader、torch和numpy随机种子固定方法等等自定义的方法
3. `src/train_utils.py`中定义了`train_one_epoch`和`test_model`,用于模型训练的基本function
4. 确保实验结果可复现，应在`apm_run.py`中指定torch和numpy的随机数种子
5. `bpdd_src`文件夹定义的是针对`CQU-BPDD`数据集的模型定义文件和训练文件-作为方法鲁棒性的证明补充实验代码
