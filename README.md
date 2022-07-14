<!--
 * @Author       : LiAo
 * @Date         : 2022-07-06 15:17:42
 * @LastEditTime : 2022-07-14 10:37:25
 * @LastAuthor   : LiAo
 * @Description  : Please add file description
-->


# **Attention**
# **Updating, Not The Final Version**
## Currenting: 调参 & 更换数据集的鲁棒性实验

# Pavement disease detection method with adaptive perception of high-resolution image content

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
