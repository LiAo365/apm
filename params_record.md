<!--
 * @Author       : LiAo
 * @Date         : 2022-07-14 10:38:36
 * @LastEditTime : 2022-07-14 22:02:29
 * @LastAuthor   : LiAo
 * @Description  : Please add file description
-->
## Record For CQU2021

| commit id |     method     |    transforms    | optimizer  |        lr_scheduler         | learning_rate | weight_decay | batch_size | Accuracy |
| :-------: | :------------: | :--------------: | :--------: | :-------------------------: | :-----------: | :----------: | :--------: | :------: |
|  d499643  |      apm       |    ToTensor()    | RangerLars |      ReduceLROnPlateau      |     0.005     |     0.00     |     8      |  0.7677  |
|  8d6fb4e  |      apm       | CLAHE+ToTensor() | RangerLars |      ReduceLROnPlateau      |     0.005     |     0.00     |     16     |  0.8256  |
|  af1c294  |      apm       | CLAHE+ToTensor() | RangerLars | CosineAnnealingWarmRestarts |     0.005     |     1e-5     |     16     |  0.7363  |
|  86a14dd  |      apm       | CLAHE+ToTensor() |  Adadelta  |           StepLR            |     0.005     |     1e-5     |     16     |  0.7830  |
|  b98fe70  |    apm_cat     | CLAHE+ToTensor() | RangerLars |           StepLR            |     0.005     |     0.00     |     16     |  0.8103  |
|  82d21f0  |      apm       | CLAHE+ToTensor() | RangerLars |           StepLR            |     0.003     |     0.00     |     16     |  0.8316  |
|  93c95b6  |    apm_cat     | CLAHE+ToTensor() | RangerLars |           StepLR            |     0.005     |     1e-5     |     16     |          |
|  93c95b6  |      apm       | CLAHE+ToTensor() | RangerLars |           StepLR            |     0.003     |     1e-5     |     16     |  0.8229  |
|           | apm+focal_loss | CLAHE+ToTensor() |    Adam    |           StepLR            |     0.001     |     1e-5     |     16     |          |

## Record For CQU-BPDD
| commit id | method |    transforms    | optimizer  | lr_scheduler | learning_rate | weight_decay | batch_size | Accuracy |
| :-------: | :----: | :--------------: | :--------: | :----------: | :-----------: | :----------: | :--------: | :------: |
|           |  apm   | CLAHE+ToTensor() | RangerLars |    StepLR    |     0.003     |     1e-5     |     32     |          |