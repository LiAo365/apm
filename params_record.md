<!--
 * @Author       : LiAo
 * @Date         : 2022-07-14 10:38:36
 * @LastEditTime : 2022-07-14 14:24:49
 * @LastAuthor   : LiAo
 * @Description  : Please add file description
-->
| commit id | method |    transforms    | optimizer  |        lr_scheduler         | learning_rate | weight_decay | batch_size | Accuracy |
| :-------: | :----: | :--------------: | :--------: | :-------------------------: | :-----------: | :----------: | :--------: | :------: |
|  d499643  |  apm   |    ToTensor()    | RangerLars |      ReduceLROnPlateau      |     0.005     |     0.00     |     8      |  0.7677  |
|  8d6fb4e  |  apm   | CLAHE+ToTensor() | RangerLars |      ReduceLROnPlateau      |     0.005     |     0.00     |     16     |  0.8256  |
|  af1c294  |  apm   | CLAHE+ToTensor() | RangerLars | CosineAnnealingWarmRestarts |     0.005     |     1e-5     |     16     |  0.7363  |
|           |  apm   | CLAHE+ToTensor() |  Adadelta  |           StepLR            |     0.005     |     1e-5     |     16     |          |
