# Windows 10 Anconda python3.8
# coding=utf-8
'''
Author       : LiAo
Date         : 2022-07-10 01:26:53
LastEditTime : 2022-07-10 01:31:23
LastAuthor   : LiAo
Description  : Please add file description
'''
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
import torch
import sys
sys.path.append("../")
##########################################################

if __name__ == '__main__':
    from src import apm
    longitudinal_path = "/data/liao/code/apm/images/longitudinal_crack/061_102639666.jpg"
    transverse_path = "/data/liao/code/apm/images/transverse_crack/325_101535555.jpg"
    repair_path = "/data/liao/code/apm/images/mend/000_095533539.jpg"
    best_weight = "/data/liao/code/apm/result/bicubic/weight/best_weight_back.pth"

    class SelfCLAHE(object):
        def __init__(self, clip_limit=4.0, tile_grid_size=(32, 32)):
            self.ops = cv2.createCLAHE(
                clipLimit=clip_limit, tileGridSize=tile_grid_size)

        def __call__(self, img):
            image_np = np.array(img)
            return Image.fromarray(self.ops.apply(image_np))

    data_transform = transforms.Compose([
        SelfCLAHE(clip_limit=4.0, tile_grid_size=(64, 64)),
        transforms.ToTensor()
    ])

    def loader(path: str):
        return Image.open(path).convert('L')  # .resize((1200, 900))

    def rgb_loader(path: str):
        return Image.open(path).convert('RGB')  # .resize((1200, 900))

    # 1. 加载模型
    model = torch.load(best_weight, map_location=torch.device('cpu'))
    model.eval()
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())
    # 2. 选择目标层
    target_layer = [
        model.backbone.blocks[0]
    ]
    # 3. 构建输入图像的Tensor形式
    image = loader(longitudinal_path)
    # 获取原始图像经过Resizer Module之后的图像可视化
    image_tensor = data_transform(image)
    image_tensor = image_tensor.unsqueeze(dim=0)
    rgb_image = np.float32(rgb_loader(longitudinal_path)) / 255

    # Construct the CAM object once, and then re-use it on many images:
    # 4.初始化GradCAM，包括模型，目标层以及是否使用cuda
    cam = GradCAM(model=model, target_layers=target_layer, use_cuda=False)

    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    # 5.选定目标类别，如果不设置，则默认为分数最高的那一类
    target_category = None

    #  You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    # 6. 计算cam
    grayscale_cam = cam(input_tensor=image_tensor,
                        targets=target_category,
                        aug_smooth=True,
                        eigen_smooth=True
                        )

    # In this example grayscale_cam has only one image in the batch:
    # 7.展示热力图并保存, grayscale_cam是一个batch的结果，只能选择一张进行展示
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_image, grayscale_cam)

    save_path = "/data/liao/code/apm/images/longitudinal_crack/bicubic/061_backbone_blocks_0_long.jpg"
    cv2.imwrite(save_path, visualization)
