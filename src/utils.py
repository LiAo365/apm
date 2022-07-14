# Windows 10 Anconda python3.8
# coding=utf-8
'''
Author       : LiAo
Date         : 2022-07-05 19:52:11
LastEditTime : 2022-07-14 19:27:04
LastAuthor   : LiAo
Description  : Please add file description
'''
import os
import cv2
import shutil
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import dataset
from typing import Callable, Optional, Any, Tuple, List, Dict
from torch.nn.modules.utils import _pair, _quadruple


def setup_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


def path_exist(path) -> None:
    if os.path.exists(path) is False:
        os.mkdir(path)


def single_dir_move(origin_dir_path: str, target_dir_path: str):
    """移动将源文件夹下所有子文件移动到目标文件夹下面

    Args:
        origin_dir_path (str): 源文件夹路径
        target_dir_path (str): 目标文件夹路径
    """
    path_exist(target_dir_path)
    for root, _, files in tqdm(list(os.walk(origin_dir_path))):
        for file in files:
            shutil.copy(os.path.join(root, file),
                        os.path.join(target_dir_path, file))


def move_data(origin_path: str, target_path: str):
    """移动将源文件夹下所有的子文件夹及子文件移动到目标文件夹下面

    Args:
        origin_path (str): 源文件夹路径
        target_path (str): 目标文件夹路径
    """
    path_exist(target_path)
    for label in os.listdir(origin_path):
        label_origin_path = os.path.join(origin_path, label)
        lebel_target_path = os.path.join(target_path, label)
        if os.path.isdir(label_origin_path):
            single_dir_move(label_origin_path, lebel_target_path)
        elif os.path.isfile(label_origin_path):
            shutil.copy(label_origin_path, lebel_target_path)


class SelfCLAHE(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(64, 64)):
        """自适应直方图均衡化

        Args:
            clip_limit (float, optional): clip_limit参数. Defaults to 4.0.
            tile_grid_size (tuple, optional): 每次均衡化的小patch大小. Defaults to (32, 32).
        """
        self.ops = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        image_np = np.array(img)
        return Image.fromarray(self.ops.apply(image_np))


def gray_loader(path: str, size=None) -> Image:
    img = Image.open(path).convert('L')
    if isinstance(size, tuple) or isinstance(size, list):
        img = img.resize(size=size)
    return img


def rgb_loader(path: str, size=None) -> Image:
    img = Image.open(path).convert('RGB')
    if isinstance(size, tuple) or isinstance(size, list):
        img = img.resize(size=size)
    return img


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


class SplitDataSet(dataset.Dataset):
    def __init__(self, classes, class_to_idx, samples: list, transform: Optional[Callable] = None, loader: Callable[[str], Any] = gray_loader, shuffle=True) -> None:
        self.samples = np.array(samples)
        self.loader = loader
        self.transform = transform
        self.classes = classes
        self.class_to_idx = class_to_idx
        if shuffle:
            np.random.shuffle(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, int(target)

    def __len__(self) -> int:
        return len(self.samples)


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(
        directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(
            f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class AllImageFolder():
    def __init__(self, root: str) -> None:
        self.root = root
        classes, class_to_idx = self.find_classes(self.root)
        self.classes = classes
        self.class_to_idx = class_to_idx

    def get_classes(self):
        return self.classes

    def get_class_to_idx(self):
        return self.class_to_idx

    def split(self, ratio: float = 0.9):
        train_set = []
        test_set = []
        directory = os.path.expanduser(self.root)
        for target_class in sorted(self.class_to_idx.keys()):
            target_instances = []
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = path, class_index
                        target_instances.append(item)
            np.random.shuffle(target_instances)
            offset = int(ratio * len(target_instances))
            train_set.extend(target_instances[:offset])
            test_set.extend(target_instances[offset:])

        return train_set, test_set

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)


class StochasticPool2d(nn.Module):
    """ Stochastic 2D pooling, where prob(selecting index)~value of the activation
    IM_SIZE should be divisible by 2, not best implementation.
    based off:
    https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598#file-median_pool-py-L5
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=2, stride=2, padding=0, same=False):
        super(StochasticPool2d, self).__init__()
        # I don't know what this is but it works
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # because multinomial likes to fail on GPU when all values are equal
        # Try randomly sampling without calling the get_random function a million times
        init_size = x.shape

        # x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.kernel_size[0], self.stride[0]).unfold(
            3, self.kernel_size[1], self.stride[1])
        x = x.contiguous().view(-1, 4)
        idx = torch.randint(0, x.shape[1], size=(
            x.shape[0],)).type(torch.cuda.LongTensor)
        x = x.contiguous().view(-1)
        x = torch.take(x, idx)
        x = x.contiguous().view(init_size[0], init_size[1], int(
            init_size[2] / 2), int(init_size[3] / 2))
        return x


def d(x):
    return 1


class ConcatLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, scheduler1, scheduler2, total_steps, pct_start=0.5, last_epoch=-1):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.step_start = float(pct_start * total_steps) - 1
        super(ConcatLR, self).__init__(optimizer, last_epoch)

    def step(self):
        if self.last_epoch <= self.step_start:
            self.scheduler1.step()
        else:
            self.scheduler2.step()
        super().step()

    def get_lr(self):
        if self.last_epoch <= self.step_start:
            return self.scheduler1.get_lr()
        else:
            return self.scheduler2.get_lr()


def flat_and_anneal(optimizer, total_steps: int, ann_start: float = 0.5):
    """_summary_

    Args:
        optimizer (_type_): optimizer
        total_steps (int): (dataset.len() / batch_szie)) * epoch
        ann_start (float): default = 0.5

    Returns:
        scheduler: a scheduler
    """
    dummy = torch.optim.lr_scheduler.LambdaLR(optimizer, d)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(total_steps * (1 - ann_start)))
    scheduler = ConcatLR(optimizer, dummy, cosine,
                         total_steps, ann_start)
    return scheduler


def test_shuffle_list():
    files = ["1xs", "2dsd", 3, 4, 5, 6, 7, 8, 9]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    stack = np.column_stack((np.array(files), np.array(labels)))
    np.random.shuffle(stack)
    files = stack[:, -1]
    labels = stack[:, 0]
    print(files)
    print(labels)
