from glob import glob
from typing import Tuple
import cv2
import numpy as np
from torch.utils.data import Dataset

# 定义ToothDataset数据集类
class Liver(Dataset):
    # 初始化函数
    def __init__(self, base_path: str = '../../data/liver/', out_shape: Tuple[int] = (128, 128)):
        self.base_path = base_path
        self.image_paths = sorted(glob(f'{base_path}/image_with_GA/*.png'))
        self.label_paths = sorted(glob(f'{base_path}/label_with_GA/*.png'))
        self.out_shape = out_shape

    # 获取数据集长度
    def __len__(self) -> int:
        return len(self.image_paths)

    # 获取单个数据项
    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)

        # 图像和标签缩放
        image = cv2.resize(src=image, dsize=self.out_shape, interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(src=label, dsize=self.out_shape, interpolation=cv2.INTER_NEAREST)
        
        # 归一化图像
        image = image / 255.0  # 将像素值从[0, 255]映射到[0, 1]
        # 假设标签是二进制的
        assert len(np.unique(label)) <= 2
        label = label != np.unique(label)[0]

        # 维度修正
        # 为了符合Torch的要求，将通道放在前面
        image = image[None, :, :]
        label = label[None, :, :]

        return image, label

    # 获取图像通道数
    def num_image_channel(self) -> int:
        return 1

    # 获取类别数
    def num_classes(self) -> int:
        return 1
