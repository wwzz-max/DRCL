import argparse
import sys
import warnings
from typing import Tuple
import os
import numpy as np
import phate
from spectralnet import SpectralNet
import torch

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.metrics import per_class_dice_coeff
from utils.segmentation import label_hint_seg

warnings.filterwarnings("ignore")


def generate_spectralnet(shape: Tuple[int],
                    latent: np.array,
                    label_true: np.array,
                    num_workers: int = 1,
                    random_seed: int = 1) -> Tuple[float, np.array, np.array]:

    H, W, C = shape
    assert latent.shape == (H * W, C)

    seg_true = label_true > 0

    # Very occasionally, SVD won't converge.
    try:
        clusters = phate_clustering(latent=latent,
                                    random_seed=random_seed,
                                    num_workers=num_workers)
    except:
        clusters = phate_clustering(latent=latent,
                                    random_seed=random_seed + 1,
                                    num_workers=num_workers)

    # [H x W, C] to [H, W, C]
    label_pred = clusters.reshape((H, W))

    seg_pred = label_hint_seg(label_pred=label_pred, label_true=label_true)

    return per_class_dice_coeff(seg_pred, seg_true), label_pred, seg_pred




def add_spatial_info_to_feature_map(feature_map, weight=2.8):
    height, width, channels = feature_map.shape
    
    # 确定使用的设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    
    # 生成 x 和 y 坐标网格
    x = torch.linspace(-1, 1, width).view(1, width).repeat(height, 1) * weight
    y = torch.linspace(-1, 1, height).view(height, 1).repeat(1, width) * weight
    xy_grid = torch.stack((x, y), dim=2).to(device)  # 确保xy_grid在正确的设备上
    
    # 转换 feature_map 为 Tensor并确保它在正确的设备上
    feature_map_tensor = torch.from_numpy(feature_map).to(device) if isinstance(feature_map, np.ndarray) else feature_map.to(device)

    # 将坐标网格连接到特征图上
    return torch.cat((feature_map_tensor, xy_grid), dim=2)



def phate_clustering(latent: np.array, random_seed: int,
                     num_workers: int) -> np.array:

    
    
  

    # 将 latent 转换为 torch.Tensor 对象
    latent_tensor = torch.tensor(latent)
   
    #spectral_batch_size=437,
    spectralnet = SpectralNet(
        n_clusters=10,
        spectral_batch_size=437,
        spectral_epochs=2
      )
    
    
    spectralnet.fit(latent_tensor) # X is the dataset and it should be a torch.Tensor
    cluster_assignments = spectralnet.predict(latent_tensor) # Get the final assignments to clusters
    
    return cluster_assignments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--num_workers', type=int, required=True)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    numpy_array = np.load(args.load_path)
    image = numpy_array['image']
    label_true = numpy_array['label']
    latent = numpy_array['latent']

    image = (image + 1) / 2

    H, W = label_true.shape[:2]
    C = latent.shape[-1]
    X = latent

    dice_score, label_pred, seg_pred = generate_kmeans(
        (H, W, C), latent, label_true, num_workers=args.num_workers)

    with open(args.save_path, 'wb+') as f:
        np.savez(f,
                 image=image,
                 label=label_true,
                 latent=latent,
                 label_kmeans=label_pred,
                 seg_kmeans=seg_pred)

    sys.stdout.write('SUCCESS! %s, dice: %s' %
                     (args.load_path.split('/')[-1], dice_score))
