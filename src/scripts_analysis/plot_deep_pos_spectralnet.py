import argparse
import os
import sys
import warnings
from glob import glob

import numpy as np
import phate
import scprep
import yaml
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from tqdm import tqdm

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.diffusion_condensation import continuous_renumber
from utils.parse import parse_settings
from utils.segmentation import label_hint_seg

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    random_seed = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config = parse_settings(config, log_settings=False)

    files_folder = '%s/%s' % (config.output_save_path,
                              'numpy_files_seg_pos_spectralnet')
    figure_folder = '%s/%s' % (config.output_save_path, 'deep_figures')
    phate_folder = '%s/%s' % (config.output_save_path, 'numpy_files_phate')

    os.makedirs(figure_folder, exist_ok=True)
    os.makedirs(phate_folder, exist_ok=True)

    np_files_path = sorted(glob('%s/%s' % (files_folder, '*.npz')))

    for image_idx in tqdm(range(21,len(np_files_path))):
        numpy_array = np.load(np_files_path[image_idx])
        image = numpy_array['image']
        label_true = numpy_array['label'].astype(np.int16)
        latent = numpy_array['latent']
        label_pos_spectralnet = numpy_array['label_pos_spectralnet']
        seg_pos_spectralnet = label_hint_seg(label_true=label_true,
                                    label_pred=label_pos_spectralnet)

        H, W = label_true.shape[:2]

        # 1. PHATE plot.
        phate_path = '%s/sample_%s.npz' % (phate_folder,
                                           str(image_idx).zfill(5))
        if os.path.exists(phate_path):
            # Load the phate data if exists.
            data_phate_numpy = np.load(phate_path)
            data_phate = data_phate_numpy['data_phate']
        else:
            # Otherwise, generate the phate data.
            phate_op = phate.PHATE(random_state=random_seed,
                                   n_jobs=config.num_workers)

            data_phate = phate_op.fit_transform(normalize(latent, axis=1))
            with open(phate_path, 'wb+') as f:
                np.savez(f, data_phate=data_phate)

        fig1 = plt.figure(figsize=(15, 4))
        ax = fig1.add_subplot(1, 3, 1)
        # Plot the ground truth.
        scprep.plot.scatter2d(data_phate,
                              c=continuous_renumber(
                                  label_true.reshape((H * W, -1))),
                              legend_anchor=(1, 1),
                              ax=ax,
                              title='Ground truth label',
                              xticks=False,
                              yticks=False,
                              label_prefix="PHATE",
                              fontsize=10,
                              s=3)
        ax = fig1.add_subplot(1, 3, 2)
        # Plot the deep spectranl net.
        scprep.plot.scatter2d(data_phate,
                              c=continuous_renumber(
                                  label_pos_spectralnet.reshape((H * W, -1))),
                              legend_anchor=(1, 1),
                              ax=ax,
                              title='LGGUTS',
                              xticks=False,
                              yticks=False,
                              label_prefix="PHATE",
                              fontsize=10,
                              s=3)
        ax = fig1.add_subplot(1, 3, 3)
        # Plot the segmented deep spectranl net.
        scprep.plot.scatter2d(data_phate,
                              c=seg_pos_spectralnet.reshape((H * W, -1)),
                              legend_anchor=(1, 1),
                              ax=ax,
                              title='LGCUTS',
                              xticks=False,
                              yticks=False,
                              label_prefix="PHATE",
                              fontsize=10,
                              s=3)

        # 2. Segmentation plot.
        fig2 = plt.figure(figsize=(20, 6))
        ax = fig2.add_subplot(1, 4, 1)
        ax.imshow(image)
        ax.set_axis_off()
        ax = fig2.add_subplot(1, 4, 2)
        gt_cmap = 'gray' if len(np.unique(label_true)) <= 2 else 'tab20'
        ax.imshow(continuous_renumber(label_true), cmap=gt_cmap)
        ax.set_axis_off()
        ax = fig2.add_subplot(1, 4, 3)
        ax.imshow(seg_pos_spectralnet, cmap='gray')
        ax.set_title('LGGUTS')
        ax.set_axis_off()
        ax = fig2.add_subplot(1, 4, 4)
        ax.imshow(continuous_renumber(label_pos_spectralnet), cmap='tab20')
        ax.set_title('LGGUTS')
        ax.set_axis_off()

        fig_path = '%s/sample_%s' % (figure_folder, str(image_idx).zfill(5))

        fig1.tight_layout()
        fig1.savefig('%s_phate_deep spectralnet.png' % fig_path)

        fig2.tight_layout()
        fig2.savefig('%s_segmentation_deep spectralnet.png' % fig_path)
