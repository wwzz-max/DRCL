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
    parser.add_argument('--config', help='Path to config yaml file.', required=True)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config = parse_settings(config, log_settings=False)

    files_folder = '%s/%s' % (config.output_save_path, 'numpy_files_seg_spectralnet')
    files_folder1 = '%s/%s' % (config.output_save_path, 'numpy_files_seg_pos_spectralnet')
    files_folder2 = '%s/%s' % (config.output_save_path, 'numpy_files_seg_pixel_spectralnet')
    files_folder3 = '%s/%s' % (config.output_save_path, 'numpy_files_seg_kmeans')
    files_folder4 = '%s/%s' % (config.output_save_path, 'numpy_files_seg_pixel_kmeans')
    files_folder5 = '%s/%s' % (config.output_save_path, 'numpy_files_seg_pixel_pos_spectralnet')

    figure_folder = '%s/%s' % (config.output_save_path, 'figures')
    phate_folder = '%s/%s' % (config.output_save_path, 'numpy_files_phate')

    os.makedirs(figure_folder, exist_ok=True)
    os.makedirs(phate_folder, exist_ok=True)

    np_files_path = sorted(glob('%s/%s' % (files_folder, '*.npz')))
    np_files_path_1 = sorted(glob('%s/%s' % (files_folder1, '*.npz')))
    np_files_path_2 = sorted(glob('%s/%s' % (files_folder2, '*.npz')))
    np_files_path_3 = sorted(glob('%s/%s' % (files_folder3, '*.npz')))
    np_files_path_4 = sorted(glob('%s/%s' % (files_folder4, '*.npz')))
    np_files_path_5 = sorted(glob('%s/%s' % (files_folder5, '*.npz')))

    for image_idx in tqdm(range(32, 33)):
        numpy_array = np.load(np_files_path[image_idx])
        numpy_array_1 = np.load(np_files_path_1[image_idx])
        numpy_array_2 = np.load(np_files_path_2[image_idx])
        numpy_array_3 = np.load(np_files_path_3[image_idx])
        numpy_array_4 = np.load(np_files_path_4[image_idx])
        numpy_array_5 = np.load(np_files_path_5[image_idx])

        image = numpy_array['image']
        label_true = numpy_array['label'].astype(np.int16)
        latent = numpy_array['latent']

        label_kmeans = numpy_array_3['label_kmeans']
        seg_kmeans = label_hint_seg(label_true=label_true, label_pred=label_kmeans)

        label_spectralnet = numpy_array['label_spectralnet']
        seg_spectralnet = label_hint_seg(label_true=label_true, label_pred=label_spectralnet)
                                    
        label_pos_spectralnet = numpy_array_1['label_pos_spectralnet']
        seg_pos_spectralnet = label_hint_seg(label_true=label_true, label_pred=label_pos_spectralnet)

        label_pixel_spectralnet = numpy_array_2['label_spectralnet']
        seg_pixel_spectralnet = label_hint_seg(label_true=label_true, label_pred=label_pixel_spectralnet)

        label_pixel_kmeans = numpy_array_4['label_kmeans']
        seg_pixel_kmeans = label_hint_seg(label_true=label_true, label_pred=label_pixel_kmeans)

        label_pixel_pos_spectralnet = numpy_array_5['label_pos_spectralnet']
        seg_pixel_pos_spectralnet = label_hint_seg(label_true=label_true, label_pred=label_pixel_pos_spectralnet)

        H, W = label_true.shape[:2]

        phate_op = phate.PHATE(random_state=random_seed, n_jobs=config.num_workers)
        image_flattened = image.reshape(-1, image.shape[-1])
        data_phate_image = phate_op.fit_transform(normalize(image_flattened, axis=1))

        phate_path = '%s/sample_%s.npz' % (phate_folder, str(image_idx).zfill(5))
        if os.path.exists(phate_path):
            data_phate_numpy = np.load(phate_path)
            data_phate = data_phate_numpy['data_phate']
        else:
            phate_op = phate.PHATE(random_state=random_seed, n_jobs=config.num_workers)
            data_phate = phate_op.fit_transform(normalize(latent.reshape(-1, latent.shape[-1]), axis=1))
            with open(phate_path, 'wb+') as f:
                np.savez(f, data_phate=data_phate)

        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']

        fig_all = plt.figure(figsize=(20, 16))
        titles = [
            'Ground truth label\nin image space', 'SK', 'SN', 'SN+SC',
            'Image space', 'Segmented SK', 'Segmented SN', 'Segmented SN+SC',
            'Ground truth label\nin latent space', 'SK', 'SN', 'SN+SC',
            'Latent space', 'Segmented SK', 'Segmented SN', 'Segmented SN+SC'
        ]
        data = [
            continuous_renumber(label_true.reshape((H * W, -1))),
            continuous_renumber(label_kmeans.reshape((H * W, -1))),
            continuous_renumber(label_pixel_spectralnet.reshape((H * W, -1))),
            continuous_renumber(label_pos_spectralnet.reshape((H * W, -1))),
            continuous_renumber(label_true.reshape((H * W, -1))),
            seg_pixel_kmeans.reshape((H * W, -1)),
            seg_pixel_spectralnet.reshape((H * W, -1)),
            seg_pixel_pos_spectralnet.reshape((H * W, -1)),
            continuous_renumber(label_true.reshape((H * W, -1))),
            continuous_renumber(label_kmeans.reshape((H * W, -1))),
            continuous_renumber(label_pixel_spectralnet.reshape((H * W, -1))),
            continuous_renumber(label_pos_spectralnet.reshape((H * W, -1))),
            continuous_renumber(label_true.reshape((H * W, -1))),
            seg_kmeans.reshape((H * W, -1)),
            seg_spectralnet.reshape((H * W, -1)),
            seg_pos_spectralnet.reshape((H * W, -1))
        ]
        
        for i in range(16):
            ax = fig_all.add_subplot(4, 4, i + 1)
            legend_anchor = (1, 1) if (i + 1) % 4 == 0 else None
            scprep.plot.scatter2d(data_phate_image if i < 8 else data_phate, c=data[i],
                                  legend_anchor=legend_anchor,
                                  ax=ax, title=titles[i],
                                  xticks=False, yticks=False, label_prefix="PHATE", fontsize=12, s=3)
            if legend_anchor is None:
                ax.legend([], [], frameon=False)

        fig_path = '%s/sample_%s' % (figure_folder, str(image_idx).zfill(5))
        fig_all.tight_layout()
        #fig_all.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.2, wspace=0.4)
        fig_all.savefig('%s_phate_combined.png' % fig_path)
