import argparse
import os
import sys
import warnings
from glob import glob
from tqdm import tqdm
from typing import Dict

import numpy as np
import scprep
import cv2
import yaml
from matplotlib import pyplot as plt
import multiscale_phate
from sklearn.preprocessing import normalize

sys.path.append('../')
from utils.attribute_hashmap import AttributeHashmap
from utils.diffusion_condensation import continuous_renumber, get_persistent_structures
from utils.parse import parse_settings
from utils.segmentation import label_hint_seg
from utils.metrics import dice_coeff, per_class_dice_coeff

warnings.filterwarnings("ignore")


def grayscale_3channel(image: np.array) -> np.array:
    if len(image.shape) == 2:
        image = image[..., None]
    assert len(image.shape) == 3
    if image.shape[-1] == 1:
        image = np.repeat(image, repeats=3, axis=-1)
    return image


def pop_blue_channel(label: np.array) -> np.array:
    assert label.min() >= 0 and label.max() <= 1
    if len(label.shape) == 3:
        label = label.squeeze(-1)
    assert len(label.shape) == 2

    output = np.zeros((*label.shape, 4))
    output[..., 2] = label

    output[label > 0, 3] = 0.6
    return np.uint8(255 * output)


def find_nearest_idx(arr: np.array, num: float) -> int:
    return np.abs(arr - num).argmin()


def draw_multiline_text(image, text, position, font, font_scale, color, thickness, line_type):
    lines = text.split('\n')
    x, y = position
    for i, line in enumerate(lines):
        y_offset = y + i * (cv2.getTextSize(line, font, font_scale, thickness)[0][1] + 5)
        cv2.putText(image, line, (x, y_offset), font, font_scale, color, thickness, line_type)


def plot_overlaid_comparison(fig: plt.Figure,
                             num_samples: int,
                             sample_idx: int,
                             data_hashmap: dict,
                             image_grayscale: bool,
                             pred_color: str = 'blue'):
    H, W = data_hashmap['label_true'].shape[:2]

    label_keys = [
        'label_true',
        'seg_pos_spectralnet',
        'label_random',
        'label_watershed',
        'label_felzenszwalb',
        'label_slic',
        'label_dfc',
        'label_stego',
        'seg_cuts_kmeans',
        'seg_cuts_persistent',
        'seg_cuts_best',
        'label_sam',
        'label_supervised_unet',
        'label_supervised_nnunet',
    ]

    title_map = {
        'label_true': 'Label',
        'seg_pos_spectralnet': 'OURS',
        'label_random': 'Random',
        'label_watershed': 'Wtrshd.',
        'label_felzenszwalb': 'Felzen.',
        'label_slic': 'SLIC',
        'label_dfc': 'DFC',
        'label_stego': 'STEGO',
        'seg_cuts_kmeans': 'CUTS+Spectral\nkmeans',
        'seg_cuts_persistent': 'CUTS+Diffusion\n(per.)',
        'seg_cuts_best': 'CUTS+Diffusion\n(best)',
        'label_sam': '[Sup.Pre-train]\n:SAM',
        'label_supervised_unet': '[Sup.]:UNet',
        'label_supervised_nnunet': '[Sup.]:nn-UNet'
    }

    num_labels = len(label_keys)
    cols_per_row = 3
    rows_per_sample = (num_labels + 1 + cols_per_row - 1) // cols_per_row

    true_color = (0, 255, 0)
    if pred_color == 'blue':
        pred_color = (0, 0, 255)
    elif pred_color == 'red':
        pred_color = (255, 0, 0)

    # image = np.uint8(255 * data_hashmap['image'].copy())
    # if image_grayscale:
    #     image = grayscale_3channel(image)

    for i, key in enumerate(['Original Image'] + label_keys):
        row_idx = (sample_idx * rows_per_sample) + (i // cols_per_row)
        col_idx = i % cols_per_row
        ax = fig.add_subplot(num_samples * rows_per_sample, cols_per_row, (row_idx * cols_per_row) + col_idx + 1)

        if i == 0:
            image = np.uint8(255 * data_hashmap['image'].copy())
            if image_grayscale:
                image = grayscale_3channel(image)
                if image.ndim == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            draw_multiline_text(image, 'Inputs', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            #draw_multiline_text(image, 'Inputs', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            ax.imshow(image)
        else:
            key = label_keys[i - 1]
            image = np.uint8(255 * data_hashmap['image'].copy())
            if image_grayscale:
                image = grayscale_3channel(image)
                if image.ndim == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            true_contours, _ = cv2.findContours(np.uint8(data_hashmap['label_true']), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for contour in true_contours:
                cv2.drawContours(image, [contour], -1, true_color, 4)
            pred_contours, _ = cv2.findContours(np.uint8(data_hashmap[key]), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for contour in pred_contours:
                weight = 1 if key == 'label_random' else 4
                cv2.drawContours(image, [contour], -1, pred_color, weight)

            # Calculate Dice coefficient
            dice = dice_coeff(data_hashmap[key], data_hashmap['label_true'])
            #title = f"{title_map[key]}\nDice: {dice:.2f}"
            title = f"{title_map[key]}"
            Dice = f"Dice: {dice:.2f}"
            draw_multiline_text(image, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            draw_multiline_text(image, Dice, (5, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # draw_multiline_text(image, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            # draw_multiline_text(image, Dice, (5, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            ax.imshow(image)

        ax.set_axis_off()

    fig.set_size_inches(12, 4 * num_samples * rows_per_sample)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.01, wspace=0.001)

    return fig


if __name__ == '__main__':
    random_seed = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('--image-idx',
                        help='Image index.',
                        type=int,
                        nargs='+',
                        required=True)
    parser.add_argument(
        '--grayscale',
        help='Use this flag if the image is expected to be grayscale.',
        action='store_true')
    parser.add_argument(
        '--binary',
        help='Use this flag if the label is expected to be binary.',
        action='store_true')
    parser.add_argument(
        '--comparison',
        help='Whether or not to include the comparison against other methods.',
        action='store_true')
    parser.add_argument(
        '--separate',
        help=
        'If true, do not overlay with contour, and show the segmentations separately. Default to true for multi-class segmentation',
        action='store_true')
    parser.add_argument(
        '-r',
        '--rerun',
        action='store_true',
        help=
        'If true, will rerun the script until succeeds to circumvent deadlock.'
    )
    parser.add_argument(
        '-t',
        '--max-wait-sec',
        help='Max wait time in seconds for each process (only relevant if `--rerun`).' + \
            'Consider increasing if you hit too many TimeOuts.',
        type=int,
        default=60)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config = parse_settings(config, log_settings=False)

    files_folder_raw = '%s/%s' % (config.output_save_path, 'numpy_files')
    files_folder_baselines = '%s/%s' % (config.output_save_path,
                                        'numpy_files_seg_baselines')
    files_folder_dfc = '%s/%s' % (config.output_save_path,
                                  'numpy_files_seg_DFC')
    files_folder_stego = '%s/%s' % (config.output_save_path,
                                    'numpy_files_seg_STEGO')
    files_folder_sam = '%s/%s' % (config.output_save_path,
                                  'numpy_files_seg_SAM')
    files_folder_supervised_unet = '%s/%s' % (
        config.output_save_path, 'numpy_files_seg_supervised_unet')
    files_folder_supervised_nnunet = '%s/%s' % (
        config.output_save_path, 'numpy_files_seg_supervised_nnunet')
   

    files_folder_cuts_kmeans = '%s/%s' % ('/root/cuts0.1/results/retina_cuts_seed2',
                                         'numpy_files_seg_kmeans')

    # files_folder_cuts_kmeans = '%s/%s' % ('/root/cuts0.1/results/brain_tumor_cuts_seed2',
    #                                      'numpy_files_seg_kmeans')

    # files_folder_cuts_kmeans = '%s/%s' % ('/root/cuts0.1/results/brain_ventricles_cuts_seed2/',
    #                                      'numpy_files_seg_kmeans')

    files_folder_kmeans = '%s/%s' % (config.output_save_path,
                                     'numpy_files_seg_kmeans')

    # files_folder_cuts_diffusion = '%s/%s' % ('/root/cuts0.1/results/brain_tumor_cuts_seed2',
    #                                         'numpy_files_seg_diffusion')


    files_folder_cuts_diffusion = '%s/%s' % ('/root/cuts0.1/results/retina_cuts_seed2',
                                            'numpy_files_seg_diffusion')
    
    
    # files_folder_cuts_diffusion = '%s/%s' % ('/root/cuts0.1/results/brain_ventricles_cuts_seed2/',
    #                                         'numpy_files_seg_diffusion')

    files_folder_diffusion = '%s/%s' % (config.output_save_path,
                                        'numpy_files_seg_diffusion')

    files_folder_spectralnet = '%s/%s' % (config.output_save_path,
                                    'numpy_files_seg_spectralnet')
    print('*****',files_folder_spectralnet)
    files_folder_pos_spectralnet = '%s/%s' % (config.output_save_path,
                                        'numpy_files_seg_pos_spectralnet')
    figure_folder = '%s/%s' % (config.output_save_path, 'paper_figure')

    os.makedirs(figure_folder, exist_ok=True)

    files_path_raw = sorted(glob('%s/%s' % (files_folder_raw, '*.npz')))
    files_path_baselines = sorted(
        glob('%s/%s' % (files_folder_baselines, '*.npz')))
    files_path_dfc = sorted(glob('%s/%s' % (files_folder_dfc, '*.npz')))
    files_path_stego = sorted(glob('%s/%s' % (files_folder_stego, '*.npz')))
    files_path_sam = sorted(glob('%s/%s' % (files_folder_sam, '*.npz')))
    files_path_supervised_unet = sorted(
        glob('%s/%s' % (files_folder_supervised_unet, '*.npz')))
    files_path_supervised_nnunet = sorted(
        glob('%s/%s' % (files_folder_supervised_nnunet, '*.npz')))
    
    files_path_cuts_kmeans = sorted(glob('%s/%s' % (files_folder_cuts_kmeans, '*.npz')))
    files_path_kmeans = sorted(glob('%s/%s' % (files_folder_kmeans, '*.npz')))

    files_path_cuts_diffusion = sorted(
        glob('%s/%s' % (files_folder_cuts_diffusion, '*.npz')))
    files_path_diffusion = sorted(
        glob('%s/%s' % (files_folder_diffusion, '*.npz')))

    files_path_diffusion = sorted(
        glob('%s/%s' % (files_folder_diffusion, '*.npz')))
    
    files_path_spectralnet = sorted(
        glob('%s/%s' % (files_folder_spectralnet, '*.npz')))
 

    files_path_pos_spectralnet = sorted(
        glob('%s/%s' % (files_folder_pos_spectralnet, '*.npz')))
    num_samples = len(args.image_idx)

    if args.comparison:
        fig = plt.figure(figsize=(25, 2 * num_samples))
    else:
        fig = plt.figure(figsize=(25, 4 * num_samples))

    for sample_idx, image_idx in enumerate(tqdm(args.image_idx)):

        numpy_array_raw = np.load(files_path_raw[image_idx])

        image = numpy_array_raw['image']
        image = (image + 1) / 2

        recon = numpy_array_raw['recon']
        recon = (recon + 1) / 2

        label_true = numpy_array_raw['label']
        if np.isnan(label_true).all():
            print('\n\n[Major Warning !!!] We found that the true label is all `NaN`s.' + \
            '\nThis shall only happen if you are not providing labels. Please double check!\n\n')
            label_true = np.ones_like(label_true)
        label_true = label_true.astype(np.int16)
        latent = numpy_array_raw['latent']

        try:
            numpy_array_baselines = np.load(files_path_baselines[image_idx])
            label_random = numpy_array_baselines['label_random']
            label_watershed = numpy_array_baselines['label_watershed']
            if args.binary:
                label_watershed = label_hint_seg(label_pred=label_watershed,
                                                 label_true=label_true)
            label_felzenszwalb = numpy_array_baselines['label_felzenszwalb']
            if args.binary:
                label_felzenszwalb = label_hint_seg(
                    label_pred=label_felzenszwalb, label_true=label_true)
            label_slic = numpy_array_baselines['label_slic']
            if args.binary:
                label_slic = label_hint_seg(label_pred=label_slic,
                                            label_true=label_true)
        except:
            print(
                'Warning! `baselines` results not found. Placeholding with blank labels.'
            )
            label_random = np.zeros_like(label_true)
            label_watershed = np.zeros_like(label_true)
            label_felzenszwalb = np.zeros_like(label_true)
            label_slic = np.zeros_like(label_true)

        try:
            numpy_array_cuts_kmeans = np.load(files_path_cuts_kmeans[image_idx])
            label_cuts_kmeans = numpy_array_cuts_kmeans['label_kmeans']
        except:
            print(
                'Warning! `CUTS + k-means` results not found. Placeholding with blank labels.'
            )
            label_cuts_kmeans = np.zeros_like(label_true)

        try:
            numpy_array_kmeans = np.load(files_path_kmeans[image_idx])
            label_kmeans = numpy_array_kmeans['label_kmeans']
        except:
            print(
                'Warning! `OURS + k-means` results not found. Placeholding with blank labels.'
            )
            label_kmeans = np.zeros_like(label_true)

        
        
        try:
            numpy_array_spectralnet = np.load(files_path_spectralnet[image_idx])
            label_spectralnet = numpy_array_spectralnet['label_spectralnet']
        except:
            print(
                'Warning! `OURS + spectralnet` results not found. Placeholding with blank labels.'
            )
            label_spectralnet = np.zeros_like(label_true)
     
        try:
            numpy_array_pos_spectralnet = np.load(files_path_pos_spectralnet[image_idx])
            label_pos_spectralnet = numpy_array_pos_spectralnet['label_pos_spectralnet']
        except:
            print(
                'Warning! `OURS + pos + spectralnet` results not found. Placeholding with blank labels.'
            )
            label_pos_spectralnet = np.zeros_like(label_true)


        try:
            numpy_array_cuts_diffusion = np.load(files_path_cuts_diffusion[image_idx])
            labels_cuts_diffusion = numpy_array_cuts_diffusion['labels_diffusion']
        except:
            print(
                'Warning! `CUTS + diffusion condensation` results not found. Placeholding with blank labels.'
            )
            labels_cuts_diffusion = np.zeros((10, *label_true.shape))
            granularities = None

        try:
            numpy_array_diffusion = np.load(files_path_diffusion[image_idx])
            labels_diffusion = numpy_array_diffusion['labels_diffusion']
        except:
            print(
                'Warning! `OURS + diffusion condensation` results not found. Placeholding with blank labels.'
            )
            labels_diffusion = np.zeros((10, *label_true.shape))
            granularities = None

        try:
            numpy_array_dfc = np.load(files_path_dfc[image_idx])
            label_dfc = numpy_array_dfc['label_dfc']
            label_dfc = label_hint_seg(label_pred=label_dfc,
                                       label_true=label_true)
        except:
            print(
                'Warning! `DFC` results not found. Placeholding with blank labels.'
            )
            label_dfc = np.zeros_like(label_true)

        try:
            numpy_array_stego = np.load(files_path_stego[image_idx])
            label_stego = numpy_array_stego['label_stego']
            label_stego = label_hint_seg(label_pred=label_stego,
                                         label_true=label_true)
        except:
            print(
                'Warning! `STEGO` results not found. Placeholding with blank labels.'
            )
            label_stego = np.zeros_like(label_true)

        try:
            numpy_array_sam = np.load(files_path_sam[image_idx])
            label_sam = numpy_array_sam['label_sam']
        except:
            print(
                'Warning! `SAM` results not found. Placeholding with blank labels.'
            )
            label_sam = np.zeros_like(label_true)

        try:
            numpy_array_unet = np.load(files_path_supervised_unet[image_idx])
            label_supervised_unet = numpy_array_unet['label_pred']
        except:
            print(
                'Warning! `Supervised UNet` results not found. Placeholding with blank labels.'
            )
            label_supervised_unet = np.zeros_like(label_true)

        try:
            numpy_array_nnunet = np.load(
                files_path_supervised_nnunet[image_idx])
            label_supervised_nnunet = numpy_array_nnunet['label_pred']
        except:
            print(
                'Warning! `Supervised nn-UNet` results not found. Placeholding with blank labels.'
            )
            label_supervised_nnunet = np.zeros_like(label_true)

        H, W = label_true.shape[:2]
        
        B = labels_diffusion.shape[0]  

        B2 = labels_cuts_diffusion.shape[0]  

        label_persistent = get_persistent_structures(
            labels_diffusion.reshape((B, H, W)))

        label_cuts_persistent = get_persistent_structures(
            labels_cuts_diffusion.reshape((B2, H, W)))

        label_best = np.zeros_like(label_true)
        label_cuts_best = np.zeros_like(label_true)
        best_dice_val = 0
        for curr_granularity_idx in range(B):
            label_curr_granularity = labels_diffusion.reshape((B, H, W))[curr_granularity_idx]
            seg_curr_granularity = label_hint_seg(label_pred=label_curr_granularity,
                                                  label_true=label_true)
            if args.binary:
                dice_metric = dice_coeff
            else:
                dice_metric = per_class_dice_coeff
            curr_dice_val = dice_metric(label_pred=seg_curr_granularity,
                                        label_true=label_true)
            if curr_dice_val > best_dice_val:
                best_dice_val = curr_dice_val
                label_best = label_curr_granularity
                seg_best = seg_curr_granularity
        
        for curr_granularity_idx in range(B2):
            label_curr_granularity = labels_cuts_diffusion.reshape((B2, H, W))[curr_granularity_idx]
            seg_curr_granularity = label_hint_seg(label_pred=label_curr_granularity,
                                                  label_true=label_true)
            if args.binary:
                dice_metric = dice_coeff
            else:
                dice_metric = per_class_dice_coeff
            curr_dice_val = dice_metric(label_pred=seg_curr_granularity,
                                        label_true=label_true)
            if curr_dice_val > best_dice_val:
                best_dice_val = curr_dice_val
                label_cuts_best = label_curr_granularity
                seg_cuts_best = seg_curr_granularity
        


        seg_cuts_kmeans = label_hint_seg(label_pred=label_kmeans,
                                    label_true=label_true)

        seg_kmeans = label_hint_seg(label_pred=label_kmeans,
                                    label_true=label_true)

        seg_spectralnet = label_hint_seg(label_pred=label_spectralnet,
                                    label_true=label_true)

        seg_pos_spectralnet = label_hint_seg(label_pred=label_pos_spectralnet,
                                    label_true=label_true)

        seg_cuts_persistent = label_hint_seg(label_pred=label_cuts_persistent,
                                        label_true=label_true)

        seg_cuts_best = label_hint_seg(label_pred=label_cuts_best,
                                  label_true=label_true)

        seg_persistent = label_hint_seg(label_pred=label_persistent,
                                        label_true=label_true)
        seg_best = label_hint_seg(label_pred=label_best,
                                  label_true=label_true)

        data_hashmap = {
            'image': image,
            'recon': recon,
            'latent': latent,
            'label_true': label_true,
            'label_random': label_random,
            'label_watershed': label_watershed,
            'label_felzenszwalb': label_felzenszwalb,
            'label_slic': label_slic,
            'label_dfc': label_dfc,
            'label_stego': label_stego,
            'label_sam': label_sam,
            'label_supervised_unet': label_supervised_unet,
            'label_supervised_nnunet': label_supervised_nnunet,
            'label_cuts_kmeans': label_cuts_kmeans,
            'label_pos_spectralnet': label_pos_spectralnet,
            'label_spectralnet': label_spectralnet,
            'label_kmeans': label_kmeans,
            'seg_cuts_kmeans': seg_cuts_kmeans,
            'seg_kmeans': seg_kmeans,
            'seg_spectralnet': seg_spectralnet,
            'seg_pos_spectralnet': seg_pos_spectralnet,
            'labels_cuts_diffusion': labels_cuts_diffusion,
            'labels_diffusion': labels_diffusion,
            'label_persistent': label_persistent,
            'seg_cuts_persistent': seg_cuts_persistent,
            'seg_persistent': seg_persistent,
            'label_best': label_best,
            'seg_cuts_best': seg_cuts_best,
            'seg_best': seg_best,
        }

        if args.comparison:
            if args.separate:
                fig = plot_comparison(fig=fig,
                                      num_samples=num_samples,
                                      sample_idx=sample_idx,
                                      data_hashmap=data_hashmap,
                                      image_grayscale=args.grayscale,
                                      label_binary=args.binary)
            else:
                assert args.binary
                fig = plot_overlaid_comparison(
                    fig=fig,
                    num_samples=num_samples,
                    sample_idx=sample_idx,
                    data_hashmap=data_hashmap,
                    image_grayscale=args.grayscale,
                    pred_color='blue'
                    if config.dataset_name == 'retina' else 'red')
        else:
            fig = plot_results(fig=fig,
                               num_samples=num_samples,
                               sample_idx=sample_idx,
                               data_hashmap=data_hashmap,
                               image_grayscale=args.grayscale)

    figure_str = ''
    for image_idx in args.image_idx:
        figure_str += str(image_idx) + '-'
    figure_str = figure_str.rstrip('-')

    fig_path = '%s/sample_%s' % (figure_folder, figure_str)
    fig.tight_layout()

    if args.comparison:
        if args.separate:
            fig.savefig('%s_figure_plot_comparison_separate.png' % fig_path)
        else:
            fig.savefig('%s_figure_plot_comparison.png' % fig_path)
    else:
        fig.savefig('%s_figure_plot.png' % fig_path)

    os._exit(0)
