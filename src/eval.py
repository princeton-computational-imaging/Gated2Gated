from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import matplotlib.cm as cm

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
# from utils import readlines
# from options import MonodepthOptions
import networks
import argparse

from torchvision.transforms import ToTensor

gated_transform = ToTensor()
from tqdm.contrib import tzip

import matplotlib.pyplot as plt

import visualize2D
import math
import PIL.Image as pil


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


def read_sample_files(train_samples_files):
    samples = []
    with open(train_samples_files, 'r') as f:
        samples += f.read().splitlines()
    samples = [sample.replace(',', '_') for sample in samples]
    return samples


def threshold(y1, y2, thr=1.25):
    max_ratio = np.maximum(y1 / y2, y2 / y1)
    return np.mean(max_ratio < thr, dtype=np.float64) * 100.


def rmse(y1, y2):
    diff = y1 - y2
    return math.sqrt(np.mean(diff * diff, dtype=np.float64))


def ard(y1, y2):
    return np.mean(np.abs(y1 - y2) / y2, dtype=np.float64)


def mae(y1, y2):
    return np.mean(np.abs(y1 - y2), dtype=np.float64)


def compute_errors(groundtruth, output, min_distance=3., max_distance=150.):
    output = output[groundtruth > 0]
    groundtruth = groundtruth[groundtruth > 0]
    output = np.clip(output, min_distance, max_distance)
    groundtruth = np.clip(groundtruth, min_distance, max_distance)

    return rmse(output, groundtruth), \
           mae(output, groundtruth), ard(output, groundtruth), \
           threshold(output, groundtruth, thr=1.25), \
           threshold(output, groundtruth, thr=1.25 ** 2), threshold(output, groundtruth, thr=1.25 ** 3)


def calc_bins(clip_min, clip_max, nb_bins):
    bins = np.linspace(clip_min, clip_max, num=nb_bins + 1)
    mean_bins = np.array([0.5 * (bins[i + 1] + bins[i]) for i in range(0, nb_bins)])
    return bins, mean_bins


def read_img(img_path,
             num_bits=10,
             crop_height=512, crop_width=1024, dataset='g2d'):
    gated_imgs = []
    normalizer = 2 ** num_bits - 1.

    for gate_id in range(3):
        path = img_path.format(gate_id)
        assert os.path.exists(path), "No such file : %s" % path
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img[((img.shape[0] - crop_height) // 2):((img.shape[0] + crop_height) // 2),
              ((img.shape[1] - crop_width) // 2):((img.shape[1] + crop_width) // 2)]
        img = img.copy()
        img[img > 2 ** 10 - 1] = normalizer
        img = np.float32(img / normalizer)
        gated_imgs.append(np.expand_dims(img, axis=2))
    img = np.concatenate(gated_imgs, axis=2)
    return img


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 3.0
    MAX_DEPTH = 80.0

    # Load dataset items
    dataset_dir = opt.data_dir
    eval_files_name = os.path.basename(opt.eval_files_path).replace('.txt', '')

    val_ids = sorted(read_sample_files(opt.eval_files_path))
    if opt.dataset == 'g2d':
        lidar_paths = [os.path.join(dataset_dir, "depth_hdl64_gated_compressed", "{}.npz".format(_id)) for _id in
                       val_ids]
        gated_paths = [os.path.join(dataset_dir, "gated{}_10bit", "{}.{}".format(_id,opt.img_ext)) for _id in val_ids]
    elif opt.dataset == 'stf':
        lidar_paths = [os.path.join(dataset_dir, "lidar_hdl64_strongest_filtered_gated", "{}.npz".format(_id)) for _id
                       in val_ids]
        gated_paths = [os.path.join(dataset_dir, "gated{}_10bit", "{}.{}".format(_id,opt.img_ext)) for _id in val_ids]

    # Load weights
    assert os.path.isdir(opt.load_weights_folder), "Cannot find a folder at {}".format(opt.load_weights_folder)
    print("-> Loading weights from {}".format(opt.load_weights_folder))
    depth_path = os.path.join(opt.load_weights_folder, "depth.pth")
    depth_dict = torch.load(depth_path)

    depth_net = networks.PackNetSlim01(dropout=0.5, version="1A")
    model_dict = depth_net.state_dict()
    depth_net.load_state_dict({k: v for k, v in depth_dict.items() if k in model_dict})
    depth_net.cuda()
    depth_net.eval()

    print("-> Computing predictions with size {}x{}".format(opt.height, opt.width))
    if opt.g2d_crop:
        g2d_width = 980
        g2d_height = 420
        assert opt.width >= g2d_width and opt.height >= g2d_height, 'Gated2Depth Crop can only be applied for width >= {} and height >= {}'.format(
            g2d_height, g2d_height)
        print("-> Computing metrics for Gated2Depth crop 420x980".format(opt.height, opt.width))

    if not os.path.exists(os.path.join(opt.results_dir)):
        os.makedirs(os.path.join(opt.results_dir))

    errors = []

    if opt.binned_metrics:
        average_points = 15000
        results_counter = 0
        results = np.zeros((average_points * len(lidar_paths), 2), dtype=np.float32)

    with torch.no_grad():
        for lidar_path, gated_path in tzip(lidar_paths, gated_paths):

            img_id = os.path.basename(gated_path).split('.')[0]

            gated_img = read_img(gated_path, crop_height=opt.height, crop_width=opt.width, dataset=opt.dataset)

            lidar = np.load(lidar_path)['arr_0']
            gt_depth = lidar[((lidar.shape[0] - opt.height) // 2):((lidar.shape[0] + opt.height) // 2),
                       ((lidar.shape[1] - opt.width) // 2):((lidar.shape[1] + opt.width) // 2)]

            input_patch = gated_transform(gated_img).unsqueeze(0).cuda()
            output = depth_net(input_patch)

            _, pred_depth = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_depth = pred_depth[0, 0].cpu().numpy() * opt.depth_normalizer

            ### Generate graphics for results ###
            if opt.gen_figs:
                # Making directory for storing results
                result_dirs = ['gated2gated_imgs', 'all', 'gated2gated']
                for result_folder in result_dirs:
                    if not os.path.exists(os.path.join(opt.results_dir, result_folder)):
                        os.makedirs(os.path.join(opt.results_dir, result_folder))
                input_patch = input_patch.permute(0, 2, 3, 1).cpu().numpy()

                # Generate colorized pointcloud from Lidar
                depth_lidar1_color = visualize2D.colorize_pointcloud(gt_depth, min_distance=MIN_DEPTH,
                                                                     max_distance=MAX_DEPTH, radius=3, cmap=cm.plasma)

                # Generate colorized depth map
                depth_map_color = visualize2D.colorize_depth(pred_depth, min_distance=MIN_DEPTH, max_distance=MAX_DEPTH,
                                                             cmap=cm.plasma)

                in_out_shape = (int(depth_map_color.shape[0] + depth_map_color.shape[0] / 3. + gt_depth.shape[0]),
                                depth_map_color.shape[1], 3)

                input_output = np.zeros(shape=in_out_shape)
                scaled_input = cv2.resize(input_patch[0, :, :, :],
                                          dsize=(int(input_patch.shape[2] / 3), int(input_patch.shape[1] / 3)),
                                          interpolation=cv2.INTER_AREA) * 255

                for i in range(3):
                    input_output[:scaled_input.shape[0], :scaled_input.shape[1], i] = scaled_input[:, :, 0]
                    input_output[:scaled_input.shape[0], scaled_input.shape[1]: 2 * scaled_input.shape[1],
                    i] = scaled_input[:, :, 1]
                    input_output[:scaled_input.shape[0], scaled_input.shape[1] * 2:scaled_input.shape[1] * 3,
                    i] = scaled_input[:, :, 2]

                input_output[scaled_input.shape[0]: scaled_input.shape[0] + depth_map_color.shape[0], :,
                :] = depth_map_color
                input_output[scaled_input.shape[0] + depth_map_color.shape[0]:, :, :] = depth_lidar1_color
                depth_map_color = pil.fromarray(depth_map_color.astype(np.uint8))
                input_output = pil.fromarray(input_output.astype(np.uint8))
                depth_map_color.save(os.path.join(opt.results_dir, 'gated2gated_imgs', '{}.jpg'.format(img_id)))
                input_output.save(os.path.join(opt.results_dir, 'all', '{}.jpg'.format(img_id)))

                np.savez_compressed(os.path.join(opt.results_dir, 'gated2gated', '{}'.format(img_id)), pred_depth)

            # check whether groundtruth depthmap contains any lidar point


            if opt.g2d_crop:
                gt_depth = gt_depth[((gt_depth.shape[0] - g2d_height) // 2):((gt_depth.shape[0] + g2d_height) // 2),
                           ((gt_depth.shape[1] - g2d_width) // 2):((gt_depth.shape[1] + g2d_width) // 2)]
                pred_depth = pred_depth[
                             ((pred_depth.shape[0] - g2d_height) // 2):((pred_depth.shape[0] + g2d_height) // 2),
                             ((pred_depth.shape[1] - g2d_width) // 2):((pred_depth.shape[1] + g2d_width) // 2)]

            if np.sum(gt_depth > 0.0) > 0.:

                error = compute_errors(gt_depth, pred_depth, min_distance=MIN_DEPTH, max_distance=MAX_DEPTH)
                errors.append(error)

                if opt.binned_metrics:
                    pred_depth = pred_depth[gt_depth > 0]
                    gt_depth = gt_depth[gt_depth > 0]

                    if results_counter + len(gt_depth) > results.shape[0]:
                        print('Overflow')
                        break

                    results[results_counter:results_counter + len(gt_depth), 0] = gt_depth
                    results[results_counter:results_counter + len(gt_depth), 1] = pred_depth

                    results_counter += len(gt_depth)

        # Print and save metrics
        print('### Metrics ###')
        res = np.array(errors).mean(0)
        metric_str = ['rmse', 'mae', 'ard', 'delta1', 'delta2', 'delta3']
        res_str = ''
        for i in range(res.shape[0]):
            res_str += '{}={:.2f} \n'.format(metric_str[i], res[i])
        print(res_str)
        with open(os.path.join(opt.results_dir, '{}_results.txt'.format(eval_files_name)), 'w') as f:
            f.write(res_str)
        with open(os.path.join(opt.results_dir, '{}_results.tex'.format(eval_files_name)), 'w') as f:
            f.write(' & '.join(metric_str) + '\n')
            f.write(' & '.join(['{:.2f}'.format(r) for r in res]))

        # Print and save binned metrics
        if opt.binned_metrics:
            print('### Binned Metrics ###')
            results = results[results[:, 0] != 0]

            bins = np.linspace(MIN_DEPTH, MAX_DEPTH, num=12)
            inds = np.digitize(results[:, 0], bins)

            binned_results = np.zeros((len(bins), 6 + 1))
            for i, bin in enumerate(bins):
                metrics = compute_errors(results[inds == i + 1, 0], results[inds == i + 1, 1], min_distance=MIN_DEPTH,
                                         max_distance=MAX_DEPTH)
                binned_results[i, 0] = bin
                binned_results[i, 1:] = metrics

            with open(os.path.join(opt.results_dir, '{}_binned_distance_results.txt'.format(eval_files_name)),
                      'w') as f:
                np.savetxt(f, binned_results, delimiter=' ')

            mean_error_binned = np.zeros((6, 1))
            for i in range(0, 6):
                mean_error_binned[i] = np.mean(binned_results[~np.isnan(binned_results[:, i + 1]), i + 1])
            res_str = ''
            for i in range(res.shape[0]):
                res_str += '{}={:.2f} \n'.format(metric_str[i], float(mean_error_binned[i]))
            print(res_str)
            with open(os.path.join(opt.results_dir, '{}_binned_results.txt'.format(eval_files_name)), 'w') as f:
                f.write(res_str)
            with open(os.path.join(opt.results_dir, '{}_binned_results.tex'.format(eval_files_name)), 'w') as f:
                f.write(' & '.join(metric_str) + '\n')
                np.savetxt(f, np.transpose(mean_error_binned), delimiter=' & ', fmt='%.2f')


if __name__ == "__main__":
    options = argparse.ArgumentParser()
    options.add_argument("--data_dir", required=True,
                         help="Path to the dataset directory")
    options.add_argument("--min_depth", default=0.1,
                         type=float,
                         help="Minimum depth value to evaluate")
    options.add_argument("--max_depth", default=100.0,
                         type=float,
                         help="Max depth value to evaluate")
    options.add_argument("--height", default=512,
                         type=int,
                         help="height of crop for gated image")
    options.add_argument("--width", default=1024,
                         type=int,
                         help="width of crop for gated image")
    options.add_argument("--img_ext", default='png',
                         help="image extension (without .)")
    options.add_argument("--depth_normalizer", default=70.0,
                         type=float,
                         help="depth normalizer to multiply predicted depth with")
    options.add_argument("--load_weights_folder", required=True,
                         help="Path where weights are stored")
    options.add_argument("--results_dir", required=True,
                         help="Path where results are stored")
    options.add_argument("--gen_figs", action='store_true',
                         help="Whether to generate figures or not")
    options.add_argument("--eval_files_path",
                         help="Path to file with validation/evaluation file names.",
                         required=True)
    options.add_argument("--dataset", default='stf',
                         choices=['stf', 'g2d'],
                         help="Which dataset is used for evaluation.")
    options.add_argument('--g2d_crop', help='Use same crop as used for Evaluation in Gated2Depth Paper.',
                         action='store_true', required=False)
    options.add_argument('--binned_metrics', help='Calculate additional binned metrics',
                         action='store_true', required=False)

    options = options.parse_args()
    evaluate(options)