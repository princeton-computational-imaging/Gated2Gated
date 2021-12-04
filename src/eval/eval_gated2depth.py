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

cmap_dict = {
    'jet': cm.jet,
    'jet_r': cm.jet_r,
    'plasma': cm.plasma,
    'plasma_r': cm.plasma_r
}

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

def compute_errors_mdp(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

min_val = 1e-7

def threshold(y1, y2, thr=1.25):
    max_ratio = np.maximum(y1 / y2, y2 / y1)
    return np.mean(max_ratio < thr, dtype=np.float64) * 100.


def rmse(y1, y2):
    diff = y1 - y2
    return math.sqrt(np.mean(diff * diff, dtype=np.float64))


def rmse_log(y1, y2):
    return rmse(np.log(y1), np.log(y2))


def ard(y1, y2):
    return np.mean(np.abs(y1 - y2) / y2, dtype=np.float64)


def mae(y1, y2):
    return np.mean(np.abs(y1 - y2), dtype=np.float64)

def compute_errors_gated(groundtruth, output, min_distance=3., max_distance=150.):
    output = output[groundtruth > 0]
    groundtruth = groundtruth[groundtruth > 0]
    output = np.clip(output, min_distance, max_distance)
    groundtruth = np.clip(groundtruth, min_distance, max_distance)

    return rmse(output, groundtruth), rmse_log(output, groundtruth), \
           ard(output, groundtruth), mae(output, groundtruth), \
           threshold(output, groundtruth, thr=1.25), \
           threshold(output, groundtruth, thr=1.25 ** 2), threshold(output, groundtruth, thr=1.25 ** 3)


def read_img(img_path,
             num_bits=10,  
             scale_images=False,
             scaled_img_width=None, scaled_img_height=None,
             crop_height=512,crop_width=1024):
    
    gated_imgs = []
    normalizer = 2 ** num_bits - 1.

    for gate_id in range(3):
        path = img_path.format(gate_id)
        assert os.path.exists(path),"No such file : %s"%path 
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img[((img.shape[0]-crop_height)//2):((img.shape[0]+crop_height)//2),((img.shape[1]-crop_width)//2):((img.shape[1] + crop_width)//2)] 
        img = img.copy()
        img[img > 2 ** 10 - 1] = normalizer
        img = np.float32(img / normalizer)
        gated_imgs.append(np.expand_dims(img, axis=2))
    img = np.concatenate(gated_imgs, axis=2)
    if scale_images:
        img = cv2.resize(img, dsize=(scaled_img_width, scaled_img_height), interpolation=cv2.INTER_AREA)
    return img

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 3.0
    MAX_DEPTH = 80.0
    
    # Load dataset items
    dataset_dir = opt.data_dir                       # "/mnt/storage03-srv2/users/aman/data/MonocularGated"

    with open("splits/gated2depth/val_files_ids.txt", "r") as f:
        val_ids = f.read().split('\n')

    lidar_paths = [os.path.join(dataset_dir,"depth_hdl64_gated_compressed","{}.npz".format(_id)) for _id in val_ids]

    # gated_ids = [os.path.splitext(os.path.basename(item))[0] for item in lidar_paths]
    gated_paths = [os.path.join(dataset_dir,"gated{}_10bit","{}.png".format(_id)) for _id in val_ids]    

    # Load weights
    assert os.path.isdir(opt.load_weights_folder), "Cannot find a folder at {}".format(opt.load_weights_folder)
    print("-> Loading weights from {}".format(opt.load_weights_folder))
    depth_path = os.path.join(opt.load_weights_folder, "depth.pth")
    depth_dict = torch.load(depth_path)

    depth_net = networks.PackNetSlim01(dropout=0.5,version="1A")
    model_dict = depth_net.state_dict()
    depth_net.load_state_dict({k: v for k, v in depth_dict.items() if k in model_dict})
    depth_net.cuda()

    pred_disps = []
    errors = []

    print("-> Computing predictions with size {}x{}".format(opt.height, opt.width))

    # Making directory for storing results
    result_dirs = ['gated2depth', 'gated2depth_img', 'all']
    for result_folder in result_dirs:
        if not os.path.exists(os.path.join(opt.results_dir,opt.cmap, result_folder)):
            os.makedirs(os.path.join(opt.results_dir,opt.cmap, result_folder))

    with torch.no_grad():
        for lidar_path,gated_path in tzip(lidar_paths,gated_paths):

            img_id = os.path.basename(gated_path).split('.')[0]

            gated_img = read_img(gated_path,crop_height=opt.height,crop_width=opt.width)
            
            lidar = np.load(lidar_path)['arr_0']
            gt_depth = lidar[((lidar.shape[0]-opt.height)//2):((lidar.shape[0]+opt.height)//2),
                            ((lidar.shape[1]-opt.width)//2):((lidar.shape[1] + opt.width)//2)] 
            
            input_patch = gated_transform(gated_img).unsqueeze(0).cuda()
            output = depth_net(input_patch)
            
            _, pred_depth = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_depth = pred_depth[0,0].cpu().numpy() * opt.depth_normalizer

            
            # Generate mask for depth evaluation
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            np.savez_compressed(os.path.join(opt.results_dir,opt.cmap, 'gated2depth', '{}'.format(img_id)), pred_depth)

            '''
                Generate graphics for results
            '''
            if opt.gen_figs:    
                input_patch = input_patch.permute(0,2,3,1).cpu().numpy()

                # Generate colorized pointcloud from Lidar
                depth_lidar1_color = visualize2D.colorize_pointcloud(gt_depth, min_distance=MIN_DEPTH,max_distance=MAX_DEPTH, radius=3, cmap=cm.plasma)

                #Generate colorized depth map
                depth_map_color = visualize2D.colorize_depth(pred_depth, min_distance=MIN_DEPTH, max_distance=MAX_DEPTH, cmap=cm.plasma)

                in_out_shape = (int(depth_map_color.shape[0] + depth_map_color.shape[0] / 3. + gt_depth.shape[0]), depth_map_color.shape[1], 3)


                input_output = np.zeros(shape=in_out_shape)
                scaled_input = cv2.resize(input_patch[0, :, :, :],
                                        dsize=(int(input_patch.shape[2] / 3), int(input_patch.shape[1] / 3)),
                                        interpolation=cv2.INTER_AREA) * 255

                for i in range(3):
                    input_output[:scaled_input.shape[0], :scaled_input.shape[1], i] = scaled_input[:, :, 0]
                    input_output[:scaled_input.shape[0], scaled_input.shape[1]: 2 * scaled_input.shape[1], i] = scaled_input[:, :, 1]
                    input_output[:scaled_input.shape[0], scaled_input.shape[1] * 2:scaled_input.shape[1] * 3, i] = scaled_input[:, :, 2]

                input_output[scaled_input.shape[0]: scaled_input.shape[0] + depth_map_color.shape[0], :, :] = depth_map_color
                input_output[scaled_input.shape[0] + depth_map_color.shape[0]:, :, :] = depth_lidar1_color
                
                cv2.imwrite(os.path.join(opt.results_dir,opt.cmap, 'gated2depth_img', '{}.jpg'.format(img_id)), depth_map_color.astype(np.uint8))
                cv2.imwrite(os.path.join(opt.results_dir,opt.cmap, 'all', '{}.jpg'.format(img_id)), input_output.astype(np.uint8))
            
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            # error = compute_errors(gt_depth, pred_depth)
            error = compute_errors_gated(gt_depth, pred_depth)
            
            if not np.isnan(np.sum(np.array(error))):
                errors.append(error)
            else:
                print(img_id)
    
        mean_errors = np.array(errors).mean(0)

        # print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print("\n  " + ("{:>8} | " * 7).format('rmse', 'rmse_log', 'ard', 'mae', 'delta1', 'delta2', 'delta3'))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")



if __name__ == "__main__":
    options = argparse.ArgumentParser()
    options.add_argument("--data_dir",required=True,
                        help="Path to the dataset directory")
    options.add_argument("--min_depth",default=0.1,
                        type=float,
                        help="Minimum depth value to evaluate")
    options.add_argument("--max_depth",default=100.0,
                        type=float,
                        help="Max depth value to evaluate")
    options.add_argument("--height",default=512,
                        type=int,
                        help="height of crop for gated image")
    options.add_argument("--width",default=1024,
                        type=int,
                        help="width of crop for gated image")
    options.add_argument("--depth_normalizer",default=70.0,
                        type=float,
                        help="depth normalizer to multiply predicted depth with")
    options.add_argument("--load_weights_folder",required=True,
                         help="Path where weights are stored")
    options.add_argument("--results_dir",required=True,
                         help="Path where results are stored")
    options.add_argument("--gen_figs",action='store_true',
                        help="Whether to generate figures or not")
    options.add_argument("--cmap",default='jet',
                         choices=['jet','jet_r','plasma','plasma_r'],   
                        help="Which colormap to use for generating results")

    options = options.parse_args()
    evaluate(options)