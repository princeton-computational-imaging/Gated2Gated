# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib
import torch
import cv2
import io
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))

def fig2img(fig_buffer):
    buf = io.BytesIO()
    plt.axis('off')
    plt.savefig(buf, format="png",transparent = True, bbox_inches = 'tight', pad_inches = 0,dpi=100, facecolor=(0, 0, 0))
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img

def disp_to_mpimg(disp,colormap='jet_r'):
    fig = plt.figure(figsize=(30,20),dpi=100)
    plt.subplots_adjust(wspace=0.00,hspace=0.00)
    
    fig.add_subplot(111)
    plt.imshow(disp,cmap=colormap)
    plt.axis('off')
    
    img = fig2img(fig)
    plt.close()
    # im_pil = Image.fromarray(img)
    # cv2.imwrite("debug.png",img)
    return img

def snr_binary_mask(gated_img, min_intns = 0.04, max_intns = 0.98):
    """[snr_mask_binary calculates a binary mask based on the SNR and the maximum intensity of the input gated image]

    Args:
        gated_img ([torch.Tensor]): [gated image of dimension B x 3 x H x W]
    Returns:
        [torch.Tensor]: [Mask with dimension B x 1 x H x W]
    """
    max_intensity,_ = torch.max(gated_img, dim=1, keepdims=True)
    min_intensity,_ = torch.min(gated_img, dim=1, keepdims=True)
    snr = max_intensity - min_intensity
    snr_binary_mask = torch.logical_and(snr > min_intns, max_intensity < max_intns).float() 
    return snr_binary_mask

def intensity_mask(gated_img, depth):
    """[intensity_mask calculates a mask based on the intensities of the input gated image and the utilized range intensity profiles and the depth of the flat world]
    Args:
        gated_img ([torch.Tensor]): [gated image of dimension B x 3 x H x W]
    Returns:
        [torch.Tensor]: [Mask with dimension B x 1 x H x W]
    """
    max_intensity,_ = torch.max(gated_img, dim=1, keepdims=True)
    mask1 = max_intensity == gated_img[:,0:1,:,:]
    mask2 = torch.logical_and(max_intensity == gated_img[:,1:2,:,:], depth > 30. * torch.normal(1., 0.1, size=(depth.size())).to(device=depth.device))
    mask3 = torch.logical_and(max_intensity == gated_img[:,2:3,:,:], depth > 73. * torch.normal(1., 0.1, size=(depth.size())).to(device=depth.device))
    intensity_mask = mask1 + mask2 + mask3
    intensity_mask = (intensity_mask > 0.0).float()
    return intensity_mask

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and (p.grad is not None):
            layers.append(n)
            ave_grads.append(p.grad.abs().detach().cpu().numpy().mean())
            max_grads.append(p.grad.abs().detach().cpu().numpy().max())
    
    fig = plt.figure(figsize=(30,20),dpi=100)
    plt.subplots_adjust(wspace=0.00,hspace=0.00)
    
    fig.add_subplot(111)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    # plt.savefig('grad.png')
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches = 'tight', pad_inches = 0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    # cv2.imwrite('grad_img.png',img)
    plt.close()
    return img

def depth_image(depth, min_depth=0.1, max_depth = 100.0, colormap='jet_r'):
    fig = plt.figure(figsize=(20,10),dpi=100)
    plt.subplots_adjust(wspace=0.00,hspace=0.00)

    depth = np.clip(depth, min_depth, max_depth)    
    depth[0,0] = min_depth
    depth[-1,-1] = max_depth

     
    
    fig.add_subplot(111)
    plt.imshow(depth,cmap=colormap)
    plt.axis('off')
    plt.colorbar(aspect=80,orientation='horizontal',pad=0.01)
    
    img = fig2img(fig)
    plt.close()
    return img