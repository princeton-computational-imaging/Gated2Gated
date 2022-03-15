from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
import cv2

import json


def passive_loader(base_dir, img_id, crop_size_h, crop_size_w, cent_fnum,
                    img_ext='png',
                    num_bits=10, data_type='real',
                    scale_images=False,
                    scaled_img_width=None, scaled_img_height=None):
    normalizer = 2 ** num_bits - 1.

    if cent_fnum == 0:
        dir = os.path.join(base_dir, 'gated_passive_10bit')
    else:
        dir = os.path.join(base_dir, 'gated_passive_10bit_history_%d' % (cent_fnum))
    path = os.path.join(dir, img_id + f'.{img_ext}')
    assert os.path.exists(path), "No such file : %s" % path
    img = cv2.imread(os.path.join(dir, img_id + f'.{img_ext}'), cv2.IMREAD_UNCHANGED)
    if data_type == 'real':
        img = img[crop_size_h:(img.shape[0] - crop_size_h),
              crop_size_w:(img.shape[1] - crop_size_w)
              ]

        img = img.copy()
        img[img > 2 ** 10 - 1] = normalizer

    img = np.float32(img / normalizer, 0., 1.)
    if scale_images:
        img = cv2.resize(img, dsize=(scaled_img_width, scaled_img_height), interpolation=cv2.INTER_AREA)
    return img


def gated_loader(base_dir, img_id, crop_size_h, crop_size_w, history=None,
                 img_ext='png',   
                 num_bits=10, data_type='real',   
                 scale_images=False,
                 scaled_img_width=None, scaled_img_height=None):
    gated_imgs = []
    normalizer = 2 ** num_bits - 1.



    for gate_id in range(3):
        if history is None:
            gate_dir = os.path.join(base_dir,'gated%d_10bit' % gate_id)
        else:
            gate_dir = os.path.join(base_dir,'gated%d_10bit_history_%d'%(gate_id,history))
        path = os.path.join(gate_dir, img_id + f'.{img_ext}')
        assert os.path.exists(path),"No such file : %s"%path 
        img = cv2.imread(os.path.join(gate_dir, img_id + f'.{img_ext}'), cv2.IMREAD_UNCHANGED)
        if data_type == 'real':
            img = img[ crop_size_h:(img.shape[0] - crop_size_h),
                       crop_size_w:(img.shape[1] - crop_size_w)
                     ]
            
            img = img.copy()
            img[img > 2 ** 10 - 1] = normalizer
        
        img = np.float32(img / normalizer)
        gated_imgs.append(np.expand_dims(img, axis=2))
    img = np.concatenate(gated_imgs, axis=2)
    if scale_images:
        img = cv2.resize(img, dsize=(scaled_img_width, scaled_img_height), interpolation=cv2.INTER_AREA)
    return img

class GatedDataset(data.Dataset):

    def __init__(self,
                 gated_dir,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='png',
                 load_passive = False):
        super(GatedDataset, self).__init__()
        
        self.root_dir = gated_dir
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.img_ext = img_ext

        self.full_res_shape = (1280, 720)
        self.crop_size_h, self.crop_size_w = int((self.full_res_shape[1]-self.height)/2), int((self.full_res_shape[0]-self.width)/2),

        self.frame_idxs = frame_idxs

        self.is_train = is_train

        self.loader = gated_loader
        self.interp = Image.ANTIALIAS
        self.load_passive = load_passive
        if self.load_passive:
            self.passive_loader = passive_loader
            

        self.to_tensor = transforms.ToTensor()

        self.resize = {}

        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.K = np.array([[1.81,0.0, 0.52, 0.0 ],
                           [0.0, 3.23, 0.36, 0.0 ],
                           [0.0, 0.0, 1.0, 0.0 ],
                           [0.0, 0.0, 0.0, 1.0 ]], dtype=np.float32)

        
        self.load_depth = self.check_depth()

    def __getitem__(self, index):
        
        inputs = {}
        do_flip = self.is_train and random.random() > 0.5

        # line = self.filenames[index].split()
        line = self.filenames[index].split(',')
        frame_index = line[0]
        cent_fnum = int(line[1])

        inputs['frame_info'] = "{}-{}".format(frame_index,cent_fnum)

        for i in self.frame_idxs:
            history = i + cent_fnum     # Get temporal next or previous frame depending on frame_indx 
            history = None if history == 0 else history             
            inputs[("gated", i, -1)] = self.get_gated(frame_index,history,do_flip)


        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        gated_aug = (lambda x: x)
        self.preprocess(inputs, gated_aug)

        for i in self.frame_idxs:
            del inputs[("gated", i, -1)]
            del inputs[("gated_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(frame_index, cent_fnum, do_flip)
            inputs["depth_gt"] = torch.from_numpy(depth_gt)

        if self.load_passive:
            passive = self.get_passive(frame_index, cent_fnum, do_flip)
            inputs["passive"] = torch.from_numpy(passive)

        

        return inputs        

    def preprocess(self, inputs, gated_aug):
        """
            Resize colour images to the required scales and augment if required

            We create the gated_aug object in advance and apply the same augmentation to all
            images in this item. This ensures that all images input to the pose network receive the
            same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "gated" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    # inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
                    s = 2 ** i
                    scaled_img_width, scaled_img_height = self.width // s, self.height // s
                    inputs[(n, im, i)] = cv2.resize(inputs[(n, im, i - 1)], dsize=(scaled_img_width, scaled_img_height), interpolation=cv2.INTER_AREA) 

        for k in list(inputs):
            f = inputs[k]
            if "gated" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(gated_aug(f))

    def __len__(self):
        return len(self.filenames)

    def get_gated(self, frame_index, history, do_flip):
        gated = self.loader(self.root_dir, frame_index, self.crop_size_h, self.crop_size_w, history=history, img_ext=self.img_ext)
        if do_flip:
            gated = np.fliplr(gated).copy()
        return gated

    def get_passive(self, frame_index, cent_fnum, do_flip):
        passive = self.passive_loader(self.root_dir, frame_index, self.crop_size_h, self.crop_size_w, cent_fnum=cent_fnum, img_ext=self.img_ext)
        if do_flip:
            passive = np.fliplr(passive).copy()
        passive = np.expand_dims(passive, 0).astype(np.float32)
        return passive

    def get_depth(self, frame_index, cent_fnum, do_flip):
        if cent_fnum ==  0:
            lidar_filename = os.path.join(self.root_dir, 'lidar_hdl64_strongest_filtered_gated', frame_index + '.npz')
            depth_gt = np.load(lidar_filename)['arr_0']
            depth_gt = depth_gt[self.crop_size_h:self.full_res_shape[1] - self.crop_size_h, self.crop_size_w:self.full_res_shape[0] - self.crop_size_w]
        else:
            depth_gt = np.zeros((self.height, self.width))
        
        if do_flip:
            depth_gt = np.fliplr(depth_gt).copy()

        depth_gt = np.expand_dims(depth_gt, 0).astype(np.float32)
        return depth_gt



    def check_depth(self):
        sample = self.filenames[0].split(',')[0]
        lidar_filename = os.path.join(self.root_dir, 'lidar_hdl64_strongest_filtered_gated', '{}.npz'.format(sample))
        return os.path.isfile(lidar_filename)


    
        

