from . import gated_dataset
import numpy as np
import os
import cv2
import random
import torch

def read_gt_image(base_dir, img_id, data_type, depth_normalizer = 150.0, min_distance=0.1, max_distance=100.0, scale_images=False,
                  scaled_img_width=None,
                  crop_size_h= 104,crop_size_w = 128,
                  scaled_img_height=None, raw_values_only=False):
    
    if data_type == 'real':
        depth_lidar1 = np.load(os.path.join(base_dir, "depth_hdl64_gated_compressed", img_id + '.npz'))['arr_0']
        depth_lidar1 = depth_lidar1[crop_size_h:(depth_lidar1.shape[0] - crop_size_h),
                                    crop_size_w:(depth_lidar1.shape[1] - crop_size_w)]
        if raw_values_only:
            return depth_lidar1, None

        gt_mask = (depth_lidar1 > 0.)

        depth_lidar1 = np.float32(np.clip(depth_lidar1, min_distance, max_distance) / depth_normalizer)

        return depth_lidar1, gt_mask

    img = np.load(os.path.join(base_dir, 'depth_compressed', img_id + '.npz'))['arr_0']

    if raw_values_only:
        return img, None

    img = np.clip(img, min_distance, max_distance) / max_distance

    if scale_images:
        img = cv2.resize(img, dsize=(scaled_img_width, scaled_img_height), interpolation=cv2.INTER_AREA)

    return np.expand_dims(np.expand_dims(img, axis=2), axis=0), None

def read_gated_image(base_dir, img_id, num_bits=10, data_type='real', 
                     scale_images=False, scaled_img_width=None,crop_size_h= 104,crop_size_w = 128, scaled_img_height=None):
    gated_imgs = []
    normalizer = 2 ** num_bits - 1.

    for gate_id in range(3):
        gate_dir = os.path.join(base_dir,'gated%d_10bit' % gate_id)
        path = os.path.join(gate_dir, img_id + '.png')
        assert os.path.exists(path),"No such file : %s"%path 
        img = cv2.imread(os.path.join(gate_dir, img_id + '.png'), cv2.IMREAD_UNCHANGED)
        if data_type == 'real':
            img = img[crop_size_h:(img.shape[0] - crop_size_h),
                      crop_size_w:(img.shape[1] - crop_size_w)]
            img = img.copy()
            img[img > 2 ** 10 - 1] = normalizer
        
        img = np.float32(img / normalizer)
        gated_imgs.append(np.expand_dims(img, axis=2))
    img = np.concatenate(gated_imgs, axis=2)
    if scale_images:
        img = cv2.resize(img, dsize=(scaled_img_width, scaled_img_height), interpolation=cv2.INTER_AREA)
    return img

class Gated2DepthDataset(gated_dataset.GatedDataset):
    
    def __init__(self, gated_dir, filenames,
                 height, width, num_scales, depth_normalizer = 150.0,
                 frame_idxs = [0],
                 is_train=False):
        super().__init__(gated_dir, filenames, height, width, frame_idxs,
                         num_scales, is_train=is_train)
        assert frame_idxs == [0], "Gated2depth dataset has no temporal frames"
        self.depth_normalizer = depth_normalizer
        self.load_depth = self.check_depth()
        self.depth_loader = read_gt_image
        self.loader = read_gated_image

    def __getitem__(self, index):
        
        inputs = {}
        do_flip = self.is_train and random.random() > 0.5

        # line = self.filenames[index].split()
        line = self.filenames[index].split(',')
        frame_index = line[0]
        
        # there is no temporal data for gated2depth dataset    
        inputs[("gated", 0, -1)] = self.get_gated(frame_index,do_flip)
        inputs["depth_gt"] = self.get_depth(frame_index,do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        color_aug = (lambda x: x)
        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("gated", i, -1)]
            del inputs[("gated_aug", i, -1)]
           

        return inputs 

    def preprocess(self, inputs, color_aug):
       
        for k in list(inputs):
            frame = inputs[k]
            if "gated" in k :
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
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def get_depth(self,frame_index,do_flip):
        depth_gt,_ = self.depth_loader(self.root_dir, frame_index, 'real', depth_normalizer=self.depth_normalizer)
        if do_flip:
            depth_gt = np.fliplr(depth_gt).copy()
        return depth_gt
    
    def get_gated(self, frame_index, do_flip):
        gated = self.loader(self.root_dir,frame_index)

        if do_flip:
            gated = np.fliplr(gated).copy()

        return gated

    def check_depth(self):
        return True # Gated2Depth dataset has lidar data