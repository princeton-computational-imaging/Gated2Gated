from __future__ import absolute_import, division, print_function

import numpy as np
import time
import os
import json
import matplotlib as mpl
import matplotlib.cm as cm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from collections import OrderedDict

import networks
import dataset

from layers import *
from utils import *

class Trainer:

    def __init__(self, options) -> None:
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.exp_name, "exp-{}_{}".format(self.opt.exp_num,self.opt.exp_metainfo))

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        norm = mpl.colors.Normalize(vmin=self.opt.min_depth, vmax=self.opt.max_depth)
        cmap = cm.jet
        self.m = cm.ScalarMappable(norm=norm, cmap=cmap)

        self.models = {}
        self.parameters_to_train = []

        # num_bits : number of bits used to store gated image in png format
        self.normalizer = 2**self.opt.num_bits - 1
        self.dark_levels = np.array([87.,70.,87.]).reshape(1,3,1,1).astype(np.float32)
        self.load_passive = True if self.opt.passive_supervision else False

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.frame_ids == [0])

        if self.opt.model_type == "multioutput":
            self.models["gated2gated"] = networks.PackNetSlim01MultiDecoder(dropout = self.opt.dropout,
                                                                            version="{}{}".format(1,self.opt.feat_stack),
                                                                            use_batchnorm=self.opt.use_batchnorm)
            self.models["gated2gated"].to(self.device)
            self.parameters_to_train += list(self.models["gated2gated"].parameters())

        elif self.opt.model_type == "multinetwork":
            
            #===============================================
            #                    PackNet
            #===============================================
            if self.opt.depth_model == "packnet":
                self.models["depth"] = networks.PackNetSlim01(dropout = self.opt.dropout,
                                                            version="{}{}".format(1,self.opt.feat_stack))
                self.models["depth"].to(self.device)
                if not self.opt.train_depth_normalizer:
                    self.parameters_to_train += list(self.models["depth"].parameters())
            elif self.opt.depth_model == "packnet_full":
                self.models["depth"] = networks.PackNet01(dropout = self.opt.dropout,
                                                            version="{}{}".format(1,self.opt.feat_stack))
                self.models["depth"].to(self.device)
                if not self.opt.train_depth_normalizer:
                    self.parameters_to_train += list(self.models["depth"].parameters())
            #===============================================
            #                    ResNet
            #===============================================
            elif self.opt.depth_model == "resnet":
                self.models["depth_encoder"] = networks.ResnetEncoder(self.opt.num_layers,
                                                                    self.opt.weights_init == "pretrained")
                self.models["depth_encoder"].to(self.device)

                self.models["depth"] = networks.DepthDecoder(
                    self.models["depth_encoder"].num_ch_enc, self.opt.scales)
                self.models["depth"].to(self.device)

                if not self.opt.train_depth_normalizer:
                    self.parameters_to_train += list(self.models["depth_encoder"].parameters())
                    self.parameters_to_train += list(self.models["depth"].parameters())

           
            self.models["encoder"] = networks.Encoder(num_convs=self.opt.num_convs)
            self.models["encoder"].to(self.device)
            if not self.opt.train_depth_normalizer:
                self.parameters_to_train += list(self.models["encoder"].parameters())

            self.models["albedo"] = networks.Decoder(name = "albedo", scales = range(1), out_channels=1)
            self.models["albedo"].to(self.device)
            if not self.opt.train_depth_normalizer:
                self.parameters_to_train += list(self.models["albedo"].parameters())
            
            self.models["ambient"] = networks.Decoder(name = "ambient", scales = range(1), out_channels=1)
            self.models["ambient"].to(self.device)
            if not self.opt.train_depth_normalizer:
                self.parameters_to_train += list(self.models["ambient"].parameters())

            # self.depth_normalizer = nn.Parameter(data = torch.tensor(self.opt.depth_normalizer),requires_grad=self.opt.train_depth_normalizer).to(self.device)
            self.depth_normalizer = torch.tensor(self.opt.depth_normalizer,requires_grad=self.opt.train_depth_normalizer, device=self.device)
            
            if self.opt.train_depth_normalizer:
                self.parameters_to_train += [self.depth_normalizer]

        if self.use_pose_net and self.opt.temporal_loss:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(self.opt.num_layers,
                                                                     self.opt.weights_init == "pretrained",
                                                                     num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                if not self.opt.train_depth_normalizer:
                    if not self.opt.freeze_pose_net:
                        self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc,
                                                           num_input_features=1,
                                                           num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            if not self.opt.train_depth_normalizer:
                if not self.opt.freeze_pose_net:
                    self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("experiment number:\n  Training model named:\n  ", self.opt.exp_num, self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"gated":      dataset.GatedDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))


        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(self.opt.data_dir,
                                     train_filenames,
                                     self.opt.height, self.opt.width,
                                     self.opt.frame_ids,
                                     num_scales = 4,
                                     is_train=True,
                                     load_passive=self.load_passive)

        self.train_loader = DataLoader(train_dataset, self.opt.batch_size, shuffle=True,
                                       num_workers=self.opt.num_workers,
                                       pin_memory=True, drop_last=True)
        
        val_dataset = self.dataset(self.opt.data_dir,
                                   val_filenames,
                                   self.opt.height, self.opt.width,
                                   self.opt.frame_ids,
                                   num_scales = 4,
                                   is_train=False,
                                   load_passive=self.load_passive)

        self.val_loader = DataLoader(val_dataset, self.opt.batch_size, shuffle=True,
                                     num_workers=self.opt.num_workers,
                                     pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))


        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        
        if self.opt.cycle_loss:
            self.sim_gated = SimulateGated(cheb_path=self.opt.coeff_fpath,
                                           dark_levels = self.dark_levels, 
                                           depth_normalizer=self.opt.depth_normalizer,
                                           num_bits=self.opt.num_bits,
                                           min_depth=self.opt.min_depth,
                                           max_depth=self.opt.max_depth).to(self.device)
        if self.opt.intensity_mask:
            full_res_shape = (1280, 720)
            crop_size_h, crop_size_w = int((full_res_shape[1] - self.opt.height) / 2), int((full_res_shape[0] - self.opt.width) / 2)
            assert os.path.isfile(os.path.join(self.opt.depth_flat_world_fpath, 'depth_flat_world.npz')), "path to flat depthmap is required for using intensity mask!"
            self.depth_flat_world = np.load(os.path.join(self.opt.depth_flat_world_fpath, 'depth_flat_world.npz'))['arr_0']
            self.depth_flat_world = self.depth_flat_world[:,:,crop_size_h:(self.depth_flat_world.shape[2] - crop_size_h), crop_size_w:(self.depth_flat_world.shape[3] - crop_size_w)]
            self.depth_flat_world = torch.tensor(self.depth_flat_world)  # [1,1,H,W]
            self.depth_flat_world = self.depth_flat_world.to(self.device)
       
        if self.opt.temporal_loss:
            for scale in self.opt.scales:
                h = self.opt.height // (2 ** scale)
                w = self.opt.width // (2 ** scale)

                self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
                self.backproject_depth[scale].to(self.device)

                self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
                self.project_3d[scale].to(self.device)
                

        self.depth_metric_names = ["de/abs", "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        if not self.opt.train_depth_normalizer:
            for name, model in self.models.items():
                if "pose" not in name:
                    model.train()
                elif not self.opt.freeze_pose_net:
                    model.train()
            if self.opt.freeze_pose_net:
                self.freeze_pose_network()

            self.depth_normalizer.requires_grad = False
        else:
            self.freeze_all_networks()
            self.depth_normalizer.requires_grad = True


    def freeze_pose_network(self):
        """set gradient flow for all pose networks to be zero
        """
        for name, model in self.models.items():
            if "pose" in name:
                for param in model.parameters():
                    param.requires_grad = False

    def freeze_all_networks(self):
        """set gradient flow for all networks to be zero
        """
        for name, model in self.models.items():
            for param in model.parameters():
                param.requires_grad = False

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()
        self.depth_normalizer.requires_grad = False

        
    def train(self):
        """
            Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.start_epoch, self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
    
    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs and torch.sum(inputs["depth_gt"]) > 0. and ("sc_depth", 0) in outputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def run_epoch(self):
        
        self.set_train()
       
        for batch_id,inputs in enumerate(self.train_loader):
            
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            if self.opt.clip_depth_grad > 0.0:

                if self.opt.depth_model == "resnet":
                    torch.nn.utils.clip_grad_norm_(self.models["depth_encoder"].parameters(),
                                                   self.opt.clip_depth_grad)

                torch.nn.utils.clip_grad_norm_(self.models["depth"].parameters(),
                                                   self.opt.clip_depth_grad)
                    
            
            self.model_optimizer.step()
            # self.model_lr_scheduler.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_id % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if batch_id % self.opt.chkpt_frequency == 0:
                self.save_model(is_chkpt=True)
                
            if early_phase or late_phase :
                self.log_time(batch_id, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs and torch.sum(inputs["depth_gt"]) > 0. and ("sc_depth", 0) in outputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1
    
    def process_batch(self,inputs):

        # transfer all data to GPU device
        for key, ipt in inputs.items():
            if key != 'frame_info':
                inputs[key] = ipt.to(self.device)
        
        # Get outputs for inverse depth, albedo and ambient by passing gated image
        if self.opt.model_type == "multioutput":
            outputs = self.models["gated2gated"](inputs["gated_aug", 0, 0])
        elif self.opt.model_type == "multinetwork":
            if self.opt.depth_model == "packnet" or self.opt.depth_model == "packnet_full":
                # Get inverse depth output
                outputs = self.models["depth"](inputs["gated_aug", 0, 0])
            elif self.opt.depth_model == "resnet":
                depth_feats = self.models["depth_encoder"](inputs["gated_aug", 0, 0])
                outputs = self.models["depth"](depth_feats)

            # Get albedo and ambient  
            features = self.models["encoder"](inputs["gated_aug", 0, 0])
            outputs.update(self.models["albedo"](features))
            outputs.update(self.models["ambient"](features))

        # Calculate pose transformation from central frame to neighbouring frames i.e. -1 and 1, using pose network.
        if self.use_pose_net and self.opt.temporal_loss:
            outputs.update(self.predict_poses(inputs))

        # Generate SNR masks if cycle loss is used
        if self.opt.cycle_loss and self.opt.snr_mask:
            for scale in self.opt.scales:
                if self.opt.v1_multiscale:
                    source_scale = scale
                else:
                    source_scale = 0
                mask = snr_binary_mask(inputs["gated", 0, source_scale],min_intns=self.opt.min_snr_val)
                outputs[("snr_mask", scale)] = mask

                if self.opt.intensity_mask:
                    mask = intensity_mask(inputs["gated", 0, source_scale], self.depth_flat_world)
                    outputs[("intensity_mask", scale)] = mask

        self.generate_images_pred(inputs,outputs)
        losses = self.compute_losses(inputs, outputs)
        
        return outputs, losses

    def generate_images_pred(self, inputs, outputs):
        """
            Generate the warped (reprojected) gated images for a minibatch.
            Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                # if not multiple scales, resize disparity to original scale from all outputs 
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            # obtain scaled depth from disparity
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            

            
            outputs[("depth", 0, scale)] = depth

            ################################################################################################################################
            '''
                Calculate temporal reprojections from adjacent frames to regenerate central frame i.e. frame_0
            '''
            if self.opt.temporal_loss:
                for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                    T = outputs[("cam_T_cam", 0, frame_id)]

                    # from the authors of https://arxiv.org/abs/1712.00175
                    if self.opt.pose_model_type == "posecnn":

                        axisangle = outputs[("axisangle", 0, frame_id)]
                        translation = outputs[("translation", 0, frame_id)]

                        inv_depth = 1 / depth
                        mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                        
                        '''
                            Get transformation from central frame, i.e. frame_0 to frame_-1 and frame_+1
                        '''
                        T = transformation_from_parameters(axisangle[:, 0],
                                                        translation[:, 0] * mean_inv_depth[:, 0],
                                                        frame_id < 0)
                    
                    # Convert depthmap to pointcloud in central frame coordinate system
                    cam_points = self.backproject_depth[source_scale](depth, inputs[("inv_K", source_scale)])

                    # generate pixel coordinates of the warped image                  
                    pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)

                    outputs[("sample", frame_id, scale)] = pix_coords
                    
                    # Do inverse warping to warp the neighbouring frames to central frame for calculating reprojection loss
                    outputs[("gated", frame_id, scale)] = F.grid_sample(inputs[("gated", frame_id, source_scale)],
                                                                        outputs[("sample", frame_id, scale)],
                                                                        padding_mode="border",align_corners=True)

                    if not self.opt.disable_automasking:
                        outputs[("gated_identity", frame_id, scale)] = inputs[("gated", frame_id, source_scale)]
            ################################################################################################################################

            ################################################################################################################################
            '''
                Calculate reconstruction from intensity profiles based on depth to regenerate central frame
            '''
            if self.opt.cycle_loss and scale == 0:   # Only do gated reconstruction for original resolution i.e. scale = 0
                
                albedo = outputs[("albedo", 0)]
                ambient = outputs[("ambient", 0)]
                
                sc_depth = torch.clamp(depth * self.depth_normalizer, self.opt.min_depth, self.opt.max_depth)
                
                sim_gated, _, sc_albedo = self.sim_gated(sc_depth, albedo, ambient) 
                outputs[("sim_gated", 0)] = sim_gated
                outputs[("albedo", 0)]    = sc_albedo
                outputs[("sc_depth", 0)]  = sc_depth
    
    def compute_reprojection_loss(self, pred, target):
        """
            Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """
            Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            depth = outputs[("depth",0, scale)].detach() # using detach to avoid passing gradient
            gated = inputs[("gated", 0, scale)]
            target = inputs[("gated", 0, source_scale)]

            if self.opt.temporal_loss:
                for frame_id in self.opt.frame_ids[1:]:
                    pred = outputs[("gated", frame_id, scale)]
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))

                reprojection_losses = torch.cat(reprojection_losses, 1)

                if not self.opt.disable_automasking:
                    identity_reprojection_losses = []
                    for frame_id in self.opt.frame_ids[1:]:
                        pred = inputs[("gated", frame_id, source_scale)]
                        identity_reprojection_losses.append(
                            self.compute_reprojection_loss(pred, target))

                    identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                    if self.opt.avg_reprojection:
                        identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                    else:
                        # save both images, and do min all at once below
                        identity_reprojection_loss = identity_reprojection_losses

                
                if self.opt.infty_hole_mask and self.epoch >= self.opt.infty_epoch_start :       # mask to remove infinite depth holes
                    
                    # identifying pixels which are close to camera using gated intensities
                    close_pixel_mask = (target[:,0,:,:]>=self.opt.close_px_fact*target[:,2,:,:]) | (target[:,1,:,:]>=self.opt.close_px_fact*target[:,2,:,:]) 
                    
                    # Constructing mask for pixel which lies between 2*median and max depth      
                    depth_lower_thresh  = 2.0*torch.median(depth.reshape(self.opt.batch_size,-1),dim=1)[0]
                    depth_higher_thresh = torch.max(depth.reshape(self.opt.batch_size,-1),dim=1)[0]    

                    depth_lower_thresh = depth_lower_thresh.reshape(self.opt.batch_size,1,1,1) * torch.ones_like(depth)
                    depth_higher_thresh = depth_higher_thresh.reshape(self.opt.batch_size,1,1,1) * torch.ones_like(depth)      
                    far_depth_mask = (depth >= depth_lower_thresh) & (depth <= depth_higher_thresh)

                    # The idea here is that pixels which are identified to be closer from gated intensities should have low depth
                    # as well. Any pixels which are identified to be close from gated intensities but have depth greater than 2*median
                    # depth probably correspond to infinity holes and should be taken out for reprojection loss calculation.   
                    infty_holes_mask = far_depth_mask * close_pixel_mask.unsqueeze(1) 

                    # not of infinity mask because we want to preserve pixels which are not the
                    # infinity pixels (most likely infinite pixels)
                    reprojection_losses *= ~infty_holes_mask 
                    outputs["close_pix_mask/{}".format(scale)] = close_pixel_mask.cpu().data
                    outputs["inf_holes_mask/{}".format(scale)] = infty_holes_mask.cpu().data

                if self.opt.avg_reprojection:
                    reprojection_loss = reprojection_losses.mean(1, keepdim=True)
                else:
                    reprojection_loss = reprojection_losses

                if not self.opt.disable_automasking:
                    # add random numbers to break ties
                    identity_reprojection_loss += torch.randn(
                        identity_reprojection_loss.shape).cuda() * 0.00001

                    combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
                else:
                    combined = reprojection_loss

                if combined.shape[1] == 1:
                    to_optimise = combined
                else:
                    to_optimise, idxs = torch.min(combined, dim=1)

                if not self.opt.disable_automasking:
                    outputs["identity_selection/{}".format(scale)] = (idxs > identity_reprojection_loss.shape[1] - 1).float()

                temporal_loss = to_optimise.mean()
                losses["temporal_loss/{}".format(scale)] = temporal_loss
                loss += self.opt.temporal_weight * temporal_loss

            if self.opt.cycle_loss and scale == 0:
                
                target = inputs[("gated",0,scale)]
                pred = outputs[("sim_gated",scale)]
                
                reprojection_losses = self.compute_reprojection_loss(pred,target)

                if self.opt.snr_mask:
                    snr_mask = outputs[("snr_mask", scale)]
                    reprojection_losses *= snr_mask
                
                if self.opt.intensity_mask:
                    int_mask = outputs[("intensity_mask", scale)]
                    reprojection_losses *= int_mask

                if self.opt.snr_mask and self.opt.intensity_mask:
                    outputs[("cycle_mask", scale)] = snr_mask * int_mask

                cycle_loss = reprojection_losses.mean()
                losses["cycle_loss/{}".format(scale)] = cycle_loss
                loss += self.opt.cycle_weight * cycle_loss

            if self.opt.passive_supervision and scale == 0:
                passive = inputs[("passive")]
                ambient = outputs[("ambient", 0)]
                passive_loss = self.compute_reprojection_loss(passive, ambient).mean()
                losses["passive_loss/0"] = passive_loss
                loss += self.opt.passive_weight * passive_loss

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            # smooth_loss = get_smooth_loss(norm_disp, gated)
            smooth_loss = get_smooth_loss(disp, gated)
            losses["smoothness/{}".format(scale)] = smooth_loss

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses
    
    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            
            pose_feats = {f_i: inputs["gated_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("gated_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, i], translation[:, i])

        return outputs

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        # Add visualization for depth normalizer
        writer.add_scalar("depth_normalizer", self.depth_normalizer, self.step)

        
        # frame_id = 0
        
        if mode == "train":
            for model_name, model in self.models.items():
                gradflow = plot_grad_flow(model.named_parameters())
                writer.add_image("gradflow/{}".format(model_name), gradflow, self.step, dataformats='HWC')
        
        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                # Log temporal inputs and prediction
                if self.opt.temporal_loss:
                    for frame_id in self.opt.frame_ids:
                        writer.add_image(
                            "gated_{}_{}/{}".format(frame_id, s, j),
                            inputs[("gated", frame_id, s)][j].data, self.step)
                        if s == 0 and frame_id != 0:
                            writer.add_image(
                                "gated_temp_pred_{}_{}/{}".format(frame_id, s, j),
                                outputs[("gated", frame_id, s)][j].data, self.step)

                    if not self.opt.disable_automasking:
                        writer.add_image("automask_{}/{}".format(s, j), outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

                    if self.opt.infty_hole_mask and self.epoch >= self.opt.infty_epoch_start:
                        writer.add_image("close_pix_mask_{}/{}".format(s, j), outputs["close_pix_mask/{}".format(s)][j][None, ...], self.step)
                        writer.add_image("infty_hole_mask_{}/{}".format(s, j), outputs["inf_holes_mask/{}".format(s)][j], self.step)
                    
                    disp = disp_to_mpimg(normalize_image(outputs[("disp", s)][j])[0].detach().cpu())
                    writer.add_image("disp_{}/{}".format(s, j),disp, self.step, dataformats='HWC')
                else:
                    # if the temporal loss is not opted, then show the central frame i.e. Frame 0 
                    writer.add_image("gated_{}_{}/{}".format(0, s, j),inputs[("gated", 0, s)][j].data, self.step)
                
                # Log albedo and ambient
                if self.opt.cycle_loss and s == 0:
                    writer.add_image("gated_cycle_pred_{}_{}/{}".format(frame_id, s, j), 
                                    torch.clamp(outputs[("sim_gated", s)][j].data, 0.0, 1.0), self.step)
                    writer.add_image("albedo_pred_{}_{}/{}".format(frame_id, s, j), 
                                    torch.clamp(outputs[("albedo", s)][j].data, 0.0, 1.0), self.step)
                    writer.add_image("ambient_pred_{}_{}/{}".format(frame_id, s, j), 
                                    normalize_image(outputs[("ambient", s)][j].data), self.step)
                    writer.add_image("scaled_depth_pred_{}_{}/{}".format(frame_id, s, j), 
                                    depth_image(outputs[("sc_depth", s)][j].data[0].detach().cpu()), self.step, dataformats='HWC')

                    if self.opt.snr_mask:
                        writer.add_image("snr_mask_{}_{}/{}".format(frame_id, s, j), 
                                        normalize_image(outputs[("snr_mask", s)][j].data), self.step)
                    
                    if self.opt.intensity_mask:
                        writer.add_image("intensity_mask_0_{}/{}".format(s, j),
                                        normalize_image(outputs[("intensity_mask", s)][j].data), self.step)

                    if self.opt.snr_mask and self.opt.intensity_mask:
                        writer.add_image("cycle_mask_0_{}/{}".format(s, j),
                                        normalize_image(outputs[("cycle_mask", s)][j].data), self.step)
                
                depth = outputs[("depth", 0, s)][j].data
                depth_img = disp_to_mpimg(depth[0].detach().cpu())
                writer.add_image("depth_pred_{}_{}/{}".format(frame_id, s, j), depth_img[:, :, 0:3], self.step, dataformats='HWC')
            writer.add_text("file_metadata/{}".format(j), inputs['frame_info'][j], self.step)

            if "depth_gt" in inputs:
                depth_gt = inputs["depth_gt"][j,0,:,:].cpu().detach().numpy()
                zero_pos = (depth_gt == 0.0)
                depth_gt = self.m.to_rgba(depth_gt)[:,:, 0:3]
                depth_gt[zero_pos, :] = 0.0
                writer.add_image("depth_gt_0_0/{}".format(j), torch.tensor(depth_gt), self.step, dataformats='HWC')
            
            if "passive" in inputs:
                writer.add_image("passive_0_0/{}".format(j), torch.clamp(inputs[("passive")][j], 0., 1.), self.step)

    def compute_depth_losses(self, inputs, outputs, losses):
        depth_pred = outputs[("sc_depth", 0)]
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0


        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        if not self.opt.cycle_loss:
            depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_gt = torch.clamp(depth_gt, min=self.opt.min_depth, max=self.opt.max_depth)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def save_opts(self):
        """
            Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self,is_chkpt=False):
        """Save model weights to disk
        """
        if is_chkpt:
            save_folder = os.path.join(self.log_path, "models", "chkpt_latest")
        else:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'gated2gated':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
               
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

        param_dict = OrderedDict()
        save_path = os.path.join(save_folder, "{}.pth".format("depth_normalizer"))
        param_dict['depth_normalizer'] = self.depth_normalizer
        torch.save(param_dict, save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            
            if os.path.isfile(path):
                print("Loading {} weights...".format(n))
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)
            else:
                print("Cannot find weights for {}".format(n))
        

        # loading adam state
        if "adam" in self.opt.models_to_load:
            optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
            if os.path.isfile(optimizer_load_path):
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            else:
                print("Cannot find Adam weights so Adam is randomly initialized")

        if "depth_normalizer" in self.opt.models_to_load:
            param_dict = torch.load("depth_normalizer.pth")
            self.depth_normalizer = param_dict['depth_normalizer']