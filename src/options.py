from __future__ import absolute_import, division, print_function

import os
import argparse

class GatedOptions:

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description="Depth From Gated Profile Options")

        # PATH options
        self.parser.add_argument("--data_dir",
                                    type=str,
                                    required=True,
                                    help="directory gated dataset")
        self.parser.add_argument("--log_dir",
                                    type=str,
                                    required=True,
                                    help="directory to store logs")
        self.parser.add_argument("--coeff_fpath",
                                    type=str,
                                    required=True,
                                    help="file with stored chebychev coefficients")
        self.parser.add_argument("--depth_flat_world_fpath",
                                    type=str,
                                    required=False,
                                    help="path to flat world npz file")
        
        # TRAINING options
        self.parser.add_argument("--model_name",
                                    type=str,
                                    help="the name of the folder to save the model in",
                                    default="gated2gated")
        self.parser.add_argument("--model_type",
                                    type=str,
                                    help="model structure to use",
                                    default="multinetwork",
                                    choices=["multinetwork","multioutput"])
        self.parser.add_argument("--depth_model",
                                    type=str,
                                    help="depth model to use",
                                    default="packnet",
                                    choices=["packnet","resnet","packnet_full"])
        self.parser.add_argument("--img_ext",
                                    type=str,
                                    help="image extension to use",
                                    default="png",
                                    choices=["png","tiff"])
        self.parser.add_argument("--exp_num",
                                    type=int,
                                    help="experiment number",
                                    default=-1)
        self.parser.add_argument("--exp_name",
                                    type=str,
                                    help="the name of the folder to save the model in",
                                    default="gated2gated")
        self.parser.add_argument("--exp_metainfo",
                                    type=str,
                                    default="Main Experiment",
                                    help="additional info regarding experiment")
        self.parser.add_argument("--height",
                                    type=int,
                                    default=512,
                                    help="crop height of the image")
        self.parser.add_argument("--width",
                                    type=int,
                                    default=1024,
                                    help="crop width of the image")
        self.parser.add_argument("--num_bits",
                                    type=int,
                                    help="number of bits for gated image intensity",
                                    default=10)
        self.parser.add_argument("--scales",
                                    nargs="+",
                                    type=int,
                                    help="scales used in the loss",
                                    default=[0,1,2,3])
        self.parser.add_argument("--frame_ids",
                                    nargs="+",
                                    type=int,
                                    help="frames to load",
                                    default=[0, -1, 1])
        self.parser.add_argument("--pose_model_type",
                                    type=str,
                                    help="normal or shared",
                                    default="separate_resnet",
                                    choices=["posecnn", "separate_resnet"])
        self.parser.add_argument("--num_layers",
                                    type=int,
                                    help="number of resnet layers",
                                    default=18,
                                    choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--weights_init",
                                    type=str,
                                    help="pretrained or scratch",
                                    default="pretrained",
                                    choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                    type=str,
                                    help="how many images the pose network gets",
                                    default="pairs",
                                    choices=["pairs", "all"])
        self.parser.add_argument("--depth_normalizer",
                                    type=float,
                                    help="constant to normalize depth",
                                    default=150.0)
        self.parser.add_argument("--train_depth_normalizer",
                                    action='store_true',
                                    help="train only a single scalar constant,\
                                          while  freezing depth, pose, ambient, and albedo head")
        self.parser.add_argument("--min_depth",
                                    type=float,
                                    help="minimum depth",
                                    default=0.1)
        self.parser.add_argument("--max_depth",
                                    type=float,
                                    help="maximum depth",
                                    default=100.0)
        self.parser.add_argument("--snr_mask",
                                    action='store_true',
                                    help="whether to use SNR based mask for reprojection loss")
        self.parser.add_argument("--intensity_mask",
                                    action='store_true',
                                    help="whether to use Intensity based mask for reprojection loss")
        self.parser.add_argument("--min_snr_val",
                                    type=float,
                                    default=0.04,
                                    help="Minimum SNR value for SNR mask")
        self.parser.add_argument("--dataset",
                                    type=str,
                                    help="dataset to train on",
                                    default="gated",
                                    choices=["gated"])
        self.parser.add_argument("--split",
                                    type=str,
                                    help="which training split to use",
                                    choices=["gated2gated"],
                                    default="gated2gated")
        self.parser.add_argument("--dropout",
                                    type=float,
                                    help="dropout rate for packnet",
                                    default=0.5)   
        self.parser.add_argument("--feat_stack",
                                    type=str,
                                    help="whether to use concatenation(A) or Addition (B)",
                                    default="A",
                                    choices=["A", "B"])
        self.parser.add_argument("--num_convs",
                                    type=int,
                                    help="number of up/down levels in UNet",
                                    default=4)

        # OPTIMIZATION OPTION
        self.parser.add_argument("--batch_size",
                                    type=int,
                                    help="batch size",
                                    default=1)
        self.parser.add_argument("--learning_rate",
                                    type=float,
                                    help="learning rate",
                                    default=1e-4)
        self.parser.add_argument("--start_epoch",
                                    type=int,
                                    help="start epoch to have non-zero starting option for continuing training",
                                    default=0)
        self.parser.add_argument("--num_epochs",
                                    type=int,
                                    help="number of epochs",
                                    default=20)
        self.parser.add_argument("--scheduler_step_size",
                                    type=int,
                                    help="step size of the scheduler",
                                    default=15)    

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                    type=str,
                                    help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                    nargs="+",
                                    type=str,
                                    help="models to load",
                                    default=["depth", "pose_encoder", "pose"])

        # ABLATION options
        self.parser.add_argument("--no_ssim",
                                    action="store_true",
                                    help="if not to use SSIM loss")
        self.parser.add_argument("--cycle_loss",
                                    help="if set, cycle loss is used",
                                    action="store_true")
        self.parser.add_argument("--cycle_weight",
                                    type=float,
                                    default=0.1,
                                    help="cycle loss weight")
        self.parser.add_argument("--temporal_loss",
                                    help="if set, temporal reprojection loss is used",
                                    action="store_true")
        self.parser.add_argument("--temporal_weight",
                                    type=float,
                                    default=1.0,
                                    help="temporal loss weight")
        self.parser.add_argument("--sim_gated",
                                    action="store_true",
                                    help="whether to generate gated simulation image")
        self.parser.add_argument("--disparity_smoothness",
                                    type=float,
                                    default=1e-3,
                                    help="disparity smoothnes weight")
        self.parser.add_argument("--v1_multiscale",
                                    help="if set, uses monodepth v1 multiscale",
                                    action="store_true")
        self.parser.add_argument("--disable_automasking",
                                    help="if set, doesn't do auto-masking",
                                    action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                    help="if set, uses average reprojection loss",
                                    action="store_true")
        self.parser.add_argument("--infty_hole_mask",
                                    help="if set, uses a masking scheme to filter out points with infinite depth close to camera",
                                    action="store_true")
        self.parser.add_argument("--infty_epoch_start",
                                    type=int,
                                    help="start epoch to use infinity masks",
                                    default=0)
        self.parser.add_argument("--close_px_fact",
                                    type=float,
                                    help="factor to select close pixels to the image",
                                    default=0.995)
        self.parser.add_argument("--infty_hole_thresh",
                                    type=float,
                                    help="threshold to consider infinity points",
                                    default=0.01)
        self.parser.add_argument("--use_batchnorm",
                                    action="store_true",
                                    help="whether to use batchnorm2D in packnet module or not")
        self.parser.add_argument("--albedo_offset",
                                    type=float,
                                    default=0.0,
                                    help="constant factor to add to albedo to avoid gradient cutoff")
        self.parser.add_argument("--freeze_pose_net",
                                    action="store_true",
                                    help="whether to freeze the training for pose network")
        self.parser.add_argument("--clip_depth_grad",
                                    type=float,
                                    default=-1.0,
                                    help="clip depth gradient to a certain value if value > 0")
        self.parser.add_argument("--passive_supervision",
                                    action="store_true",
                                    help="supervise learning of passive image with real one")
        self.parser.add_argument("--passive_weight",
                                    type=float,
                                    default=0.1,
                                    help="passive supervision loss weight")



        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                    type=int,
                                    help="number of batches between each tensorboard log",
                                    default=250)
        self.parser.add_argument("--chkpt_frequency",
                                    type=int,
                                    help="number of batches between each checkpoint",
                                    default=250)
        self.parser.add_argument("--save_frequency",
                                    type=int,
                                    help="number of epochs between each save",
                                    default=1)                

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                    action="store_true",
                                    help="whether to train on cpu")
        self.parser.add_argument("--num_workers",
                                    type=int,
                                    help="number of dataloader workers",
                                    default=12)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options