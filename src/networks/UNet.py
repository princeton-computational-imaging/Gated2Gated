from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from layers import Conv3x3
import torch.nn.functional as F
import math


class Conv1x1(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 1)

    def forward(self, x):
        out = self.conv(x)
        return out

class ConvBlock(nn.Module):
  def __init__(self,in_channels, out_channels):
    super(ConvBlock,self).__init__()
    self.conv1 = Conv3x3(in_channels,out_channels)
    self.conv2 = Conv3x3(out_channels,out_channels)
    self.nonlin = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    self.batchnorm1 = nn.BatchNorm2d(out_channels)
    self.batchnorm2 = nn.BatchNorm2d(out_channels)
  
  def forward(self,x):
    out = self.conv1(x)
    out = self.nonlin(out)
    out = self.batchnorm1(out)
    
    out = self.conv2(out)
    out = self.nonlin(out)
    out = self.batchnorm2(out)
    
    return out

class MultiChannelG2D(nn.Module):
    def __init__(self, num_convs = 4, scales = range(1), out_channels = 1,
                 init_channels = 32, height = 512, width = 1024,
                 use_depth = True):
        super(MultiChannelG2D,self).__init__()

        self.scales = scales
        self.height = height
        self.width  = width

        self.encoder = Encoder(num_convs, init_channels)
        self.use_depth = use_depth # Whether to use UNet for depth output or not
        
        if self.use_depth:
            self.depth_decoder = Decoder(scales, num_convs, init_channels, out_channels, 'depth')
        
        self.albedo_decoder = Decoder(scales, num_convs, init_channels, out_channels, 'albedo')
        self.ambient_decoder = Decoder(scales, num_convs, init_channels, out_channels, 'ambient')

    
    def forward(self,x):
        output = {}
         
        enc_feats = self.encoder(x)
        if self.use_depth:
            output.update(self.depth_decoder(enc_feats))
        output.update(self.albedo_decoder(enc_feats))
        output.update(self.ambient_decoder(enc_feats))

        # for scale in self.scales:
        #     if self.use_depth:
        #         output[('depth',scale)] = F.interpolate(output[('depth',scale)], [self.height, self.width], mode="bilinear", align_corners=False)
            
        #     output[('albedo',scale)] = F.interpolate(output[('albedo',scale)], [self.height, self.width], mode="bilinear", align_corners=False)
        #     output[('ambient',scale)] = F.interpolate(output[('ambient',scale)], [self.height, self.width], mode="bilinear", align_corners=False)
        return output


class Encoder(nn.Module):

    def __init__(self, num_convs = 4, init_channels=32):
        """[UNet Encoder for gated images]

        Args:
            num_convs (int, optional): [number of up/down levels]. Defaults to 4.
            init_channels (int, optional): [initial number of encoding channels]. Defaults to 32.
        """
        super(Encoder, self).__init__()
        self.channels = [init_channels*2**(i) for i in range(0,num_convs+1)]
        self.channels = [3] + self.channels # number of channels in gated image appended in the beginning
        self.enc_blocks = nn.ModuleList([ConvBlock(self.channels[i], self.channels[i+1]) for i in range(len(self.channels)-1)])
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        
        skips = []
        for i,enc_block in enumerate(self.enc_blocks):
            # print("input shape {} = {}".format(i,x.shape))
            x = enc_block(x)
            # print("conv block {} = {}".format(i,x.shape))
            skips.append(x)
            x = self.maxpool(x)
            # print("maxpool block {} = {}".format(i,x.shape))
            
        return skips

class Decoder(nn.Module):
    
    def __init__(self, name = "output", scales = range(1), num_convs = 4, init_channels=32, out_channels = 1):
        """[UNet Decoder for multi-headed output]

        Args:
            scales (list(int), optional): [scales to get output]. Defaults to [0].
            num_convs (int, optional): [number of up/down levels]. Defaults to 4.
            init_channels (int, optional): [initial number of encoding channels]. Defaults to 32.
            out_channels (int, optional): [number of channels in the output]. Defaults to 1.
            name (str, optional): [name of the output]. Defaults to "output".
        """
        super(Decoder,self).__init__()
        self.channels   = [init_channels*2**(i) for i in range(0,num_convs+1)] # [32,64,128,256,512]
        self.channels   = self.channels[::-1]  # Reverse the list to up sample in opposite way # [512,256,128,64,32]
        self.scales     = scales  
        self.num_convs  = num_convs    
        self.name = name
        self.up_convs = nn.ModuleList([nn.Sequential(nn.ConvTranspose2d(in_channels=self.channels[i], out_channels=self.channels[i+1], kernel_size=2,
                                      stride=2), nn.BatchNorm2d(self.channels[i+1])) for i in range(len(self.channels)-1)])  # [(512->256),(256->128),(128->64),(64->32)]

        # [(256+256 -> 256),(128+128 -> 128),(64+64 -> 64),(32+32 -> 32)] = [(512 -> 256),(256 -> 128),(128 -> 64),(64 -> 32)]
        self.conv_blocks = nn.ModuleList([ConvBlock(
            self.channels[i], self.channels[i+1]) for i in range(len(self.channels)-1)])
        self.out_convs = nn.ModuleList([Conv1x1(in_channels=self.channels[-(
            s+1)], out_channels=out_channels) for s in self.scales])  # in_channels = [32,64,128,256]

    def forward(self,encoder_feats):
        output = {}
        x = encoder_feats[-1]
        
        for i in range(len(self.channels)-1):
            # print("input shape = {}".format(x.shape))    
            x = self.up_convs[i](x)
            # print("upsample shape = {}".format(x.shape))
            enc_ftrs = encoder_feats[-(i+2)]
            x = torch.cat([x,enc_ftrs],dim=1)
            # print("concat shape = {}".format(x.shape))
            x = self.conv_blocks[i](x)
            # print("up conv shape = {}".format(x.shape))
            curr_scale = self.num_convs-i-1
            if  curr_scale in self.scales:
                output[(self.name,curr_scale)] = self.out_convs[curr_scale](x)
                # print("output shape = {}".format(output[(self.name,curr_scale)].shape)) 
        return output


