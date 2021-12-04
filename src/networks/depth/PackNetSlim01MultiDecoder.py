import torch
import torch.nn as nn
from networks.layers.packnet.layers01 import PackLayerConv3d, UnpackLayerConv3d, Conv2D, ResidualBlock, InvDepth, activations

class Decoder(nn.Module):
    def __init__(self, name, version, ni, no, n1, n2, n3, n4, n5, unpack_kernel, iconv_kernel, num_3d_feat, out_channels, activation = 'sigmoid', use_batchnorm=False):
        super().__init__()
        self.name = name
        self.version = version

        # Support for different versions
        if self.version == 'A':  # Channel concatenation
            n1o, n1i = n1, n1 + ni + no
            n2o, n2i = n2, n2 + n1 + no
            n3o, n3i = n3, n3 + n2 + no
            n4o, n4i = n4, n4 + n3
            n5o, n5i = n5, n5 + n4
        elif self.version == 'B':  # Channel addition
            n1o, n1i = n1, n1 + no
            n2o, n2i = n2, n2 + no
            n3o, n3i = n3//2, n3//2 + no
            n4o, n4i = n4//2, n4//2
            n5o, n5i = n5//2, n5//2
        else:
            raise ValueError('Unknown PackNet version {}'.format(self.version))

        # Decoder
        self.unpack5 = UnpackLayerConv3d(n5, n5o, unpack_kernel[0], d=num_3d_feat)
        self.unpack4 = UnpackLayerConv3d(n5, n4o, unpack_kernel[1], d=num_3d_feat)
        self.unpack3 = UnpackLayerConv3d(n4, n3o, unpack_kernel[2], d=num_3d_feat)
        self.unpack2 = UnpackLayerConv3d(n3, n2o, unpack_kernel[3], d=num_3d_feat)
        self.unpack1 = UnpackLayerConv3d(n2, n1o, unpack_kernel[4], d=num_3d_feat)

        self.iconv5 = Conv2D(n5i, n5, iconv_kernel[0], 1, use_batchnorm=use_batchnorm)
        self.iconv4 = Conv2D(n4i, n4, iconv_kernel[1], 1, use_batchnorm=use_batchnorm)
        self.iconv3 = Conv2D(n3i, n3, iconv_kernel[2], 1, use_batchnorm=use_batchnorm)
        self.iconv2 = Conv2D(n2i, n2, iconv_kernel[3], 1, use_batchnorm=use_batchnorm)
        self.iconv1 = Conv2D(n1i, n1, iconv_kernel[4], 1, use_batchnorm=use_batchnorm)

        # Depth Layers

        self.unpack_out4 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_out3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_out2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

        self.out4_layer = InvDepth(n4, out_channels=out_channels, activation=activation)
        self.out3_layer = InvDepth(n3, out_channels=out_channels, activation=activation)
        self.out2_layer = InvDepth(n2, out_channels=out_channels, activation=activation)
        self.out1_layer = InvDepth(n1, out_channels=out_channels, activation=activation)


    def forward(self, x5p, skip1, skip2, skip3, skip4, skip5):
        # Decoder

        unpack5 = self.unpack5(x5p)
        if self.version == 'A':
            concat5 = torch.cat((unpack5, skip5), 1)
        else:
            concat5 = unpack5 + skip5
        iconv5 = self.iconv5(concat5)

        unpack4 = self.unpack4(iconv5)
        if self.version == 'A':
            concat4 = torch.cat((unpack4, skip4), 1)
        else:
            concat4 = unpack4 + skip4
        iconv4 = self.iconv4(concat4)
        out4 = self.out4_layer(iconv4)
        uout4 = self.unpack_out4(out4)

        unpack3 = self.unpack3(iconv4)
        if self.version == 'A':
            concat3 = torch.cat((unpack3, skip3, uout4), 1)
        else:
            concat3 = torch.cat((unpack3 + skip3, uout4), 1)
        iconv3 = self.iconv3(concat3)
        out3 = self.out3_layer(iconv3)
        uout3 = self.unpack_out3(out3)

        unpack2 = self.unpack2(iconv3)
        if self.version == 'A':
            concat2 = torch.cat((unpack2, skip2, uout3), 1)
        else:
            concat2 = torch.cat((unpack2 + skip2, uout3), 1)
        iconv2 = self.iconv2(concat2)
        out2 = self.out2_layer(iconv2)
        uout2 = self.unpack_out2(out2)

        unpack1 = self.unpack1(iconv2)
        if self.version == 'A':
            concat1 = torch.cat((unpack1, skip1, uout2), 1)
        else:
            concat1 = torch.cat((unpack1 +  skip1, uout2), 1)
        iconv1 = self.iconv1(concat1)
        out1 = self.out1_layer(iconv1)

        outs = [out1, out2, out3, out4]
        outputs = {}
        for i,out in enumerate(outs):
            outputs[(self.name,i)] = out
        return outputs

class PackNetSlim01MultiDecoder(nn.Module):
    """
    PackNet network with 3d convolutions (version 01, from the CVPR paper).
    Slimmer version, with fewer feature channels
    https://arxiv.org/abs/1905.02693
    Parameters
    ----------
    dropout : float
        Dropout value to use
    version : str
        Has a XY format, where:
        X controls upsampling variations (not used at the moment).
        Y controls feature stacking (A for concatenation and B for addition)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, dropout=None, version=None, cycle_loss=False, use_batchnorm = False,  **kwargs):
        super().__init__()
        self.version = version[1:]
        name = 'depth' if cycle_loss else 'disp'

        # Input/output channels
        in_channels = 3
        out_channels = 1
        # Hyper-parameters
        ni, no = 32, out_channels
        n1, n2, n3, n4, n5 = 32, 64, 128, 256, 512
        num_blocks = [2, 2, 3, 3]
        pack_kernel = [5, 3, 3, 3, 3]
        unpack_kernel = [3, 3, 3, 3, 3]
        iconv_kernel = [3, 3, 3, 3, 3]
        num_3d_feat = 4
        # Initial convolutional layer
        self.pre_calc = Conv2D(in_channels, ni, 5, 1,use_batchnorm=use_batchnorm)


        # Encoder

        self.pack1 = PackLayerConv3d(n1, pack_kernel[0], d=num_3d_feat)
        self.pack2 = PackLayerConv3d(n2, pack_kernel[1], d=num_3d_feat)
        self.pack3 = PackLayerConv3d(n3, pack_kernel[2], d=num_3d_feat)
        self.pack4 = PackLayerConv3d(n4, pack_kernel[3], d=num_3d_feat)
        self.pack5 = PackLayerConv3d(n5, pack_kernel[4], d=num_3d_feat)

        self.conv1 = Conv2D(ni, n1, 7, 1, use_batchnorm=use_batchnorm)
        self.conv2 = ResidualBlock(n1, n2, num_blocks[0], 1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv3 = ResidualBlock(n2, n3, num_blocks[1], 1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv4 = ResidualBlock(n3, n4, num_blocks[2], 1, dropout=dropout, use_batchnorm=use_batchnorm)
        self.conv5 = ResidualBlock(n4, n5, num_blocks[3], 1, dropout=dropout, use_batchnorm=use_batchnorm)

        # Decoder

        self.depth_decoder = Decoder(name, self.version, ni, no, n1, n2, n3, n4, n5, unpack_kernel, iconv_kernel, num_3d_feat, out_channels, activation='sigmoid', use_batchnorm=use_batchnorm)
        self.albedo_decoder = Decoder('albedo', self.version, ni, no, n1, n2, n3, n4, n5, unpack_kernel, iconv_kernel, num_3d_feat, out_channels, activation='sigmoid', use_batchnorm=use_batchnorm)
        self.ambient_decoder = Decoder('ambient', self.version, ni, no, n1, n2, n3, n4, n5, unpack_kernel, iconv_kernel, num_3d_feat, out_channels, activation='sigmoid', use_batchnorm=use_batchnorm)

        self.init_weights()

    def init_weights(self):
        """Initializes network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        x = self.pre_calc(x)

        # Encoder

        x1 = self.conv1(x)
        x1p = self.pack1(x1)
        x2 = self.conv2(x1p)
        x2p = self.pack2(x2)
        x3 = self.conv3(x2p)
        x3p = self.pack3(x3)
        x4 = self.conv4(x3p)
        x4p = self.pack4(x4)
        x5 = self.conv5(x4p)
        x5p = self.pack5(x5)

        # Skips

        skip1 = x
        skip2 = x1p
        skip3 = x2p
        skip4 = x3p
        skip5 = x4p

        # Decoder
        outputs = {}
        outputs.update(self.depth_decoder(x5p, skip1, skip2, skip3, skip4, skip5))
        outputs.update(self.albedo_decoder(x5p, skip1, skip2, skip3, skip4, skip5))
        outputs.update(self.ambient_decoder(x5p, skip1, skip2, skip3, skip4, skip5))

        return outputs



if __name__ == '__main__':
    import sys
    import os
    sys.path.append("..")
    from layers.packnet.layers01 import PackLayerConv3d, UnpackLayerConv3d, Conv2D, ResidualBlock, InvDepth
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    dropout = 0.5
    version = '1A'
    model = PackNetSlim01MultiDecoder(dropout,version)
    img = torch.rand(1,3, 512, 1024)
    outputs = model(img)
    print(outputs)