import torch
import numpy as np

from torch import nn

from maxent_irl_costmaps.networks.misc import ScaledSigmoid

class DownsampleBlock(nn.Module):
    """
    Generic CNN to downsample an image
    TODO: Figure out if VGG/Resnet blocks matter.
    """
    def __init__(self, in_channels, out_channels, pool=2, activation=nn.ReLU, device='cpu'):
        super(DownsampleBlock, self).__init__()
        #For now, let's leave out batchnorm.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = activation()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.device = device

    def forward(self, x):
        extra_dims = x.shape[:-3]
        _x = self.conv(x.flatten(end_dim=-4))
        _x = _x.view(extra_dims + _x.shape[-3:])
        _x = self.activation(_x)
        _x = self.pool(_x)
        return _x

    def to(self, device):
        self.device = device
        self.conv = self.conv.to(device)
        self.activation = self.activation.to(device)
        self.pool = self.pool.to(device)
        return self

class UpsampleBlock(nn.Module):
    """
    Simple generic upsampling block
    TODO: Same as DownsampleBlock.
    """
    def __init__(self, in_channels, out_channels, scale=2, out_shape=None, activation=nn.ReLU, device='cpu'):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = activation()
        self.scale = scale
        self.out_shape = out_shape
        self.device = device

    def forward(self, x):
        extra_dims = x.shape[:-3]
        _x = x.flatten(end_dim=-4)

        if self.out_shape is None:
            _x = nn.functional.interpolate(_x, scale_factor=self.scale)
        else:
            _x = nn.functional.interpolate(_x, size=self.out_shape)

        _x = self.conv(_x)
        _x = _x.view(extra_dims + _x.shape[-3:])
        _x = self.activation(_x)
        return _x

    def to(self, device):
        self.device = device
        self.conv = self.conv.to(device)
        self.activation = self.activation.to(device)
        return self

class UNet(nn.Module):
    """
    Implementation of U-Net. Generally, this is as follows:
        1. Apply k downsample blocks to the image
        2. Apply 1-d convolution to the most downsampled image
        3. Concatenate original image and features and upsample.
    """
    def __init__(self, insize, outsize, n_blocks=3, hidden_channels=[8, 16, 32], pool=2, activation_scale=1.0, device='cpu'):
        """
        Args:
            insize: The dimension of the image to process (not really needed for unet but keep for consistency with other architectures.
            outsize: The dimension of the output image. We actually need this bc we're combining with a mask
        """
        super(UNet, self).__init__()

        self.insize = insize
        self.outsize = outsize
        self.n_blocks = n_blocks
        self.hidden_channels = hidden_channels

        #Compute layer sizes for simplicity
        self.layer_sizes = [self.insize]
        self.downsample_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()

        for i in range(n_blocks):
            last_size = self.layer_sizes[-1]
            layer_size = torch.Size([self.hidden_channels[i], last_size[1]//pool, last_size[2]//pool])
            self.layer_sizes.append(layer_size)
            self.downsample_blocks.append(DownsampleBlock(
                in_channels=last_size[0],
                out_channels=layer_size[0],
            ))

            self.upsample_blocks.append(UpsampleBlock(
                in_channels=layer_size[0],
                out_channels=layer_size[0],
                out_shape=last_size[1:]
            ))

            self.conv_blocks.append(nn.Conv2d(
                in_channels = last_size[0] + layer_size[0],
                out_channels = last_size[0],
                kernel_size = 3,
                padding = 1
            ))

        self.channel_wise_conv = nn.Linear(np.prod(self.layer_sizes[-1][1:]), np.prod(self.layer_sizes[-1][1:]))
        self.conv1d = nn.Conv2d(self.hidden_channels[-1], self.hidden_channels[-1], kernel_size=1)
        self.last_conv = nn.Conv2d(self.insize[0], self.outsize[0], kernel_size=1)
        self.activation = ScaledSigmoid(scale=activation_scale)

        self.device = device

    def forward(self, x):
        """
        steps:
            1. run all downsample blocks and cache outputs
            2. perform the 1d conv
            3. Upsample, combine with input and re-conv
        """
        feature_cache = [x]
        _x = x

        #Downsample
        for i in range(self.n_blocks):
            _x = self.downsample_blocks[i].forward(_x)
            feature_cache.append(_x)
        
        #1d-conv
        xshape = _x.shape
        _x = _x.view(*_x.shape[:-2], -1)
        _x = self.channel_wise_conv(_x)
        _x = _x.view(xshape)
        _x = self.conv1d.forward(_x)

        #Upsample
        #Design decision: I'd rather have indices correspond to layer size, so iterate backwards for upsampling
        for i in range(self.n_blocks-1, -1, -1):
            _x = self.upsample_blocks[i].forward(_x)
            _x = torch.cat([_x, feature_cache[i]], dim=-3)
            _x = self.conv_blocks[i].forward(_x)
        
        _x = self.last_conv(_x)
        return _x
#        return self.activation.forward(_x) #squash to [0, 1]

    def to(self, device):   
        self.device = device
        self.downsample_blocks = self.downsample_blocks.to(device)
        self.upsample_blocks = self.upsample_blocks.to(device)
        self.conv_blocks = self.conv_blocks.to(device)
        self.channel_wise_conv = self.channel_wise_conv.to(device)
        self.conv1d = self.conv1d.to(device)
        self.last_conv = self.last_conv.to(device)
        return self
