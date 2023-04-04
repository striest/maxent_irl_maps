
import numpy as np
import torch
from torch import nn

from maxent_irl_costmaps.networks.mlp import MLP
from maxent_irl_costmaps.networks.fourier_layer import FourierLayer
from maxent_irl_costmaps.networks.resnet import ResnetCostmapBlock
from maxent_irl_costmaps.networks.misc import ScaledSigmoid, Exponential

"""
A collection of basic CNN blocks to try.
"""
class FourierResnetCostmapSpeedmapCNNEnsemble2(nn.Module):
    def __init__(self, in_channels, hidden_channels, nf=8, ensemble_dim=100, hidden_activation='tanh', dropout=0.0, activation_type='sigmoid', activation_scale=1.0, device='cpu'):
        """
        Args:
            in_channels: The number of channels in the input image
            out_channels: The number of channels in the output image
            hidden_channels: A list containing the intermediate channels

        Note that in contrast to regular resnet, there is no end MLP nor pooling

        Same as the first ensemble, but now make the first layer the ensemble
        """
        super(FourierResnetCostmapSpeedmapCNNEnsemble2, self).__init__()
        self.channel_sizes = [in_channels * nf * 2] + hidden_channels + [1]

        if hidden_activation == 'tanh':
            hidden_activation = nn.Tanh
        elif hidden_activation == 'relu':
            hidden_activation = nn.ReLU

        self.ensemble_dim = ensemble_dim

        self.fourier_layer = FourierLayer(C=in_channels, nf=nf, device=device)

        self.cnn = nn.ModuleList()
        for i in range(len(self.channel_sizes) - 2):
            if i == 0:
                self.cnn_base = nn.ModuleList([ResnetCostmapBlock(in_channels=self.channel_sizes[i], out_channels=self.channel_sizes[i+1], activation=hidden_activation, dropout=dropout) for _ in range(self.ensemble_dim)])
            else:
                self.cnn.append(ResnetCostmapBlock(in_channels=self.channel_sizes[i], out_channels=self.channel_sizes[i+1], activation=hidden_activation, dropout=dropout))

        #last conv to avoid activation (for cost head)
        self.cost_head = nn.Conv2d(in_channels=self.channel_sizes[-2], out_channels=1, kernel_size=1, bias=True)
        self.speed_head = nn.Conv2d(in_channels=self.channel_sizes[-2], out_channels=2, kernel_size=1, bias=True)

        self.cnn = torch.nn.Sequential(*self.cnn)

        if activation_type == 'sigmoid':
            self.activation = ScaledSigmoid(scale=activation_scale)
        elif activation_type == 'exponential':
            self.activation = Exponential(scale=activation_scale)
        elif activation_type == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation_type == 'none':
            self.activation = nn.Identity()

    def forward(self, x, return_features=True):
        idx = torch.randint(self.ensemble_dim, size=(1, ))
        base_layer = self.cnn_base[idx]

        ffeats = self.fourier_layer.forward(x)
        features = self.cnn.forward(base_layer.forward(ffeats))
        costmap = self.activation(self.cost_head(features))
        speed_logits = self.speed_head(features)

        #exponentiate the mean value too, as speeds are always positive
        speed_dist = torch.distributions.Normal(loc=speed_logits[...,0, :, :].exp(), scale=(speed_logits[..., 1, :, :].exp() + 1e-6))

        return {
                    'costmap': costmap,
                    'speedmap': speed_dist,
                    'features': features
                }

    def ensemble_forward(self, x, return_features=True):
        ffeats = self.fourier_layer(x)
        features_batch = torch.stack([layer.forward(ffeats) for layer in self.cnn_base], dim=-4)

        #have to reshape for cnn
        data_dims = features_batch.shape[-3:]
        batch_dims = features_batch.shape[:-3]
        features_batch_flat = features_batch.view(-1, *data_dims)

        features = self.cnn.forward(features_batch_flat)

        #unsqueeze to make [B x E x C x H x W]
        costmap = self.activation(self.cost_head(features)).view(*batch_dims, 1, *data_dims[1:])
        speed_logits = self.speed_head(features).view(*batch_dims, 2, *data_dims[1:])

        #exponentiate the mean value too, as speeds are always positive
        speed_dist = torch.distributions.Normal(loc=speed_logits[..., 0, :, :].exp(), scale=(speed_logits[..., 1, :, :].exp() + 1e-6))

        return {
                    'costmap': costmap,
                    'speedmap': speed_dist,
                    'features': features
                }

    def to(self, device):
        super().to(device)
        self.fourier_layer = self.fourier_layer.to(device)
        return self
