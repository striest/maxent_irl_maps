#miscellaneous network utils

import torch

from torch import nn

class ShiftedELU(nn.Module):
    """
    ELU, but have the min value be +1 for value iteration
    """
    def __init__(self, shift=2.0):
        super(ShiftedELU, self).__init__()
        self.shift = shift

    def forward(self, x):
        return torch.nn.functional.elu(x) + self.shift

class Exponential(nn.Module):
    """
    Try exponentiating
    """
    def __init__(self, scale=1.0):
        super(Exponential, self).__init__()
        self.scale = scale

    def forward(self, x):
        return (self.scale * x).exp()

class ScaledSigmoid(nn.Module):
    """
    Try adding a scaling factor to the sigmois activation to make it less sharp
    """
    def __init__(self, scale=1.0):
        super(ScaledSigmoid, self).__init__()
        self.scale = scale

    def forward(self, x):
        return (self.scale * x).sigmoid()
