import torch
from torch import nn

import numpy as np

class FourierLayer(nn.Module):
    """
    Network layer that applies a Fourier transform (A {cos/sin}(Bx)) to an image/map
    """
    def __init__(self, C, nf=16, device='cpu'):
        """
        Args:
            C: Number of channels in the input image
            nf: number of fourier transformations to apply to each layer (output will have 2*nf*C channels)
        """
        super(FourierLayer, self).__init__()
        self.nf = nf
        self.C = C
        self.A = torch.randn(C, nf, device=device) * 2 * np.pi
        self.B = torch.randn(C, nf, device=device) * 2 * np.pi
        self.device = device

    def forward(self, x):
        """
        Args:
            x: [B x C x W x H] Tensor of features to transform

        Returns:
            y: [B x (C * 2 * nf) x W x H] Tensor of transformed features
        """
        _x = x.view(*x.shape[:2], 1, *x.shape[2:]) #[B x C x Fd x W x H]
        _A = self.A.view(1, self.C, self.nf, 1, 1)
        _B = self.B.view(1, self.C, self.nf, 1, 1)

        cos_embeddings = _A * (_B * _x).cos()
        sin_embeddings = _A * (_B * _x).sin()

        flat_cos_embeddings = cos_embeddings.view(x.shape[0], self.C*self.nf, *x.shape[2:])
        flat_sin_embeddings = sin_embeddings.view(x.shape[0], self.C*self.nf, *x.shape[2:])

        return torch.cat([flat_cos_embeddings, flat_sin_embeddings], dim=1)

    def to(self, device):
        self.A = self.A.to(device)
        self.B = self.B.to(device)
        self.device = device
        return self

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = torch.ones(4, 6, 100, 200)

    layer = FourierLayer(C = x.shape[1], nf=8)

    y = layer.forward(x)

    X = torch.stack(torch.meshgrid(
        torch.linspace(0., 1., 100),
        torch.linspace(0., 1., 100),
        indexing='ij'), dim=0).unsqueeze(0)

    layer = FourierLayer(C = X.shape[1], nf=8)

    Y = layer.forward(X)

    print(X.shape, Y.shape)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(X[0, 0])
    axs[1].imshow(X[0, 1])
    axs[2].imshow(Y[0].sum(dim=0))
    plt.show()

    for i in range(Y.shape[1]):
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(X[0, 0])
        axs[1].imshow(X[0, 1])
        axs[2].imshow(Y[0, i])
        plt.show()
