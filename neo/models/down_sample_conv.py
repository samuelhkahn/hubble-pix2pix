"""Downsampling convolution block used in the encoder and discriminator.

Each block applies a stride-2 convolution that halves the spatial dimensions,
followed by optional batch normalization and LeakyReLU activation.  This is
the standard building block for the contracting path of the U-Net generator
and for the PatchGAN discriminator.

The sequence is: ``Conv2d(stride=2) -> [BatchNorm] -> [LeakyReLU(0.2)]``
"""

import torch
import torch.nn as nn


class DownSampleConv(nn.Module):
    """Strided convolution block for spatial downsampling.

    Applies ``Conv2d -> [BatchNorm2d] -> [LeakyReLU]`` where each optional
    component can be toggled independently.

    The first encoder block typically disables batch normalization
    (following the original pix2pix convention), and the bottleneck block
    may disable activation.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        kernel: Convolution kernel size (default: 4, standard for pix2pix).
        strides: Convolution stride (default: 2, halves spatial dimensions).
        padding: Zero-padding size (default: 1).
        activation: Whether to apply LeakyReLU with slope 0.2.
        batchnorm: Whether to apply batch normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 4,
        strides: int = 2,
        padding: int = 1,
        activation: bool = True,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            # LeakyReLU with slope 0.2 (standard for GAN discriminators and
            # pix2pix encoders -- allows small gradients for negative inputs).
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x
