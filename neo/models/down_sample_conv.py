"""Downsampling convolution block used in encoder and discriminator."""

import torch.nn as nn


class DownSampleConv(nn.Module):
    """Strided convolution block for spatial downsampling.

    Applies Conv2d -> BatchNorm (optional) -> LeakyReLU (optional).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel: Kernel size (default: 4).
        strides: Stride for downsampling (default: 2, halves spatial dims).
        padding: Padding size (default: 1).
        activation: Whether to apply LeakyReLU activation.
        batchnorm: Whether to apply batch normalization.
    """

    def __init__(self, in_channels, out_channels, kernel=4, strides=2,
                 padding=1, activation=True, batchnorm=True):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x
