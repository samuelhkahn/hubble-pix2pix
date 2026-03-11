"""Upsampling convolution block used in the generator decoder."""

import torch.nn as nn


class UpSampleConv(nn.Module):
    """Transposed convolution block for spatial upsampling.

    Applies ConvTranspose2d -> BatchNorm (optional) -> Dropout (optional) -> ReLU (optional).
    Optionally uses resize convolution (nearest-neighbor upsample + conv) instead
    of transposed convolution to reduce checkerboard artifacts.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel: Kernel size (default: 4).
        strides: Stride for upsampling (default: 2, doubles spatial dims).
        padding: Padding size (default: 1).
        output_padding: Additional output padding for ConvTranspose2d.
        activation: Whether to apply ReLU activation.
        batchnorm: Whether to apply batch normalization.
        dropout: Whether to apply 50% dropout (used in first decoder layers).
        resize_convolution: If True, use resize convolution instead of transposed conv.
    """

    def __init__(self, in_channels, out_channels, kernel=4, strides=2,
                 padding=1, output_padding=0, activation=True, batchnorm=True,
                 dropout=False, resize_convolution=False):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout

        if resize_convolution:
            self.deconv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
            )
        else:
            self.deconv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel, strides, padding, output_padding
            )

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.deconv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.dropout:
            x = self.drop(x)
        return x
