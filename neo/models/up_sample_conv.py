"""Upsampling convolution block used in the generator decoder.

Each block doubles the spatial dimensions using either a transposed
convolution or a resize convolution (nearest-neighbor upsample + conv).
The sequence is:

    ``ConvTranspose2d(stride=2) -> [BatchNorm] -> [Dropout(0.5)] -> [ReLU]``

The first three decoder blocks use 50% dropout for regularization,
following the original pix2pix design.
"""

import torch
import torch.nn as nn


class UpSampleConv(nn.Module):
    """Transposed convolution block for spatial upsampling.

    Applies ``ConvTranspose2d -> [BatchNorm2d] -> [Dropout(0.5)] -> [ReLU]``
    where each optional component can be toggled.

    Optionally supports resize convolution (nearest-neighbor interpolation
    followed by a regular convolution) as an alternative to transposed
    convolution, which can reduce checkerboard artifacts.

    Args:
        in_channels: Number of input feature channels.  Note that decoder
            blocks receiving skip connections have doubled input channels
            (e.g., 512 + 512 = 1024).
        out_channels: Number of output feature channels.
        kernel: Kernel size for transposed convolution (default: 4).
        strides: Stride for upsampling (default: 2, doubles spatial dims).
        padding: Padding size (default: 1).
        output_padding: Additional output padding for ConvTranspose2d.
        activation: Whether to apply ReLU activation.
        batchnorm: Whether to apply batch normalization.
        dropout: Whether to apply 50% dropout (used in the first 3 decoder
            blocks for regularization).
        resize_convolution: If ``True``, use nearest-neighbor upsample +
            3x3 conv instead of transposed convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 4,
        strides: int = 2,
        padding: int = 1,
        output_padding: int = 0,
        activation: bool = True,
        batchnorm: bool = True,
        dropout: bool = False,
        resize_convolution: bool = False,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout

        if resize_convolution:
            # Resize convolution: upsample by 2x with nearest-neighbor, then
            # apply a regular 3x3 convolution.  This avoids the uneven overlap
            # patterns that cause checkerboard artifacts in transposed convolutions.
            self.deconv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.ReflectionPad2d(1),
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=0
                ),
            )
        else:
            # Standard transposed convolution for learned upsampling.
            self.deconv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel, strides, padding, output_padding
            )

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            # ReLU with inplace=True to save memory.
            self.act = nn.ReLU(inplace=True)

        if dropout:
            # 50% spatial dropout for regularization (zeroes entire channels).
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.dropout:
            x = self.drop(x)
        return x
