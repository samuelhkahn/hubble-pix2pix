"""PatchGAN discriminator for conditional image generation.

The PatchGAN classifies whether overlapping image patches are real or fake,
rather than classifying the entire image. This encourages high-frequency
detail preservation in the generated images.
"""

import torch
import torch.nn as nn

from neo.models.down_sample_conv import DownSampleConv


class PatchGAN(nn.Module):
    """PatchGAN discriminator that operates on image patches.

    Takes a pair of images (generated/real + condition) concatenated along
    the channel dimension and outputs a spatial map of real/fake predictions.

    Args:
        input_channels: Number of input channels (2 for image + condition pair).
    """

    def __init__(self, input_channels):
        super().__init__()
        self.d1 = DownSampleConv(input_channels, 64, batchnorm=False)
        self.d2 = DownSampleConv(64, 128)
        self.d3 = DownSampleConv(128, 256)
        self.d4 = DownSampleConv(256, 512)
        self.final = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x, y):
        """Forward pass through the discriminator.

        Args:
            x: Generated or real image tensor.
            y: Conditioning image tensor (upsampled HSC).

        Returns:
            Spatial map of logits indicating real/fake for each patch.
        """
        x = torch.cat([x, y], axis=1)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return self.final(x)
