"""PatchGAN discriminator for conditional image generation.

The PatchGAN classifies whether overlapping image *patches* are real or fake,
rather than producing a single scalar for the entire image.  This encourages
the generator to produce high-frequency detail across the full spatial extent
of the image, which is critical for preserving fine astronomical structures
like galaxy arms, tidal tails, and point sources.

The discriminator receives the concatenation of two images along the channel
dimension: ``concat(image, condition)`` where ``image`` is either a real HST
image or a generated super-resolved image, and ``condition`` is the upsampled
HSC input.
"""

import torch
import torch.nn as nn

from neo.models.down_sample_conv import DownSampleConv


class PatchGAN(nn.Module):
    """PatchGAN discriminator that operates on image patches.

    Architecture::

        Input: concat(image, condition) -> (B, 2, H, W)
          |-- DownSampleConv: 2 -> 64   (no batchnorm)
          |-- DownSampleConv: 64 -> 128
          |-- DownSampleConv: 128 -> 256
          |-- DownSampleConv: 256 -> 512
          |-- 1x1 Conv: 512 -> 1        (patch logit map)
        Output: (B, 1, H', W')  -- spatial map of real/fake predictions

    Each spatial location in the output corresponds to a receptive field
    (patch) in the input image.  The discriminator loss is computed over
    all patches, encouraging consistent realism across the full image.

    Args:
        input_channels: Number of input channels after concatenation.
            For conditional Pix2Pix this is 2 (1-channel image + 1-channel
            condition).
    """

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.d1 = DownSampleConv(input_channels, 64, batchnorm=False)
        self.d2 = DownSampleConv(64, 128)
        self.d3 = DownSampleConv(128, 256)
        self.d4 = DownSampleConv(256, 512)
        # 1x1 convolution produces a single logit per spatial patch.
        self.final = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass through the discriminator.

        Args:
            x: Generated or real image tensor ``(B, 1, H, W)``.
            y: Conditioning image tensor ``(B, 1, H, W)`` (upsampled HSC).

        Returns:
            Spatial map of logits ``(B, 1, H', W')`` where each value
            indicates the discriminator's confidence that the corresponding
            input patch is real (positive) or fake (negative).
        """
        # Concatenate image and condition along the channel dimension.
        x = torch.cat([x, y], axis=1)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return self.final(x)
