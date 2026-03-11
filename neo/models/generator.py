"""U-Net generator with sub-pixel convolution upsampling.

Architecture overview::

    Input: (B, 1, 128, 128)  -- low-resolution HSC image
      |
      |-- Encoder: 7 DownSampleConv blocks (stride-2 convolutions)
      |   C64 -> C128 -> C256 -> C512 -> C512 -> C512 -> C512
      |   Each block halves spatial dimensions.
      |
      |-- Decoder: 7 UpSampleConv blocks (transpose convolutions) + skip connections
      |   CD512 -> CD512 -> CD512 -> C256 -> C128 -> C64 -> C32
      |   Each block doubles spatial dimensions.
      |   First 3 blocks use 50% dropout for regularization.
      |   Skip connections concatenate encoder features at matching resolution.
      |
      |-- PixelShuffle upsampler: 3x then 2x = 6x total (128 -> 384 -> 768)
      |
      |-- Final: 1x1 Conv -> Tanh (output in [-1, 1])
      |
    Output: (B, 1, 768, 768)  -- super-resolved HST-quality image

The 6x super-resolution factor matches the HST/HSC pixel scale ratio.
PixelShuffle (sub-pixel convolution) is used instead of interpolation for
the final upsampling to learn the upsampling kernels directly.
"""

import torch
import torch.nn as nn
from torchlayers.upsample import ConvPixelShuffle

from neo.models.down_sample_conv import DownSampleConv
from neo.models.gaussian_noise import GaussianNoise
from neo.models.up_sample_conv import UpSampleConv


class Pix2PixGenerator(nn.Module):
    """U-Net generator with sub-pixel convolution for astronomical super-resolution.

    Takes a low-resolution input (128x128) and generates a high-resolution
    output (768x768) using an encoder-decoder with skip connections and
    PixelShuffle upsampling.

    The skip connections allow the decoder to access fine-grained spatial
    information from the encoder, which is critical for preserving galaxy
    morphology and faint point sources during super-resolution.

    Args:
        in_channels: Number of input image channels (1 for grayscale FITS).
        out_channels: Number of output image channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Encoder (downsampling path)
        # Each DownSampleConv halves the spatial dimensions via stride-2 conv.
        # Spatial dimensions shown for a 128x128 input:
        # ------------------------------------------------------------------
        self.encoders = nn.ModuleList([
            DownSampleConv(in_channels, 64, batchnorm=False),   # -> 64 x 64 x 64
            DownSampleConv(64, 128),                             # -> 128 x 32 x 32
            DownSampleConv(128, 256),                            # -> 256 x 16 x 16
            DownSampleConv(256, 512),                            # -> 512 x 8 x 8
            DownSampleConv(512, 512),                            # -> 512 x 4 x 4
            DownSampleConv(512, 512),                            # -> 512 x 2 x 2
            DownSampleConv(512, 512, batchnorm=False),           # -> 512 x 1 x 1 (bottleneck)
        ])

        # ------------------------------------------------------------------
        # Decoder (upsampling path with skip connections)
        # Input channels are doubled (except first) because skip connections
        # concatenate encoder features along the channel dimension.
        # ------------------------------------------------------------------
        self.decoders = nn.ModuleList([
            UpSampleConv(512, 512, dropout=True),    # -> 512 x 2 x 2
            UpSampleConv(1024, 512, dropout=True),   # -> 512 x 4 x 4
            UpSampleConv(1024, 512, dropout=True),   # -> 512 x 8 x 8
            UpSampleConv(1024, 256),                 # -> 256 x 16 x 16
            UpSampleConv(512, 128),                  # -> 128 x 32 x 32
            UpSampleConv(256, 64),                   # -> 64 x 64 x 64
            UpSampleConv(128, 32),                   # -> 32 x 128 x 128
        ])

        # ------------------------------------------------------------------
        # Sub-pixel upsampling: two PixelShuffle stages
        # Stage 1: 3x upsampling (128 -> 384)
        # Stage 2: 2x upsampling (384 -> 768)
        # Total: 6x upsampling to match HST/HSC resolution ratio.
        # ------------------------------------------------------------------
        self.ps_blocks = nn.Sequential(
            ConvPixelShuffle(in_channels=32, out_channels=32, upscale_factor=3),
            nn.PReLU(),
            ConvPixelShuffle(in_channels=32, out_channels=32, upscale_factor=2),
            nn.PReLU(),
        )

        # 1x1 convolution to project from 32 feature channels to 1 output channel.
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

        # Gaussian noise layer for data augmentation during training.
        self.noise = GaussianNoise()

        # Tanh activation constrains output to [-1, 1], matching the DS9-scaled
        # input range.
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, identity_map: bool) -> torch.Tensor:
        """Forward pass through the generator.

        Args:
            x: Low-resolution input tensor of shape ``(B, 1, 128, 128)``.
            identity_map: If ``True``, skip noise injection (use at inference
                time to get deterministic outputs).

        Returns:
            Super-resolved output tensor of shape ``(B, 1, 768, 768)``.
        """
        # ------------------------------------------------------------------
        # Encoder: store intermediate features for skip connections.
        # ------------------------------------------------------------------
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        # Reverse the skip list (excluding the bottleneck itself) so that
        # each decoder block gets the matching-resolution encoder features.
        skips = list(reversed(skips[:-1]))

        # ------------------------------------------------------------------
        # Decoder: upsample and concatenate skip connections.
        # All decoders except the last one receive skip-connected features.
        # ------------------------------------------------------------------
        for decoder, skip in zip(self.decoders[:-1], skips):
            x = decoder(x)
            x = torch.cat((x, skip), axis=1)  # Channel-wise concatenation.

        # Final decoder block (no skip connection at the output resolution).
        x = self.decoders[-1](x)

        # ------------------------------------------------------------------
        # Sub-pixel upsampling: 128x128 -> 768x768 (6x total).
        # ------------------------------------------------------------------
        x = self.ps_blocks(x)

        # ------------------------------------------------------------------
        # Output projection: 32 channels -> 1 channel, then Tanh.
        # ------------------------------------------------------------------
        x = self.final_conv(x)
        x = self.tanh(x)

        return x
