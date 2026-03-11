"""U-Net based generator with sub-pixel convolution upsampling.

The generator follows a U-Net encoder-decoder architecture with skip connections,
combined with PixelShuffle layers for 6x super-resolution (128x128 -> 768x768).

Architecture:
    Encoder: C64-C128-C256-C512-C512-C512-C512 (4x4 convs, stride 2)
    Decoder: CD512-CD512-CD512-C256-C128-C64-C32 (transpose convs, stride 2)
    Upsampler: PixelShuffle 3x then 2x (total 6x from 128 -> 768)
"""

import torch
import torch.nn as nn
from torchlayers.upsample import ConvPixelShuffle

from neo.models.down_sample_conv import DownSampleConv
from neo.models.up_sample_conv import UpSampleConv
from neo.models.gaussian_noise import GaussianNoise


class Pix2PixGenerator(nn.Module):
    """U-Net generator with sub-pixel convolution for astronomical super-resolution.

    Takes a low-resolution input (128x128) and generates a high-resolution
    output (768x768) using an encoder-decoder architecture with skip connections
    and PixelShuffle upsampling.

    Args:
        in_channels: Number of input image channels (1 for grayscale FITS).
        out_channels: Number of output image channels.
        n_ps_blocks: Number of PixelShuffle blocks (unused, kept for compatibility).
        resize_conv: Whether to use resize convolution (unused, kept for compatibility).
    """

    def __init__(self, in_channels, out_channels, n_ps_blocks=2, resize_conv=True):
        super().__init__()

        # Encoder (downsampling path)
        self.encoders = nn.ModuleList([
            DownSampleConv(in_channels, 64, batchnorm=False),   # -> 64 x 64 x 64
            DownSampleConv(64, 128),                             # -> 128 x 32 x 32
            DownSampleConv(128, 256),                            # -> 256 x 16 x 16
            DownSampleConv(256, 512),                            # -> 512 x 8 x 8
            DownSampleConv(512, 512),                            # -> 512 x 4 x 4
            DownSampleConv(512, 512),                            # -> 512 x 2 x 2
            DownSampleConv(512, 512, batchnorm=False),           # -> 512 x 1 x 1
        ])

        # Decoder (upsampling path with skip connections)
        self.decoders = nn.ModuleList([
            UpSampleConv(512, 512, dropout=True),    # -> 512 x 2 x 2
            UpSampleConv(1024, 512, dropout=True),   # -> 512 x 4 x 4
            UpSampleConv(1024, 512, dropout=True),   # -> 512 x 8 x 8
            UpSampleConv(1024, 256),                 # -> 256 x 16 x 16
            UpSampleConv(512, 128),                  # -> 128 x 32 x 32
            UpSampleConv(256, 64),                   # -> 64 x 64 x 64
            UpSampleConv(128, 32),                   # -> 32 x 128 x 128
        ])

        # Sub-pixel upsampling: 3x then 2x = 6x total (128 -> 768)
        self.ps_blocks = nn.Sequential(
            ConvPixelShuffle(in_channels=32, out_channels=32, upscale_factor=3),
            nn.PReLU(),
            ConvPixelShuffle(in_channels=32, out_channels=32, upscale_factor=2),
            nn.PReLU(),
        )

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.noise = GaussianNoise()
        self.tanh = nn.Tanh()

    def forward(self, x, identity_map):
        """Forward pass through the generator.

        Args:
            x: Low-resolution input tensor of shape (B, 1, 128, 128).
            identity_map: If True, skip noise injection (used during inference).

        Returns:
            Super-resolved output tensor of shape (B, 1, 768, 768).
        """
        # Encode with skip connections
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        # Reverse skips (exclude bottleneck) for decoder
        skips = list(reversed(skips[:-1]))

        # Decode with skip connections (all but last decoder)
        for decoder, skip in zip(self.decoders[:-1], skips):
            x = decoder(x)
            x = torch.cat((x, skip), axis=1)

        # Final decoder (no skip connection)
        x = self.decoders[-1](x)

        # Sub-pixel upsampling (6x)
        x = self.ps_blocks(x)

        # Output projection
        x = self.final_conv(x)
        x = self.tanh(x)

        return x
