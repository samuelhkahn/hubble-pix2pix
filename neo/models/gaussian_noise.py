"""Gaussian noise injection layer for realistic noise augmentation.

Adds instance-level Gaussian noise scaled by the per-image pixel intensity
standard deviation.  This simulates the photon (Poisson-like) noise present
in astronomical images, helping the generator learn to be robust to noise
variations between exposures.

During inference (``identity_map=True``), noise injection is skipped so
that the generator produces deterministic, clean outputs.
"""

import numpy as np
import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    """Adds instance-level Gaussian noise scaled by pixel intensity std.

    The noise magnitude is proportional to the standard deviation of each
    image in the batch, so brighter images (with higher std) receive more
    noise -- mimicking the relationship between signal level and photon
    noise in astronomical detectors.

    Note:
        This layer is defined but currently not called in the generator's
        forward pass.  It is retained for experimental use and can be
        inserted into the generator pipeline to test noise robustness.
    """

    def __init__(self) -> None:
        super().__init__()
        # Register a buffer so the noise tensor follows the model to the
        # correct device (CPU/GPU) automatically.
        self.register_buffer("noise", torch.tensor(0.0))

    @staticmethod
    def _ds9_unscaling(
        x: torch.Tensor, a: float = 1000, offset: float = 0
    ) -> torch.Tensor:
        """Invert DS9 logarithmic scaling to recover linear flux values.

        This reverses the ``ds9_scaling`` transform applied in the dataset,
        converting from log-scaled pixel values back to linear flux so that
        noise statistics can be computed on physically meaningful values.
        """
        return (((a + 1) ** x - 1) / a) + offset

    def forward(
        self, x: torch.Tensor, identity_map: bool, ds9: bool = True
    ) -> torch.Tensor:
        """Apply Gaussian noise to the input tensor.

        Args:
            x: Input tensor of shape ``(B, H, W)`` or ``(B, C, H, W)``.
            identity_map: If ``True``, return input unchanged (for inference).
            ds9: If ``True``, undo DS9 scaling before computing noise
                statistics, then apply noise in linear flux space.

        Returns:
            Noise-augmented tensor (same shape as input), or the unchanged
            input if ``identity_map`` is ``True``.
        """
        if identity_map:
            return x

        if ds9:
            x = self._ds9_unscaling(x)

        # Compute per-image standard deviation (across H, W dimensions).
        std = x.std(axis=(1, 2))

        # Generate random noise and scale by per-image std.
        noise = torch.randn_like(x)
        stds = std[:, np.newaxis, np.newaxis] * noise

        x = x + stds
        return x
