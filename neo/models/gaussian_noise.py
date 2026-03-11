"""Gaussian noise layer for realistic noise injection during training."""

import numpy as np
import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    """Adds instance-level Gaussian noise scaled by pixel intensity standard deviation.

    Used to inject realistic photon noise into generated images during training.
    Can optionally undo DS9 log scaling before computing noise statistics.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('noise', torch.tensor(0.0))

    @staticmethod
    def _ds9_unscaling(x, a=1000, offset=0):
        """Invert DS9 logarithmic scaling."""
        return (((a + 1) ** x - 1) / a) + offset

    def forward(self, x, identity_map, ds9=True):
        """Apply Gaussian noise to the input tensor.

        Args:
            x: Input tensor.
            identity_map: If True, return input unchanged (used at inference).
            ds9: If True, undo DS9 scaling before computing noise statistics.

        Returns:
            Noise-augmented tensor, or unchanged input if identity_map is True.
        """
        if identity_map:
            return x
        if ds9:
            x = self._ds9_unscaling(x)
        std = x.std(axis=(1, 2))
        noise = torch.randn_like(x)
        stds = std[:, np.newaxis, np.newaxis] * noise
        x = x + stds
        return x
