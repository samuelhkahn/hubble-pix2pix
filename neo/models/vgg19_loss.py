"""VGG-19 perceptual loss for encouraging high-level feature similarity."""

import torch.nn as nn

from neo.models.vgg19 import Vgg19


class VGGLoss(nn.Module):
    """Weighted multi-scale perceptual loss using VGG-19 features.

    Computes MSE between VGG feature maps of generated and target images
    at 5 hierarchical levels, with configurable per-layer weights.

    Args:
        device: Torch device for the VGG model.
        weights: List of 5 floats weighting each VGG layer's contribution to the loss.
    """

    def __init__(self, device, weights):
        super().__init__()
        self.vgg = Vgg19().to(device)
        self.criterion = nn.MSELoss()
        self.weights = weights

    def forward(self, x, y):
        """Compute perceptual loss between generated image x and target y.

        Args:
            x: Generated image tensor.
            y: Target (ground truth) image tensor.

        Returns:
            Weighted sum of MSE losses across VGG feature layers.
        """
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
