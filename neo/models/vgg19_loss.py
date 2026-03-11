"""VGG-19 perceptual loss for encouraging high-level feature similarity.

Perceptual loss (also called content loss or feature loss) measures the
difference between generated and target images in the feature space of a
pretrained VGG-19 network, rather than in pixel space.  This encourages
the generator to match the high-level structure and textures of the
ground-truth image, producing perceptually sharper results than pixel-wise
losses alone.

The loss is computed as a weighted sum of MSE between feature maps at 5
hierarchical VGG layers, allowing control over which levels of abstraction
are prioritized.  Typically, lower layers (edges, textures) receive higher
weights for super-resolution tasks.
"""

import torch.nn as nn

from neo.models.vgg19 import Vgg19


class VGGLoss(nn.Module):
    """Weighted multi-scale perceptual loss using VGG-19 features.

    Computes MSE between VGG feature maps of generated and target images
    at 5 hierarchical levels, with configurable per-layer weights.

    Args:
        device: Torch device for the VGG model.
        weights: List of 5 floats weighting each VGG layer's contribution.
            For example, ``[1.0, 1.0, 0.0, 0.0, 0.0]`` uses only the first
            two (lowest-level) feature blocks.
    """

    def __init__(self, device: str, weights: list) -> None:
        super().__init__()
        self.vgg = Vgg19().to(device)
        self.criterion = nn.MSELoss()
        self.weights = weights

    def forward(self, x, y):
        """Compute perceptual loss between generated image ``x`` and target ``y``.

        Args:
            x: Generated image tensor ``(B, 1, H, W)``.
            y: Target (ground truth) image tensor ``(B, 1, H, W)``.

        Returns:
            Scalar weighted sum of MSE losses across VGG feature layers.
            Target features are ``.detach()``-ed so gradients only flow
            through the generated image path.
        """
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
