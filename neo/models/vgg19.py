"""VGG-19 feature extractor for perceptual loss computation.

Extracts multi-scale features from a pretrained VGG-19 network. Single-channel
astronomical images are duplicated to 3 channels and normalized to match
ImageNet statistics before feature extraction.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms


class Vgg19(nn.Module):
    """VGG-19 feature extractor that returns activations at 5 intermediate layers.

    Features are extracted after each of the 5 max-pooling blocks:
        - relu1_1 (64 channels)
        - relu2_1 (128 channels)
        - relu3_1 (256 channels)
        - relu4_1 (512 channels)
        - relu5_1 (512 channels)

    Args:
        requires_grad: If False, freeze all parameters (default for loss computation).
    """

    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    @staticmethod
    def _tensor_zero_one_transform(tensor):
        """Rescale tensor from [-1, 1] (tanh output) to [0, 1]."""
        return (tensor + 1) / 2

    @staticmethod
    def _duplicate_channels(tensor):
        """Duplicate single-channel grayscale to 3-channel RGB for VGG input."""
        return torch.repeat_interleave(tensor, 3, dim=1)

    def forward(self, X):
        """Extract multi-scale VGG features.

        Args:
            X: Input tensor of shape (B, 1, H, W) with values in [-1, 1].

        Returns:
            List of 5 feature tensors from successive VGG blocks.
        """
        X = self._tensor_zero_one_transform(X)
        X = self._duplicate_channels(X)
        X = self.normalize(X)

        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
