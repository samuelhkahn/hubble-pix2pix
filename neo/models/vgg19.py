"""VGG-19 feature extractor for perceptual loss computation.

Extracts multi-scale features from a pretrained VGG-19 network at five
intermediate layers (one per max-pooling block).  These features capture
increasingly abstract image representations -- from edges and textures
in early layers to object parts and semantic content in deeper layers.

Since astronomical images are single-channel (grayscale), the input is
duplicated to 3 channels and normalized to ImageNet statistics before
feature extraction, so that the pretrained VGG weights remain valid.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms


class Vgg19(nn.Module):
    """VGG-19 feature extractor returning activations at 5 intermediate layers.

    Features are extracted after the first convolution in each of the 5
    VGG blocks::

        - Block 1: relu1_1 (64 channels)   -- edges, simple textures
        - Block 2: relu2_1 (128 channels)  -- textures, patterns
        - Block 3: relu3_1 (256 channels)  -- complex textures
        - Block 4: relu4_1 (512 channels)  -- object parts
        - Block 5: relu5_1 (512 channels)  -- high-level semantics

    All VGG parameters are frozen by default (``requires_grad=False``)
    since this module is used only as a fixed feature extractor for
    computing perceptual loss.

    Args:
        requires_grad: If ``False`` (default), freeze all VGG parameters.
    """

    def __init__(self, requires_grad: bool = False) -> None:
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        # ImageNet normalization (applied after converting to 3-channel RGB).
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # Split VGG-19 features into 5 sequential slices, one per block.
        # The layer indices correspond to the VGG-19 architecture:
        #   Block 1: layers 0-1   (conv + relu, before first max pool)
        #   Block 2: layers 2-6   (max pool + conv + relu + conv + relu)
        #   Block 3: layers 7-11  (max pool + 4 conv/relu layers)
        #   Block 4: layers 12-20 (max pool + 8 conv/relu layers)
        #   Block 5: layers 21-29 (max pool + 8 conv/relu layers)
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
    def _tensor_zero_one_transform(tensor: torch.Tensor) -> torch.Tensor:
        """Rescale tensor from [-1, 1] (tanh output range) to [0, 1].

        The generator outputs values in [-1, 1] due to the Tanh activation.
        VGG expects inputs normalized to ImageNet statistics starting from
        [0, 1], so we apply this affine transform first.
        """
        return (tensor + 1) / 2

    @staticmethod
    def _duplicate_channels(tensor: torch.Tensor) -> torch.Tensor:
        """Duplicate single-channel grayscale to 3-channel pseudo-RGB.

        VGG-19 was trained on 3-channel RGB images.  For single-channel
        astronomical images, we replicate the grayscale channel three times
        so the pretrained convolutional filters can still extract meaningful
        features.
        """
        return torch.repeat_interleave(tensor, 3, dim=1)

    def forward(self, X: torch.Tensor) -> list:
        """Extract multi-scale VGG-19 features.

        Args:
            X: Input tensor of shape ``(B, 1, H, W)`` with values in
                ``[-1, 1]``.

        Returns:
            List of 5 feature tensors from successive VGG blocks, each with
            shape ``(B, C_i, H_i, W_i)`` where ``C_i`` and ``H_i`` decrease
            with depth.
        """
        # Prepare input: [-1,1] -> [0,1] -> 3-channel -> ImageNet normalized.
        X = self._tensor_zero_one_transform(X)
        X = self._duplicate_channels(X)
        X = self.normalize(X)

        # Extract features at each block boundary.
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
