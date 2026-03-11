"""Pix2Pix conditional GAN for astronomical image super-resolution.

Implements a Pix2Pix-style model with a multi-component loss function
specifically designed for astronomical imagery.  The composite loss combines
five complementary objectives:

    1. **BCE adversarial loss** -- a PatchGAN discriminator classifies image
       patches as real or fake, encouraging the generator to produce realistic
       high-frequency detail.
    2. **L1 reconstruction loss** -- pixel-wise absolute error between the
       generated and ground-truth images enforces overall brightness fidelity.
    3. **VGG-19 perceptual loss** -- MSE between multi-scale VGG feature maps
       encourages similarity in high-level image features.
    4. **Wavelet scattering transform loss** -- L1 error between scattering
       coefficients (via Kymatio) preserves multi-scale morphological structure
       that is critical for galaxy shape measurements.
    5. **Segmentation-masked L1 loss** -- an L1 loss weighted by a binary
       source segmentation map, so that detected astronomical sources receive
       higher reconstruction priority than empty background.

The total generator loss is a weighted sum of these five terms, with each
weight controllable via the ``lambda_*`` hyperparameters.
"""

import os
from typing import Tuple, Union

import torch
import torch.nn as nn
import torchlayers as tl
from kymatio.torch import Scattering2D
from torchvision.transforms import CenterCrop

from neo.models.generator import Pix2PixGenerator
from neo.models.patchgan import PatchGAN
from neo.models.vgg19_loss import VGGLoss


class Pix2Pix:
    """Pix2Pix conditional GAN for HSC -> HST super-resolution.

    Trains a U-Net generator to produce HST-quality images from HSC inputs,
    using a PatchGAN discriminator and a composite loss function that balances
    adversarial, reconstruction, perceptual, scattering, and segmentation-
    weighted objectives.

    Args:
        in_channels: Number of input image channels (1 for grayscale FITS).
        out_channels: Number of output image channels.
        input_size: Spatial size of the center-cropped training images.
            Used to configure the scattering transform (must match the crop
            applied in ``_gen_step`` and ``_disc_step``).
        device: Torch device (``"cuda"`` or ``"cpu"``).
        vgg_loss_weights: Per-layer weights for VGG perceptual loss.
            A list of 5 floats corresponding to VGG blocks 1--5.
        learning_rate: Adam learning rate for the generator.
        disc_learning_rate: Adam learning rate for the discriminator.
        lambda_recon: Weight for L1 reconstruction loss.
        lambda_segmap: Weight for segmentation-masked L1 loss.
        lambda_vgg: Weight for VGG perceptual loss.
        lambda_scattering: Weight for wavelet scattering loss.
        lambda_adv: Weight for adversarial loss.
        display_step: Logging frequency (in training steps).
        pretrained_generator: Filename of a pretrained generator checkpoint
            in the ``models/`` directory, or ``""`` to train from scratch.
        pretrained_discriminator: Filename of a pretrained discriminator
            checkpoint in the ``models/`` directory, or ``""`` to train
            from scratch.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_size: int,
        device: str,
        vgg_loss_weights: Tuple[float, ...] = (1.0, 1.0, 0.0, 0.0, 0.0),
        learning_rate: float = 0.0002,
        disc_learning_rate: float = 0.0002,
        lambda_recon: float = 200,
        lambda_segmap: float = 200,
        lambda_vgg: float = 200,
        lambda_scattering: float = 1,
        lambda_adv: float = 5,
        display_step: int = 25,
        pretrained_generator: str = "",
        pretrained_discriminator: str = "",
    ) -> None:
        super().__init__()

        self.device = device
        self.display_step = display_step

        # ------------------------------------------------------------------
        # Generator initialization
        # ------------------------------------------------------------------
        if pretrained_generator:
            print(f"Loading Pretrained Generator: {pretrained_generator}")
            path = os.path.join(os.getcwd(), "models", pretrained_generator)
            self.gen = torch.load(path)
        else:
            self.gen = Pix2PixGenerator(in_channels, out_channels)
            # ``torchlayers.build`` performs a dummy forward pass to infer
            # shapes for any lazy (shape-deferred) layers.
            tl.build(self.gen, torch.randn(1, 1, 128, 128), True)

        # ------------------------------------------------------------------
        # Discriminator initialization
        # ------------------------------------------------------------------
        if pretrained_discriminator:
            print(f"Loading Pretrained Discriminator: {pretrained_discriminator}")
            path = os.path.join(os.getcwd(), "models", pretrained_discriminator)
            self.patch_gan = torch.load(path)
        else:
            # Input channels = 2: generated/real image + HSC conditioning image
            # concatenated along the channel dimension.
            self.patch_gan = PatchGAN(2)

        # ------------------------------------------------------------------
        # Loss weights
        # ------------------------------------------------------------------
        self.lr = learning_rate
        self.disc_lr = disc_learning_rate
        self.lambda_recon = lambda_recon
        self.lambda_vgg = lambda_vgg
        self.lambda_scattering = lambda_scattering
        self.lambda_adv = lambda_adv
        self.lambda_segmap = lambda_segmap

        # ------------------------------------------------------------------
        # Loss functions
        # ------------------------------------------------------------------
        # BCE with logits for the adversarial objective (PatchGAN outputs
        # raw logits, not probabilities).
        self.adversarial_criterion = nn.BCEWithLogitsLoss()

        # Pixel-wise reconstruction losses.
        self.recon_criterion_l1 = nn.L1Loss()
        self.recon_criterion_l2 = nn.MSELoss()

        # VGG-19 perceptual loss (multi-scale feature matching).
        self.vgg_criterion = VGGLoss(self.device, weights=vgg_loss_weights)

        # Wavelet scattering transform for multi-scale structural loss.
        # J=3 gives 3 octaves of wavelet scales, L=8 gives 8 orientations,
        # max_order=2 computes up to second-order scattering coefficients.
        self.scattering_f = Scattering2D(
            J=3,
            L=8,
            shape=(input_size, input_size),
            out_type="array",
            max_order=2,
        ).to(device)

        # ------------------------------------------------------------------
        # Optimizers
        # ------------------------------------------------------------------
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr)
        self.disc_opt = torch.optim.Adam(
            self.patch_gan.parameters(), lr=self.disc_lr
        )

        # Move models to the target device.
        self.gen.to(self.device)
        self.patch_gan.to(self.device)

    # ------------------------------------------------------------------
    # Static loss helpers
    # ------------------------------------------------------------------

    @staticmethod
    def l1_loss_with_mask(
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        seg_map_real: torch.Tensor,
    ) -> torch.Tensor:
        """L1 loss weighted by a binary segmentation mask.

        Only pixels where ``seg_map_real == 1`` (detected sources) contribute
        to the loss.  The result is normalized by the total mask area so that
        the loss magnitude is independent of the number of detected sources.

        Args:
            x_real: Ground-truth image tensor.
            x_fake: Generated image tensor.
            seg_map_real: Binary mask (1 = source, 0 = background).

        Returns:
            Scalar masked L1 loss.
        """
        return (
            torch.sum(torch.abs(x_real - x_fake) * seg_map_real)
            / torch.sum(seg_map_real)
        )

    @staticmethod
    def l2_loss_with_mask(
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        seg_map_real: torch.Tensor,
    ) -> torch.Tensor:
        """L2 loss weighted by a binary segmentation mask.

        Same masking logic as ``l1_loss_with_mask`` but with squared errors.
        """
        return (
            torch.sum(((x_real - x_fake) * seg_map_real) ** 2.0)
            / torch.sum(seg_map_real)
        )

    # ------------------------------------------------------------------
    # Generator forward / loss
    # ------------------------------------------------------------------

    def _gen_step(
        self,
        real_images: torch.Tensor,
        conditioned_images: torch.Tensor,
        hsc_hr: torch.Tensor,
        seg_map_real: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Compute the full generator loss (all five components).

        All images are center-cropped to ``input_size`` (600 px) before loss
        computation.  This removes the reflective-padding border which can
        cause edge artifacts in the generated output.

        Args:
            real_images: Ground-truth HST images ``(B, 1, 768, 768)``.
            conditioned_images: Low-resolution HSC input ``(B, 1, 128, 128)``.
            hsc_hr: Upsampled HSC for discriminator conditioning
                ``(B, 1, 768, 768)``.
            seg_map_real: Binary source masks ``(B, 1, 768, 768)``.

        Returns:
            Tuple of ``(total_loss, adv_loss, recon_loss, vgg_loss,
            scattering_loss, segmap_loss)``.
        """
        # Generate super-resolved image from the low-resolution input.
        fake_images = self.gen(conditioned_images, identity_map=True)

        # Center crop all tensors to 600x600 to remove border padding.
        fake_images = CenterCrop(600)(fake_images)
        real_images = CenterCrop(600)(real_images)
        seg_map_real = CenterCrop(600)(seg_map_real)
        hsc_hr = CenterCrop(600)(hsc_hr)

        # --- Adversarial loss ---
        # The generator wants the discriminator to classify its fakes as real
        # (target = 1).
        disc_logits = self.patch_gan(fake_images, hsc_hr)
        adversarial_loss = self.adversarial_criterion(
            disc_logits.flatten(), torch.ones_like(disc_logits).flatten()
        )

        # --- Pixel-wise reconstruction loss (L1) ---
        recon_loss = self.recon_criterion_l1(fake_images, real_images)

        # --- VGG perceptual loss ---
        vgg_loss = self.vgg_criterion(fake_images, real_images)

        # --- Segmentation-masked L1 loss ---
        segmap_loss = self.l1_loss_with_mask(
            fake_images, real_images, seg_map_real
        )

        # --- Wavelet scattering loss ---
        # Compute scattering coefficients for real and fake images.
        # Squeeze out the singleton scale dimension and skip the zeroth-order
        # coefficients (index 0) which just encode the mean intensity.
        scat_real = self.scattering_f(real_images.contiguous()).squeeze(1)[
            :, 1:, :, :
        ]
        scat_fake = self.scattering_f(fake_images.contiguous()).squeeze(1)[
            :, 1:, :, :
        ]
        scattering_loss = (
            (scat_real - scat_fake).abs().sum(axis=(1, 2, 3)).mean()
        )

        # --- Weighted total loss ---
        # Each component is scaled by its lambda weight. The ``self.lr``
        # factor is a legacy scaling that uniformly scales all loss terms.
        total_loss = self.lr * (
            self.lambda_adv * adversarial_loss
            + self.lambda_recon * recon_loss
            + self.lambda_vgg * vgg_loss
            + self.lambda_scattering * scattering_loss
            + self.lambda_segmap * segmap_loss
        )

        return (
            total_loss,
            adversarial_loss,
            recon_loss,
            vgg_loss,
            scattering_loss,
            segmap_loss,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate_fake_images(
        self,
        conditioned_images: torch.Tensor,
        identity_map: bool = False,
    ) -> torch.Tensor:
        """Generate super-resolved images from low-resolution inputs.

        Args:
            conditioned_images: Low-resolution HSC input ``(B, 1, 128, 128)``.
            identity_map: If ``True``, skip noise injection (use at inference).

        Returns:
            Super-resolved output ``(B, 1, 768, 768)``.
        """
        return self.gen(conditioned_images, identity_map=identity_map)

    # ------------------------------------------------------------------
    # Discriminator forward / loss
    # ------------------------------------------------------------------

    def _disc_step(
        self,
        real_images: torch.Tensor,
        conditioned_images: torch.Tensor,
        hsc_hr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute discriminator loss on real and fake images.

        The discriminator receives ``concat(image, hsc_hr)`` as input and must
        classify each patch as real (target = 1) or fake (target = 0).  The
        generator's output is ``.detach()``-ed so that gradients do not flow
        back through the generator during this step.

        Args:
            real_images: Ground-truth HST images ``(B, 1, 768, 768)``.
            conditioned_images: Low-resolution HSC input ``(B, 1, 128, 128)``.
            hsc_hr: Upsampled HSC conditioning ``(B, 1, 768, 768)``.

        Returns:
            Tuple of ``(disc_loss, fake_logits, real_logits)``.
        """
        # Generate fakes and detach to prevent generator gradient updates.
        fake_images = self.gen(conditioned_images, identity_map=True).detach()

        # Center crop to 600x600 (same crop used in generator loss).
        fake_images = CenterCrop(600)(fake_images)
        real_images = CenterCrop(600)(real_images)
        hsc_hr = CenterCrop(600)(hsc_hr)

        # Discriminator predictions on fake and real images.
        fake_logits = self.patch_gan(fake_images, hsc_hr)
        real_logits = self.patch_gan(real_images, hsc_hr)

        # BCE loss: fakes should be classified as 0, reals as 1.
        fake_loss = self.adversarial_criterion(
            fake_logits.flatten(), torch.zeros_like(fake_logits).flatten()
        )
        real_loss = self.adversarial_criterion(
            real_logits.flatten(), torch.ones_like(real_logits).flatten()
        )
        return real_loss + fake_loss, fake_logits, real_logits

    # ------------------------------------------------------------------
    # Training and validation interface
    # ------------------------------------------------------------------

    def training_step(
        self,
        real: torch.Tensor,
        condition: torch.Tensor,
        hsc_hr: torch.Tensor,
        seg_map_real: torch.Tensor,
        optimizer: str,
    ) -> Union[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Execute one training step for the specified optimizer.

        Computes the loss, performs backpropagation, and updates the weights
        of either the generator or the discriminator.

        Args:
            real: Ground-truth HR images (HST), ``(B, 1, 768, 768)``.
            condition: Low-resolution input images (HSC), ``(B, 1, 128, 128)``.
            hsc_hr: Upsampled HSC for discriminator conditioning,
                ``(B, 1, 768, 768)``.
            seg_map_real: Binary segmentation maps, ``(B, 1, 768, 768)``.
            optimizer: Which model to update -- ``"generator"`` or
                ``"discriminator"``.

        Returns:
            For ``"generator"``: ``(total_loss, adv_loss, recon_loss,
            vgg_loss, scat_loss, seg_loss)``.
            For ``"discriminator"``: ``(disc_loss, fake_logits, real_logits)``.
        """
        if optimizer == "discriminator":
            loss, fake_logits, real_logits = self._disc_step(
                real, condition, hsc_hr
            )
            self.disc_opt.zero_grad()
            loss.backward()
            self.disc_opt.step()
            return loss, fake_logits, real_logits
        elif optimizer == "generator":
            losses = self._gen_step(real, condition, hsc_hr, seg_map_real)
            self.gen_opt.zero_grad()
            losses[0].backward()
            self.gen_opt.step()
            return losses

    def validation_step(
        self,
        real: torch.Tensor,
        condition: torch.Tensor,
        hsc_hr: torch.Tensor,
        seg_map_real: torch.Tensor,
        optimizer: str,
    ) -> Union[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Evaluate losses without updating model weights.

        Same interface and return values as ``training_step``, but does not
        call ``backward()`` or ``step()``.
        """
        if optimizer == "discriminator":
            return self._disc_step(real, condition, hsc_hr)
        elif optimizer == "generator":
            return self._gen_step(real, condition, hsc_hr, seg_map_real)
