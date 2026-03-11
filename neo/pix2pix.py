"""Pix2Pix conditional GAN for astronomical image super-resolution.

Implements a Pix2Pix model with a multi-component loss function designed for
astronomical imagery:
    - BCE adversarial loss (PatchGAN discriminator)
    - L1 reconstruction loss
    - VGG-19 perceptual loss
    - Wavelet scattering transform loss (preserves multi-scale structure)
    - Segmentation-masked L1 loss (emphasizes source regions)
"""

import os

import torch
import torch.nn as nn
import torchlayers as tl
from kymatio.torch import Scattering2D
from torchvision import transforms
from torchvision.transforms import CenterCrop
from torchvision.transforms.functional import InterpolationMode as IMode

from neo.models.generator import Pix2PixGenerator
from neo.models.patchgan import PatchGAN
from neo.models.vgg19_loss import VGGLoss


class Pix2Pix:
    """Pix2Pix conditional GAN for HSC -> HST super-resolution.

    Trains a U-Net generator to produce HST-quality images from HSC inputs,
    using a PatchGAN discriminator and a composite loss function that
    balances adversarial, reconstruction, perceptual, scattering, and
    segmentation-weighted objectives.

    Args:
        in_channels: Number of input image channels.
        out_channels: Number of output image channels.
        input_size: Spatial size of the cropped training images (for scattering transform).
        device: Torch device ('cuda' or 'cpu').
        vgg_loss_weights: Per-layer weights for VGG perceptual loss (5 values).
        learning_rate: Generator learning rate.
        disc_learning_rate: Discriminator learning rate.
        lambda_recon: Weight for L1 reconstruction loss.
        lambda_segmap: Weight for segmentation-masked L1 loss.
        lambda_vgg: Weight for VGG perceptual loss.
        lambda_scattering: Weight for wavelet scattering loss.
        lambda_adv: Weight for adversarial loss.
        display_step: Logging frequency (in training steps).
        pretrained_generator: Filename of pretrained generator checkpoint (or empty string).
        pretrained_discriminator: Filename of pretrained discriminator checkpoint (or empty string).
    """

    def __init__(self, in_channels, out_channels, input_size, device,
                 vgg_loss_weights=(1.0, 1.0, 0.0, 0.0, 0.0),
                 learning_rate=0.0002, disc_learning_rate=0.0002,
                 lambda_recon=200, lambda_segmap=200, lambda_vgg=200,
                 lambda_scattering=1, lambda_adv=5, display_step=25,
                 pretrained_generator="", pretrained_discriminator=""):

        super().__init__()

        self.device = device
        self.display_step = display_step

        # Initialize generator
        if pretrained_generator:
            print(f"Loading Pretrained Generator: {pretrained_generator}")
            path = os.path.join(os.getcwd(), "models", pretrained_generator)
            self.gen = torch.load(path)
        else:
            self.gen = Pix2PixGenerator(in_channels, out_channels)
            tl.build(self.gen, torch.randn(1, 1, 128, 128), True)

        # Initialize discriminator
        if pretrained_discriminator:
            print(f"Loading Pretrained Discriminator: {pretrained_discriminator}")
            path = os.path.join(os.getcwd(), "models", pretrained_discriminator)
            self.patch_gan = torch.load(path)
        else:
            self.patch_gan = PatchGAN(2)

        # Loss weights
        self.lr = learning_rate
        self.disc_lr = disc_learning_rate
        self.lambda_recon = lambda_recon
        self.lambda_vgg = lambda_vgg
        self.lambda_scattering = lambda_scattering
        self.lambda_adv = lambda_adv
        self.lambda_segmap = lambda_segmap

        # Loss functions
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion_l1 = nn.L1Loss()
        self.recon_criterion_l2 = nn.MSELoss()
        self.vgg_criterion = VGGLoss(self.device, weights=vgg_loss_weights)
        self.scattering_f = Scattering2D(
            J=3, L=8, shape=(input_size, input_size),
            out_type="array", max_order=2,
        ).to(device)

        # Optimizers
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr)
        self.disc_opt = torch.optim.Adam(self.patch_gan.parameters(), lr=self.disc_lr)

        self.hr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(768, interpolation=IMode.BILINEAR),
            transforms.ToTensor(),
        ])

        # Move models to device
        self.gen.to(self.device)
        self.patch_gan.to(self.device)

    @staticmethod
    def l1_loss_with_mask(x_real, x_fake, seg_map_real):
        """L1 loss weighted by a binary segmentation mask."""
        return torch.sum(torch.abs(x_real - x_fake) * seg_map_real) / torch.sum(seg_map_real)

    @staticmethod
    def l2_loss_with_mask(x_real, x_fake, seg_map_real):
        """L2 loss weighted by a binary segmentation mask."""
        return torch.sum(((x_real - x_fake) * seg_map_real) ** 2.0) / torch.sum(seg_map_real)

    def _gen_step(self, real_images, conditioned_images, hsc_hr, seg_map_real):
        """Compute generator loss with all components."""
        fake_images = self.gen(conditioned_images, identity_map=True)

        # Center crop to remove border artifacts
        fake_images = CenterCrop(600)(fake_images)
        real_images = CenterCrop(600)(real_images)
        seg_map_real = CenterCrop(600)(seg_map_real)
        hsc_hr = CenterCrop(600)(hsc_hr)

        # Adversarial loss
        disc_logits = self.patch_gan(fake_images, hsc_hr)
        adversarial_loss = self.adversarial_criterion(
            disc_logits.flatten(), torch.ones_like(disc_logits).flatten()
        )

        # Reconstruction losses
        recon_loss = self.recon_criterion_l1(fake_images, real_images)
        vgg_loss = self.vgg_criterion(fake_images, real_images)
        segmap_loss = self.l1_loss_with_mask(fake_images, real_images, seg_map_real)

        # Wavelet scattering loss
        scat_real = self.scattering_f(real_images.contiguous()).squeeze(1)[:, 1:, :, :]
        scat_fake = self.scattering_f(fake_images.contiguous()).squeeze(1)[:, 1:, :, :]
        scattering_loss = (scat_real - scat_fake).abs().sum(axis=(1, 2, 3)).mean()

        # Weighted total loss
        total_loss = self.lr * (
            self.lambda_adv * adversarial_loss
            + self.lambda_recon * recon_loss
            + self.lambda_vgg * vgg_loss
            + self.lambda_scattering * scattering_loss
            + self.lambda_segmap * segmap_loss
        )

        return total_loss, adversarial_loss, recon_loss, vgg_loss, scattering_loss, segmap_loss

    def generate_fake_images(self, conditioned_images, identity_map=False):
        """Generate super-resolved images from low-resolution inputs."""
        return self.gen(conditioned_images, identity_map=identity_map)

    def _disc_step(self, real_images, conditioned_images, hsc_hr):
        """Compute discriminator loss on real and fake images."""
        fake_images = self.gen(conditioned_images, identity_map=True).detach()

        # Center crop to remove border artifacts
        fake_images = CenterCrop(600)(fake_images)
        real_images = CenterCrop(600)(real_images)
        hsc_hr = CenterCrop(600)(hsc_hr)

        fake_logits = self.patch_gan(fake_images, hsc_hr)
        real_logits = self.patch_gan(real_images, hsc_hr)

        fake_loss = self.adversarial_criterion(
            fake_logits.flatten(), torch.zeros_like(fake_logits).flatten()
        )
        real_loss = self.adversarial_criterion(
            real_logits.flatten(), torch.ones_like(real_logits).flatten()
        )
        return real_loss + fake_loss, fake_logits, real_logits

    def training_step(self, real, condition, hsc_hr, seg_map_real, optimizer):
        """Execute one training step for the specified optimizer.

        Args:
            real: Ground truth HR images (HST).
            condition: Low-resolution input images (HSC).
            hsc_hr: Upsampled HSC images for discriminator conditioning.
            seg_map_real: Binary segmentation maps of detected sources.
            optimizer: Which model to train - "generator" or "discriminator".

        Returns:
            For generator: (total_loss, adv_loss, recon_loss, vgg_loss, scat_loss, seg_loss)
            For discriminator: (disc_loss, fake_logits, real_logits)
        """
        if optimizer == "discriminator":
            loss, fake_logits, real_logits = self._disc_step(real, condition, hsc_hr)
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

    def validation_step(self, real, condition, hsc_hr, seg_map_real, optimizer):
        """Evaluate losses without updating model weights.

        Same interface and return values as ``training_step``.
        """
        if optimizer == "discriminator":
            return self._disc_step(real, condition, hsc_hr)
        elif optimizer == "generator":
            return self._gen_step(real, condition, hsc_hr, seg_map_real)
