from generator import Generator
from patchgan import PatchGAN
import torch.nn as nn
import torch
class Pix2Pix:

    def __init__(self, in_channels, out_channels, device,learning_rate=0.0002, lambda_recon=200, display_step=25):

        super().__init__()
        
        self.display_step = display_step
        self.gen = Generator(in_channels, out_channels)
        self.patch_gan = PatchGAN(in_channels + out_channels)

        self.lr = learning_rate
        self.lambda_recon = lambda_recon
        # intializing weights
        self.gen = self.gen.apply(self._weights_init)
        self.patch_gan = self.patch_gan.apply(self._weights_init)
        # initialize Weights
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

        #Optimizers 
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr)
        self.disc_opt = torch.optim.Adam(self.patch_gan.parameters(), lr=self.lr)

        # put on proper device
        self.gen.to(device)
        self.patch_gan.to(device)


    def _weights_init(self,m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
    @staticmethod
    def l1_loss_with_mask(x_real, x_fake,seg_map_real):
        return torch.sum(((torch.abs(x_real-x_fake))*seg_map_real))/torch.sum(seg_map_real)

    def _gen_step(self, real_images, conditioned_images,seg_map_real):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        fake_images = self.gen(conditioned_images)
        disc_logits = self.patch_gan(fake_images, conditioned_images)
        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        recon_loss = self.l1_loss_with_mask(fake_images, real_images,seg_map_real)
        return adversarial_loss + self.lambda_recon * recon_loss

    def generate_fake_images(self, conditioned_images):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        fake_images = self.gen(conditioned_images)

        return fake_images
    def _disc_step(self, real_images, conditioned_images):
        fake_images = self.gen(conditioned_images).detach()
        fake_logits = self.patch_gan(fake_images, conditioned_images)

        real_logits = self.patch_gan(real_images, conditioned_images)

        fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
        return (real_loss + fake_loss) / 2


    def training_step(self, real,condition,seg_map_real, optimizer):

        loss = None
        if optimizer == "discriminator":
            loss = self._disc_step(real, condition)
            self.disc_opt.zero_grad()
            loss.backward()
            self.disc_opt.step()
        elif optimizer == "generator":
            loss = self._gen_step(real, condition, seg_map_real)
            self.gen_opt.zero_grad()
            loss.backward()
            self.gen_opt.step()
        return loss