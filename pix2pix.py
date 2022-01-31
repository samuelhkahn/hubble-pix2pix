from generator import Generator
from patchgan import PatchGAN
import torch.nn as nn
import torch
from torchvision.transforms import CenterCrop
from vgg19_loss import VGGLoss
from kymatio.torch import Scattering2D

class Pix2Pix:

    def __init__(self,in_channels, 
                      out_channels,
                      input_size,
                      device,
                      learning_rate=0.0002, 
                      lambda_recon=200, 
                      lambda_vgg = 200,
                      lambda_scattering = 1,

                      display_step=25):

        super().__init__()

        self.device = device
        self.display_step = display_step
        self.gen = Generator(in_channels, out_channels)
        self.patch_gan = PatchGAN(in_channels + out_channels)

        self.lr = learning_rate
        self.lambda_recon = lambda_recon
        self.lambda_vgg = lambda_vgg
        self.lambda_scattering = lambda_scattering


        # intializing weights
        self.gen = self.gen.apply(self._weights_init)
        self.patch_gan = self.patch_gan.apply(self._weights_init)

        #Loss functions 
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion_l1 = nn.L1Loss()
        self.recon_criterion_l2 = nn.MSELoss()
        self.vgg_criterion = VGGLoss(self.device)
        self.scattering_f = Scattering2D(J=3, L=8,shape=(input_size, input_size), out_type="array",max_order=2).to(device)

        #Optimizers 
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr)
        self.disc_opt = torch.optim.Adam(self.patch_gan.parameters(), lr=self.lr)

        # put models on proper device
        self.gen.to(self.device)
        self.patch_gan.to(self.device)


    def _weights_init(self,m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
    @staticmethod
    def l1_loss_with_mask(x_real, x_fake,seg_map_real):
        return torch.sum(((torch.abs(x_real-x_fake))*seg_map_real))/torch.sum(seg_map_real)
    @staticmethod
    def l2_loss_with_mask(x_real, x_fake,seg_map_real):
        return torch.sum(((x_real-x_fake)*seg_map_real)**2.0)/torch.sum(seg_map_real)

    def _gen_step(self, real_images, conditioned_images,seg_map_real):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        fake_images = self.gen(conditioned_images)

        #Crop off sides so not computed in loss 
        fake_images = CenterCrop(100)(fake_images)
        conditioned_images = CenterCrop(100)(conditioned_images)
        real_images = CenterCrop(100)(real_images)

        disc_logits = self.patch_gan(fake_images, conditioned_images)

        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        #recon_loss = self.recon_criterion_l1(fake_images, real_images)
        recon_loss = self.recon_criterion_l1(fake_images, real_images)
        vgg_loss = self.vgg_criterion(fake_images, real_images)

        #wavlet scattering loss
        scat_real = self.scattering_f(real_images.contiguous()).squeeze(1)[:,1:,:,:]
        scat_fake = self.scattering_f(fake_images.contiguous()).squeeze(1)[:,1:,:,:]


        scattering_loss = (scat_real - scat_fake).abs().sum(axis=(1, 2, 3)).mean()

        total_loss = self.lr*(adversarial_loss\
                     + self.lambda_recon*recon_loss\
                     + self.lambda_vgg*vgg_loss\
                     + self.lambda_scattering*scattering_loss)

        
        return total_loss,adversarial_loss,recon_loss,vgg_loss,scattering_loss

    def generate_fake_images(self, conditioned_images):
        # Generate image for plotting
        fake_images = self.gen(conditioned_images)
        return fake_images

    def _disc_step(self, real_images, conditioned_images):
        fake_images = self.gen(conditioned_images).detach()

        #Crop off sides so not computed in loss 
        fake_images = CenterCrop(100)(fake_images)
        conditioned_images = CenterCrop(100)(conditioned_images)
        real_images = CenterCrop(100)(real_images)

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
            return loss
        elif optimizer == "generator":
            total_loss,adversarial_loss,recon_loss,vgg_loss,scattering_loss = self._gen_step(real, condition, seg_map_real)
            self.gen_opt.zero_grad()
            total_loss.backward()
            self.gen_opt.step()
            return total_loss,adversarial_loss,recon_loss,vgg_loss,scattering_loss