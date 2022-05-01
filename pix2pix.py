from generator import Pix2PixGenerator
from patchgan import PatchGAN
import torch.nn as nn
import torch
from torchvision.transforms import CenterCrop
from vgg19_loss import VGGLoss
from kymatio.torch import Scattering2D
import os
from torchvision import transforms 
from torchvision.transforms.functional import InterpolationMode as IMode
import torchlayers as tl
class Pix2Pix:

    def __init__(self,in_channels, 
                      out_channels,
                      input_size,
                      device,
                      vgg_loss_weights = [1.0,1.0,0.0,0.0,0.0],
                      learning_rate=0.0002,
                      disc_learning_rate=0.0002,
                      lambda_recon=200,
                      lambda_segmap=200,
                      lambda_vgg = 200,
                      lambda_scattering = 1,
                      lambda_adv = 5,
                      display_step=25,
                      pretrained_generator = "",
                      pretrained_discriminator = "" ):

        super().__init__()

        self.device = device
        self.display_step = display_step

        ## Initialize UNET & Patchgan - Load pretrained model if specfied
        if len(pretrained_generator) != 0:
            print(f"Loading Pretrained Generator: {pretrained_generator}")
            pretrained_generator = os.path.join(os.getcwd(),"models",pretrained_generator)
            self.gen = torch.load(pretrained_generator)
        else:
            self.gen = Pix2PixGenerator(in_channels, out_channels)
            tl.build(self.gen,torch.randn(1, 1, 128, 128))
            # self.gen = self.gen.apply(self._weights_init)

        if len(pretrained_discriminator) != 0:
            print(f"Loading Pretrained Discriminator: {pretrained_discriminator}")
            pretrained_discriminator = os.path.join(os.getcwd(),"models",pretrained_discriminator)
            self.patch_gan = torch.load(pretrained_discriminator)
        else:
            self.patch_gan = PatchGAN(2)
            # self.patch_gan = self.patch_gan.apply(self._weights_init)


        # Loss component weights
        self.lr = learning_rate
        self.disc_lr = disc_learning_rate
        self.lambda_recon = lambda_recon
        self.lambda_vgg = lambda_vgg
        self.lambda_scattering = lambda_scattering
        self.lambda_adv = lambda_adv
        self.lambda_segmap = lambda_segmap

        #Loss functions 
        self.adversarial_criterion = nn.BCEWithLogitsLoss()

        self.recon_criterion_l1 = nn.L1Loss()
        self.recon_criterion_l2 = nn.MSELoss()
        self.vgg_criterion = VGGLoss(self.device,weights=vgg_loss_weights)
        self.scattering_f = Scattering2D(J=3, L=8,shape=(input_size, input_size), out_type="array",max_order=2).to(device)


        #Optimizers 
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr)
        self.disc_opt = torch.optim.Adam(self.patch_gan.parameters(), lr=self.disc_lr)


        self.hr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(768, interpolation=IMode.BILINEAR),
            transforms.ToTensor()
        ])

        # put models on proper device
        self.gen.to(self.device)
        self.patch_gan.to(self.device)


    # def _weights_init(self,m):
    #     if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
    #         torch.nn.init.normal_(m.weight, 0.0, 0.02)
    #     if isinstance(m, nn.BatchNorm2d):
    #         torch.nn.init.normal_(m.weight, 0.0, 0.02)
    #         torch.nn.init.constant_(m.bias, 0)
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
        fake_images = CenterCrop(600)(fake_images)
        real_images = CenterCrop(600)(real_images)
        seg_map_real =  CenterCrop(600)(seg_map_real)

        # # Upsample LR image so wecan input as second channel of discriminator
        # conditioned_images = conditioned_images.squeeze(1)
        # conditioned_images =  self.hr_transforms(conditioned_images)
        # conditioned_images = conditioned_images.unsqueeze(1)
        
        conditioned_images = CenterCrop(600)(conditioned_images)

        disc_logits = self.patch_gan(fake_images,conditioned_images)

        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        #recon_loss = self.recon_criterion_l1(fake_images, real_images)
        recon_loss = self.recon_criterion_l1(fake_images, real_images)
        vgg_loss = self.vgg_criterion(fake_images, real_images)

        # vgg_loss = vgg_loss.sum()/len(weights)
        #segmap loss
        segmap_loss = self.l1_loss_with_mask(fake_images, real_images,seg_map_real)


        #wavlet scattering loss
        scat_real = self.scattering_f(real_images.contiguous()).squeeze(1)[:,1:,:,:]
        scat_fake = self.scattering_f(fake_images.contiguous()).squeeze(1)[:,1:,:,:]


        scattering_loss = (scat_real - scat_fake).abs().sum(axis=(1, 2, 3)).mean()

        total_loss = self.lr*(self.lambda_adv*adversarial_loss+ self.lambda_recon*recon_loss\
                     + self.lambda_vgg*vgg_loss\
                     + self.lambda_scattering*scattering_loss\
                     + self.lambda_segmap*segmap_loss)

        return total_loss,adversarial_loss,recon_loss,vgg_loss,scattering_loss,segmap_loss

    def generate_fake_images(self, conditioned_images):
        # Generate image for plotting
        fake_images = self.gen(conditioned_images)
        return fake_images

    def _disc_step(self, real_images, conditioned_images,hsc_hr):
        fake_images = self.gen(conditioned_images).detach()

        #Crop off sides so not computed in loss 
        fake_images = CenterCrop(600)(fake_images)
        real_images = CenterCrop(600)(real_images)
        hsc_hr = CenterCrop(600)(hsc_hr)

        # Upsample LR image so wecan input as second channel of discriminator
        # conditioned_images = conditioned_images.squeeze(1)
        # print(conditioned_images.shape)

        # conditioned_images =  self.hr_transforms(conditioned_images)
        # print(conditioned_images.shape)

        # conditioned_images = conditioned_images.unsqueeze(1)
        # conditioned_images = CenterCrop(600)(conditioned_images)


        ### NOTE TO SELF; I removed the second channel of PATCHGAN, which
        ### is the conditioned image. In the context of SuperResolution
        ### It doesn't make too mch sense since we don't have a HR input image. 
        ### We'd need to upsample the input and that would likely cause shifting
        ### It will be worth trying it though as an expereiment!
        fake_logits = self.patch_gan(fake_images,hsc_hr)
        real_logits = self.patch_gan(real_images,hsc_hr)

        fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
        return real_loss+fake_loss, fake_logits, real_logits


    def training_step(self, real, condition, hsc_hr, seg_map_real, optimizer):

        loss = None
        if optimizer == "discriminator":
            loss,fake_logits, real_logits = self._disc_step(real,condition, hsc_hr)
            self.disc_opt.zero_grad()
            loss.backward()
            self.disc_opt.step()
            return loss,fake_logits, real_logits
        elif optimizer == "generator":
            total_loss,adversarial_loss,recon_loss,vgg_loss,scattering_loss,segmap_loss = self._gen_step(real,condition, seg_map_real)
            self.gen_opt.zero_grad()
            total_loss.backward()
            self.gen_opt.step()
            return total_loss,adversarial_loss,recon_loss,vgg_loss,scattering_loss,segmap_loss
