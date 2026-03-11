"""Training script for the Neo super-resolution model.

Trains a Pix2Pix conditional GAN to translate ground-based HSC images to
space-based HST quality using paired FITS image cutouts.

Usage:
    python train.py <config_file>

Example:
    python train.py configs/example.ini
"""

import configparser
import os
import sys

import numpy as np
import torch
from comet_ml import Experiment
from torchvision.transforms import CenterCrop
from tqdm import tqdm

from neo.data.collate_fn import collate_fn
from neo.data.dataset import SR_HST_HSC_Dataset
from neo.log_figure import log_figure
from neo.pix2pix import Pix2Pix


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load configuration
    config_file = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)

    # Data paths
    hst_path_train = config["DEFAULT"]["hst_path_train"]
    hsc_path_train = config["DEFAULT"]["hsc_path_train"]
    hst_path_val = config["DEFAULT"]["hst_path_val"]
    hsc_path_val = config["DEFAULT"]["hsc_path_val"]

    # Image dimensions
    hst_dim = int(config["HST_DIM"]["hst_dim"])
    hsc_dim = int(config["HSC_DIM"]["hsc_dim"])

    # Training parameters
    comet_tag = config["COMET_TAG"]["comet_tag"]
    batch_size = int(config["BATCH_SIZE"]["batch_size"])
    total_steps = int(config["GAN_STEPS"]["gan_steps"])
    save_steps = int(config["SAVE_STEPS"]["save_steps"])
    data_aug = eval(config["DATA_AUG"]["data_aug"])
    identifier = eval(config["IDENTIFIER"]["identifier"])
    display_step = eval(config["DISPLAY_STEPS"]["display_steps"])

    # Optimizer parameters
    lr = eval(config["LR"]["lr"])
    disc_lr = eval(config["DISC_LR"]["disc_lr"])

    # Loss weights
    lambda_recon = eval(config["LAMBDA_RECON"]["lambda_recon"])
    lambda_segmap = eval(config["LAMBDA_SEGMAP"]["lambda_segmap"])
    lambda_vgg = eval(config["LAMBDA_VGG"]["lambda_vgg"])
    lambda_scattering = eval(config["LAMBDA_SCATTERING"]["lambda_scattering"])
    lambda_adv = eval(config["LAMBDA_ADV"]["lambda_adv"])

    # Discriminator settings
    disc_update_freq = int(config["DISC_UPDATE_FREQ"]["disc_update_freq"])

    # Pretrained models
    pretrained_generator = config["PRETRAINED_GENERATOR"]["pretrained_generator"]
    pretrained_discriminator = config["PRETRAINED_DISCRIMINATOR"]["pretrained_discriminator"]
    vgg_loss_weights = eval(config["VGG_LOSS_WEIGHTS"]["vgg_loss_weights"])

    # Initialize Comet ML experiment tracking
    api_key = os.environ['COMET_ML_ASTRO_API_KEY']
    experiment = Experiment(
        api_key=api_key,
        project_name="Pix2Pix Image Translation: HSC->HST",
        workspace="samkahn-astro",
    )

    experiment.add_tag(comet_tag)
    experiment.log_parameter("batch_size", batch_size)
    experiment.log_parameter("total_steps", total_steps)
    experiment.log_parameter("save_steps", save_steps)
    experiment.log_parameter("data_aug", data_aug)
    experiment.log_parameter("display_step", display_step)
    experiment.log_parameter("lr", lr)
    experiment.log_parameter("disc_lr", disc_lr)
    experiment.log_parameter("lambda_recon", lambda_recon)
    experiment.log_parameter("lambda_vgg", lambda_vgg)
    experiment.log_parameter("lambda_scattering", lambda_scattering)
    experiment.log_parameter("lambda_segrecon", lambda_segmap)
    experiment.log_parameter("lambda_adv", lambda_adv)
    experiment.log_parameter("disc_update_freq", disc_update_freq)
    for i in range(5):
        experiment.log_parameter(f"vgg_layer_{i+1}", vgg_loss_weights[i])

    model_name = (
        f"gaussian_bcegan_{identifier}_global_lr={lr}_recon={lambda_recon}"
        f"_segrecon={lambda_segmap}_vgg={lambda_vgg}_scatter={lambda_scattering}"
        f"_adv={lambda_adv}_discupdate={disc_update_freq}"
        f"_vgglayer_weights_{str(vgg_loss_weights)}"
    )
    print(model_name)

    # Create dataloaders
    dataloader_train = torch.utils.data.DataLoader(
        SR_HST_HSC_Dataset(
            hst_path=hst_path_train, hsc_path=hsc_path_train,
            hr_size=[hst_dim, hst_dim], lr_size=[hsc_dim, hsc_dim],
            transform_type="ds9_scale", data_aug=data_aug, experiment=experiment,
        ),
        batch_size=batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn,
    )

    dataloader_val = torch.utils.data.DataLoader(
        SR_HST_HSC_Dataset(
            hst_path=hst_path_val, hsc_path=hsc_path_val,
            hr_size=[hst_dim, hst_dim], lr_size=[hsc_dim, hsc_dim],
            transform_type="ds9_scale", data_aug=data_aug, experiment=experiment,
        ),
        batch_size=batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn,
    )
    dataloader_val = iter(dataloader_val)

    # Initialize model
    pix2pix = Pix2Pix(
        in_channels=1, out_channels=1, input_size=600, device=device,
        learning_rate=lr, disc_learning_rate=disc_lr,
        vgg_loss_weights=vgg_loss_weights, lambda_recon=lambda_recon,
        lambda_segmap=lambda_segmap, lambda_vgg=lambda_vgg,
        lambda_scattering=lambda_scattering, lambda_adv=lambda_adv,
        display_step=display_step, pretrained_generator=pretrained_generator,
        pretrained_discriminator=pretrained_discriminator,
    )

    # Training loop
    cur_step = 0
    while cur_step < total_steps:
        for hr_real, lr, hsc_hr, seg_map_real in tqdm(dataloader_train, position=0):
            # Add channel dimension: (B, H, W) -> (B, 1, H, W)
            hr_real = hr_real.unsqueeze(1).to(device)
            hsc_hr = hsc_hr.unsqueeze(1).to(device)
            lr = lr.unsqueeze(1).to(device)
            seg_map_real = seg_map_real.unsqueeze(1).to(device)

            # Generator step
            losses = pix2pix.training_step(hr_real, lr, hsc_hr, seg_map_real, "generator")
            gen_loss = losses[0].item()
            adv_loss = losses[1].item()
            recon_loss = losses[2].item()
            vgg_loss = losses[3].item()
            scattering_loss = losses[4].item()
            segmap_loss = losses[5].item()

            # Discriminator step (at specified frequency)
            if cur_step % disc_update_freq == 0:
                disc_losses = pix2pix.training_step(hr_real, lr, hsc_hr, seg_map_real, "discriminator")
                disc_loss = disc_losses[0].item()
                fake_disc_logits = disc_losses[1]
                real_disc_logits = disc_losses[2]

            # Validation and logging
            if cur_step % display_step == 0 and cur_step > 0:
                hr_real_val, lr_val, hsc_hr_val, seg_map_real_val = next(dataloader_val)

                hr_real_val = hr_real_val.unsqueeze(1).to(device)
                hsc_hr_val = hsc_hr_val.unsqueeze(1).to(device)
                lr_val = lr_val.unsqueeze(1).to(device)
                seg_map_real_val = seg_map_real_val.unsqueeze(1).to(device)

                val_losses = pix2pix.validation_step(hr_real_val, lr_val, hsc_hr_val, seg_map_real_val, "generator")
                gen_val_loss = val_losses[0].item()
                adv_val_loss = val_losses[1].item()
                recon_val_loss = val_losses[2].item()
                vgg_val_loss = val_losses[3].item()
                scattering_val_loss = val_losses[4].item()
                segmap_val_loss = val_losses[5].item()

                disc_val_losses = pix2pix.validation_step(hr_real_val, lr_val, hsc_hr_val, seg_map_real_val, "discriminator")
                disc_val_loss = disc_val_losses[0].item()
                fake_disc_val_logits = disc_val_losses[1]
                real_disc_val_logits = disc_val_losses[2]

                fake_val_images = pix2pix.generate_fake_images(lr_val, identity_map=True)
                print(f'Step: {cur_step}, Generator loss: {gen_val_loss:.5f}, Discriminator loss: {disc_val_loss:.5f}')

                # Extract single images for visualization
                hr_val = hr_real_val[0, :, :, :].squeeze(0).cpu()
                lr_val_img = lr_val[0, :, :, :].squeeze(0).cpu()
                fake_val = fake_val_images[0, 0, :, :].double().cpu()
                real_disc_val_map = real_disc_val_logits[0, 0, :, :].cpu()
                fake_disc_val_map = fake_disc_val_logits[0, 0, :, :].cpu()

                # Log visualization figures
                img_diff = CenterCrop(600)(fake_val - hr_val).cpu().detach().numpy()
                vmax = np.abs(img_diff).max()

                log_figure(CenterCrop(100)(lr_val_img).detach().numpy(), "100x100 Conditioned Val Image (HSC)", experiment)
                log_figure(CenterCrop(600)(fake_val).detach().numpy(), "600x600 Generated Val Image (SR)", experiment)
                log_figure(CenterCrop(600)(hr_val).detach().numpy(), "600x600 Real Val Image (HST)", experiment)
                log_figure(real_disc_val_map.detach().numpy(), "Real Disc Val Logits", experiment)
                log_figure(fake_disc_val_map.detach().numpy(), "Fake Disc Val Logits", experiment)
                log_figure(img_diff, "Paired Image Difference", experiment, cmap="bwr_r", set_lims=True, lims=[-vmax, vmax])

                # Log training metrics
                experiment.log_metric("Generator Loss", gen_loss)
                experiment.log_metric("Discriminator Loss", disc_loss)
                experiment.log_metric("VGG Loss", vgg_loss)
                experiment.log_metric("L1 Reconstruction Loss", recon_loss)
                experiment.log_metric("L1 Scattering Loss", scattering_loss)
                experiment.log_metric("L1 Segmap Reconstruction Loss", segmap_loss)
                experiment.log_metric("L1 Segmap/L1 Recon Ratio", segmap_loss / recon_loss)
                experiment.log_metric("Adversarial Loss", adv_loss)

                # Log validation metrics
                experiment.log_metric("Generator Val Loss", gen_val_loss)
                experiment.log_metric("Discriminator Val Loss", disc_val_loss)
                experiment.log_metric("VGG Val Loss", vgg_val_loss)
                experiment.log_metric("L1 Val Reconstruction Loss", recon_val_loss)
                experiment.log_metric("L1 Val Scattering Loss", scattering_val_loss)
                experiment.log_metric("L1 Val Segmap Reconstruction Loss", segmap_val_loss)
                experiment.log_metric("L1 Val Segmap/L1 Recon Ratio", segmap_val_loss / recon_val_loss)
                experiment.log_metric("Adversarial Val Loss", adv_val_loss)

            # Save checkpoints
            if cur_step % save_steps == 0 and cur_step > 0:
                torch.save(pix2pix.gen, f'models/gen_pix2pixsr_{model_name}_checkpoint_{cur_step}.pt')
                torch.save(pix2pix.patch_gan, f'models/patchgan_pix2pixsr_{model_name}_checkpoint_{cur_step}.pt')

            cur_step += 1


if __name__ == "__main__":
    main()
