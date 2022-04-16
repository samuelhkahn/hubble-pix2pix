from comet_ml import Experiment
from tqdm import tqdm
from pix2pix import Pix2Pix
import os
import torch
from dataset import SR_HST_HSC_Dataset
import numpy as np
import configparser
from log_figure import log_figure
from torchvision.transforms import CenterCrop
import sys


def collate_fn(batch):

	hrs, lrs,hsc_hrs, hr_segs = [], [], [], []
	for hr,lr,hsc_hr,hr_seg in batch:
		if any(el is None for el in [hr,lr,hsc_hr,hr_seg]): #Skip corrupted files
			continue

		hr_nan = torch.isnan(hr).any()
		lr_nan = torch.isnan(lr).any()

		hr_inf = torch.isinf(hr).any()
		lr_inf = torch.isinf(lr).any()

		good_vals = [hr_nan,lr_nan,hr_inf,lr_inf]

		# print(f"HR Shape: {hr.shape}")
		# print(f"HR Up Shape: {hsc_hr.shape}")
		# print(f"LR Shape: {lr.shape}")
		# print(f"HR Seg Shape: {hr_seg.shape}")
		if hr.shape == (768,768) and lr.shape == (128,128) and hsc_hr.shape == (768,768) and True not in good_vals:
			hrs.append(hr)
			lrs.append(lr)
			hsc_hrs.append(hsc_hr)
			hr_segs.append(hr_seg)

	hrs = torch.stack(hrs, dim=0)
	lrs = torch.stack(lrs, dim=0)
	hsc_hrs = torch.stack(hsc_hrs, dim=0)
	hr_segs = torch.stack(hr_segs, dim=0)
	return hrs, lrs, hsc_hrs, hr_segs


def main():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Load file paths from config
	config_file = sys.argv[1]
	config = configparser.ConfigParser()
	config.read(config_file)

	# Configuration Information
	hst_path = config["DEFAULT"]["hst_path"]
	hsc_path = config["DEFAULT"]["hsc_path"]

	hst_dim = int(config["HST_DIM"]["hst_dim"])
	hsc_dim = int(config["HSC_DIM"]["hsc_dim"])

	comet_tag = config["COMET_TAG"]["comet_tag"]

	batch_size = int(config["BATCH_SIZE"]["batch_size"])
	total_steps = int(config["GAN_STEPS"]["gan_steps"])
	save_steps = int(config["SAVE_STEPS"]["save_steps"])


	data_aug = eval(config["DATA_AUG"]["data_aug"])



	display_step = eval(config["DISPLAY_STEPS"]["display_steps"])

	lr = eval(config["LR"]["lr"])
	lambda_recon = eval(config["LAMBDA_RECON"]["lambda_recon"])
	lambda_segmap = eval(config["LAMBDA_SEGMAP"]["lambda_segmap"])
	lambda_vgg = eval(config["LAMBDA_VGG"]["lambda_vgg"])
	lambda_scattering = eval(config["LAMBDA_SCATTERING"]["lambda_scattering"])
	lambda_adv = eval(config["LAMBDA_ADV"]["lambda_adv"])


	disc_update_freq = int(config["DISC_UPDATE_FREQ"]["disc_update_freq"])

	pretrained_generator = config["PRETRAINED_GENERATOR"]["pretrained_generator"]
	pretrained_discriminator = 	config["PRETRAINED_DISCRIMINATOR"]["pretrained_discriminator"]

	# Adding Comet Logging
	api_key = os.environ['COMET_ML_ASTRO_API_KEY']


	experiment = Experiment(
	    api_key=api_key,
	    project_name="Pix2Pix Image Translation: HSC->HST",
	    workspace="samkahn-astro",
	)

	experiment.add_tag(comet_tag)
	# Log Hyperparemeters
	experiment.log_parameter("batch_size",batch_size)
	experiment.log_parameter("total_steps",total_steps)
	experiment.log_parameter("save_steps",save_steps)
	experiment.log_parameter("data_aug",data_aug)
	experiment.log_parameter("display_step",display_step)
	experiment.log_parameter("lr",lr)
	experiment.log_parameter("lambda_recon",lambda_recon)
	experiment.log_parameter("lambda_vgg",lambda_vgg)
	experiment.log_parameter("lambda_scattering",lambda_scattering)
	experiment.log_parameter("lambda_segrecon",lambda_segmap)
	experiment.log_parameter("lambda_adv",lambda_adv)
	experiment.log_parameter("disc_update_freq",disc_update_freq)

	model_name = f"transposed_conv_global_lr={lr}_recon={lambda_recon}_segrecon={lambda_segmap}_vgg={lambda_vgg}_scatter={lambda_scattering}_adv={lambda_adv}_discupdate={disc_update_freq}"
	print(model_name)
	# Create Dataloader
	dataloader = torch.utils.data.DataLoader(
	    SR_HST_HSC_Dataset(hst_path = hst_path , hsc_path = hsc_path, hr_size=[hst_dim, hst_dim], 
	    	lr_size=[hsc_dim, hsc_dim], transform_type = "ds9_scale",data_aug = data_aug,experiment = experiment), 
	    batch_size=batch_size, pin_memory=True, shuffle=True, collate_fn = collate_fn)


	pix2pix = Pix2Pix(in_channels = 1, 
					 out_channels = 1,
					 input_size =600, 
					 device = device,
					 learning_rate=lr, 
					 lambda_recon=lambda_recon, 
					 lambda_segmap=lambda_segmap,
					 lambda_vgg=lambda_vgg, 
					 lambda_scattering=lambda_scattering,
					 lambda_adv=lambda_adv, 
					 display_step=display_step,
					 pretrained_generator = pretrained_generator,
					 pretrained_discriminator = pretrained_discriminator)

	cur_step = 0

	while cur_step < total_steps:
		for hr_real,lr,hsc_hr, seg_map_real in tqdm(dataloader, position=0):
			# Conv2d expects (n_samples, channels, height, width)
			# So add the channel dimension
			hr_real = hr_real.unsqueeze(1).to(device)
			hsc_hr = hsc_hr.unsqueeze(1).to(device)
			lr = lr.unsqueeze(1).to(device) # condition
			seg_map_real = seg_map_real.unsqueeze(1).to(device)
			# print(f"HR Shape: {hr_real.shape}")
			# print(f"LR Up Shape: {lr_up.shape}")
			# print(f"LR  Shape: {lr.shape}")
			# print(f"HR Seg Realy Shape: {seg_map_real.shape}")
			if cur_step%disc_update_freq==0:
				disc_loss = pix2pix.training_step(hr_real,lr,hsc_hr,seg_map_real,"discriminator").item()

			losses = pix2pix.training_step(hr_real,lr,hsc_hr,seg_map_real,"generator")


			gen_loss,adv_loss,recon_loss,vgg_loss,scattering_loss,segmap_loss = losses[0].item(),\
																				losses[1].item(),\
																				losses[2].item(),\
																				losses[3].item(),\
																				losses[4].item(),\
																				losses[5].item()


			if cur_step % display_step == 0 and cur_step > 0:
				fake_images = pix2pix.generate_fake_images(lr)

				print('Step: {}, Generator loss: {:.5f}, Discriminator loss: {:.5f}'.format(cur_step,gen_loss, disc_loss))
				hr = hr_real[0,:,:,:].squeeze(0).cpu()
				lr = lr[0,:,:,:].squeeze(0).cpu()
				fake = fake_images[0,0,:,:].double().cpu()


				img_diff = CenterCrop(600)(fake - hr).cpu().detach().numpy()
				vmax = np.abs(img_diff).max()

				log_figure(hr.detach().numpy(),"HST Full Image",experiment)
				log_figure(lr.detach().numpy(),"LR Image",experiment)
				log_figure(fake.detach().numpy(),"Generated Image",experiment)
				log_figure(CenterCrop(100)(lr).detach().numpy(),"100x100 Conditioned Image (HSC)",experiment)
				log_figure(CenterCrop(600)(fake).detach().numpy(),"600x600 Generated Image (SR)",experiment)
				log_figure(CenterCrop(600)(hr).detach().numpy(),"600x600 Real Image (HST)",experiment)

				log_figure(img_diff,"Paired Image Difference",experiment,cmap="bwr_r",set_lims=True,lims=[-vmax,vmax])


				experiment.log_metric("Generator Loss",gen_loss)
				experiment.log_metric("Discriminator Loss",disc_loss)
				experiment.log_metric("VGG Loss",vgg_loss)
				experiment.log_metric("L1 Reconstriction Loss",recon_loss)
				experiment.log_metric("L1 Scattereing Loss",scattering_loss)
				experiment.log_metric("L1 Reconstriction Segmap Loss",segmap_loss)
				experiment.log_metric("L1 Segmap/L1 Recon Loss",segmap_loss/recon_loss)


				experiment.log_metric("Adversarial Loss",adv_loss)


			if cur_step % save_steps == 0 and cur_step > 0:
				torch.save(pix2pix.gen, f'models/gen_pix2pixsr_{model_name}_checkpoint_{cur_step}.pt')
				torch.save(pix2pix.patch_gan, f'models/patchgan_pix2pixsr_{model_name}_checkpoint_{cur_step}.pt')


			cur_step+=1

if __name__=="__main__":
    main()
