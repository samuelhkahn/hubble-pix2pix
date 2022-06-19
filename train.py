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
	hst_path_train = config["DEFAULT"]["hst_path_train"]
	hsc_path_train = config["DEFAULT"]["hsc_path_train"]
	hst_path_val = config["DEFAULT"]["hst_path_val"]
	hsc_path_val = config["DEFAULT"]["hsc_path_val"]

	hst_dim = int(config["HST_DIM"]["hst_dim"])
	hsc_dim = int(config["HSC_DIM"]["hsc_dim"])

	comet_tag = config["COMET_TAG"]["comet_tag"]

	batch_size = int(config["BATCH_SIZE"]["batch_size"])
	total_steps = int(config["GAN_STEPS"]["gan_steps"])
	save_steps = int(config["SAVE_STEPS"]["save_steps"])


	data_aug = eval(config["DATA_AUG"]["data_aug"])
	identifier = eval(config["IDENTIFIER"]["identifier"])



	display_step = eval(config["DISPLAY_STEPS"]["display_steps"])

	lr = eval(config["LR"]["lr"])
	disc_lr = eval(config["DISC_LR"]["disc_lr"])

	lambda_recon = eval(config["LAMBDA_RECON"]["lambda_recon"])
	lambda_segmap = eval(config["LAMBDA_SEGMAP"]["lambda_segmap"])
	lambda_vgg = eval(config["LAMBDA_VGG"]["lambda_vgg"])
	lambda_scattering = eval(config["LAMBDA_SCATTERING"]["lambda_scattering"])
	lambda_adv = eval(config["LAMBDA_ADV"]["lambda_adv"])


	disc_update_freq = int(config["DISC_UPDATE_FREQ"]["disc_update_freq"])

	pretrained_generator = config["PRETRAINED_GENERATOR"]["pretrained_generator"]
	pretrained_discriminator = 	config["PRETRAINED_DISCRIMINATOR"]["pretrained_discriminator"]
	vgg_loss_weights = eval(config["VGG_LOSS_WEIGHTS"]["vgg_loss_weights"])

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
	experiment.log_parameter("disc_lr",disc_lr)
	experiment.log_parameter("lambda_recon",lambda_recon)
	experiment.log_parameter("lambda_vgg",lambda_vgg)
	experiment.log_parameter("lambda_scattering",lambda_scattering)
	experiment.log_parameter("lambda_segrecon",lambda_segmap)
	experiment.log_parameter("lambda_adv",lambda_adv)
	experiment.log_parameter("disc_update_freq",disc_update_freq)

	for i in range(5):
		experiment.log_parameter(f"vgg_layer_{i+1}",vgg_loss_weights[i])

	model_name = f"gaussian_bcegan_{identifier}_global_lr={lr}_recon={lambda_recon}_segrecon={lambda_segmap}_vgg={lambda_vgg}_scatter={lambda_scattering}_adv={lambda_adv}_discupdate={disc_update_freq}_vgglayer_weieghts_{str(vgg_loss_weights)}"
	print(model_name)


	# Create Dataloader
	dataloader_train = torch.utils.data.DataLoader(
	    SR_HST_HSC_Dataset(hst_path = hst_path_train , hsc_path = hsc_path_train, hr_size=[hst_dim, hst_dim], 
	    	lr_size=[hsc_dim, hsc_dim], transform_type = "ds9_scale",data_aug = data_aug,experiment = experiment), 
	    	batch_size=batch_size, pin_memory=True, shuffle=True, collate_fn = collate_fn)

	dataloader_val = torch.utils.data.DataLoader(
	    SR_HST_HSC_Dataset(hst_path = hst_path_val , hsc_path = hsc_path_val, hr_size=[hst_dim, hst_dim], 
	    	lr_size=[hsc_dim, hsc_dim], transform_type = "ds9_scale",data_aug = data_aug,experiment = experiment), 
	    	batch_size=batch_size, pin_memory=True, shuffle=True, collate_fn = collate_fn)
	dataloader_val = iter(dataloader_val)

	pix2pix = Pix2Pix(in_channels = 1, 
					 out_channels = 1,
					 input_size =600, 
					 device = device,
					 learning_rate=lr,
					 disc_learning_rate=disc_lr,
					 vgg_loss_weights = vgg_loss_weights,
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
		for hr_real,lr,hsc_hr, seg_map_real in tqdm(dataloader_train, position=0):
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
			losses = pix2pix.training_step(hr_real,lr,hsc_hr,seg_map_real,"generator")


			gen_loss,adv_loss,recon_loss,vgg_loss,scattering_loss,segmap_loss = losses[0].item(),\
																				losses[1].item(),\
																				losses[2].item(),\
																				losses[3].item(),\
																				losses[4].item(),\
																				losses[5].item()
			if cur_step%disc_update_freq==0:
				disc_losses = pix2pix.training_step(hr_real,lr,hsc_hr,seg_map_real,"discriminator")
				disc_loss,fake_disc_logits, real_disc_logits = disc_losses[0].item(),disc_losses[1],disc_losses[2]




			if cur_step % display_step == 0 and cur_step > 0:
				hr_real_val,lr_val,hsc_hr_val, seg_map_real_val = next(dataloader_val)

				hr_real_val = hr_real_val.unsqueeze(1).to(device)
				hsc_hr_val = hsc_hr_val.unsqueeze(1).to(device)
				lr_val = lr_val.unsqueeze(1).to(device) # condition
				seg_map_real_val = seg_map_real_val.unsqueeze(1).to(device)
				val_losses = pix2pix.validation_step(hr_real_val,lr_val,hsc_hr_val,seg_map_real_val,"generator")


				gen_val_loss,adv_val_loss,recon_val_loss,vgg_val_loss,scattering_val_loss,segmap_val_loss = val_losses[0].item(),\
																					val_losses[1].item(),\
																					val_losses[2].item(),\
																					val_losses[3].item(),\
																					val_losses[4].item(),\
																					val_losses[5].item()
				disc_val_losses = pix2pix.validation_step(hr_real_val,lr,hsc_hr_val,seg_map_real_val,"discriminator")
				disc_val_loss,fake_disc_val_logits, real_disc_val_logits = disc_val_losses[0].item(),disc_val_losses[1],disc_val_losses[2]
				fake_val_images = pix2pix.generate_fake_images(lr_val,identity_map=True)
				print('Step: {}, Generator loss: {:.5f}, Discriminator loss: {:.5f}'.format(cur_step,gen_val_loss, disc_val_loss))
				hr_val = hr_real_val[0,:,:,:].squeeze(0).cpu()
				lr_val = lr_val[0,:,:,:].squeeze(0).cpu()
				fake_val = fake_val_images[0,0,:,:].double().cpu()
				real_disc_val_logits = real_disc_val_logits[0,0,:,:].cpu()
				fake_disc_val_logits = fake_disc_val_logits[0,0,:,:].cpu()



				# print("FAKE AVG: ",fake_avg)
				# print("MEANS: ",means)

				img_diff = CenterCrop(600)(fake_val - hr_val).cpu().detach().numpy()
				vmax = np.abs(img_diff).max()
				log_figure(CenterCrop(100)(lr_val).detach().numpy(),"100x100 Conditioned Val Image (HSC)",experiment)
				log_figure(CenterCrop(600)(fake_val).detach().numpy(),"600x600 Generated Val Image (SR)",experiment)
				log_figure(CenterCrop(600)(hr_val).detach().numpy(),"600x600 Real Val Image (HST)",experiment)
				log_figure(real_disc_val_logits.detach().numpy(),"Real Disc Val Logits",experiment)
				log_figure(real_disc_val_logits.detach().numpy(),"Fake Disc Val Logits",experiment)

				log_figure(img_diff,"Paired Image Difference",experiment,cmap="bwr_r",set_lims=True,lims=[-vmax,vmax])


				experiment.log_metric("Generator Loss",gen_loss)
				experiment.log_metric("Discriminator Loss",disc_loss)
				experiment.log_metric("VGG Loss",vgg_loss)
				experiment.log_metric("L1 Reconstriction Loss",recon_loss)
				experiment.log_metric("L1 Scattereing Loss",scattering_loss)
				experiment.log_metric("L1 Reconstriction Segmap Loss",segmap_loss)
				experiment.log_metric("L1 Segmap/L1 Recon Loss",segmap_loss/recon_loss)
				experiment.log_metric("Adversarial Loss",adv_loss)


				experiment.log_metric("Generator Val Loss",gen_val_loss)
				experiment.log_metric("Discriminator Val Loss",disc_val_loss)
				experiment.log_metric("VGG Val Loss",vgg_val_loss)
				experiment.log_metric("L1 Val Reconstriction Loss",recon_val_loss)
				experiment.log_metric("L1 Val Scattereing Loss",scattering_val_loss)
				experiment.log_metric("L1 Val Reconstriction Segmap Loss",segmap_val_loss)
				experiment.log_metric("L1 Val Segmap/L1 Recon Loss",segmap_val_loss/recon_val_loss)
				experiment.log_metric("Adversarial Loss",adv_val_loss)


			if cur_step % save_steps == 0 and cur_step > 0:
				torch.save(pix2pix.gen, f'models/gen_pix2pixsr_{model_name}_checkpoint_{cur_step}.pt')
				torch.save(pix2pix.patch_gan, f'models/patchgan_pix2pixsr_{model_name}_checkpoint_{cur_step}.pt')


			cur_step+=1

if __name__=="__main__":
    main()
