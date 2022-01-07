from comet_ml import Experiment
from tqdm import tqdm
from pix2pix import Pix2Pix
import os
import torch
from dataset import SR_HST_HSC_Dataset
import numpy as np
import configparser
from log_figure import log_figure


# Load file paths from config
config = configparser.ConfigParser()
config.read('paths.config')
hst_dim = int(config["HST_DIM"]["hst_dim"])
hsc_dim = int(config["HSC_DIM"]["hsc_dim"])


def collate_fn(batch):
	hrs, hr_downs, lrs, hr_segs = [], [], [], []

	for hr,hr_down, lr,hr_seg in batch:
		hr_nan = torch.isnan(hr).any()
		hr_down_nan = torch.isnan(hr_down).any()
		lr_nan = torch.isnan(lr).any()

		hr_inf = torch.isinf(hr).any()
		hr_down_inf = torch.isnan(hr_down).any()
		lr_inf = torch.isinf(lr).any()

		good_vals = [hr_nan,hr_down_nan,lr_nan,hr_inf,hr_down_inf,lr_inf]
		if hr.shape == (600,600) and hr_down.shape == (128,128) and lr.shape == (128,128) and True not in good_vals:
			hrs.append(hr)
			hr_downs.append(hr_down)
			lrs.append(lr)
			hr_segs.append(hr_seg)

	hrs = torch.stack(hrs, dim=0)
	hr_downs = torch.stack(hr_downs, dim=0)
	lrs = torch.stack(lrs, dim=0)
	hr_segs = torch.stack(hr_segs, dim=0)
	return hrs, hr_downs, lrs, hr_segs


def main():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Load file paths from config
	config = configparser.ConfigParser()
	config.read('paths.config')

	# Configuration Information
	hst_path = config["DEFAULT"]["hst_path"]
	hsc_path = config["DEFAULT"]["hsc_path"]

	comet_tag = config["COMET_TAG"]["comet_tag"]
	model_name = config["MODEL_NAME"]["model_name"]


	batch_size = int(config["BATCH_SIZE"]["batch_size"])
	total_steps = int(config["GAN_STEPS"]["gan_steps"])
	save_steps = int(config["SAVE_STEPS"]["save_steps"])


	data_aug = eval(config["DATA_AUG"]["data_aug"])



	display_step = eval(config["DISPLAY_STEPS"]["display_steps"])

	lr = eval(config["LR"]["lr"])
	lambda_recon = eval(config["LAMBDA_RECON"]["lambda_recon"])
	lambda_vgg = eval(config["LAMBDA_VGG"]["lambda_vgg"])

	disc_update_freq = int(config["DISC_UPDATE_FREQ"]["disc_update_freq"])


	# Adding Comet Logging
	api_key = os.environ['COMET_ML_ASTRO_API_KEY']


	experiment = Experiment(
	    api_key=api_key,
	    project_name="Pix2Pix Image Translation: HSC->HST",
	    workspace="samkahn-astro",
	)

	experiment.add_tag(comet_tag)
	

	# Create Dataloader
	dataloader = torch.utils.data.DataLoader(
	    SR_HST_HSC_Dataset(hst_path = hst_path , hsc_path = hsc_path, hr_size=[hst_dim, hst_dim], 
	    	lr_size=[hsc_dim, hsc_dim], transform_type = "paired_image_translation",data_aug = data_aug), 
	    batch_size=batch_size, pin_memory=True, shuffle=True, collate_fn = collate_fn)


	pix2pix = Pix2Pix(1, 1, device,learning_rate=lr, lambda_recon=lambda_recon, lambda_vgg=lambda_vgg, display_step=display_step)

	cur_step = 0

	while cur_step < total_steps:
		for hr_real,hr_down,lr_condition, seg_map_real in tqdm(dataloader, position=0):
			# Conv2d expects (n_samples, channels, height, width)
			# So add the channel dimension
			hr_real = hr_real.to(device)
			hr_down = hr_down.unsqueeze(1).to(device) # real
			lr_condition = lr_condition.unsqueeze(1).to(device) # condition
			seg_map_real = seg_map_real.unsqueeze(1).to(device)

			if cur_step%disc_update_freq==0:
				disc_loss = pix2pix.training_step(hr_down,lr_condition,seg_map_real,"discriminator").item()

			gen_loss = pix2pix.training_step(hr_down,lr_condition,seg_map_real,"generator").item()

			if cur_step % display_step == 0 and cur_step > 0:
				fake_images = pix2pix.generate_fake_images(lr_condition)

				print('Step: {}, Generator loss: {:.5f}, Discriminator loss: {:.5f}'.format(cur_step,gen_loss, disc_loss))

				real = hr_down[0,:,:,:].squeeze(0).cpu()
				lr_condition = lr_condition[0,:,:,:].squeeze(0).cpu()
				fake = fake_images[0,0,:,:].double().cpu()


				img_diff = (fake - real).cpu().detach().numpy()
				vmax = np.abs(img_diff).max()


				log_figure(real.detach().numpy(),"Real Image",experiment)
				log_figure(lr_condition.detach().numpy(),"Condition Image",experiment)
				log_figure(fake.detach().numpy(),"Generate Image",experiment)
				log_figure(img_diff,"Paired Image Difference",experiment,cmap="bwr_r",lims=[-vmax,vmax])


				experiment.log_metric("Generator Loss",gen_loss)
				experiment.log_metric("Discriminator Loss",disc_loss)
			if cur_step % save_steps == 0 and cur_step > 0:
				torch.save(pix2pix.gen, f'gen_pix2pix_{model_name}_checkpoint_{cur_step}.pt')
				torch.save(pix2pix.patch_gan, f'patchgan_pix2pix_{model_name}_checkpoint_{cur_step}.pt')


			cur_step+=1

if __name__=="__main__":
    main()
