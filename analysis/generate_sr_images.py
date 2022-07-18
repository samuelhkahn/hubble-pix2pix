import sys
sys.path.append("..")
import configparser
import os
import numpy as np
from astropy.io import fits
from comet_ml import Experiment
import torch
from torchvision import models
from dataset import SR_HST_HSC_Dataset
import torch.nn as nn
from torchvision.transforms import CenterCrop
from collate_fn import collate_fn



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'



    # Load file paths from config
    config_file = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)


    # Adding Comet Logging
    api_key = os.environ['COMET_ML_ASTRO_API_KEY']

    experiment = Experiment(
        api_key=api_key,
        project_name="Pix2Pix Image Translation: HSC->HST",
        workspace="samkahn-astro",
    )
    pretrained_generator = config["PRETRAINED_GENERATOR"]["pretrained_generator"]

    print(f"Loading Pretrained Generator: {pretrained_generator}")
    pretrained_generator = os.path.join(os.getcwd(),pretrained_generator)
    gen = torch.load(pretrained_generator,map_location=torch.device(device))

    # Create Dataloader

    hst_path_val = config["DEFAULT"]["hst_path_val"]
    hsc_path_val = config["DEFAULT"]["hsc_path_val"]


    hst_dim = int(config["HST_DIM"]["hst_dim"])
    hsc_dim = int(config["HSC_DIM"]["hsc_dim"])
    batch_size=1

    sr_hst_hsc_dataset = SR_HST_HSC_Dataset(hst_path = hst_path_val , hsc_path = hsc_path_val, hr_size=[hst_dim, hst_dim], 
    lr_size=[hsc_dim, hsc_dim], transform_type = "ds9_scale",data_aug = False,experiment=experiment)

    dataloader = torch.utils.data.DataLoader(sr_hst_hsc_dataset, 
    batch_size=batch_size, pin_memory=True, shuffle=True, collate_fn = collate_fn)


    index = 0
    # while cur_step < total_steps:
    for hr_real,lr,hsc_hr, seg_map_real in dataloader: # real
        hr_real = sr_hst_hsc_dataset.ds9_unscaling(CenterCrop(600)(hr_real.squeeze(0).squeeze(0)),offset=1)

        lr = lr.unsqueeze(1)

        sr = gen(lr,True)
        sr = sr_hst_hsc_dataset.ds9_unscaling(sr.squeeze(0).squeeze(0),offset=1)
        sr = CenterCrop(600)(sr).detach().numpy().copy(order='C')

        lr = sr_hst_hsc_dataset.ds9_unscaling(CenterCrop(100)(lr.squeeze(0).squeeze(0)),offset=1)

        fits.PrimaryHDU(data=sr).writeto(f"results/sr{index}.fits")
        fits.PrimaryHDU(data=hr_real).writeto(f"results/hst{index}.fits")
        fits.PrimaryHDU(data=lr).writeto(f"results/hsc{index}.fits")
        index+=1
if __name__=="__main__":
    main()