import sys
import os
import numpy as np
from astropy.io import fits
import os
from comet_ml import Experiment
import torch
from torchvision import models
from .. dataset import SR_HST_HSC_Dataset
import torch.nn as nn
from torchvision.transforms import CenterCrop
from collate_fn import collate_fn


#unscaling
def ds9_unscaling(x, a=1000,offset = 0 ): 
    return (((a + 1)**(x+offset) - 1) / a)
api_key = os.environ['COMET_ML_ASTRO_API_KEY']


experiment = Experiment(
    api_key=api_key,
    project_name="Pix2Pix Image Translation: HSC->HST",
    workspace="samkahn-astro",
)
pretrained_generator = "gen_pix2pixsr_subpixel_checkerboard_free_global_lr=0.0002_recon=200_segrecon=0_vgg=1_scatter=0.0_adv=1.0_discupdate=1_vgglayer_weieghts_[0.0, 1.0, 1.0, 0.0, 0.0]_checkpoint_425000.pt"                               

print(f"Loading Pretrained Generator: {pretrained_generator}")
pretrained_generator = os.path.join(os.getcwd(),pretrained_generator)
gen = torch.load(pretrained_generator,map_location=torch.device('cpu'))

#### total_steps = 2
# sep.set_extract_pixstack(768*768)
# sep.set_sub_object_limit(768*768)
# Create Dataloader
hsc_path = "../data/samples/hsc/"
hst_path = "../data/samples/hst/"


hst_dim = 600
hsc_dim = 100
batch_size=1


dataloader = torch.utils.data.DataLoader(
SR_HST_HSC_Dataset(hst_path = hst_path , hsc_path = hsc_path, hr_size=[hst_dim, hst_dim], 
lr_size=[hsc_dim, hsc_dim], transform_type = "ds9_scale",data_aug = False,experiment=experiment), 
batch_size=batch_size, pin_memory=True, shuffle=True, collate_fn = collate_fn)


index = 0
# while cur_step < total_steps:
for hr_real,lr,hsc_hr, seg_map_real in dataloader: # real
    hr_real = ds9_unscaling(CenterCrop(600)(hr_real.squeeze(0).squeeze(0)),offset=1)
    lr = lr.unsqueeze(1)
    sr = gen(lr,True)
    sr = ds9_unscaling(sr.squeeze(0).squeeze(0),offset=1)
    sr = CenterCrop(600)(sr).detach().numpy().copy(order='C')
    lr = ds9_unscaling(CenterCrop(100)(lr.squeeze(0).squeeze(0)),offset=1)
    fits.PrimaryHDU(data=sr).writeto(f"sr-comparison-analysis/results/sr{index}.fits")
    fits.PrimaryHDU(data=hr_real).writeto(f"sr-comparison-analysis/results/hst{index}.fits")
    fits.PrimaryHDU(data=lr).writeto(f"sr-comparison-analysis/results/hsc{index}.fits")
    index+=1
