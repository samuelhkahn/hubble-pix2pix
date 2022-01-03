from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as transforms
from astropy.io import fits
from torch.utils.data import Dataset, DataLoader
import os
import torch
import random
import torchvision.transforms.functional as TF
import sep
from torchvision.transforms import CenterCrop
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from skimage.filters import gaussian
from sklearn.preprocessing import minmax_scale
from torchvision.transforms.functional import InterpolationMode as IMode
class SquarePad:
    def __call__(self, image):

        return TF.pad(image, 14, 0, 'constant')


class SR_HST_HSC_Dataset(Dataset):
    '''
    Dataset Class
    Values:
        hr_size: spatial size of high-resolution image, a list/tuple
        lr_size: spatial size of low-resolution image, a list/tuple
        *args/**kwargs: all other arguments for subclassed torchvision dataset
    '''

    def __init__(self, hst_path: str, hsc_path:str, hr_size: list, lr_size: list, transform_type: str, data_aug: bool ) -> None:
        super().__init__()

        # sep.set_extract_pixstack(1000000)

        if hr_size is not None and lr_size is not None:
            assert hr_size[0] == 6 * lr_size[0]
            assert hr_size[1] == 6 * lr_size[1]


        self.hsc_std = 0.04180176637927356
        self.hst_std = 0.0010912614529011736

        self.median_scale = 0.32154497558051215
        self.mean_scale = 0.31601302214882165

        self.hst_median = 2.696401406865334e-05
        self.hsc_median = 1.4194287359714508e-05

        self.hst_path = hst_path
        self.hsc_path = hsc_path

        self.transform_type = transform_type
        self.data_aug = data_aug

        self.filenames = os.listdir(hst_path)


        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

        # HST clip range - (0,99.996)
        self.hst_min,self.hst_max = (-4.656636714935303, 0.11904790546745403)

        # HSC clip range - (0,99.9)
        self.hsc_min,self.hsc_max = (-0.4692089855670929, 12.432257434845326)

        self.lr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(100, interpolation=IMode.BICUBIC)
        ])

        # now use it as the replacement of transforms.Pad class
        self.pad_array=transforms.Compose([
            transforms.ToPILImage(),
            SquarePad(),
            transforms.Resize(128)
        ])
        self.pad_pil=transforms.Compose([
            SquarePad(),
            transforms.Resize(128)
        ])
        
    def load_fits(self, file_path: str) -> np.ndarray:
        cutout = fits.open(file_path)
        array = cutout[0].data
        array = array.astype(np.float32) # Big->little endian
        return array

    def sigmoid_array(self,x:np.ndarray) -> np.ndarray:                                        
        return 1 / (1 + np.exp(-x))

    def sigmoid_transformation(self,x:np.ndarray) -> np.ndarray:
        x = self.sigmoid_array(x) #shift to make noise more apparent
        x = 2*(x-0.5)
        return x
    def sigmoid_rms_transformation(self,x:np.ndarray,std_scale:float) -> np.ndarray:
        x = self.scale_tensor(x,std_scale,"div")
        x = self.sigmoid_array(x)
        return x

    def scale_tensor(self,tensor:np.ndarray, scale:float,scale_type: str) -> np.ndarray:
        if scale_type == "prod":
            return scale*tensor
        elif scale_type == "div":
            return scale/tensor

    def log_transformation(self,tensor:np.ndarray,min_pix:float,eps:float) -> np.ndarray:
        transformed = tensor+np.abs(min_pix)+eps
        transformed = np.log10(transformed)
        return transformed

    # Local (image level) median tansformation
    def median_transformation(self,tensor:np.ndarray) -> np.ndarray:
        y = tensor - np.median(tensor)
        y_std = np.std(y)
        normalized = y/y_std
        return normalized

    # Global Median Transformation
    def global_median_transformation(self,tensor:np.ndarray,median: float, std:float) -> np.ndarray:
        y = tensor - median
        normalized = y/std
        return normalized

    # Min max normalization with clipping
    @staticmethod
    def min_max_normalization(tensor:np.ndarray, min_val:float, max_val:float) -> np.ndarray:
        tensor =  np.clip(tensor, min_val, max_val)
        numerator = tensor-min_val
        denominator = max_val-min_val
        tensor = numerator/denominator
        return tensor

    @staticmethod
    def invert_min_max_normalization(tensor:np.ndarray, min_val:float, max_val:float) -> np.ndarray:
        denominator = max_val-min_val
        unnormalized=tensor*denominator+min_val
        return unnormalized

    # segmentation map
    def get_segmentation_map(self,pixels:np.ndarray) -> np.ndarray:
            # pixels = pixels.byteswap().newbyteorder()
            bkg = sep.Background(pixels)
            mask = sep.extract(pixels, 3, 
                                err=bkg.globalrms,
                                segmentation_map=True)[1]
            mask[mask>0]=1
            return  mask

    @staticmethod
    def create_hr_lr_pair(tensor:np.ndarray,alpha:float) -> np.ndarray:
        img_low = minmax_scale(gaussian(tensor, sigma=5).flatten(), 
            feature_range=(tensor.min(), tensor.max()),).reshape(tensor.shape)
        img_high = tensor - img_low
        alpha = np.max(np.abs(img_high)) 
        img_high = img_high / alpha
        return img_low,img_high

    @staticmethod
    def ds9_scaling(x, a=1000,offset = 0):
        return (np.log10(a*x + 1)/np.log10(a + 1)) - offset

    @staticmethod
    def ds9_unscaling(x, a=1000,offset = 0 ):
        return (((a + 1)**x - 1) / a) + offset

    @staticmethod
    def clip(arr, use_data=True):
        min_offset = max(arr.mean() - arr.std()*3, arr.min()) * use_data
    
        clipped_array = np.clip(
            arr, 
            min_offset, 
            np.percentile(arr, 99.999)
        ) 
    
        return clipped_array, min_offset



    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> tuple:


        hst_image = os.path.join(self.hst_path,self.filenames[idx])
        hsc_image = os.path.join(self.hsc_path,self.filenames[idx])
        
        hst_array = self.load_fits(hst_image)
        hsc_array = self.load_fits(hsc_image)

        hst_array = self.to_pil(hst_array)
        hsc_array = self.to_pil(hsc_array)

        # hsc_array = self.to_tensor(hst_array)
        # hst_array = self.to_tensor(hsc_array)

        if self.data_aug == True:
            if random.random() > 0.5:

                # Rotate 
                rotation = random.randint(0,359)
                # 2 = BiLinear
                hsc_array = TF.rotate(hsc_array,rotation,
                                    interpolation = TF.InterpolationMode.BILINEAR)
                hst_array = TF.rotate(hst_array,rotation,
                                    interpolation = TF.InterpolationMode.BILINEAR)

            #Center Crop 
            hsc_array = TF.center_crop(hsc_array,[100,100])
            hst_array = TF.center_crop(hst_array,[600,600])

        ## Flip Augmentations
            if random.random() > 0.5:
                hsc_array  = TF.vflip(hsc_array)
                hst_array  = TF.vflip(hst_array)
                
            if random.random() >0.5:
                hsc_array  = TF.hflip(hsc_array)
                hst_array  = TF.hflip(hst_array)
        else:
             #Center Crop 
            hsc_array = TF.center_crop(hsc_array,[100,100])
            hst_array = TF.center_crop(hst_array,[600,600])

        hsc_array = np.array(hsc_array)
        hst_array = np.array(hst_array)
        hst_seg_map = self.get_segmentation_map(hst_array)
        

        # Sigmoid Scaling
        if self.transform_type == "sigmoid":
            hst_transformation = self.sigmoid_transformation(hst_array)
            hsc_transformation = self.sigmoid_transformation(hsc_array)
        # Log scaling
        elif self.transform_type == "log_scale":
            hst_transformation = self.log_transformation(hst_array,self.hst_min,1e-6)
            hsc_transformation = self.log_transformation(hsc_array,self.hst_min,1e-6)
        # Median Scaling
        elif self.transform_type == "median_scale":
            hst_transformation = self.median_transformation(hst_array)
            hsc_transformation = self.median_transformation(hsc_array)
        # Sigmoid + RMS scale  
        elif self.transform_type == "sigmoid_rms":
            hst_transformation = self.sigmoid_rms_transformation(hst_array,self.hst_std)
            hsc_transformation = self.sigmoid_rms_transformation(hsc_array,self.hsc_std)

        # Scaled based of median global value
        elif self.transform_type == "global_median_scale":
            hst_transformation = self.global_median_transformation(hst_array,self.hst_median,self.hst_std)
            hsc_transformation = self.global_median_transformation(hsc_array,self.hsc_median,self.hsc_std)
        # min
        elif self.transform_type == "clip_min_max_norm":
            hst_transformation = self.min_max_normalization(hst_array,self.hst_min,self.hst_max)
            hsc_transformation = self.min_max_normalization(hsc_array,self.hsc_min,self.hsc_max)

        elif self.transform_type == "ds9_scale":
            hst_clipped = self.clip(hst_array,use_data=False)[0]
            hst_transformation = self.ds9_scaling(hst_clipped,offset = 1)

            hsc_clipped = self.clip(hsc_array,use_data=False)[0]
            hsc_transformation = self.ds9_scaling(hsc_clipped,offset = 1)
        elif self.transform_type == "hst_downscale":
            hst_clipped = self.clip(hst_array,use_data=False)[0]
            hst_transformation = self.ds9_scaling(hst_clipped,offset = 1)
            hsc_transformation =self.lr_transforms(hst_transformation)
        elif self.transform_type == "paired_image_translation":
            hst_clipped = self.clip(hst_array,use_data=False)[0]
            hst_transformation = self.ds9_scaling(hst_clipped,offset = 1)
            hst_lr_transformation =self.lr_transforms(hst_transformation)
            hst_down_seg_map = self.lr_transforms(hst_seg_map)
            
            hsc_clipped = self.clip(hsc_array,use_data=False)[0]
            hsc_transformation = self.ds9_scaling(hsc_clipped,offset = 1)



        # Add Segmap to second channel to ensure proper augmentations
        # hst_seg_stack = np.dstack((hst_transformation,hst_seg_map))
        # hst_seg_stack = self.to_tensor(hst_seg_stack)        


        # Collapse First Dimension and extract hst/seg_map
        
        hst_down_seg_map = self.to_tensor(self.pad_pil(hst_down_seg_map)).squeeze(0)
        hsc_tensor = self.to_tensor(self.pad_array(hsc_transformation)).squeeze(0)
        hst_tensor = self.to_tensor(hst_transformation).squeeze(0)
        hst_lr_tensor = self.to_tensor(self.pad_pil(hst_lr_transformation)).squeeze(0)

        return  hst_tensor,hst_lr_tensor,hsc_tensor,hst_down_seg_map