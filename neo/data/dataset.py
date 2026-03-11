"""Dataset for paired HST (high-resolution) and HSC (low-resolution) FITS images.

Loads astronomical FITS image pairs, applies configurable intensity scaling
transformations, data augmentation, and source segmentation map extraction
using SEP (Source Extractor as a Python library).
"""

import os
import random

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian
from sklearn.preprocessing import minmax_scale
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode as IMode

import sep


class SquarePad:
    """Pad an image to a square with reflective padding.

    Args:
        padding: Number of pixels to pad on each side.
        padding_mode: Padding mode (e.g., "reflect", "constant").
    """

    def __init__(self, padding, padding_mode):
        self.padding = padding
        self.padding_mode = padding_mode

    def __call__(self, image):
        return TF.pad(image, padding=self.padding, padding_mode=self.padding_mode)


class Decimate:
    """Downsample an image by an integer factor with optional Gaussian blur.

    Args:
        factor: Decimation factor (default: 6 for HST->HSC resolution ratio).
        blur: Whether to apply Gaussian blur before decimation.
        sigma: Standard deviation of Gaussian blur kernel.
    """

    def __init__(self, factor=6, blur=True, sigma=1):
        self.factor = factor
        self.blur = blur
        self.sigma = sigma

    def __call__(self, image):
        if self.blur:
            image = gaussian_filter(image[..., :, :], sigma=self.sigma)
        return image[..., ::self.factor, ::self.factor]


class OpenCVResize:
    """Resize an image using OpenCV interpolation methods.

    Args:
        dim: Target spatial dimension (square output).
        method: OpenCV interpolation flag (e.g., cv2.INTER_LINEAR).
    """

    def __init__(self, dim, method):
        self.dim = dim
        self.method = method

    def __call__(self, image):
        return cv.resize(image, (self.dim, self.dim), interpolation=self.method)


class SR_HST_HSC_Dataset(Dataset):
    """Paired HST/HSC dataset for super-resolution training.

    Loads aligned FITS image pairs from HST (Hubble Space Telescope, high-resolution)
    and HSC (Hyper Suprime-Cam, low-resolution) directories. Applies intensity
    transformations, optional data augmentation (flips), and extracts source
    segmentation maps using SEP for weighted loss computation.

    The resolution ratio between HST and HSC is 6:1.

    Args:
        hst_path: Directory containing HST FITS cutouts.
        hsc_path: Directory containing HSC FITS cutouts (filenames must match HST).
        hr_size: [height, width] of high-resolution images (e.g., [768, 768]).
        lr_size: [height, width] of low-resolution images (e.g., [128, 128]).
        transform_type: Intensity scaling method. One of:
            - "sigmoid": Sigmoid normalization
            - "log_scale": Log10 transformation
            - "median_scale": Local median normalization
            - "sigmoid_rms": Sigmoid scaled by RMS
            - "global_median_scale": Global median normalization
            - "clip_min_max_norm": Clipped min-max normalization
            - "ds9_scale": DS9-style logarithmic scaling (recommended)
            - "hst_downscale": Downscale HST as synthetic LR
            - "paired_image_translation": DS9 scaling for both HST and HSC
        data_aug: Whether to apply random flip augmentations.
        experiment: Comet ML Experiment instance for logging.
    """

    def __init__(self, hst_path, hsc_path, hr_size, lr_size, transform_type,
                 data_aug, experiment):
        super().__init__()

        if hr_size is not None and lr_size is not None:
            assert hr_size[0] == 6 * lr_size[0]
            assert hr_size[1] == 6 * lr_size[1]

        self.hst_path = hst_path
        self.hsc_path = hsc_path
        self.transform_type = transform_type
        self.data_aug = data_aug
        self.filenames = os.listdir(hst_path)
        self.experiment = experiment

        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

        self.hr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(600, interpolation=IMode.NEAREST),
            transforms.ToTensor(),
        ])

        self.square_pad_hr = SquarePad(84, "reflect")
        self.square_pad_lr = SquarePad(14, "reflect")

        self.pad_array_hr = transforms.Compose([
            transforms.ToPILImage(),
            self.square_pad_hr,
        ])
        self.pad_array_lr = transforms.Compose([
            transforms.ToPILImage(),
            self.square_pad_lr,
        ])

    def load_fits(self, file_path):
        """Load a FITS file and return its primary data as a float32 array."""
        cutout = fits.open(file_path)
        array = cutout[0].data
        return array.astype(np.float32)

    @staticmethod
    def sigmoid_array(x):
        """Apply element-wise sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_transformation(self, x):
        """Sigmoid normalization shifted to [-1, 1] range."""
        return 2 * (self.sigmoid_array(x) - 0.5)

    def sigmoid_rms_transformation(self, x, std_scale):
        """Sigmoid normalization after RMS-based scaling."""
        x = self.scale_tensor(x, std_scale, "div")
        return self.sigmoid_array(x)

    @staticmethod
    def scale_tensor(tensor, scale, scale_type):
        """Multiply or divide tensor by a scale factor."""
        if scale_type == "prod":
            return scale * tensor
        elif scale_type == "div":
            return scale / tensor

    @staticmethod
    def log_transformation(tensor, min_pix, eps):
        """Log10 transformation with offset to handle negative values."""
        return np.log10(tensor + np.abs(min_pix) + eps)

    @staticmethod
    def median_transformation(tensor):
        """Normalize by subtracting median and dividing by standard deviation."""
        y = tensor - np.median(tensor)
        return y / np.std(y)

    @staticmethod
    def global_median_transformation(tensor, median, std):
        """Normalize using global (dataset-level) median and std."""
        return (tensor - median) / std

    @staticmethod
    def min_max_normalization(tensor, min_val, max_val):
        """Clip and normalize to [0, 1] range."""
        tensor = np.clip(tensor, min_val, max_val)
        return (tensor - min_val) / (max_val - min_val)

    @staticmethod
    def invert_min_max_normalization(tensor, min_val, max_val):
        """Invert min-max normalization back to original scale."""
        return tensor * (max_val - min_val) + min_val

    def get_segmentation_map(self, pixels):
        """Extract a binary source segmentation map using SEP.

        Args:
            pixels: 2D numpy array of pixel values.

        Returns:
            Binary mask where detected sources are 1, background is 0.
        """
        bkg = sep.Background(pixels)
        mask = sep.extract(pixels, 3, err=bkg.globalrms, segmentation_map=True)[1]
        mask[mask > 0] = 1
        return mask

    @staticmethod
    def create_hr_lr_pair(tensor, alpha):
        """Create a high/low frequency decomposition of an image."""
        img_low = minmax_scale(
            gaussian(tensor, sigma=5).flatten(),
            feature_range=(tensor.min(), tensor.max()),
        ).reshape(tensor.shape)
        img_high = tensor - img_low
        alpha = np.max(np.abs(img_high))
        img_high = img_high / alpha
        return img_low, img_high

    @staticmethod
    def ds9_scaling(x, a=1000, offset=0):
        """DS9-style logarithmic intensity scaling.

        Maps pixel values using: log10(a*x + 1) / log10(a + 1) - offset

        This is the standard scaling used in the DS9 astronomical image viewer,
        which compresses the dynamic range while preserving faint structure.
        """
        return (np.log10(a * x + 1) / np.log10(a + 1)) - offset

    @staticmethod
    def ds9_unscaling(x, a=1000, offset=0):
        """Invert DS9 logarithmic scaling."""
        return (((a + 1) ** (x + offset) - 1) / a)

    @staticmethod
    def clip(arr, use_data=True):
        """Clip array to remove extreme outliers.

        Clips to [mean - 3*std, 99.999th percentile]. If use_data is False,
        the lower bound is set to 0.
        """
        min_offset = max(arr.mean() - arr.std() * 3, arr.min()) * use_data
        clipped_array = np.clip(arr, min_offset, np.percentile(arr, 99.999))
        return clipped_array, min_offset

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """Load and preprocess an HST/HSC image pair.

        Returns:
            Tuple of (hst, hsc, hsc_hr, hst_seg_map) tensors, or
            (None, None, None, None) for corrupted files (handled by collate_fn).
        """
        hst_image = os.path.join(self.hst_path, self.filenames[idx])
        hsc_image = os.path.join(self.hsc_path, self.filenames[idx])

        try:
            hst_array = self.load_fits(hst_image)
            hsc_array = self.load_fits(hsc_image)
        except TypeError:
            return (None, None, None, None)

        hst_array = self.to_pil(hst_array)
        hsc_array = self.to_pil(hsc_array)

        # Data augmentation (random flips)
        if self.data_aug:
            if random.random() > 0.5:
                hsc_array = TF.vflip(hsc_array)
                hst_array = TF.vflip(hst_array)
            if random.random() > 0.5:
                hsc_array = TF.hflip(hsc_array)
                hst_array = TF.hflip(hst_array)

        # Center crop to standard sizes
        hsc_array = TF.center_crop(hsc_array, [100, 100])
        hst_array = TF.center_crop(hst_array, [600, 600])

        hsc_array = np.array(hsc_array)
        hst_array = np.array(hst_array)

        try:
            hst_seg_map = self.get_segmentation_map(hst_array)
        except Exception:
            return (None, None, None, None)

        # Apply intensity transformation
        if self.transform_type == "sigmoid":
            hst_transformation = self.sigmoid_transformation(hst_array)
            hsc_transformation = self.sigmoid_transformation(hsc_array)
        elif self.transform_type == "log_scale":
            hst_transformation = self.log_transformation(hst_array, self.hst_min, 1e-6)
            hsc_transformation = self.log_transformation(hsc_array, self.hst_min, 1e-6)
        elif self.transform_type == "median_scale":
            hst_transformation = self.median_transformation(hst_array)
            hsc_transformation = self.median_transformation(hsc_array)
        elif self.transform_type == "sigmoid_rms":
            hst_transformation = self.sigmoid_rms_transformation(hst_array, self.hst_std)
            hsc_transformation = self.sigmoid_rms_transformation(hsc_array, self.hsc_std)
        elif self.transform_type == "global_median_scale":
            hst_transformation = self.global_median_transformation(hst_array, self.hst_median, self.hst_std)
            hsc_transformation = self.global_median_transformation(hsc_array, self.hsc_median, self.hsc_std)
        elif self.transform_type == "clip_min_max_norm":
            hst_transformation = self.min_max_normalization(hst_array, self.hst_min, self.hst_max)
            hsc_transformation = self.min_max_normalization(hsc_array, self.hsc_min, self.hsc_max)
        elif self.transform_type == "hst_downscale":
            hst_clipped = self.clip(hst_array, use_data=False)[0]
            hst_transformation = self.ds9_scaling(hst_clipped, offset=1)
            hsc_transformation = self.lr_transforms(hst_transformation)
        elif self.transform_type == "paired_image_translation":
            hst_clipped = self.clip(hst_array, use_data=False)[0]
            hst_transformation = self.ds9_scaling(hst_clipped, offset=1)
            hsc_hr_clipped = self.clip(hsc_array, use_data=False)[0]
            hsc_transformation = self.ds9_scaling(hsc_hr_clipped, offset=1)
        elif self.transform_type == "ds9_scale":
            hst_clipped = self.clip(hst_array, use_data=False)[0]
            hst_transformation = self.ds9_scaling(hst_clipped, offset=1)
            hsc_clipped = self.clip(hsc_array, use_data=False)[0]
            hsc_transformation = self.ds9_scaling(hsc_clipped, offset=1)
            hsc_hr = self.hr_transforms(hsc_transformation)

        # Pad and convert to tensors
        hst_seg_map = self.to_tensor(self.pad_array_hr(hst_seg_map)).squeeze(0)
        hsc = self.to_tensor(self.pad_array_lr(hsc_transformation)).squeeze(0)
        hst = self.to_tensor(self.pad_array_hr(hst_transformation)).squeeze(0)
        hsc_hr = self.to_tensor(self.pad_array_hr(hsc_hr)).squeeze(0)

        return hst, hsc, hsc_hr, hst_seg_map
