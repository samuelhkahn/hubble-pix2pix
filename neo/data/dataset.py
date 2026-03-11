"""Dataset for paired HST (high-resolution) and HSC (low-resolution) FITS images.

Loads astronomical FITS image pairs, applies DS9-style logarithmic intensity
scaling, optional data augmentation (random flips), and extracts source
segmentation maps using SEP (Source Extractor as a Python library).

The segmentation maps are used as binary masks for the segmentation-weighted
L1 loss, which emphasizes reconstruction quality on detected astronomical
sources over empty background pixels.

The HST/HSC resolution ratio is 6:1 (e.g., 768x768 HST vs 128x128 HSC).
"""

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from astropy.io import fits
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode as IMode

import sep


# ---------------------------------------------------------------------------
# Helper transform classes
# ---------------------------------------------------------------------------


class SquarePad:
    """Pad an image to a square using reflective boundary conditions.

    Reflective padding avoids introducing artificial edges at image borders,
    which is important for astronomical images where edge effects can
    produce artifacts in the generator output.

    Args:
        padding: Number of pixels to pad on each side.
        padding_mode: Padding mode (e.g., ``"reflect"``, ``"constant"``).
    """

    def __init__(self, padding: int, padding_mode: str) -> None:
        self.padding = padding
        self.padding_mode = padding_mode

    def __call__(self, image):
        return TF.pad(image, padding=self.padding, padding_mode=self.padding_mode)


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------


class SR_HST_HSC_Dataset(Dataset):
    """Paired HST/HSC dataset for super-resolution training.

    Loads aligned FITS image pairs from HST (Hubble Space Telescope,
    high-resolution target) and HSC (Hyper Suprime-Cam, low-resolution input)
    directories. Each pair undergoes:

        1. FITS loading -> float32 numpy arrays
        2. Optional random horizontal/vertical flips (data augmentation)
        3. Center crop to 600x600 (HST) and 100x100 (HSC)
        4. Source detection via SEP -> binary segmentation map
        5. DS9 logarithmic intensity scaling (compresses dynamic range)
        6. Bilinear upsampling of HSC to 600x600 (``hsc_hr``, used as
           discriminator conditioning input)
        7. Reflective padding to 768x768 (HST) and 128x128 (HSC)

    Args:
        hst_path: Directory containing HST FITS cutouts.
        hsc_path: Directory containing corresponding HSC FITS cutouts.
            Filenames must match between directories.
        hr_size: ``[height, width]`` of high-resolution images after padding
            (e.g., ``[768, 768]``).
        lr_size: ``[height, width]`` of low-resolution images after padding
            (e.g., ``[128, 128]``). Must satisfy ``hr_size == 6 * lr_size``.
        data_aug: Whether to apply random flip augmentations.
        experiment: Comet ML ``Experiment`` instance for logging.
    """

    def __init__(
        self,
        hst_path: str,
        hsc_path: str,
        hr_size: List[int],
        lr_size: List[int],
        data_aug: bool,
        experiment,
    ) -> None:
        super().__init__()

        # Enforce the 6:1 resolution ratio between HST and HSC.
        if hr_size is not None and lr_size is not None:
            assert hr_size[0] == 6 * lr_size[0], (
                f"HR height {hr_size[0]} must be 6x LR height {lr_size[0]}"
            )
            assert hr_size[1] == 6 * lr_size[1], (
                f"HR width {hr_size[1]} must be 6x LR width {lr_size[1]}"
            )

        self.hst_path = hst_path
        self.hsc_path = hsc_path
        self.data_aug = data_aug
        self.filenames = os.listdir(hst_path)
        self.experiment = experiment

        # Conversion utilities between numpy/PIL/tensor.
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

        # Bilinear upsampling of HSC from 100x100 -> 600x600 (to match the
        # cropped HST size). This upsampled HSC is concatenated with the
        # generated/real image as conditioning input to the PatchGAN
        # discriminator.
        self.hr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(600, interpolation=IMode.NEAREST),
            transforms.ToTensor(),
        ])

        # Reflective padding: 600x600 -> 768x768 (HST) and 100x100 -> 128x128
        # (HSC). The padding adds (768-600)/2 = 84 px on each side for HR and
        # (128-100)/2 = 14 px on each side for LR.
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

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load_fits(self, file_path: str) -> np.ndarray:
        """Load a FITS file and return its primary HDU data as float32.

        Args:
            file_path: Path to a ``.fits`` file.

        Returns:
            2D numpy array of pixel values in float32.
        """
        cutout = fits.open(file_path)
        array = cutout[0].data
        return array.astype(np.float32)

    # ------------------------------------------------------------------
    # Intensity transforms
    # ------------------------------------------------------------------

    @staticmethod
    def ds9_scaling(x: np.ndarray, a: float = 1000, offset: float = 0) -> np.ndarray:
        """DS9-style logarithmic intensity scaling.

        Applies the transformation used by the SAOImageDS9 astronomical image
        viewer to compress the large dynamic range of astronomical images while
        preserving both bright sources and faint extended emission::

            scaled = log10(a * x + 1) / log10(a + 1) - offset

        An ``offset=1`` shifts the output range so that the background (where
        ``x ~ 0``) maps to approximately ``-1``, matching the tanh output range
        of the generator.

        Args:
            x: Input pixel values (should be non-negative after clipping).
            a: Scaling parameter controlling compression strength.
                Higher values compress the bright end more aggressively.
            offset: Constant subtracted from the result.

        Returns:
            Scaled pixel values.
        """
        return (np.log10(a * x + 1) / np.log10(a + 1)) - offset

    @staticmethod
    def clip(arr: np.ndarray) -> Tuple[np.ndarray, float]:
        """Clip array to remove extreme outliers before intensity scaling.

        Sets the lower bound to 0 (since astronomical pixel values below 0 are
        noise) and the upper bound to the 99.999th percentile (to suppress
        cosmic rays and hot pixels).

        Args:
            arr: 2D pixel array.

        Returns:
            Tuple of (clipped_array, lower_clip_value).
        """
        clipped_array = np.clip(arr, 0, np.percentile(arr, 99.999))
        return clipped_array, 0

    # ------------------------------------------------------------------
    # Source detection
    # ------------------------------------------------------------------

    def get_segmentation_map(self, pixels: np.ndarray) -> np.ndarray:
        """Extract a binary source segmentation map using SEP.

        Runs Source Extractor (Python implementation) on the pixel array to
        detect astronomical sources at 3-sigma above the global background
        RMS. The resulting segmentation map is binarized so that all detected
        source pixels are 1 and background pixels are 0.

        This mask is used by the segmentation-weighted L1 loss to focus
        reconstruction quality on scientifically relevant source regions.

        Args:
            pixels: 2D numpy array of pixel values (float32, C-contiguous).

        Returns:
            Binary mask of shape ``(H, W)`` with 1 at source pixels.
        """
        bkg = sep.Background(pixels)
        mask = sep.extract(pixels, 3, err=bkg.globalrms, segmentation_map=True)[1]
        mask[mask > 0] = 1
        return mask

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Load and preprocess one HST/HSC image pair.

        Processing pipeline:
            1. Load paired FITS files from HST and HSC directories.
            2. Convert to PIL images for torchvision transforms.
            3. Apply random flips (if ``data_aug`` is enabled).
            4. Center crop to 600x600 (HST) and 100x100 (HSC).
            5. Run SEP source detection on the HST image to get a binary
               segmentation map.
            6. Clip outliers and apply DS9 logarithmic scaling to both images.
            7. Upsample HSC to 600x600 via bilinear interpolation (``hsc_hr``).
            8. Pad all images with reflective padding to final sizes.

        Returns:
            Tuple of ``(hst, hsc, hsc_hr, hst_seg_map)`` tensors:
                - ``hst``: HST target image, shape ``(768, 768)``.
                - ``hsc``: HSC input image, shape ``(128, 128)``.
                - ``hsc_hr``: Upsampled HSC for discriminator conditioning,
                  shape ``(768, 768)``.
                - ``hst_seg_map``: Binary source mask, shape ``(768, 768)``.

            Returns ``(None, None, None, None)`` for corrupted or unreadable
            files; the custom ``collate_fn`` filters these out.
        """
        hst_image = os.path.join(self.hst_path, self.filenames[idx])
        hsc_image = os.path.join(self.hsc_path, self.filenames[idx])

        # ------------------------------------------------------------------
        # Step 1: Load FITS files. Return None tuple on corrupted files.
        # ------------------------------------------------------------------
        try:
            hst_array = self.load_fits(hst_image)
            hsc_array = self.load_fits(hsc_image)
        except TypeError:
            return (None, None, None, None)

        # Convert to PIL for torchvision transform compatibility.
        hst_array = self.to_pil(hst_array)
        hsc_array = self.to_pil(hsc_array)

        # ------------------------------------------------------------------
        # Step 2: Data augmentation (random flips applied identically to both
        # images to preserve spatial alignment).
        # ------------------------------------------------------------------
        if self.data_aug:
            if random.random() > 0.5:
                hsc_array = TF.vflip(hsc_array)
                hst_array = TF.vflip(hst_array)
            if random.random() > 0.5:
                hsc_array = TF.hflip(hsc_array)
                hst_array = TF.hflip(hst_array)

        # ------------------------------------------------------------------
        # Step 3: Center crop to standard sizes (removes border effects from
        # the original FITS cutouts).
        # ------------------------------------------------------------------
        hsc_array = TF.center_crop(hsc_array, [100, 100])
        hst_array = TF.center_crop(hst_array, [600, 600])

        # Convert back to numpy for SEP and intensity scaling.
        hsc_array = np.array(hsc_array)
        hst_array = np.array(hst_array)

        # ------------------------------------------------------------------
        # Step 4: Source detection on the raw HST image (before intensity
        # scaling, since SEP expects linear flux values).
        # ------------------------------------------------------------------
        try:
            hst_seg_map = self.get_segmentation_map(hst_array)
        except Exception:
            return (None, None, None, None)

        # ------------------------------------------------------------------
        # Step 5: DS9 logarithmic intensity scaling. Clip outliers first
        # (cosmic rays, hot pixels), then apply log scaling with offset=1
        # so that the background maps to approximately -1 (matching the
        # generator's tanh output range of [-1, 1]).
        # ------------------------------------------------------------------
        hst_clipped = self.clip(hst_array)[0]
        hst_transformation = self.ds9_scaling(hst_clipped, offset=1)

        hsc_clipped = self.clip(hsc_array)[0]
        hsc_transformation = self.ds9_scaling(hsc_clipped, offset=1)

        # ------------------------------------------------------------------
        # Step 6: Upsample HSC to match HST spatial size (600x600). This
        # upsampled version (hsc_hr) is used as the conditioning input to the
        # PatchGAN discriminator: concat(real_or_fake, hsc_hr).
        # ------------------------------------------------------------------
        hsc_hr = self.hr_transforms(hsc_transformation)

        # ------------------------------------------------------------------
        # Step 7: Pad all images with reflective padding to final dimensions.
        # HST: 600x600 -> 768x768  (84 px each side)
        # HSC: 100x100 -> 128x128  (14 px each side)
        # Squeeze removes the channel dim added by to_tensor (1,H,W) -> (H,W).
        # ------------------------------------------------------------------
        hst_seg_map = self.to_tensor(self.pad_array_hr(hst_seg_map)).squeeze(0)
        hsc = self.to_tensor(self.pad_array_lr(hsc_transformation)).squeeze(0)
        hst = self.to_tensor(self.pad_array_hr(hst_transformation)).squeeze(0)
        hsc_hr = self.to_tensor(self.pad_array_hr(hsc_hr)).squeeze(0)

        return hst, hsc, hsc_hr, hst_seg_map
