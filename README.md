# Neo

**Astronomical image super-resolution using conditional GANs**

Neo enhances ground-based telescope imagery to approach space-based resolution using a Pix2Pix conditional GAN architecture. It was developed to translate images from the Hyper Suprime-Cam (HSC) on the Subaru Telescope to match the resolution of the Hubble Space Telescope (HST), achieving a **6x super-resolution** factor.

---

## Overview

Ground-based telescopes are limited by atmospheric seeing, producing images with lower spatial resolution compared to space-based observatories. Neo bridges this gap by learning the mapping from low-resolution (HSC) to high-resolution (HST) astronomical images using deep learning.

### Key Features

- **6x Super-Resolution**: Upscales 128x128 ground-based images to 768x768 space-quality resolution
- **U-Net Generator**: Encoder-decoder architecture with skip connections for preserving spatial information
- **Sub-Pixel Convolution**: PixelShuffle upsampling (3x then 2x) for artifact-free resolution enhancement
- **PatchGAN Discriminator**: Classifies image patches as real/fake to encourage high-frequency detail
- **Multi-Component Loss Function**: Combines 5 complementary loss terms for physically meaningful outputs

### Loss Function Components

| Loss | Purpose | Description |
|------|---------|-------------|
| **BCE Adversarial** | Realism | PatchGAN discriminator ensures generated images look realistic |
| **L1 Reconstruction** | Pixel accuracy | Direct pixel-wise comparison with ground truth |
| **VGG-19 Perceptual** | Feature similarity | Multi-scale feature matching using pretrained VGG-19 |
| **Wavelet Scattering** | Multi-scale structure | Preserves morphological structure across spatial scales via Kymatio scattering transform |
| **Segmentation-Masked L1** | Source emphasis | Weighted reconstruction that prioritizes detected astronomical sources |

---

## Project Structure

```
neo/
├── __init__.py              # Package metadata and version
├── pix2pix.py               # Pix2Pix GAN: training/validation logic, loss computation
├── log_figure.py            # Comet ML figure logging utilities
├── models/
│   ├── __init__.py           # Model exports
│   ├── generator.py          # U-Net generator with PixelShuffle upsampling
│   ├── patchgan.py           # PatchGAN discriminator
│   ├── down_sample_conv.py   # Strided conv blocks (encoder + discriminator)
│   ├── up_sample_conv.py     # Transpose conv blocks (decoder)
│   ├── gaussian_noise.py     # Gaussian noise injection layer
│   ├── vgg19.py              # VGG-19 feature extractor
│   └── vgg19_loss.py         # VGG perceptual loss module
├── data/
│   ├── __init__.py           # Data exports
│   ├── dataset.py            # HST/HSC paired FITS dataset with transforms
│   └── collate_fn.py         # Batch collation with NaN/Inf filtering
├── analysis/                 # Post-training analysis notebooks and scripts
│   ├── Examine-Batch-Detections.ipynb
│   ├── Examine-Detections.ipynb
│   ├── Mosaic Noise Propertis.ipynb
│   ├── Noise Properties Plots.ipynb
│   ├── compare-sr-fits.py
│   ├── detect_and_mask.py
│   ├── generate_sr_images.py
│   ├── jades_photutils_interface.py
│   ├── perform_batch_comparison.sh
│   ├── perform_comparison.sh
│   └── quicklook.py
└── configs/
    └── example.ini           # Example training configuration

train.py                      # Main training entry point
requirements.txt              # Python dependencies
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, 16GB+ VRAM)
- FITS image data (HST and HSC paired cutouts)

### Setup

```bash
# Clone the repository
git clone git@github.com:samuelhkahn/neo.git
cd neo

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Set your Comet ML API key for experiment tracking:

```bash
export COMET_ML_ASTRO_API_KEY="your-api-key-here"
```

---

## Usage

### Training

1. **Prepare your data**: Organize paired HST/HSC FITS cutouts into train/validation directories. File names must match between HST and HSC directories.

   ```
   data/
   ├── hst_train/       # HST training cutouts (600x600 FITS)
   ├── hsc_train/       # HSC training cutouts (100x100 FITS)
   ├── hst_val/         # HST validation cutouts
   └── hsc_val/         # HSC validation cutouts
   ```

2. **Create a configuration file** (see `neo/configs/example.ini` for a template):

   ```ini
   [DEFAULT]
   hst_path_train = /path/to/hst_train
   hsc_path_train = /path/to/hsc_train
   hst_path_val = /path/to/hst_val
   hsc_path_val = /path/to/hsc_val
   ```

3. **Run training**:

   ```bash
   python train.py neo/configs/example.ini
   ```

4. **Monitor training** on Comet ML. The following metrics are logged:
   - Generator / Discriminator loss (train + validation)
   - VGG perceptual loss
   - L1 reconstruction loss
   - Wavelet scattering loss
   - Segmentation-weighted reconstruction loss
   - Visual comparisons (HSC input, generated SR, ground truth HST, residuals)

### Model Checkpoints

Checkpoints are saved to the `models/` directory at the interval specified by `save_steps` in the config:

- `gen_pix2pixsr_<model_name>_checkpoint_<step>.pt` — Generator
- `patchgan_pix2pixsr_<model_name>_checkpoint_<step>.pt` — Discriminator

### Inference

```python
import torch
from neo.pix2pix import Pix2Pix

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load trained generator
generator = torch.load('models/gen_pix2pixsr_<model_name>_checkpoint_<step>.pt')
generator.to(device)
generator.eval()

# Generate super-resolved image from HSC input
# Input shape: (1, 1, 128, 128), Output shape: (1, 1, 768, 768)
with torch.no_grad():
    sr_image = generator(hsc_input.to(device), identity_map=True)
```

---

## Data Processing Pipeline

### Intensity Scaling

Neo uses **DS9 logarithmic scaling** (the default and recommended transform) to normalize astronomical image intensities:

```
scaled = log10(a * x + 1) / log10(a + 1)
```

where `a = 1000`. This compresses the large dynamic range of astronomical images while preserving both bright sources and faint extended emission.

### Image Preprocessing

1. **Load** FITS cutouts from paired HST/HSC directories
2. **Augment** with random horizontal/vertical flips
3. **Center crop** to 600x600 (HST) and 100x100 (HSC)
4. **Apply** DS9 intensity scaling
5. **Pad** with reflective padding to 768x768 (HST) and 128x128 (HSC)
6. **Extract** source segmentation maps using SEP (Source Extractor Python)

### Collation & Filtering

The custom `collate_fn` filters out:
- Corrupted FITS files (returns `None`)
- Images with NaN or Inf pixel values
- Images with unexpected spatial dimensions

---

## Architecture Details

### Generator (U-Net + PixelShuffle)

```
Input (1, 128, 128)
  │
  ├─ Encoder: 7 DownSampleConv blocks
  │   C64 → C128 → C256 → C512 → C512 → C512 → C512
  │   (each halves spatial dimensions via stride-2 conv)
  │
  ├─ Decoder: 7 UpSampleConv blocks with skip connections
  │   CD512 → CD512 → CD512 → C256 → C128 → C64 → C32
  │   (each doubles spatial dimensions via transpose conv)
  │   (first 3 use 50% dropout for regularization)
  │
  ├─ PixelShuffle: 3x then 2x upsampling (128 → 384 → 768)
  │
  └─ Final: 1x1 Conv → Tanh
Output (1, 768, 768)
```

### Discriminator (PatchGAN)

```
Input: concat(image, condition) → (2, 600, 600)
  │
  ├─ 4 DownSampleConv blocks: C64 → C128 → C256 → C512
  │
  └─ 1x1 Conv → spatial logit map
Output: (1, H', W') patch predictions
```

---

## Configuration Reference

All training hyperparameters are set via an INI config file. See `neo/configs/example.ini` for a complete example.

| Section | Key | Description | Example |
|---------|-----|-------------|---------|
| `DEFAULT` | `hst_path_train` | Path to HST training FITS directory | `/data/hst/train` |
| `DEFAULT` | `hsc_path_train` | Path to HSC training FITS directory | `/data/hsc/train` |
| `DEFAULT` | `hst_path_val` | Path to HST validation FITS directory | `/data/hst/val` |
| `DEFAULT` | `hsc_path_val` | Path to HSC validation FITS directory | `/data/hsc/val` |
| `HST_DIM` | `hst_dim` | HST image dimension | `768` |
| `HSC_DIM` | `hsc_dim` | HSC image dimension | `128` |
| `BATCH_SIZE` | `batch_size` | Training batch size | `4` |
| `GAN_STEPS` | `gan_steps` | Total training steps | `100000` |
| `SAVE_STEPS` | `save_steps` | Checkpoint save frequency | `5000` |
| `DISPLAY_STEPS` | `display_steps` | Logging/visualization frequency | `500` |
| `LR` | `lr` | Generator learning rate | `0.0002` |
| `DISC_LR` | `disc_lr` | Discriminator learning rate | `0.0002` |
| `LAMBDA_RECON` | `lambda_recon` | L1 reconstruction loss weight | `200` |
| `LAMBDA_SEGMAP` | `lambda_segmap` | Segmentation-masked L1 weight | `200` |
| `LAMBDA_VGG` | `lambda_vgg` | VGG perceptual loss weight | `200` |
| `LAMBDA_SCATTERING` | `lambda_scattering` | Scattering loss weight | `1` |
| `LAMBDA_ADV` | `lambda_adv` | Adversarial loss weight | `5` |
| `DISC_UPDATE_FREQ` | `disc_update_freq` | Steps between discriminator updates | `3` |
| `DATA_AUG` | `data_aug` | Enable data augmentation | `True` |
| `VGG_LOSS_WEIGHTS` | `vgg_loss_weights` | Per-VGG-layer loss weights | `[1.0,1.0,0.0,0.0,0.0]` |

---

## Dependencies

Core requirements:

- **PyTorch** (>= 1.9) — Deep learning framework
- **torchvision** — VGG-19 pretrained model, image transforms
- **torchlayers** — PixelShuffle convolution modules
- **kymatio** — Wavelet scattering transform
- **comet-ml** — Experiment tracking and visualization
- **astropy** — FITS file I/O and image normalization
- **sep** — Source Extractor (Python) for segmentation maps
- **scikit-image** — Gaussian filtering
- **scikit-learn** — Min-max scaling
- **opencv-python** — Image resizing utilities
- **matplotlib** — Figure generation for logging

---

## License

This project is provided for research and educational purposes.
