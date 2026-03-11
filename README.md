# Neo: Photometric Super-Resolution for Astronomical Imagery

<!-- TODO: Uncomment when badges are available
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
-->

**Neo** enhances ground-based telescope imagery to approach space-based resolution using a Pix2Pix conditional GAN.  It translates images from the Hyper Suprime-Cam (HSC) on the Subaru Telescope to match the resolution of the Hubble Space Telescope (HST), achieving **6x super-resolution** (128x128 -> 768x768).

> **Neo: Photometric Super-Resolution for Improving Galaxy Morphological Measurements using Conditional Generative Adversarial Networks**
>
> Samuel Kahn et al.
>
> <!-- [Paper](https://arxiv.org/abs/XXXX.XXXXX) | --> [Code](https://github.com/samuelhkahn/hubble-pix2pix)

---

## Method

Ground-based telescopes are limited by atmospheric seeing, producing images with lower spatial resolution than space-based observatories.  Neo bridges this gap by learning the mapping from low-resolution HSC to high-resolution HST images using a conditional GAN with a multi-component loss function designed for astronomical imagery.

### Architecture

| Component | Description |
|-----------|-------------|
| **Generator** | U-Net encoder-decoder (C64-C512) with skip connections + PixelShuffle 6x upsampling |
| **Discriminator** | PatchGAN -- classifies overlapping image patches as real/fake |
| **Upsampling** | Sub-pixel convolution: 3x then 2x PixelShuffle (128 -> 384 -> 768) |

### Loss Function

The generator is trained with a composite loss combining five complementary objectives:

| Loss | Weight | Purpose |
|------|--------|---------|
| **BCE Adversarial** | `lambda_adv` | PatchGAN discriminator ensures generated images look realistic |
| **L1 Reconstruction** | `lambda_recon` | Pixel-wise fidelity with ground truth |
| **VGG-19 Perceptual** | `lambda_vgg` | Multi-scale feature matching via pretrained VGG-19 |
| **Wavelet Scattering** | `lambda_scattering` | Preserves multi-scale morphological structure (Kymatio) |
| **Segmentation-Masked L1** | `lambda_segmap` | Prioritizes reconstruction of detected astronomical sources (SEP) |

---

## Results

<!-- TODO: Fill in with actual experimental results -->

| Metric | Bicubic | Neo (Ours) |
|--------|---------|------------|
| PSNR (dB) | -- | -- |
| SSIM | -- | -- |
| FID | -- | -- |

<!-- TODO: Add sample images
### Visual Examples
![Comparison](figures/comparison.png)
-->

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, 16GB+ VRAM)
- FITS image data (HST and HSC paired cutouts)

### From source

```bash
git clone https://github.com/samuelhkahn/hubble-pix2pix.git
cd hubble-pix2pix
pip install -e .
```

### Environment variable

Set your Comet ML API key for experiment tracking:

```bash
export COMET_ML_ASTRO_API_KEY="your-api-key-here"
```

---

## Usage

### Data preparation

Organize paired HST/HSC FITS cutouts into train/validation directories.  Filenames must match between HST and HSC directories.

```
data/
├── hst_train/       # HST training cutouts (~600x600 FITS)
├── hsc_train/       # HSC training cutouts (~100x100 FITS)
├── hst_val/         # HST validation cutouts
└── hsc_val/         # HSC validation cutouts
```

### Training

1. Copy and edit the example config:

   ```bash
   cp neo/configs/example.ini neo/configs/my_run.ini
   # Edit paths and hyperparameters
   ```

2. Launch training:

   ```bash
   python train.py neo/configs/my_run.ini
   ```

3. Monitor on [Comet ML](https://www.comet.ml).  Logged metrics include all loss components (train + val), discriminator logit maps, and side-by-side image comparisons (HSC input, generated SR, ground truth HST, residuals).

### Inference

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a trained generator checkpoint.
generator = torch.load("models/gen_pix2pixsr_<name>_checkpoint_<step>.pt")
generator.to(device)
generator.eval()

# Super-resolve a low-resolution HSC image.
# Input: (1, 1, 128, 128) -> Output: (1, 1, 768, 768)
with torch.no_grad():
    sr_image = generator(hsc_input.to(device), identity_map=True)
```

---

## Project Structure

```
neo/
├── __init__.py              # Package metadata and version
├── pix2pix.py               # Pix2Pix GAN: training/validation logic, loss computation
├── log_figure.py            # Comet ML figure logging utilities
├── models/
│   ├── generator.py          # U-Net generator with PixelShuffle upsampling
│   ├── patchgan.py           # PatchGAN discriminator
│   ├── down_sample_conv.py   # Strided conv blocks (encoder + discriminator)
│   ├── up_sample_conv.py     # Transpose conv blocks (decoder)
│   ├── gaussian_noise.py     # Gaussian noise injection layer
│   ├── vgg19.py              # VGG-19 feature extractor
│   └── vgg19_loss.py         # VGG perceptual loss module
├── data/
│   ├── dataset.py            # HST/HSC paired FITS dataset with DS9 scaling
│   └── collate_fn.py         # Batch collation with NaN/Inf filtering
└── configs/
    └── example.ini           # Example training configuration

train.py                      # Main training entry point
pyproject.toml                # Package metadata (pip install -e .)
requirements.txt              # Pinned dependencies
```

---

## Configuration

All training hyperparameters are set via an INI config file.  See [`neo/configs/example.ini`](neo/configs/example.ini) for a fully commented template.

Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 4 | Images per batch (reduce for less VRAM) |
| `gan_steps` | 100,000 | Total training steps |
| `lr` / `disc_lr` | 0.0002 | Generator / discriminator learning rate |
| `lambda_recon` | 200 | L1 reconstruction weight |
| `lambda_segmap` | 200 | Segmentation-masked L1 weight |
| `lambda_vgg` | 200 | VGG perceptual loss weight |
| `lambda_scattering` | 1 | Wavelet scattering loss weight |
| `lambda_adv` | 5 | Adversarial loss weight |
| `disc_update_freq` | 3 | Discriminator update frequency |
| `vgg_loss_weights` | [1,1,0,0,0] | Per-VGG-layer weights |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{kahn2024neo,
  title={Neo: Photometric Super-Resolution for Improving Galaxy Morphological
         Measurements using Conditional Generative Adversarial Networks},
  author={Kahn, Samuel},
  year={2024},
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
