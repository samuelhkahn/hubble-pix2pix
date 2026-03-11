"""Custom collate function for filtering corrupted or malformed FITS samples."""

import torch


def collate_fn(batch):
    """Filter and stack a batch of (HR, LR, HSC_HR, segmap) tuples.

    Skips samples that are None (corrupted FITS files), contain NaN/Inf values,
    or have unexpected spatial dimensions.

    Expected dimensions:
        - HR (HST): 768 x 768
        - LR (HSC): 128 x 128
        - HSC_HR (upsampled HSC): 768 x 768

    Args:
        batch: List of (hr, lr, hsc_hr, hr_seg) tuples from the dataset.

    Returns:
        Tuple of stacked tensors: (hrs, lrs, hsc_hrs, hr_segs), each with
        a leading batch dimension.
    """
    hrs, lrs, hsc_hrs, hr_segs = [], [], [], []

    for hr, lr, hsc_hr, hr_seg in batch:
        if any(el is None for el in [hr, lr, hsc_hr, hr_seg]):
            continue

        has_bad_values = (
            torch.isnan(hr).any() or torch.isnan(lr).any()
            or torch.isinf(hr).any() or torch.isinf(lr).any()
        )

        correct_shapes = (
            hr.shape == (768, 768)
            and lr.shape == (128, 128)
            and hsc_hr.shape == (768, 768)
        )

        if correct_shapes and not has_bad_values:
            hrs.append(hr)
            lrs.append(lr)
            hsc_hrs.append(hsc_hr)
            hr_segs.append(hr_seg)

    hrs = torch.stack(hrs, dim=0)
    lrs = torch.stack(lrs, dim=0)
    hsc_hrs = torch.stack(hsc_hrs, dim=0)
    hr_segs = torch.stack(hr_segs, dim=0)
    return hrs, lrs, hsc_hrs, hr_segs
