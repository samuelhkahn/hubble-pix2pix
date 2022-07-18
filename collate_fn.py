import torch
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
