import torch.nn as nn
import torch

class BatchWiseGaussianNoise(nn.Module):
    """Gaussian Wise Noise .

    Args:
        zero_mean (bool, optional): whether to generate the noise centered @ zero 
    """

    def __init__(self,device, zero_mean = True):
        super().__init__()
        self.device = device
        self.zero_mean = zero_mean

    def forward(self, x):
        std = x.std().item()
        if self.zero_mean:
            gaussian_noise = torch.empty(x.size()).normal_(mean=0,std=std).to(self.device)
        else:
            gaussian_noise = torch.empty(x.size()).normal_(mean=x.mean(),std=std).to(self.device)
        x = x + gaussian_noise
        return x 