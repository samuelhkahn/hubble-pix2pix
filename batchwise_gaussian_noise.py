import torch.nn as nn
import torch

class BatchWiseGaussianNoise(nn.Module):
    """Gaussian Wise Noise .

    Args:
        zero_mean (bool, optional): whether to generate the noise centered @ zero 
    """
    
    def __init__(self, zero_mean = True):
        super().__init__()
        self.zero_mean = zero_mean

    def forward(self, x):
        std = x.std().item()
        if self.zero_mean:
            gaussian_noise = torch.empty(x.size()).normal_(mean=0,std=std)
        else:
            gaussian_noise = torch.empty(x.size()).normal_(mean=x.mean(),std=std)
        x = x + gaussian_noise
        return x 