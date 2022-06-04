import torch.nn as nn
import torch
import numpy as np
class GaussianNoise(nn.Module):
#     """Gaussian Noise Layer. Standard Deviation calculated @ batchwise. If 
#        none zero-centered mean, mean calculated batchwise. 

#     Args:
#         zero_mean (bool, optional): whether to generate the noise centered @ zero 
#     """

    def __init__(self):
        super().__init__()
        self.register_buffer('noise', torch.tensor(0.0))

    def forward(self, x,identity_map):
        if identity_map == True:
            return x


        std = x.std(axis=(1,2))
        noise = torch.randn_like(x)
        stds = std[:,np.newaxis,np.newaxis]*noise
        print(x)
        x = x + stds
        print(x)
        return x 