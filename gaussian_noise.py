import torch.nn as nn
import torch
import numpy as np
class GaussianNoise(nn.Module):
#     """Gaussian Noise Layer. Standard Deviation calculated @ instace level. 
#        If identity_map is True, no noise applied. 

#     Args:
#          None
#     """

    def __init__(self):
        super().__init__()
        self.register_buffer('noise', torch.tensor(0.0))
    @staticmethod
    def _ds9_unscaling(x, a=1000,offset = 0 ):
        return (((a + 1)**x - 1) / a) + offset

    def forward(self, x,identity_map,ds9=True):
        if identity_map == True:
            return x
        if ds9:
            x=self._ds9_unscaling(x)
        std = x.std(axis=(1,2))
        noise = torch.randn_like(x)
        stds = std[:,np.newaxis,np.newaxis]*noise
        x = x + stds
        return x 