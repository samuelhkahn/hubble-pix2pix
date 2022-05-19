import torch.nn as nn
import torch

class BatchWiseGaussianNoise(nn.Module):
#     """Gaussian Noise Layer. Standard Deviation calculated @ batchwise. If 
#        none zero-centered mean, mean calculated batchwise. 

#     Args:
#         zero_mean (bool, optional): whether to generate the noise centered @ zero 
#     """

    def __init__(self,zero_mean = True,  is_relative_detach=True):
        super().__init__()
        self.zero_mean = zero_mean
        self.register_buffer('noise', torch.tensor(0.0))

    def forward(self, x):
        std = x.std().item()
        if self.zero_mean:
            sampled_noise = self.noise.repeat(*x.size()).normal_(mean=0,std=std)
        else:
            mean = x.mean().item()
            sampled_noise = self.noise.repeat(*x.size()).normal_(mean=mean,std=std)

        x = x + sampled_noise        
        return x 