import torch.nn as nn
import torch

# class BatchWiseGaussianNoise(nn.Module):
#     """Gaussian Wise Noise .

#     Args:
#         zero_mean (bool, optional): whether to generate the noise centered @ zero 
#     """

#     def __init__(self,device, zero_mean = True):
#         super().__init__()
#         self.device = device
#         self.zero_mean = zero_mean

#     def forward(self, x):
#         x = x.to(self.device)

#         std = x.std().item()

#         if self.zero_mean:
#             gaussian_noise = torch.empty(x.size()).normal_(mean=0,std=std).to(self.device)
#         else:
#             gaussian_noise = torch.empty(x.size()).normal_(mean=x.mean(),std=std).to(self.device)
#         print(x.device)
#         print(gaussian_noise.device)

#         x = x + gaussian_noise
#         return x

class BatchWiseGaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self,zero_mean = True,  is_relative_detach=True):
        super().__init__()
        self.zero_mean = zero_mean
        self.is_relative_detach = is_relative_detach
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