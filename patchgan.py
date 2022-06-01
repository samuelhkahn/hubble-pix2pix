import torch.nn as nn
from down_sample_conv import DownSampleConv
import torch
class PatchGAN(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.d1 = DownSampleConv(input_channels, 64, batchnorm=False)
        self.d2 = DownSampleConv(64, 128,batchnorm=False)
        self.d3 = DownSampleConv(128, 256,batchnorm=False)
        self.d4 = DownSampleConv(256, 512,batchnorm=False)
        self.final = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x,y):
        x = torch.cat([x, y], axis=1)
        # print("x:",x.shape)
        x0 = self.d1(x)
        # print("x0:",x0.shape)

        x1 = self.d2(x0)
        # print("x1:",x1.shape)

        x2 = self.d3(x1)
        # print("x2:",x2.shape)

        x3 = self.d4(x2)
        # print("x3:",x3.shape)

        xn = self.final(x3)
        # print("x4:",xn.shape)
        return xn