
import torch.nn as nn
from down_sample_conv import DownSampleConv
from up_sample_conv import UpSampleConv
import torch
class Pix2PixGenerator(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Paper details:
        - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
        """
        super().__init__()

        # encoder/donwsample convs
        self.encoders = [
            DownSampleConv(in_channels, 64, batchnorm=False),  # bs x 64 x 128 x 128
            DownSampleConv(64, 128),  # bs x 128 x 64 x 64
            DownSampleConv(128, 256),  # bs x 256 x 32 x 32
            DownSampleConv(256, 512),  # bs x 512 x 16 x 16
            DownSampleConv(512, 512),  # bs x 512 x 8 x 8
            DownSampleConv(512, 512),  # bs x 512 x 4 x 4
            # DownSampleConv(512, 512),  # bs x 512 x 2 x 2
            DownSampleConv(512, 512, batchnorm=False),  # bs x 512 x 1 x 1
        ]

        # decoder/upsample convs
        self.decoders = [
            UpSampleConv(512, 512,  dropout=True),  # bs x 512 x 2 x 2
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 4 x 4
            # UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 8 x 8
            UpSampleConv(1024, 512),  # bs x 512 x 16 x 16
            UpSampleConv(1024, 256),  # bs x 256 x 32 x 32
            UpSampleConv(512, 128),  # bs x 128 x 64 x 64
            UpSampleConv(256, 64),  # bs x 64 x 128 x 128
        ]
        self.decoder_channels = [512, 512, 512, 512, 256, 128, 64]
        self.up_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=2, padding=1,output_padding=1)
        self.final_conv = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)

        self.tanh = nn.Tanh()

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        # Original Image
        x_in = x 

        # Encode & Skip Connections
        skips_cons = []
        for encoder in self.encoders:
            x = encoder(x)
            skips_cons.append(x)
        #Reverse for expansion phase
        skips_cons = list(reversed(skips_cons[:-1]))

        # Run expansion phase through decoder n-1
        decoders = self.decoders[:-1]
        for decoder, skip in zip(decoders, skips_cons):
            x = decoder(x)
            x = torch.cat((x, skip), axis=1)
        # Last decoder 
        x = self.decoders[-1](x)

        #Up sample
        x = self.up_conv(x)

        # Add input image (HSC) as "skip connection"
        x = torch.cat((x, x_in), axis=1)

        # final conv to go from 2->1 channels
        x = self.final_conv(x)

        return self.tanh(x)