
import torch.nn as nn
from down_sample_conv import DownSampleConv
from up_sample_conv import UpSampleConv
import torch
from torchlayers.upsample import ConvPixelShuffle
class Pix2PixGenerator(nn.Module):

    def __init__(self, in_channels, out_channels,n_ps_blocks=2,resize_conv=True):
        """
        Paper details:
        - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
        """
        super().__init__()

       # Same Convolutions
        same_convs = [nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0) for _ in range(5)]
        self.same_convs = nn.Sequential(*same_convs)

       # Sub-Pixel Convolutions (PixelShuffle) 
        ps_blocks = []
        ps_blocks += [ConvPixelShuffle(in_channels = 1, out_channels = 1, upscale_factor=3),
                      nn.PReLU()]
        ps_blocks += [ConvPixelShuffle(in_channels = 1, out_channels = 1, upscale_factor=2),
                      nn.PReLU()]

        # for i in range(n_ps_blocks):

        #     if i == 0:
        #         ps_blocks += [
        #         nn.Conv2d(1, 9*1, kernel_size=3, padding=1),
        #         nn.PixelShuffle(3),
        #         nn.PReLU(),]
        #     else:
        #         ps_blocks += [
        #         nn.Conv2d(1, 4*1, kernel_size=3, padding=1),
        #         nn.PixelShuffle(2),
        #         nn.PReLU(),
        #     ]

        self.ps_blocks = nn.Sequential(*ps_blocks)
        # encoder/donwsample convs
        self.encoders = [
            DownSampleConv(in_channels, 64, batchnorm=False),  # bs x 64 x 128 x 128
            DownSampleConv(64, 128),  # bs x 128 x 64 x 64
            DownSampleConv(128, 256),  # bs x 256 x 32 x 32
            DownSampleConv(256, 512),  # bs x 512 x 16 x 16
            DownSampleConv(512, 512),  # bs x 512 x 8 x 8
            DownSampleConv(512, 512),  # bs x 512 x 4 x 4
            DownSampleConv(512, 512),  # bs x 512 x 2 x 2
            DownSampleConv(512, 512, batchnorm=False),  # bs x 512 x 1 x 1
        ]

        # decoder/upsample convs
        self.decoders = [
            UpSampleConv(512, 512,  dropout=True),  # bs x 512 x 2 x 2
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 4 x 4
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 8 x 8
            UpSampleConv(1024, 512),  # bs x 512 x 16 x 16
            UpSampleConv(1024, 256),  # bs x 256 x 32 x 32
            UpSampleConv(512, 128),  # bs x 128 x 64 x 64
            UpSampleConv(256, 64),  # bs x 64 x 128 x 128
        ]
        self.decoder_channels = [512, 512, 512, 512, 256, 128, 64]
        self.up_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=2, padding=1,output_padding=1)
        # self.up_conv = nn.Sequential(
        #                         nn.Upsample(scale_factor = 2, mode='nearest'),
        #                         nn.ReflectionPad2d(1),
        #                         nn.Conv2d(64, 16,kernel_size=3, stride=1, padding=0))

        self.final_conv = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)


        self.tanh = nn.Tanh()

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        # Original Image
        # x = self.upsample(x)
        x_in = x
        # Same convs 
       # x = self.same_convs(x)
        # Encode & Skip Connections
        x = self.ps_blocks(x)
        skips_cons = []
        for encoder in self.encoders:
            # print("Before Encoder:",x.shape)

            x = encoder(x)
            # print("After Encoder:",x.shape)

            skips_cons.append(x)
        #Reverse for expansion phase
        skips_cons = list(reversed(skips_cons[:-1]))

        # Run expansion phase through decoder n-1
        decoders = self.decoders[:-1]
        for decoder, skip in zip(decoders, skips_cons):
            # print("Before Decoder:",x.shape)

            x = decoder(x)
            # print("After Decoder",x.shape)
            x = torch.cat((x, skip), axis=1)
        # Last decoder 
        x = self.decoders[-1](x)

        #Up sample
        x = self.up_conv(x)


        # Add input image (HSC) as "skip connection"
        # x = torch.cat((x, x_in), axis=1)


        # final conv to go from 2->1 channels
        x = self.final_conv(x)

        return self.tanh(x)