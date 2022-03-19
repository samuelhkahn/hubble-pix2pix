import torch.nn as nn

class UpSampleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        strides=2,
        padding=1,
        output_padding=1,
        activation=True,
        batchnorm=True,
        dropout=False,
        resize_convolution=False
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout
        if resize_convolution:

            self.deconv =  nn.Sequential(
                                            nn.Upsample(scale_factor = 2, mode='nearest'),
                                            nn.ReflectionPad2d(1),
                                            nn.Conv2d(in_channels, out_channels,kernel_size=3, stride=1, padding=0)
                                        )
            # self.deconv = nn.ModuleList(self.deconv)
        else:
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, strides, padding,output_padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.deconv(x)
        if self.batchnorm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)
        return x