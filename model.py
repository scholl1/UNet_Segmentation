import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # in, out, kernel, stride, padding
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            # o = (input_size + 2*padding - (kernel_size - 1) - 1)/stride   +  1
            # bias=False is unnecessary due to -> batchnorm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )



def forward(self, x):
    return self.conv(x)


class UNET(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        # Downpart of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature


        # Uppart of UNET
        # Conv and 
        # http://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-output-size-of-convolutions.html
        for feature in features:
            #
            for feature in reversed(features):
                self.ups.append(
                    nn.ConvTranspose2d(
                        feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))


        # bottom part
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # final output (1x1 conv)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)





    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

