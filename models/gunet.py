import torch
import torch.nn as nn

from .groupconv import StereoZ2ConvG, StereoGMaxPool2d, StereoGConvBlock, StereoGBatchNorm2d, StereoGAveragePool, StereoGUpsample2d, UNetGConvBlock 
from .stereoconv import StereoConvBlock
from .cnn import init_weights

class GUNetModel(nn.Module):
    def __init__(self, group, num_features=[3,4,8,16,32,64]):
        super().__init__()
        c0, c1, c2, c3, c4, c5 = num_features
        self.group = group
        # Lift signal to affine group
        self.lifting_conv = nn.Sequential(
                                # (B, 3, 2, 480, 640) -> (B, 3, 2, 480, 640)
                                StereoZ2ConvG(group, c0, c0, 3, 1),
                                StereoGBatchNorm2d(group, c0),
                                nn.ReLU(),
                                )

        # "Left-side" of UNet
        self.conv_1_1 = UNetGConvBlock(group, c0, c1)
        self.conv_1_2 = UNetGConvBlock(group, c1, c2)
        self.conv_1_3 = UNetGConvBlock(group, c2, c3)
        self.conv_1_4 = UNetGConvBlock(group, c3, c4)
        self.conv_1_5 = UNetGConvBlock(group, c4, c5)
        # "Right-side" of UNet
        # Last layer includes 1x1 convolution and averaging over the group dimension
        self.conv_2_1 = UNetGConvBlock(group, c4+c5, c4)
        self.conv_2_2 = UNetGConvBlock(group, c3+c4, c3)
        self.conv_2_3 = UNetGConvBlock(group, c2+c3, c2)
        self.conv_2_4 = nn.Sequential(
                            UNetGConvBlock(group, c1+c2, c1),
                            StereoGConvBlock(group, c1, 1, 1, 0, 1),
                            StereoGAveragePool(reduction="mean")
                        )
        
        self.maxpool = StereoGMaxPool2d(group)
        self.upsample = StereoGUpsample2d(group)
        
    def forward(self, x):
        # Lift signal to the affine group G
        x = self.lifting_conv(x)
        # Downward pass
        out1 = self.conv_1_1(x)
        out2 = self.conv_1_2(self.maxpool(out1))
        out3 = self.conv_1_3(self.maxpool(out2))
        out4 = self.conv_1_4(self.maxpool(out3))
        out5 = self.conv_1_5(self.maxpool(out4))
        # Upward pass
        out6 = self.conv_2_1(torch.cat([out4, self.upsample(out5)], dim=2))
        out7 = self.conv_2_2(torch.cat([out3, self.upsample(out6)], dim=2))
        out8 = self.conv_2_3(torch.cat([out2, self.upsample(out7)], dim=2))
        out9 = self.conv_2_4(torch.cat([out1, self.upsample(out8)], dim=2))
        return out9
