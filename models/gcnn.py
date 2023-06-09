import torch
import torch.nn as nn

from .groupconv import StereoZ2ConvG, StereoGMaxPool2d, StereoGConvBlock, StereoGBatchNorm2d, StereoGAveragePool
from .cnn import init_weights

class GCNNModel(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group
        self.lifting_conv = nn.Sequential(
                                # (B, 3, 2, 200, 400)
                                StereoZ2ConvG(group, 3, 16, 3, 1),
                                StereoGBatchNorm2d(group, 16),
                                nn.ReLU(),
                                # (B, n, 16, 2, 200, 400)
                                StereoGMaxPool2d(group, 10, 10),
                                # (B, n, 16, 2, 20, 40)
                                )

        self.g_conv_1 = nn.Sequential(
                                StereoGConvBlock(group, 16, 8, 3, 1),
                                StereoGConvBlock(group, 8, 8, 3, 1),
                                StereoGMaxPool2d(group, 2, 2),
                                # (B, n, 8, 2, 10, 20)
                                )

        self.g_conv_2 = nn.Sequential(
                                StereoGConvBlock(group, 8, 4, 3, 1),
                                StereoGConvBlock(group, 4, 4, 3, 1),
                                StereoGMaxPool2d(group, 2, 2),
                                # (B, n, 4, 2, 5, 10)
                                )

        self.g_conv_3 = nn.Sequential(
                                StereoGConvBlock(group, 4, 4, 3, 1),
                                StereoGConvBlock(group, 4, 2, 3, 1),
                                # (B, n, 2, 2, 5, 10)
                                StereoGAveragePool(group, reduction="mean"),
                                # 2 * 2 * 5 * 10 = 200 features out
                                )

        self.fc = nn.Sequential(
                        nn.Linear(200, 40),
                        nn.ReLU(),
                        nn.Linear(40, 1),
                        nn.Sigmoid(),
                    )

        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.lifting_conv(x)
        x = self.g_conv_1(x)
        x = self.g_conv_2(x)
        x = self.g_conv_3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = x.view(-1)
        return x

    def get_activations(self, x):
        x1 = self.lifting_conv(x)
        x2 = self.g_conv_1(x1)
        x3 = self.g_conv_2(x2)
        x4 = self.g_conv_3(x3)
        return (x1, x2, x3, x4)
