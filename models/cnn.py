import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m: nn.Module):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

class CNNBlock(nn.Module):
    """
        Simple convolutional block: CNN -> BN -> ReLU.
        Support downsampling / max pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size: int=3, padding: int=1, downsample: int=0):
        super().__init__()
        self.layers = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
                            nn.BatchNorm2d(num_features=out_channels),
                            nn.ReLU()
                        )
        self.layers.apply(init_weights)
        self.downsample = downsample

    def forward(self, x):
        x = self.layers(x)
        if self.downsample > 0:
            x = F.max_pool2d(x, kernel_size=self.downsample, stride=self.downsample)
        return x

class CNNModel(nn.Module):
    def __init__(self):
        """
            Input size:  (B, 3, 100, 400)
        """
        super().__init__()
        self.convolutions = nn.Sequential(
                                CNNBlock(3, 32, 3, 1, 5),   # (, 20, 80)
                                CNNBlock(32, 32, 3, 1, 1),

                                CNNBlock(32, 16, 3, 1, 2),  # (, 10, 40)
                                CNNBlock(16, 16, 3, 1, 1),
                                
                                CNNBlock(16, 8, 3, 1, 2),   # (, 5, 20)
                                CNNBlock(8, 8, 3, 1, 1),

                                CNNBlock(8, 4, 3, 1, 1),   # (, 5, 20)
                                CNNBlock(4, 4, 3, 1, 1),
                            ) # Features out: 4 * 5 * 20 = 400

        self.fc = nn.Sequential(
                        nn.Linear(400, 1),
                        nn.Sigmoid()
                    )

    def forward(self, x):
        x = self.convolutions(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = x.view(-1)
        return x