import torch.nn as nn
from monai.networks.nets import UNet

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )

    def forward(self, x):
        return self.model(x)