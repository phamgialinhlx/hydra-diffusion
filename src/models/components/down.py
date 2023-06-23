import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from torch import nn
from src.models.components.double_conv import DoubleConv

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)
