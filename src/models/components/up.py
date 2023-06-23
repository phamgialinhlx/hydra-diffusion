import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.double_conv import DoubleConv
from torch import nn
from torch.nn import functional as F
import torch

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, in_channels, residual=True)
            self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.conv2(x)
        return x

if __name__ == "__main__":
    bilinear = True
    factor = 2 if bilinear else 1
    
    # up = Up(512, 256 // factor, bilinear)
    # img1 = torch.rand(1, 512, 64, 64)
    # img2 = torch.rand(1, 128, 128, 128)
    # print(up(img1, img2).shape)
    # up = Up(1, 64, bilinear=False)
    # img1 = torch.rand(1, 1, 32, 32)
    # img2 = torch.rand(1, 64, 64, 64)
    # print(up(img1, img2).shape)