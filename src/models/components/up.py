"""Module defining the Up class."""
from torch import nn
from torch.nn import functional as F
import torch
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.double_conv import DoubleConv

class Up(nn.Module):
    """
    Upscaling then double conv.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, in_channels, residual=True)
            self.conv2 = DoubleConv(
                in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, input1, input2):
        """
        Forward pass.
        """
        input1 = self.up(input1)
        # input is CHW
        diff_y = input2.size()[2] - input1.size()[2]
        diff_x = input2.size()[3] - input1.size()[3]

        input1 = F.pad(input1, [diff_x // 2, diff_x -
                       diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        output = torch.cat([input2, input1], dim=1)
        output = self.conv(output)
        output = self.conv2(output)
        return output

if __name__ == "__main__":
    BILLINEAR = True
    FACTOR = 2 if BILLINEAR else 1

    # up = Up(512, 256 // factor, bilinear)
    # img1 = torch.rand(1, 512, 64, 64)
    # img2 = torch.rand(1, 128, 128, 128)
    # print(up(img1, img2).shape)
    # up = Up(1, 64, bilinear=False)
    # img1 = torch.rand(1, 1, 32, 32)
    # img2 = torch.rand(1, 64, 64, 64)
    # print(up(img1, img2).shape)
