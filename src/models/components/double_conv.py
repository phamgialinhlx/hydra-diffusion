import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from torch import nn
from torch.nn import functional as F
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

if __name__ == "__main__":
    dc = DoubleConv(1, 64)
    img = torch.rand(1, 1, 32, 32)
    print(dc(img).shape)
    dc = DoubleConv(1, 64, residual=True)
    img = torch.rand(1, 1, 32, 32)
    print(dc(img).shape)
