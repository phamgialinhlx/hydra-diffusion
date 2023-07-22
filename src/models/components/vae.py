"""Variational Auto-Encoder"""
import torch
from torch import nn

class VAE(nn.Module):
    def __init__(
            self,
            batch_size: int = 64,
            in_channels: int = 1,
            img_size: int = 32,
    ):
        super(VAE, self).__init__()
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.img_size = img_size

        self.encoder = nn.Sequential(
            nn.Linear(in_channels * img_size * img_size, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, in_channels * img_size * img_size),
            nn.Sigmoid()
        )

    def forward_encoder(self, x):
        x = x.view(x.size()[0], -1)
        x = self.encoder(x)
        return x

    def forward_decoder(self, x):
        x = self.decoder(x)
        x = x.view(x.size()[0], self.in_channels, self.img_size, self.img_size)
        return x

    def forward(self, x):
        x = self.forward_encoder(x)
        x_hat = self.forward_decoder(x)
        return x_hat, 0.0
    
if __name__ == "__main__":
    net = VAE()
    x = torch.randn(56, 1, 32, 32)
    x_hat, kld = net(x)
    print(x_hat.shape)
    print(kld)