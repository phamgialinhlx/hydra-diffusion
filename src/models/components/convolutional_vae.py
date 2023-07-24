"""Variational Auto-Encoder"""
import torch
from torch import nn

class ConvolutionalVAE(nn.Module):
    def __init__(
            self,
            batch_size: int = 64,
            in_channels: int = 1,
            img_size: int = 32,
            latent_dim: int = 100,
    ):
        super(ConvolutionalVAE, self).__init__()
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.img_size = img_size
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(32, in_channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.conv_out_size = self._get_conv_out_size(torch.rand(batch_size, in_channels, img_size, img_size))
        self.mu = nn.Sequential(
            nn.Linear(self.conv_out_size, latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        self.log_var = nn.Sequential(
            nn.Linear(self.conv_out_size, latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, self.conv_out_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

    def sampling(self, mu, log_var):
        epsilon = torch.randn(mu.shape).to(mu.device)
        return mu + epsilon * torch.exp(log_var / 2)

    def _get_conv_out_size(self, x: torch.Tensor) -> int:
        out = self.encoder(x)
        self.conv_out_shape = out.shape
        return int(torch.prod(torch.Tensor([self.conv_out_shape[1:]])))

    def forward_encoder(self, x):
        x = self.encoder(x)
        return x

    def forward_decoder(self, x):
        # x = self.decoder_linear(x)
        # x = x.view(x.size()[0], *self.conv_out_shape[1:])
        x = self.decoder(x)
        return x

    def forward(self, x):
        x = self.forward_encoder(x)
        x_hat = self.forward_decoder(x)
        x = x.view(x.size()[0], -1)
        mu_p = self.mu(x)
        log_var_p = self.log_var(x)
        kld = 0.5 * torch.sum(
            torch.pow(mu_p, 2) + torch.pow(log_var_p, 2) - torch.log(1e-8 + torch.pow(log_var_p, 2)) - 1
        ) / (self.batch_size * self.img_size * self.img_size)
        return x_hat, kld
    
if __name__ == "__main__":
    net = ConvolutionalVAE()
    x = torch.randn(56, 1, 32, 32)
    x_hat, kld = net(x)
    print(x_hat.shape)
    print(kld)