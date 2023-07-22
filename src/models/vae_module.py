from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import math
import torch.nn as nn

class VAEModule(LightningModule):
    def __init__(self, net: torch.nn.Module, lr: float = 1e-3):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        print(net)
        self.lr = lr

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def model_step(self, batch:Any):
        batch, _ = batch
        x_hat, kld = self.net(batch)
        mse = self.criterion(x_hat, batch)
        return mse, kld
    
    def training_step(self, batch: Any, batch_idx: int):
        mse, kld = self.model_step(batch)
        self.train_loss(mse + kld)
        self.log("train/mse", mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/kld", kld, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return mse + kld

    def validation_step(self, batch: Any, batch_idx: int):
        mse, kld = self.model_step(batch)
        self.val_loss(mse + kld)
        self.log("val/mse", mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/kld", kld, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return mse # + kld
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    _ = VAEModule(None, None, None)
