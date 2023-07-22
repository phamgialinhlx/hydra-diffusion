from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import math
import torch.nn as nn

class DiffusionModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(self, net: torch.nn.Module):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def model_step(self, batch:Any):
        batch, _ = batch
        loss = self.net.get_loss(batch)
        return loss
    
    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer


if __name__ == "__main__":
    _ = DiffusionModule(None, None, None)
