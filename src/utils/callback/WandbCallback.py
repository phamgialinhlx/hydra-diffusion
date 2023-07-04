from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from torchvision.utils import make_grid
import math

class WandbCallback(Callback):
    def __init__(self, num_classes: int = 10, num_images: int = 9, condition: bool = True, use_fixed_noise: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.num_images = num_images
        self.condition = condition
        self.use_fixed_noise = use_fixed_noise
        self.images = None
        self.labels = None
        self.noise = torch.randn((self.num_images, 1, 32, 32)).to("cpu")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        with torch.no_grad():
            gif_shape = [int(math.sqrt(self.num_images)), int(math.sqrt(self.num_images))]

            model = pl_module.net.to("cpu")
            if self.use_fixed_noise:
                x = self.noise
            else:               
                x = torch.randn((self.num_images, 1, 32, 32))
            
            sample_steps = torch.arange(model.t_range - 1, 0, -1)
            for t in sample_steps:
                x = model.denoise_sample(x, t, self.labels.to("cpu"))

            pl_module.net.to("cuda")

            fake = make_grid(x, nrow=gif_shape[0])
            real = make_grid(self.images, nrow=gif_shape[0])
            trainer.logger.log_image(key='samples', images=[fake, real], caption=['fake', 'real'])
            
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.images, self.labels = batch
        if len(self.images) >= self.num_images:
            self.images = self.images[: self.num_images]
            self.labels = self.labels[: self.num_images]            