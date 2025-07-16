import warnings
from logging import warning
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from base.loss import PixelWiseInfoNCE
from model import Threshold as _Threshold
from torch.optim import Optimizer
import torchvision
from neptune.types import File


class Threshold(pl.LightningModule):
    def __init__(
        self,
        lambda_entropy: float,
        lambda_contrastive: float,
        learning_rate: float,
        out_shape: Tuple[int, int] = (256, 256),
        n_classes: int = 1,
        temp: float = 0.5,
        optimizer_configs: Dict[str, Any] = {},
        scheduler_configs: Dict[str, Any] = {},
        log_sample_interval = float('inf')
    ):
        super().__init__()
        # self.save_hyperparameters() saves arguments. Model and loss objects are complex,
        # so often ignored and recreated if loading from checkpoint, or passed as arguments.
        # But if you define init args well, it mostly works.
        self.save_hyperparameters(
            logger=False
        )  # Use logger=False to avoid issues with non-serializable objects if self.model/loss_fn are passed directly.

        self.model = _Threshold(out_shape=out_shape, num_classes=n_classes)
        self.contrastive_loss_fn = PixelWiseInfoNCE(temp)
        self.lambda_entropy = lambda_entropy
        self.lambda_contrastive = lambda_contrastive
        self.learning_rate = learning_rate
        self.optimizer_configs = optimizer_configs
        self.scheduler_configs = scheduler_configs
        self.log_sample_interval = log_sample_interval

    def forward(self, x):
        # The forward method of LightningModule just wraps the inner model's forward
        return self.model(x)

    def _shared_step(self, batch):
        image_view1, image_view2 = batch

        # Forward pass for both augmented views
        logits_v1, features_v1 = self.model(image_view1)
        prob_map_v1 = torch.sigmoid(logits_v1)
        L_entropy_v1 = -torch.mean(
            -prob_map_v1 * torch.log(prob_map_v1 + 1e-8)
            - (1 - prob_map_v1) * torch.log(1 - prob_map_v1 + 1e-8)
        )

        logits_v2, features_v2 = self.model(image_view2)
        prob_map_v2 = torch.sigmoid(logits_v2)
        L_entropy_v2 = -torch.mean(
            -prob_map_v2 * torch.log(prob_map_v2 + 1e-8)
            - (1 - prob_map_v2) * torch.log(1 - prob_map_v2 + 1e-8)
        )

        # 1. Calculate Entropy Loss
        # Apply sigmoid to logits to get probability maps

        # Calculate the entropy for each view. Add a small epsilon for numerical stability.
        # The goal is to maximize entropy, which is equivalent to minimizing the negative entropy.

        L_entropy = (L_entropy_v1 + L_entropy_v2) / 2  # Average over views

        # 2. Calculate Contrastive Loss
        L_contrastive = self.contrastive_loss_fn(features_v1, features_v2)

        # Total Loss
        # We ADD the entropy loss because we are minimizing the NEGATIVE entropy.
        total_loss = self.lambda_contrastive * L_contrastive + self.lambda_entropy * L_entropy

        return total_loss, L_entropy, L_contrastive

    def training_step(self, batch, batch_idx):
        total_loss, L_entropy, L_contrastive = self._shared_step(batch)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "train_entropy_loss",
            L_entropy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train_contrastive_loss",
            L_contrastive,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, L_entropy, L_contrastive = self._shared_step(batch)

        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "val_entropy_loss",
            L_entropy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val_contrastive_loss",
            L_contrastive,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        if self.current_epoch % self.log_sample_interval == 0:
            self.log_sample(batch, batch_idx)

        return total_loss

    def log_sample(self, batch, batch_idx):
        """
        Generates and logs binary threshold images for the first test batch.
        """
        # We only log the first batch to avoid excessive logging
        if batch_idx == 0:
            image_view1, image_view2 = batch  # We only need one view for inference

            # 1. Get model output
            logits, _ = self.model(image_view1)
            prob_map = torch.sigmoid(logits)

            # 2. Create binary mask from probabilities
            binary_mask = (prob_map > 0.5).float()

            image_view = torch.cat([image_view1, image_view2], dim=0)

            # 3. Create grids of images for visualization
            input_grid = torchvision.utils.make_grid(image_view, normalize=True)
            mask_grid = torchvision.utils.make_grid(binary_mask)

            # 4. Permute from (C, H, W) to (H, W, C) before converting to numpy
            input_grid_np = input_grid.detach().cpu().permute(1, 2, 0).numpy()
            mask_grid_np = mask_grid.detach().cpu().permute(1, 2, 0).numpy()

            # 5. Log the input images and output masks to Neptune
            self.logger.experiment["test/predictions/inputs"].log(
                File.as_image(input_grid_np)
            )
            self.logger.experiment["test/predictions/masks"].log(
                File.as_image(mask_grid_np)
            )


    def configure_optimizers(self):
        # Optimizer
        import torch.optim
        import torch.optim.lr_scheduler

        optimizer_name = getattr(self.hparams.optimizer_configs, "name")
        scheduler_name = getattr(self.hparams.scheduler_configs, "name")

        scheduler_configs = {
            k: v for (k, v) in self.hparams.scheduler_configs.items() if k != "name"
        }
        optimizer_configs = {
            k: v for (k, v) in self.hparams.optimizer_configs.items() if k != "name"
        }

        Optimizer = getattr(torch.optim, optimizer_name)

        if not Optimizer:
            warnings.warn(f"Optimizer {optimizer_name} not found! Fallback to Default: Adam")
            Optimizer = torch.optim.Adam

        optimizer = Optimizer(self.parameters(), **optimizer_configs)

        # Learning Rate Scheduler (Optional but recommended for Transformers)
        if scheduler_name is None:
            return optimizer

        Scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)
        if not Scheduler:
            warnings.warn(f"Scheduler: {scheduler_name} not found! Fallback to no scheduler")
            return optimizer

        scheduler = Scheduler(optimizer, **scheduler_configs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Call scheduler.step() every epoch
                "monitor": "val_loss",  # Monitor val_loss for potentially pausing/stopping
                "frequency": 1,
            },
        }
