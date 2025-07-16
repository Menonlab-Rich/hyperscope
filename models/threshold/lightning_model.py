import io
import itertools
import warnings
from logging import warning
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision
from base.loss import PixelWiseInfoNCE
from model import Threshold as _Threshold
from neptune.types import File
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.optim import Optimizer


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
        log_sample_interval=float("inf"),
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

        features, entropies = [], []
        for view in batch:
            # Forward pass for both augmented views
            logits, feats = self.model(view)
            prob_map = torch.sigmoid(logits)
            entropy = -torch.mean(
                -prob_map * torch.log(prob_map + 1e-8)
                - (1 - prob_map) * torch.log(1 - prob_map + 1e-8)
            )

            entropies.append(entropy)
            features.append(feats)

        # 1. Calculate Entropy Loss
        # Apply sigmoid to logits to get probability maps

        # Calculate the entropy for each view. Add a small epsilon for numerical stability.
        # The goal is to maximize entropy, which is equivalent to minimizing the negative entropy.

        L_entropy = torch.stack(entropies).mean()

        # 2. Calculate Contrastive Loss
        total_contrastive_loss = 0.0
        n_pairs = 0
        for i, j in itertools.combinations(range(len(features)), 2):
            total_contrastive_loss += self.contrastive_loss_fn(features[i], features[j])
            n_pairs += 1
        L_contrastive = total_contrastive_loss / n_pairs if n_pairs > 0 else 0.0

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
        if batch_idx > 0:
            return  # Log only on the first batch to avoid clutter

        image_view = batch[0][0:1]  # (1, C, H, W)

        with torch.no_grad():
            _, features = self.model(image_view)

        features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        features_np = features.cpu().numpy()

        # Cluster
        kmeans = KMeans(n_clusters=self.hparams.n_classes, random_state=42, n_init="auto").fit(features_np)
        labels = kmeans.labels_

        # Dimensionality reduction for plotting
        pca = PCA(n_components=2)  # h, w
        features_2d = pca.fit_transform(features_np)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = sns.scatterplot(
            x=features_2d[:, 0],
            y=features_2d[:, 1],
            hue=labels,
            s=5,
            alpha=0.7,
            legend="full",
            ax=ax,
        )

        ax.set_title(f"Feature Space Clustering (Epoch {self.current_epoch})")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        plt.tight_layout()

        # Save plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        # 6. Log the input image and the feature plot to Neptune
        input_grid = torchvision.utils.make_grid(image_view, normalize=True)
        self.logger.experiment["test/visualizations/input_image"].log(
            File.as_image(input_grid.detach().cpu().permute(1, 2, 0).numpy())
        )
        self.logger.experiment["test/visualizations/feature_clusters"].log(File(buf))

        # Close the plot to free up memory
        plt.close(fig)

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
