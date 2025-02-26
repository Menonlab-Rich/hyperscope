import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from dataset import CustomSAMDataset
from config import config
import os


class SAMDataModule(pl.LightningDataModule):
    def __init__(self, image_dir, mask_dir, batch_size=4):
        super().__init__()
        print("Initializing SAMDataModule")
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.sam = sam_model_registry["vit_h"](checkpoint=config.checkpoint)
        self.sam_transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        self.dataset = None

    def setup(self, stage):
        self.dataset = CustomSAMDataset(self.image_dir, self.mask_dir, self.sam_transform)

    def train_dataloader(self):
        return DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=1
        )

class SAMLightningModule(pl.LightningModule):
    def __init__(self, sam_model):
        super().__init__()
        self.sam = sam_model
        self.save_hyperparameters()

    def forward(self, x):
        return self.sam(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch

        # Get image embeddings
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(images)

        # Generate masks
        pred_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            multimask_output=False,
        )

        # Calculate loss
        loss = F.binary_cross_entropy_with_logits(pred_masks, masks)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"},
        }


# Training setup
def train_sam_lightning():
    logger = NeptuneLogger(
        api_key=os.environ.get("NEPTUNE_API_TOKEN"),
        project="richbai90/SAM",
        tags=["finetuning"],
    )
    # Initialize model
    sam = sam_model_registry["vit_h"](checkpoint=config.checkpoint)

    # Create Lightning modules
    model = SAMLightningModule(sam)
    data_module = SAMDataModule(config.img_dir, config.mask_dir)

    # Configure trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=10,
        precision=16,  # Mixed precision training
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="train_loss",
                dirpath="checkpoints",
                filename="sam-{epoch:02d}-{train_loss:.2f}",
                save_top_k=3,
                mode="min",
            ),
            pl.callbacks.EarlyStopping(monitor="train_loss", patience=10, mode="min"),
        ],
        logger=logger,
        log_every_n_steps=1,
    )

    # Start training
    trainer.fit(model, data_module)


# Run training
if __name__ == "__main__":
    train_sam_lightning()
