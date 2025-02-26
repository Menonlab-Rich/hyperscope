import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.predictor import SamPredictor
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from dataset import CustomSAMDataset
from config import config
import os
import cv2
import numpy as np

class Transform:
    def __init__(self, sam):
        self.sam_transform = ResizeLongestSide(sam.image_encoder.img_size)

    def pad_to_square(self, image):
        if len(image.shape) == 3:
            h, w, c = image.shape
            max_dim = max(h, w)
            new_img = np.zeros((max_dim, max_dim, c), dtype=image.dtype)
        else:  # Handle 2D mask
            h, w = image.shape
            max_dim = max(h, w)
            new_img = np.zeros((max_dim, max_dim), dtype=image.dtype)
    
        # Calculate padding
        pad_h = (max_dim - h) // 2
        pad_w = (max_dim - w) // 2
    
        # Place the image/mask in the center of the padded square
        new_img[pad_h:pad_h+h, pad_w:pad_w+w] = image
    
        return new_img

    def apply_image(self, image, mask):
        # First resize
        img = self.sam_transform.apply_image(image)
        h, w = img.shape[:2]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        # Then pad the resized image
        img = self.pad_to_square(img)
        mask = self.pad_to_square(mask)
        return img, mask

class SAMDataModule(pl.LightningDataModule):
    def __init__(self, image_dir, mask_dir, batch_size=4):
        super().__init__()
        print("Initializing SAMDataModule")
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.sam = sam_model_registry["vit_h"](checkpoint=config.checkpoint)
        self.sam_transform = Transform(self.sam)
        self.dataset = None

    def setup(self, stage):
        self.dataset = CustomSAMDataset(self.image_dir, self.mask_dir, self.sam_transform)

    def train_dataloader(self):
        return DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4
        )

class SAMLightningModule(pl.LightningModule):
    def __init__(self, sam_model):
        super().__init__()
        self.sam = sam_model
        self.save_hyperparameters()
        self.predictor = SamPredictor(self.sam)

    def forward(self, x):
        return self.sam(x)

    def build_point_grid(self, n_per_side: int) -> np.ndarray:
        """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
        offset = 1 / (2 * n_per_side)
        points_one_side = np.linspace(offset, 1 - offset, n_per_side)
        points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
        points_y = np.tile(points_one_side[:, None], (1, n_per_side))
        points = np.stack([points_x.flatten(), points_y.flatten()], axis=1)
        return points

    def training_step(self, batch, batch_idx):
        images, masks =  batch
        pred_masks = []
        for img, mask in zip(images,masks):
            with torch.no_grad():
                self.predictor.set_image(img)
                points = self.build_point_grid(32)
                points_scale = np.array(img.shape[:2])[None, ::-1]
                points = points * points_scale
                # Get transformed points and labels for prompting
                transformed_points = self.predictor.transform.apply_coords(points, img.shape[:2])
                in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
                in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)

                # Get predictions using the predictor
                masks, iou_preds, _ = self.predictor.predict_torch(in_points[:, None, :], in_labels[:, None], multimask_output=True, return_logits=True,)

                # reset the image when done
                self.predictor.reset_image()
                # Generate masks
                pred_mask, _ = self.sam.mask_decoder(
                    image_embeddings=image_embeddings,
                    multimask_output=False,
                )

                pred_masks.append(pred_mask)

        # Calculate loss
        loss = F.binary_cross_entropy_with_logits(torch.tensor(pred_masks), masks)

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
