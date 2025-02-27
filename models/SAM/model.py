import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.predictor import SamPredictor
from memory_profiling import *
import numpy as np
import gc

def print_gpu_memory(label="Current"):
    """Print GPU memory usage at a specific point"""
    torch.cuda.synchronize()
    print(f"\n[{label}] Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"[{label}] Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"[{label}] Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

def free_memory():
    """Attempt to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

class SimplifiedSAMLightningModule(pl.LightningModule):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.sam = None  # Initialize later to save memory during setup
        self.predictor = None
        self.save_hyperparameters(ignore=['sam'])
        
    def setup(self, stage):
        # Load the model only when needed
        if self.sam is None:
            print("Loading SAM model...")
            self.sam = sam_model_registry["vit_h"](checkpoint=self.checkpoint_path)
            self.predictor = SamPredictor(self.sam)
            print("SAM model loaded")
            print_gpu_memory("After loading SAM model")

    def forward(self, x):
        return self.sam(x)

    def build_point_grid(self, n_per_side: int, image_shape) -> np.ndarray:
        """Generates a 2D grid of points evenly spaced in image coordinates."""
        h, w = image_shape
        offset_y = h / (2 * n_per_side)
        offset_x = w / (2 * n_per_side)
        
        points_one_side_y = np.linspace(offset_y, h - offset_y, n_per_side)
        points_one_side_x = np.linspace(offset_x, w - offset_x, n_per_side)
        
        points_y = np.tile(points_one_side_y[:, None], (1, n_per_side))
        points_x = np.tile(points_one_side_x[None, :], (n_per_side, 1))
        
        points = np.stack([points_x.flatten(), points_y.flatten()], axis=1)
        return points


    @torch.no_grad
    def _get_predictions(self, img):
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        self.predictor.set_image(img_np)
        
        # Generate point prompts using a grid
        points = self.build_point_grid(4, img_np.shape[:2])  # Using only 4×4 grid for efficiency
        
        # Get transformed points
        transformed_points = self.predictor.transform.apply_coords(points, img_np.shape[:2])
        in_points = torch.as_tensor(transformed_points, device=self.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        
        # Get predictions
        masks_pred, _, _ = self.predictor.predict_torch(
            point_coords=in_points[:, None, :],
            point_labels=in_labels[:, None],
            multimask_output=False,
            return_logits=True
        )
    
        # Get predicted mask and calculate loss
        pred_mask = masks_pred[0]  # First mask prediction
        # Reset predictor to free memory
        self.predictor.reset_image()
        return pred_mask


    def training_step(self, batch, batch_idx):
        images, masks = batch
        batch_loss = 0
        
        # Process each image in the batch
        for i in range(len(images)):
            img = images[i]
            mask = masks[i]
            
            # Print memory usage if first image of first few batches
            if i == 0 and batch_idx < 3:
                print_gpu_memory(f"Before processing image {i} in batch {batch_idx}")
            
            pred_mask = self._get_predictions(img)
            # Convert image to numpy for SAM predictor
            loss = F.binary_cross_entropy_with_logits(pred_mask, mask)
            batch_loss += loss
            
            
            # Attempt to free GPU memory
            if i == len(images) - 1 and batch_idx < 3:
                free_memory()
                print_gpu_memory(f"After processing batch {batch_idx}")
        
        # Average loss over batch
        avg_loss = batch_loss / len(images)
        self.log("train_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return avg_loss

    def configure_optimizers(self):
        # Freeze the image encoder to reduce memory and training time
        # This dramatically reduces memory usage as the image encoder is the largest part
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
            
        # Only optimize the mask decoder parameters
        optimizer = torch.optim.Adam(
            [p for p in self.sam.mask_decoder.parameters() if p.requires_grad],
            lr=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"},
        }
