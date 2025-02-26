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

class PatchProcessor:
    def __init__(self, patch_size=256, overlap=32):
        """
        Initialize patch processor with specified patch size and overlap
        
        Args:
            patch_size: Size of square patches to extract
            overlap: Overlap between adjacent patches
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
    
    def extract_patches(self, image, mask=None):
        """
        Extract overlapping patches from image and optionally mask
        
        Args:
            image: Input image (H, W, C)
            mask: Optional corresponding mask (H, W)
            
        Returns:
            patches: List of image patches
            mask_patches: List of mask patches (if mask provided)
            patch_coords: List of (y, x) coordinates for each patch
            
        Raises:
            ValueError: If image dimensions are smaller than patch size
            AssertionError: If extracted patch dimensions don't match expected size
        """
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        # Input validation
        if h < self.patch_size or w < self.patch_size:
            raise ValueError(f"Image dimensions ({h}, {w}) must be >= patch size {self.patch_size}")
            
        patches = []
        mask_patches = [] if mask is not None else None
        patch_coords = []
        
        # Calculate number of patches in each dimension
        n_h = (h - self.patch_size + self.stride) // self.stride 
        n_w = (w - self.patch_size + self.stride) // self.stride
        
        # Handle edge cases by adding one more patch if there's remaining space
        if h > n_h * self.stride:
            n_h += 1
        if w > n_w * self.stride:
            n_w += 1
            
        expected_patch_shape = (self.patch_size, self.patch_size, channels) if channels > 1 else (self.patch_size, self.patch_size)
        expected_mask_shape = (self.patch_size, self.patch_size, 1)
            
        for i in range(n_h):
            for j in range(n_w):
                # Calculate patch coordinates with boundary handling
                y = min(i * self.stride, h - self.patch_size)
                x = min(j * self.stride, w - self.patch_size)
                
                # Extract patches
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                
                # Assert patch shape
                assert patch.shape == expected_patch_shape, \
                    f"Patch shape {patch.shape} does not match expected shape {expected_patch_shape}"
                    
                patches.append(patch)
                patch_coords.append((y, x))
                
                if mask is not None:
                    mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                    # Assert mask patch shape
                    assert mask_patch.shape == expected_mask_shape, \
                        f"Mask patch shape {mask_patch.shape} does not match expected shape {expected_mask_shape}"
                    mask_patches.append(mask_patch)
        
        return patches, mask_patches, patch_coords
    
    def merge_patches(self, patches, patch_coords, original_shape, is_mask=False):
        """
        Merge patches back to full image with blending in overlapping regions
        
        Args:
            patches: List of patches (numpy arrays)
            patch_coords: List of (y, x) coordinates for each patch
            original_shape: Shape of original image (H, W, C) or (H, W) for masks
            is_mask: Boolean flag for mask processing
            
        Returns:
            Reconstructed image/mask
        """
        h, w = original_shape[:2]
        
        # Initialize output image and weight map for blending
        c = original_shape[2] if len(original_shape) > 2 else 1
        merged = np.zeros((h, w, c), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)
        
        # Create weight kernel for smooth blending in overlap regions
        weight_kernel = self._create_blending_weights()
        
        for patch_idx, ((y, x), patch) in enumerate(zip(patch_coords, patches)):
            # Handle mask patches differently from image patches
            if is_mask:
                # Make sure patch is 2D for masks
                if len(patch.shape) > 2:
                    # If mask has extra dimensions, flatten them
                    if len(patch.shape) == 3 and patch.shape[0] == 1:  # Shape like (1, H, W)
                        patch = patch[0]
                    else:
                        # Try to get a 2D version - use the first channel if multiple
                        patch = patch.reshape(patch.shape[-2:])
            else:
                # For non-mask patches (images), ensure correct dimensions
                if len(patch.shape) == 2:
                    # If image is 2D but should be 3D
                    patch = patch[..., np.newaxis]
            
            patch_h, patch_w = patch.shape[:2]
            
            # Create weight mask for this patch (2D)
            patch_weight = weight_kernel[:patch_h, :patch_w]

           # Add patch to output with weighting
            if is_mask:
                merged[y:y+patch_h, x:x+patch_w] += patch * patch_weight
                weights[y:y+patch_h, x:x+patch_w] += patch_weight
            else:
                # For non-mask patches, expand dims for broadcasting
                merged[y:y+patch_h, x:x+patch_w] += patch * patch_weight[:, :, np.newaxis]
                weights[y:y+patch_h, x:x+patch_w] += patch_weight     
        
        # Normalize by weights to get final output
        # Avoid division by zero
        weights = np.maximum(weights, 1e-10)
        
        if is_mask:
            merged /= weights
        else:
            merged /= weights[:, :, np.newaxis]
        
        # Convert to appropriate type for masks
        if is_mask:
            merged = (merged > 0.5).astype(np.uint8)
            
        return merged
    
    def _create_blending_weights(self):
        """Create weight kernel for blending overlapping patches"""
        # Linear weighting from center (1.0) to edges (0.2)
        y = np.linspace(0.2, 1.0, self.patch_size // 2)
        center_weight = np.concatenate([y, np.flip(y)])
        
        # Create 2D weight matrix
        xx, yy = np.meshgrid(center_weight, center_weight)
        weights = np.minimum(xx, yy)
        
        return weights

class SAMDataModule(pl.LightningDataModule):
    def __init__(self, image_dir, mask_dir, batch_size=4, patch_size=256, patch_overlap=32):
        super().__init__()
        print("Initializing SAMDataModule")
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.sam = sam_model_registry["vit_h"](checkpoint=config.checkpoint)
        self.sam_transform = Transform(self.sam)
        self.patch_processor = PatchProcessor(patch_size=patch_size, overlap=patch_overlap)
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
    def __init__(self, sam_model, patch_size=256, patch_overlap=32):
        super().__init__()
        self.sam = sam_model
        self.patch_processor = PatchProcessor(patch_size=patch_size, overlap=patch_overlap)
        self.save_hyperparameters(ignore=['sam_model'])
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
        images, masks = batch
        batch_loss = 0
        
        for img, mask in zip(images, masks):
            img_np = img.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format for patch extraction
            mask_np = mask.cpu().numpy().transpose(1,2,0) # Convert to HWC format for patch extraction
            
            # Extract patches
            img_patches, mask_patches, patch_coords = self.patch_processor.extract_patches(img_np, mask_np)
            
            patch_pred_masks = []
            
            # Process each patch
            for img_patch, mask_patch in zip(img_patches, mask_patches):
                # Convert patch back to tensor format
                img_patch_tensor = torch.from_numpy(img_patch.transpose(2, 0, 1)).to(self.device)
                
                # Use predict_torch method directly
                with torch.no_grad():
                    # Set image for predictor
                    self.predictor.set_image(img_patch_tensor)
                    
                    # Create point grid for prompting
                    points = self.build_point_grid(8)  # Reduced grid size for patches
                    points_scale = np.array(img_patch.shape[:2])[None, ::-1]
                    points = points * points_scale
                    
                    # Get transformed points and labels
                    transformed_points = self.predictor.transform.apply_coords(points, img_patch.shape[:2])
                    in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
                    in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
                    
                    # Get predictions using the predictor's API
                    masks, scores, _ = self.predictor.predict_torch(
                        point_coords=in_points[:, None, :],
                        point_labels=in_labels[:, None],
                        multimask_output=False,
                        return_logits=True
                    )
                    
                    # Reset predictor
                    self.predictor.reset_image()
                
                    # Get the predicted mask and ensure it's 2D
                    pred_mask = masks[0].cpu().numpy()  # Shape should be (H, W)
                
                    # Store the prediction
                    patch_pred_masks.append(pred_mask)
            
            # Merge predicted mask patches
            merged_pred_mask = self.patch_processor.merge_patches(
                patch_pred_masks, 
                patch_coords, 
                mask_np.shape, 
                is_mask=True
            )
            
            # Convert back to tensor for loss computation
            merged_pred_mask_tensor = torch.from_numpy(merged_pred_mask).float().to(self.device)
            
            # Calculate loss for this image
            image_loss = F.binary_cross_entropy_with_logits(
                merged_pred_mask_tensor.unsqueeze(0), 
                mask.unsqueeze(0)
            )
            
            batch_loss += image_loss
        
        # Average loss over batch
        avg_loss = batch_loss / len(images)
        
        # Log metrics
        self.log("train_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return avg_loss

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
def train_sam_lightning(patch_size=256, patch_overlap=32):
    logger = NeptuneLogger(
        api_key=os.environ.get("NEPTUNE_API_TOKEN"),
        project="richbai90/SAM",
        tags=["finetuning", "patch-processing"],
    )
    
    # Initialize model
    sam = sam_model_registry["vit_h"](checkpoint=config.checkpoint)

    # Create Lightning modules
    model = SAMLightningModule(sam, patch_size=patch_size, patch_overlap=patch_overlap)
    data_module = SAMDataModule(
        config.img_dir, 
        config.mask_dir, 
        patch_size=patch_size, 
        patch_overlap=patch_overlap
    )

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
                filename="sam-patched-{epoch:02d}-{train_loss:.2f}",
                save_top_k=3,
                mode="min",
            ),
            pl.callbacks.EarlyStopping(monitor="train_loss", patience=10, mode="min"),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=1.0,  # Add gradient clipping for stability
    )

    # Start training
    trainer.fit(model, data_module)


# Run training
if __name__ == "__main__":
    # You can adjust these parameters based on your GPU memory
    # For even less memory usage, try reducing patch_size to 128
    train_sam_lightning(patch_size=128, patch_overlap=16)
