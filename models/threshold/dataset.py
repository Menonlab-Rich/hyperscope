import os
from typing import Callable, List, Tuple

import kornia
import numpy as np
import pytorch_lightning as pl
import skimage.morphology
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from transformers import AutoImageProcessor

# --- 1. Feature Extractor Pipeline (Your logic, unchanged but with minor notes) ---
# This part remains the same as it contains your core domain-specific logic.


class MedianFilter(nn.Module):
    """Differentiable Median Filter using Kornia."""

    def __init__(self, kernel_size=(3, 3), p=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.p = p

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Kornia expects a 4D tensor (B, C, H, W)
        p = self.p
        if not np.random.choice([True, False], p=[p, 1 - p]):
            return data
        is_3d = data.dim() == 3
        if is_3d:
            data = data.unsqueeze(0)

        filtered = kornia.filters.median_blur(data, self.kernel_size)

        if is_3d:
            filtered = filtered.squeeze(0)
        return filtered

class FeatureExtractorPipeline(nn.Module):
    """
    A further optimized feature extraction pipeline.
    """

    def __init__(
        self,
        vertical_ksize=(5, 1),
        horizontal_ksize=(1, 5),
        intensity_std_devs=1.0,
        opening_kernel_size=2,
    ):
        super().__init__()
        self.vertical_filter = MedianFilter(vertical_ksize)
        self.horizontal_filter = MedianFilter(horizontal_ksize)
        self.intensity_std_devs = intensity_std_devs
        
        # Register the kernel as a buffer. This automatically handles device placement.
        self.register_buffer('opening_kernel', torch.ones(opening_kernel_size, opening_kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add a batch dimension if it's not there for consistent processing
        is_3d = x.dim() == 3
        if is_3d:
            x = x.unsqueeze(0)

        # 1. Normalize the image (only once if possible)
        x_norm = kornia.enhance.normalize_min_max(x)

        # 2. Apply median filters (additive is often faster than sequential)
        filtered_h = self.vertical_filter(x_norm)
        filtered_w = self.horizontal_filter(x_norm)
        combined_background = filtered_h + filtered_w

        # 3. Create mask and get foreground pixels directly
        mask = combined_background != 0
        foreground_pixels = x_norm[mask]

        if foreground_pixels.numel() < 2:
            # Return a black image of the correct dimension
            return torch.zeros_like(x.squeeze(0) if is_3d else x)

        # 4. More efficient normalization and thresholding
        mean = foreground_pixels.mean()
        std = foreground_pixels.std()
        
        # Calculate threshold directly from original foreground pixels
        # This avoids creating a large intermediate 'normalized_img' tensor
        threshold_val = self.intensity_std_devs * std + mean
        
        # Create binary mask by thresholding original foreground pixels
        # and placing them into a new mask tensor
        binary_mask = torch.zeros_like(x_norm, dtype=torch.bool)
        binary_mask[mask] = foreground_pixels >= threshold_val
        binary_mask = binary_mask.to(x.dtype)

        # 5. Perform morphological opening
        img_opened = kornia.morphology.opening(binary_mask, self.opening_kernel)

        # 6. Apply the final mask
        # No need for a final normalization as the input is already normalized
        final_img = x_norm * img_opened

        # Remove the batch dimension if we added it
        if is_3d:
            final_img = final_img.squeeze(0)

        return final_img

class UnsupervisedNPYDataset(Dataset):
    """
    Creates multiple views of an image for unsupervised learning using a provided
    list of transformation pipelines.
    """

    def __init__(
        self,
        image_paths: List[str],
        processor: "AutoImageProcessor",
        view_pipelines: List[Callable],
    ):
        self.image_paths = image_paths
        self.processor = processor
        self.view_pipelines = view_pipelines
        # Ensure at least two views are generated
        assert len(self.view_pipelines) >= 2, "At least two view pipelines are required."

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image_as_tensor(self, path):
        """Loads a .npy file and converts it directly to a PyTorch tensor."""
        data = np.load(path)
        # Normalize to [0, 1] range
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
        else:
            data = data.astype(np.float32)

        # Ensure tensor has a channel dimension (C, H, W)
        tensor = torch.from_numpy(data)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        image_path = self.image_paths[idx]
        # Load the image once as a tensor
        base_tensor = self._load_image_as_tensor(image_path)

        # Generate each view by applying its corresponding pipeline
        views = [pipeline(base_tensor) for pipeline in self.view_pipelines]

        # Process all views for the model (resizing, normalizing, etc.)
        processed_views = []
        for view in views:
            # The model expects a 3-channel image, so we repeat the single channel if needed.
            if view.ndim == 4:
                view = view.squeeze(0)

            if view.shape[0] == 1:
                if view.ndim == 2:
                    view = view.unsqueeze(0)
                view = view.repeat(3, 1, 1)

            processed_view = self.processor(
                images=kornia.enhance.normalize_min_max(view),
                return_tensors="pt",
                do_rescale=False,  # We handle normalization manually
                do_resize=True,  # Let the processor handle resizing to the model's expected input
            )["pixel_values"].squeeze(0)
            processed_views.append(processed_view)

        return processed_views


# --- 3. Refactored PyTorch Lightning DataModule ---


class NPYDataModule(pl.LightningDataModule):
    """
    Manages the UnsupervisedNPYDataset, defining the different view
    generation pipelines and creating DataLoaders.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 4,
        num_workers: int = 0,
        image_size: Tuple[int, int] = (224, 224),
        processor_model_name: str = "microsoft/swin-base-patch4-window7-224-in22k",
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        n_views: int = 2,  # Total number of views to generate
    ):
        super().__init__()
        self.save_hyperparameters()
        self.processor = AutoImageProcessor.from_pretrained(processor_model_name)
        # The processor will resize images to the model's expected input size
        self.processor.size = {"height": image_size[0], "width": image_size[1]}

        # --- Define View Pipelines ---
        # This list will contain all the different ways to create a view.
        self.view_pipelines = []

        # Define a standard augmentation pipeline using Kornia for GPU acceleration
        # This will be used for n_views - 1 views.
        standard_view_transform = nn.Sequential(
            kornia.augmentation.Normalize(mean=0., std=1., p=1.0),
            kornia.augmentation.ColorJiggle(brightness=0.4, contrast=0.4, p=0.8),
            kornia.augmentation.RandomClahe(p=0.7),
            kornia.augmentation.RandomGaussianNoise(p=0.3),
            MedianFilter(p=0.5),
        )

        # Add the standard pipeline for n_views - 1 views
        for _ in range(self.hparams.n_views):
            self.view_pipelines.append(standard_view_transform)

    def setup(self, stage: str = None):
        all_files = [
            os.path.join(self.hparams.data_dir, f)
            for f in os.listdir(self.hparams.data_dir)
            if f.endswith(".npy")
        ]
        if not all_files:
            raise FileNotFoundError(f"No .npy files found in {self.hparams.data_dir}")

        train_set_files, val_set_files, test_set_files = random_split(
            all_files,
            self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(self.hparams.seed),
        )

        dataset_args = {
            "processor": self.processor,
            "view_pipelines": self.view_pipelines,
        }

        if stage == "fit" or stage is None:
            self.train_dataset = UnsupervisedNPYDataset(train_set_files, **dataset_args)
            self.val_dataset = UnsupervisedNPYDataset(val_set_files, **dataset_args)
        if stage == "test" or stage is None:
            self.test_dataset = UnsupervisedNPYDataset(test_set_files, **dataset_args)

    # --- DataLoader methods remain unchanged ---
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
        )
