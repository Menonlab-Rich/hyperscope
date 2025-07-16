import os
from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
from transformers import AutoImageProcessor
import kornia
import skimage.morphology

## -------------------------------------------------------------------
## 1. Custom Transform
## -------------------------------------------------------------------

# Helper module from our previous discussion
class MedianFilterTransform(nn.Module):
    def __init__(self, kernel_size=(3, 3)):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return kornia.filters.median_blur(data, self.kernel_size)

# The new all-in-one pipeline module
class FeatureExtractorPipeline(nn.Module):
    """
    Encapsulates the entire feature extraction workflow in a single module.
    """
    def __init__(self, vertical_ksize=(51, 1), horizontal_ksize=(1, 31),
                 intensity_std_devs=2.0, area_threshold=16):
        super().__init__()
        # 1. Initialize the filter layers
        self.vertical_filter = MedianFilterTransform(vertical_ksize)
        self.horizontal_filter = MedianFilterTransform(horizontal_ksize)
        
        # 2. Store thresholding parameters
        self.intensity_std_devs = intensity_std_devs
        self.area_threshold = area_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the full pipeline in a single forward pass.
        """
        # Step 1: Apply directional median filters
        filtered_h = self.vertical_filter(x)
        filtered_w = self.horizontal_filter(x)

        # Step 2: Create a mask from the combined backgrounds
        combined_background = filtered_h + filtered_w
        mask = combined_background != 0

        # Step 3: Select pixels from the original image using the mask
        selected_pixels = torch.zeros_like(x)
        selected_pixels[mask] = x[mask]

        # Step 4: Normalize the selected pixels
        non_zero_elements = selected_pixels[mask]
        # Add a small epsilon to std to prevent division by zero
        mean, std = non_zero_elements.mean(), non_zero_elements.std()
        normalized_img = (selected_pixels - mean) / (std + 1e-8)
        
        # Make sure the background remains zero after normalization
        normalized_img[~mask] = 0

        # Step 5: Apply intensity thresholding
        threshold_val = self.intensity_std_devs
        # (For a normalized distribution, the threshold is just the number of std devs)
        normalized_img[normalized_img < threshold_val] = 0

        # Step 6: Apply area opening
        # WARNING: This step requires converting to NumPy and running on the CPU.
        # It makes this module non-differentiable and unsuitable for GPU-based training.
        img_for_skimage = normalized_img.squeeze().cpu().numpy()
        img_opened = skimage.morphology.area_opening(
            img_for_skimage, area_threshold=self.area_threshold
        )
        
        # Convert back to a tensor on the original device
        final_tensor = torch.from_numpy(img_opened).to(x.device).unsqueeze(0).unsqueeze(0)

        return final_tensor

## -------------------------------------------------------------------
## 2. The Unsupervised Dataset Class
## -------------------------------------------------------------------
class UnsupervisedNPYDataset(Dataset):
    """
    Creates two views of an image for unsupervised learning.
    Intelligently uses paired event camera data for the second view when available,
    and falls back to a strong augmentation when not.
    """

    def __init__(
        self,
        image_paths: List[str],
        processor: "AutoImageProcessor",
        event_camera_dir: str,
        spatial_transform: transforms.Compose,
        photometric_transform: transforms.Compose,
        fallback_transform: object,
    ):
        self.image_paths = image_paths
        self.processor = processor
        self.event_camera_dir = event_camera_dir
        self.spatial_transform = spatial_transform
        self.photometric_transform = photometric_transform
        self.fallback_transform = fallback_transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_and_prepare_image(self, path):
        data = np.load(path)
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
        else:
            data = data.astype(np.float32)
        
        if data.ndim == 2:
            img = Image.fromarray((data * 255).astype(np.uint8)).convert("RGB")
        elif data.shape[2] == 1:
            img = Image.fromarray((data.squeeze(2) * 255).astype(np.uint8)).convert("RGB")
        else:
            img = Image.fromarray((data * 255).astype(np.uint8))
        return self.spatial_transform(img)

    def _do_rescale(self, img: torch.Tensor):
        rescale = not (img.min() == 0 and img.max() == 1)
        return rescale 

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        image_path = self.image_paths[idx]
        pil_image = self._load_and_prepare_image(image_path)

        view1_pil = self.photometric_transform(pil_image)
        processed_view1 = self.processor(
            images=view1_pil, return_tensors="pt", do_rescale=self._do_rescale(pil_to_tensor(view1_pil))
        )["pixel_values"].squeeze(0)

        view2_tensor = None
        event_image_path = os.path.join(
            self.event_camera_dir or "", os.path.basename(image_path)
        )

        if self.event_camera_dir and os.path.exists(event_image_path):
            event_pil = self._load_and_prepare_image(event_image_path)
            view2_tensor = transforms.ToTensor()(self.spatial_transform(event_pil))
        else:
            image_tensor = transforms.ToTensor()(pil_image)
            view2_tensor = self.fallback_transform(image_tensor)

        if view2_tensor.shape[0] == 1:
            view2_tensor = view2_tensor.repeat(3, 1, 1)

        processed_view2 = self.processor(
            images=view2_tensor, return_tensors="pt", do_resize=False, do_rescale=self._do_rescale(view2_tensor)
        )["pixel_values"].squeeze(0)

        return processed_view1, processed_view2


## -------------------------------------------------------------------
## 3. The Restored PyTorch Lightning DataModule
## -------------------------------------------------------------------
class NPYDataModule(pl.LightningDataModule):
    """
    The PyTorch Lightning DataModule that manages the UnsupervisedNPYDataset,
    including splitting data and creating DataLoaders.
    """

    def __init__(
        self,
        data_dir: str,
        event_camera_dir: str = None, # Add event camera directory path
        batch_size: int = 4,
        num_workers: int = 0,
        image_size: Tuple[int, int] = (224, 224),
        processor_model_name: str = "microsoft/swin-base-patch4-window7-224-in22k",
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.processor = AutoImageProcessor.from_pretrained(processor_model_name)
        self.processor.size = {"height": image_size[0], "width": image_size[1]}

        # Define transformations here to be saved with the checkpoint
        self.spatial_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
            ]
        )
        self.photometric_transform = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.4, contrast=0.4, hue=0.1),
                transforms.GaussianBlur(
                    kernel_size=(image_size[0] // 20 * 2 + 1), sigma=(0.1, 2.0)
                ),
            ]
        )
        self.fallback_transform = MedianFilterTransform(window_size=8)

    def setup(self, stage: str = None):
        all_files = [
            os.path.join(self.hparams.data_dir, f)
            for f in os.listdir(self.hparams.data_dir)
            if f.endswith(".npy")
        ]
        if not all_files:
            raise FileNotFoundError(f"No .npy files found in {self.hparams.data_dir}")

        train_set, val_set, test_set = random_split(
            all_files,
            self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(self.hparams.seed),
        )

        dataset_args = {
            "processor": self.processor,
            "event_camera_dir": self.hparams.event_camera_dir,
            "spatial_transform": self.spatial_transform,
            "photometric_transform": self.photometric_transform,
            "fallback_transform": self.fallback_transform,
        }

        if stage == "fit" or stage is None:
            self.train_dataset = UnsupervisedNPYDataset(train_set, **dataset_args)
            self.val_dataset = UnsupervisedNPYDataset(val_set, **dataset_args)
        if stage == "test" or stage is None:
            self.test_dataset = UnsupervisedNPYDataset(test_set, **dataset_args)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        # For validation, we use the test set as discussed previously
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
        )
