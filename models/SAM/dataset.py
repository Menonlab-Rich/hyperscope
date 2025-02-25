from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from pathlib import Path
import os


class CustomSAMDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted([Path(image_dir) / p for p in os.listdir(image_dir) if p.endswith('.tif')])
        self.mask_paths = sorted([Path(mask_dir) / p for p in os.listdir(mask_dir) if p.endswith('.npz')])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load 16-bit TIF image
        image = Image.open(self.image_paths[idx])

        # Convert to numpy array and normalize to 0-1
        image = np.array(image).astype(np.float32)
        image = image / image.max()  # Normalize 16-bit to 0-1
        image = image.astype(np.float16)

        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        # Load NPZ mask
        mask = np.load(self.mask_paths[idx])
        mask = mask["arr_0"] if isinstance(mask, np.lib.npyio.NpzFile) else mask
        mask = mask.astype(np.float16)

        # Apply SAM's image transforms
        if self.transform is not None:
            image = self.transform.apply_image(image)
            # The mask should be resized to match the transformed image
            mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
            mask = F.interpolate(
                mask.unsqueeze(0), size=image.shape[:2], mode="nearest"  # Add batch dimension
            ).squeeze(0)

        # Convert image to tensor and adjust dimensions
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW

        return image, mask
