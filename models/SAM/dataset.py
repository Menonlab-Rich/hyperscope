from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from pathlib import Path
import os
from segment_anything.utils.transforms import ResizeLongestSide
import pdb

class CustomSAMDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super().__init__()
        print("Initializing CustomSAMDataset")
        self.image_paths = sorted([Path(image_dir) / p for p in os.listdir(image_dir) if p.endswith('.tif')])
        self.mask_paths = sorted([Path(mask_dir) / p for p in os.listdir(mask_dir) if p.endswith('.npz')])
        self.transform = transform
        
        print(f"Found {len(self.image_paths)} images and {len(self.mask_paths)} masks")
        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks must match"
        assert len(self.image_paths) > 0, "No images found in directory"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load 16-bit TIF image
        image = Image.open(self.image_paths[idx])

        # Convert to numpy array and normalize to 0-1
        image = np.array(image).astype(np.float32)
        image = image / image.max()  # Normalize 16-bit to 0-1

        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        # Load NPZ mask
        mask = np.load(self.mask_paths[idx])
        mask = mask["arr_0"] if isinstance(mask, np.lib.npyio.NpzFile) else mask
        
        # Convert mask to float32 first
        mask = mask.astype(np.float32)

        # Apply SAM's image transforms
        if self.transform is not None:
            image = self.transform.apply_image(image)
            # Convert mask to tensor and add channel dimension
            mask = torch.from_numpy(mask).float().unsqueeze(0)  # Add channel dimension
            mask = F.interpolate(
                    mask.unsqueeze(0), size=image.shape[:2], mode="nearest"
            ).squeeze(0)

        # Convert image to tensor and adjust dimensions
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC to CHW
        image = self._pad_to_multiple_of_64(image)
        # Ensure mask is float tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()

        mask = self._pad_to_multiple_of_64(mask)

        return image, mask

    def _pad_to_multiple_of_64(self, image):
        h, w = image.shape[-2:]
        # 2nd mod64 handles the case where the dimension is already a multiple of 64
        # avoiding unnescesary padding by converting 64 to 0 in this instance
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        
        # Pad the image to make dimensions multiples of 64
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h))
        return image
