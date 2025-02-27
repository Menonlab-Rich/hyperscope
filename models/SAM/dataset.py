from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from memory_profiling import *
import os
import cv2

class OptimizedSAMDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, max_size=1024):
        """
        Create a memory-efficient dataset for SAM fine-tuning
        
        Args:
            image_dir: Directory containing the images
            mask_dir: Directory containing the masks
            transform: Optional transform function
            max_size: Maximum size for the longest side of images (for memory optimization)
        """
        super().__init__()
        print("Initializing OptimizedSAMDataset")
        self.image_paths = sorted([Path(image_dir) / p for p in os.listdir(image_dir) if p.endswith('.tif')])
        self.mask_paths = sorted([Path(mask_dir) / p for p in os.listdir(mask_dir) if p.endswith('.npz')])
        self.transform = transform
        self.max_size = max_size
        
        print(f"Found {len(self.image_paths)} images and {len(self.mask_paths)} masks")
        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks must match"
        assert len(self.image_paths) > 0, "No images found in directory"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and handle high bit depth
        with Image.open(self.image_paths[idx]) as img:
            # Convert PIL image to numpy array
            image = np.array(img).astype(np.float32)
            
        # Normalize to 0-1
        image = image / max(1.0, image.max())
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
            
        # Resize if image is too large (to save memory)
        if max(image.shape[0], image.shape[1]) > self.max_size:
            scale = self.max_size / max(image.shape[0], image.shape[1])
            new_h, new_w = int(image.shape[0] * scale), int(image.shape[1] * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # Load mask (more memory efficient by loading directly)
        try:
            mask_data = np.load(self.mask_paths[idx])
            mask = mask_data["arr_0"] if "arr_0" in mask_data else next(iter(mask_data.values()))
            mask = mask.astype(np.float32)
            
            # Resize mask to match image if needed
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(f"Error loading mask {self.mask_paths[idx]}: {e}")
            # Create an empty mask in case of error
            mask = np.zeros(image.shape[:2], dtype=np.float32)
            
        # Apply transforms if available
        if self.transform is not None:
            image, mask = self.transform(image, mask)
            
        # Convert to PyTorch tensors
        image = torch.from_numpy(image.copy().transpose(2, 0, 1)).float()  # (C, H, W)
        mask = torch.from_numpy(mask.copy()).float().unsqueeze(0)  # (1, H, W)
                
        return image, mask

# Simple transform class
class SimpleTransform:
    def __init__(self, sam_transform):
        self.sam_transform = sam_transform
        
    def __call__(self, image, mask):
        # Apply SAM's transform to the image
        transformed_image = self.sam_transform.apply_image(image)
        
        # Resize mask to match the transformed image size
        h, w = transformed_image.shape[:2]
        transformed_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return self._make_square(transformed_image), self._make_square(transformed_mask)


    def _make_square(self, image):
        """Pads an image with zeros to make it square.

        Args:
            image: A NumPy array representing the image (height, width, channels).

        Returns:
            A new NumPy array representing the padded square image.
        """
        height, width = image.shape[:2]
        max_dim = max(height, width)
        
        pad_top = (max_dim - height) // 2
        pad_bottom = max_dim - height - pad_top
        pad_left = (max_dim - width) // 2
        pad_right = max_dim - width - pad_left
        
        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)) if image.ndim == 3 else ((pad_top, pad_bottom), (pad_left, pad_right))
        
        return np.pad(image, padding, mode='constant', constant_values=0) 
