import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import torchvision.transforms as T
from transformers import AutoImageProcessor
from typing import List, Tuple, Union

class UnsupervisedNPYDataset(Dataset):
    """
    A PyTorch Dataset for loading .npy files, converting to square dimensions (e.g., 512x512),
    and generating two augmented views for unsupervised learning.
    Handles various integer and float input dtypes, converting to float32 [0,1].
    The transformation pipelines are passed during initialization.
    """
    def __init__(self, 
                 file_paths: List[str], 
                 processor: AutoImageProcessor, 
                 initial_resize_transform: T.Compose,
                 spatial_transform: T.Compose,
                 photometric_transform_base: T.Compose):
        """
        Args:
            file_paths (List[str]): List of full paths to the .npy image files.
            processor (AutoImageProcessor): Hugging Face AutoImageProcessor for final normalization.
            initial_resize_transform (T.Compose): Transform for initial resizing to square dimensions.
            spatial_transform (T.Compose): Transforms for shared spatial augmentations.
            photometric_transform_base (T.Compose): Transforms for independent photometric augmentations.
        """
        self.file_paths = file_paths
        self.processor = processor
        self.initial_resize_transform = initial_resize_transform
        self.spatial_transform = spatial_transform
        self.photometric_transform_base = photometric_transform_base

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[idx]

        # 1. Load .npy file
        data = np.load(file_path)

        # Handle potential single-channel dimension for consistency
        if data.ndim == 2:
            data = data[..., np.newaxis] # Add channel dim: (H, W) -> (H, W, 1)
        
        # Convert to float32 and normalize to [0, 1] based on original dtype
        if np.issubdtype(data.dtype, np.integer):
            max_val = np.iinfo(data.dtype).max
            data = data.astype(np.float32) / float(max_val)
        elif data.dtype != np.float32: # Ensure other float types are float32
            data = data.astype(np.float32)

        # Convert to PIL Image
        # PIL expects uint8 for Image.fromarray directly unless mode is specified for float,
        # so we convert float32 [0,1] to uint8 [0,255] for PIL.
        if data.shape[2] == 1: # Grayscale image
            pil_image = Image.fromarray((data.squeeze(2) * 255).astype(np.uint8)).convert("RGB")
        elif data.shape[2] == 3: # RGB image
            pil_image = Image.fromarray((data * 255).astype(np.uint8))
        else:
            raise ValueError(f"Unsupported number of channels in .npy data: {data.shape[2]}. Expected 1 or 3.")

        # 2. Initial Resize to 512x512 (this makes the image dimensions square and scales it)
        initial_resized_image = self.initial_resize_transform(pil_image)

        # 3. Apply a single set of spatial transformations
        spatially_augmented_image = self.spatial_transform(initial_resized_image)

        # 4. Apply two INDEPENDENT sets of photometric transformations
        image_view1_pil = self.photometric_transform_base(spatially_augmented_image)
        image_view2_pil = self.photometric_transform_base(spatially_augmented_image)

        # 5. Use the Hugging Face processor for final tensor conversion and normalization
        processed_view1 = self.processor(images=image_view1_pil, return_tensors="pt")["pixel_values"].squeeze(0)
        processed_view2 = self.processor(images=image_view2_pil, return_tensors="pt")["pixel_values"].squeeze(0)
        
        return processed_view1, processed_view2


class NPYDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for managing .npy datasets, including splitting
    and creating DataLoaders for unsupervised contrastive learning.
    Transformations are defined and stored within this DataModule.
    """
    def __init__(self, 
                 data_dir: str, 
                 batch_size: int = 4, 
                 num_workers: int = 0, 
                 image_size: Tuple[int, int] = (512, 512),
                 processor_model_name: str = "microsoft/swin-base-patch4-window7-224-in22k",
                 train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 seed: int = 42):
        super().__init__()
        self.save_hyperparameters() # Saves all init arguments as hyperparameters

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.processor_model_name = processor_model_name
        self.train_val_test_split = train_val_test_split
        self.seed = seed
        
        # Initialize the Hugging Face AutoImageProcessor
        self.processor = AutoImageProcessor.from_pretrained(processor_model_name)
        self.processor.size = {'height': self.image_size[0], 'width': self.image_size[1]}

        # Define and store all transformations here in the DataModule's state
        # These will be saved when the DataModule/Trainer checkpoint is saved.

        # 1. Initial Resize to target_size (makes dimensions square)
        self.initial_resize_transform = T.Resize(self.image_size, interpolation=T.InterpolationMode.BICUBIC)

        # 2. Shared Spatial Augmentations
        self.spatial_transform = T.Compose([
            T.RandomResizedCrop(self.image_size, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            # T.RandomRotation(degrees=15), # Optional: Uncomment for more spatial variation
        ])

        # 3. Independent Photometric Augmentations
        self.photometric_transform_base = T.Compose([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=(self.image_size[0] // 20 // 2 * 2 + 1), sigma=(0.1, 2.0)),
        ])

    def prepare_data(self):
        # This method is for operations that should only happen once per node/GPU,
        # typically downloading or processing data that isn't dependent on rank.
        # Transformations are instantiated in __init__ and passed to datasets in setup.
        pass

    def setup(self, stage: str = None):
        """
        Sets up the datasets for training, validation, and testing.
        Args:
            stage (str): 'fit' (for train/val) or 'test'.
        """
        all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.npy')]
        
        if not all_files:
            raise FileNotFoundError(f"No .npy files found in the specified data directory: {self.data_dir}")

        num_total = len(all_files)
        num_train = int(num_total * self.train_val_test_split[0])
        num_val = int(num_total * self.train_val_test_split[1])
        num_test = num_total - num_train - num_val 

        generator = torch.Generator().manual_seed(self.seed) 
        
        train_indices, val_indices, test_indices = random_split(
            range(num_total),
            [num_train, num_val, num_test], 
            generator=generator
        )
        
        train_file_paths = [all_files[i] for i in train_indices.indices]
        val_file_paths = [all_files[i] for i in val_indices.indices]
        test_file_paths = [all_files[i] for i in test_indices.indices]

        # Pass the instantiated transforms from DataModule to the Dataset
        if stage == "fit" or stage is None:
            self.train_dataset = UnsupervisedNPYDataset(
                train_file_paths, 
                self.processor, 
                self.initial_resize_transform,
                self.spatial_transform,
                self.photometric_transform_base
            )
            self.val_dataset = UnsupervisedNPYDataset(
                val_file_paths, 
                self.processor, 
                self.initial_resize_transform,
                self.spatial_transform,
                self.photometric_transform_base
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = UnsupervisedNPYDataset(
                test_file_paths, 
                self.processor, 
                self.initial_resize_transform,
                self.spatial_transform,
                self.photometric_transform_base
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )

# --- Example Usage (requires creating dummy .npy files) ---
if __name__ == "__main__":
    # Create a dummy data directory and some .npy files for testing
    dummy_data_dir = "dummy_npy_data"
    os.makedirs(dummy_data_dir, exist_ok=True)

    # Create 15 dummy .npy files with mixed uint8 and uint16 types
    # Let's create some non-square dummy images to test the resizing
    for i in range(15):
        h_ = np.random.randint(100, 300) 
        w_ = np.random.randint(100, 300) 
        # Make some images non-square to demonstrate the squaring effect of T.Resize
        if i % 3 == 0:
            h_ = int(h_ * 1.5) # Make height different from width sometimes
        
        if i % 2 == 0: # Even files are uint8
            dummy_data = np.random.randint(0, 256, size=(h_, w_), dtype=np.uint8)
        else: # Odd files are uint16
            dummy_data = np.random.randint(0, 65536, size=(h_, w_), dtype=np.uint16)
        
        np.save(os.path.join(dummy_data_dir, f"image_{i:02d}.npy"), dummy_data)
    
    print(f"Created {len(os.listdir(dummy_data_dir))} dummy .npy files in {dummy_data_dir}")

    # Instantiate the DataModule
    dm = NPYDataModule(
        data_dir=dummy_data_dir,
        batch_size=2,
        num_workers=os.cpu_count() // 2 if os.cpu_count() else 0,
        image_size=(512, 512), # All images will be resized to 512x512
        processor_model_name="microsoft/swin-base-patch4-window7-224-in22k",
        train_val_test_split=(0.7, 0.15, 0.15)
    )

    dm.setup(stage="fit")

    train_dataloader = dm.train_dataloader()
    print(f"\nTraining Dataloader length: {len(train_dataloader)} batches")
    
    for batch_idx, (sample_batch_v1, sample_batch_v2) in enumerate(train_dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Shape of image_view1: {sample_batch_v1.shape}, Dtype: {sample_batch_v1.dtype}")
        print(f"  Shape of image_view2: {sample_batch_v2.shape}, Dtype: {sample_batch_v2.dtype}")
        assert sample_batch_v1.shape == (dm.batch_size, 3, dm.image_size[0], dm.image_size[1])
        assert sample_batch_v2.shape == (dm.batch_size, 3, dm.image_size[0], dm.image_size[1])
        assert sample_batch_v1.dtype == torch.float32
        assert not torch.equal(sample_batch_v1, sample_batch_v2)
        if batch_idx >= 1:
            break

    print("\nData loading with transformations stored in DataModule seems correct!")

    # Optional: Clean up dummy data directory after testing
    # import shutil
    # shutil.rmtree(dummy_data_dir)
