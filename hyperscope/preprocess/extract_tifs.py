# Previous imports...
import pdb
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import cv2
from loguru import logger
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import supervision as sv
from supervision.annotators.utils import ColorLookup
from typing import List, Optional, Dict, Tuple
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence
import glob
from hyperscope import config
from hyperscope.helpers.func import flatten
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(
        self,
        paths: List[Path],
        darkframe: Optional[np.ndarray] = None,
        crop: Optional[List[int]] = None,
        max_size: int = 1024,  # Maximum dimension size
    ):
        self.paths = paths
        self.darkframe = darkframe if darkframe is not None else None
        self.crop = crop
        self.max_size = max_size

    def __len__(self):
        return len(self.paths)

    def resize_maintain_aspect(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Resize image maintaining aspect ratio"""
        h, w = image.shape
        original_size = (h, w)

        # Calculate new dimensions
        if h > w:
            new_h = self.max_size
            new_w = int(w * (self.max_size / h))
        else:
            new_w = self.max_size
            new_h = int(h * (self.max_size / w))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, original_size

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = np.load(img_path).astype(np.uint16)
        original_size = img.shape
        darkframe = None

        if self.crop:
            x1, y1, x2, y2 = self.crop
            img = img[y1:y2, x1:x2]
            if self.darkframe is not None:
                darkframe = self.darkframe[y1:y2, x1:x2]

        # Initial normalization
        img = self.minmax_norm(img)

        # Darkframe correction if available
        if darkframe is not None:
            img -= darkframe

        # Resize image
        img_resized, _ = self.resize_maintain_aspect(img)

        # Convert to RGB format expected by SAM
        img_resized = self.minmax_norm(img_resized, 0, 255, np.uint8)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

        # Convert to tensor
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()

        return {
            "image": img_tensor,
            "path": img_path,
            "original": torch.from_numpy(img),
            "original_size": original_size,
        }

    @staticmethod
    def collate(batch):
        """
        Custom collate function for the ImageDataset
        Args:
            batch: list of dictionary items from dataset
        """
        # Initialize empty lists for each key
        images = []
        paths = []
        originals = []
        original_sizes = []

        # Collect items from batch
        for item in batch:
            images.append(item["image"])
            paths.append(item["path"])
            originals.append(item["original"])
            original_sizes.append(item["original_size"])

        # Stack tensors where appropriate
        images = torch.stack(images, dim=0)
        originals = torch.stack(originals, dim=0)

        return {
            "image": images,
            "path": paths,  # Keep as list
            "original": originals,
            "original_size": original_sizes,  # Keep as list
        }

    @staticmethod
    def minmax_norm(image, minval=0, maxval=1, dtype=np.float32):
        return np.clip(
            ((image - image.min()) / (image.max() - image.min()) * (maxval - minval)) + minval,
            minval,
            maxval,
        ).astype(dtype)


class ImageProcessor(nn.Module):
    def __init__(self, sam_model_type: str = "vit_h", checkpoint_path: Optional[Path] = None):
        super().__init__()
        self.sam_model_type = sam_model_type
        self.checkpoint_path = checkpoint_path or config.MODELS_DIR / "sam_vit_h_4b8939.pth"
        self.mask_generator = None

    def to(self, device):
        if self.mask_generator is None:
            sam = sam_model_registry[self.sam_model_type](checkpoint=self.checkpoint_path)
            sam.to(device=device)
            self.mask_generator = SamAutomaticMaskGenerator(sam)
        return super().to(device)

    def resize_mask(self, mask: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """Resize mask back to original image size"""
        return cv2.resize(
            mask.astype(np.uint8),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = batch["image"]
        original_sizes = batch["original_size"]
        batch_size = images.shape[0]
        processed_masks = []
        signal_percentages = []

        for i in range(batch_size):
            image = images[i].permute(1, 2, 0).cpu().numpy()
            if image.max() <= 1.0:
                image *= 255
            image = image.astype(np.uint8)
            original_size = original_sizes[i]

            # Generate masks using SAM on resized image
            result = self.mask_generator.generate(image)
            detections = sv.Detections.from_sam(result)

            # Process labels
            labels = ~detections.mask[0]

            # Resize mask back to original size
            labels_resized = self.resize_mask(labels, original_size)

            sig_percent = torch.tensor(np.sum(labels_resized) / labels_resized.size)

            mask_tensor = torch.from_numpy(labels_resized).to(images.device)
            processed_masks.append(mask_tensor)
            signal_percentages.append(sig_percent)

        return torch.stack(processed_masks), torch.stack(signal_percentages)


def setup_process(local_rank: int, world_size: int):
    """Setup process group for distributed processing"""
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    return local_rank if world_size > 1 else 0


def get_dataloader(
    dataset: Dataset, world_size: int, local_rank: int, batch_size: int
) -> DataLoader:
    """Create appropriate dataloader based on number of GPUs"""
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=local_rank
        )
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=0,
        pin_memory=True,
        collate_fn=ImageDataset.collate,
    )


def process_images(
    local_rank: int,
    world_size: int,
    dataset: Dataset,
    output_dir: Path,
    catmap: dict,
    batch_size: int = 4,
    verify: bool = False,
):
    """Process images with automatic handling of distributed vs single-GPU cases"""
    device = setup_process(local_rank, world_size)

    dataloader = get_dataloader(dataset, world_size, local_rank, batch_size)

    processor = ImageProcessor().to(device)
    if world_size > 1:
        processor = DDP(processor, device_ids=[local_rank])

    visualizer = VerificationVisualizer() if verify else None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing Batches", unit="batch"):
            batch["image"] = batch["image"].to(device)

            masks, signal_percentages = processor(batch)

            if local_rank == 0 or world_size == 1:
                save_batch_results(
                    batch, masks, signal_percentages, output_dir, catmap, verify, visualizer
                )

    if world_size > 1:
        dist.destroy_process_group()


class VerificationVisualizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_annotator = sv.MaskAnnotator(color_lookup=ColorLookup.INDEX)

    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Generate verification visualizations for a batch of images and masks

        Args:
            images: Batch of images [B, C, H, W]
            masks: Batch of masks [B, H, W]

        Returns:
            Batch of annotated images [B, H, W, C]
        """
        batch_size = images.shape[0]
        annotated_images = []

        for i in range(batch_size):
            image = images[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            mask = masks[i].cpu().numpy()

            # Create fake detections object for visualization
            detections = sv.Detections(
                mask=np.expand_dims(~mask, 0),  # Invert mask back for visualization
                xyxy=np.array([[0, 0, image.shape[1], image.shape[0]]]),
                confidence=np.array([1.0]),
            )

            annotated_image = self.mask_annotator.annotate(
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR), detections
            )
            annotated_images.append(torch.from_numpy(annotated_image))

        return torch.stack(annotated_images)


def save_batch_results(
    batch: Dict[str, torch.Tensor],
    masks: torch.Tensor,
    signal_percentages: torch.Tensor,
    output_dir: Path,
    catmap: dict,
    verify: bool = False,
    visualizer: Optional[VerificationVisualizer] = None,
):
    """Save processed results for a batch"""
    batch_size = masks.shape[0]

    for i in range(batch_size):
        sig_percent = signal_percentages[i].item()
        if sig_percent < 0.005 or sig_percent > 0.1:
            continue

        img_path = batch["path"][i]
        img_name = img_path.stem
        signal_key = img_name.split("_")[0]

        # Save verification plot if requested
        if verify and visualizer is not None:
            annotated_image = visualizer(batch["image"][i].unsqueeze(0), masks[i].unsqueeze(0))

            plt.imsave(
                str(output_dir / "verification" / f"{img_name}.png"),
                annotated_image[0].cpu().numpy(),
            )

        # Save results
        labels = masks[i].cpu().numpy().astype(np.uint8)
        labels[labels == 1] = catmap.get(signal_key, 0)

        cv2.imwrite(
            str(output_dir / "imgs" / f"{img_name}.tif"), batch["original"][i].cpu().numpy()
        )
        np.savez(str(output_dir / "masks" / f"{img_name}.npz"), labels)


def process_paths(
    input_patterns: List[str], output_dir: Path, exclude_pattern: Optional[str] = None
) -> List[Path]:
    """Process and filter file paths based on input patterns and exclusions."""
    all_paths = flatten([glob.glob(input_pattern) for input_pattern in input_patterns])
    exclude_paths = glob.glob(exclude_pattern) if exclude_pattern else []
    final_paths = list(set(all_paths) - set(exclude_paths))

    # Handle multipage TIFFs
    tif_paths = [p for p in final_paths if p.endswith((".tif", ".tiff"))]
    extracted_dir = output_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    new_paths = []
    for path in tif_paths:
        img = Image.open(path)
        base_name = Path(path).stem

        for i, page in enumerate(ImageSequence.Iterator(img)):
            output_path = extracted_dir / f"{base_name}_page_{i+1}.npy"
            if not output_path.exists():
                np.save(output_path, np.array(page))
            new_paths.append(str(output_path))

    final_paths = list((set(final_paths) - set(tif_paths)) | set(new_paths))
    final_paths.sort(reverse=True)
    return [Path(p) for p in final_paths]


def main(
    input_patterns: List[str],
    output: str,
    catmap: dict,
    create_dir: bool = True,
    crop: Optional[List[int]] = None,
    verify: bool = False,
    batch_size: int = 4,
    darkframe: Optional[str] = None,
    exclude: Optional[str] = None,
    max_size: int = 1024,  # New parameter for maximum image dimension
):
    """Main execution function with automatic distributed handling"""
    output = Path(os.path.expanduser(output))

    if create_dir:
        output.mkdir(exist_ok=True)
        (output / "masks").mkdir(exist_ok=True)
        (output / "imgs").mkdir(exist_ok=True)
        if verify:
            (output / "verification").mkdir(exist_ok=True)

    paths = process_paths(input_patterns, output, exclude)

    # Load darkframe if provided
    df = None
    if darkframe:
        df_dir = Path(os.path.expanduser(darkframe))
        df_paths = list(df_dir.glob("*.npy"))
        if df_paths:
            df = np.sum([np.load(f).astype(np.uint16) for f in df_paths], axis=0)
            df = df.clip(0, np.iinfo(np.uint16).max)
            df = ImageDataset.minmax_norm(df)

    dataset = ImageDataset(paths, darkframe=df, crop=crop, max_size=max_size)

    # Determine number of GPUs and handle distributed setup
    world_size = torch.cuda.device_count()
    logger.info(f"Found {world_size} GPUs")

    if world_size > 1:
        import torch.multiprocessing as mp

        mp.spawn(
            process_images,
            args=(world_size, dataset, output, catmap, batch_size, verify),
            nprocs=world_size,
            join=True,
        )
    else:
        process_images(0, 1, dataset, output, catmap, batch_size, verify)


if __name__ == "__main__":
    main(
        input_patterns=["*.npy"],
        output="output_dir",
        catmap={"signal": 1},
        batch_size=4,
        verify=True,
        max_size=1024,  # Specify maximum dimension size
    )
