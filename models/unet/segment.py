from dotenv import load_dotenv
import torch
import pytorch_lightning as pl
from PIL import Image
import numpy as np
from typing import Tuple
from models.unet.model import UNetLightning
import neptune
from neptune.types import File
from models.unet.config import get_train_transform

load_dotenv()


def load_lightning_model(
    checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> pl.LightningModule:
    """
    Load the PyTorch Lightning UNet model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Loaded PyTorch Lightning model
    """
    # Load model from checkpoint
    model = UNetLightning.load_from_checkpoint(checkpoint_path, map_location=device, n_classes=3)

    model = model.to(device)
    model.eval()

    return model


def process_patch(
    patch: Image.Image,
    model: pl.LightningModule,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """Process a single patch through the model."""
    patch = np.array(patch)
    patch_tensor = get_train_transform()["input"](
        image=patch, mask=np.zeros_like(patch).astype(np.uint8)
    )["image"].unsqueeze(0)
    patch_tensor = patch_tensor.to(device)

    with torch.no_grad():
        output = model(patch_tensor)
        # Apply softmax and get class predictions
        output = torch.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)

    return output.squeeze(0)


def create_colored_overlay(segmentation_map: torch.Tensor, alpha: float = 0.5) -> np.ndarray:
    """Create a colored overlay from the segmentation map."""
    # Convert to numpy array
    seg_map = segmentation_map.cpu().numpy()

    # Create RGBA overlay
    overlay = np.zeros((*seg_map.shape, 4), dtype=np.uint8)

    # Set colors: black (0), red (1), green (2)
    overlay[seg_map == 1] = [255, 0, 0, int(255 * alpha)]  # Red for class 1
    overlay[seg_map == 2] = [0, 255, 0, int(255 * alpha)]  # Green for class 2
    overlay[seg_map == 3] = [0, 0, 255, int(255 * alpha)]  # blue for class 3

    return overlay


def process_image(
    image_path: str,
    checkpoint_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patch_size: Tuple[int, int] = (64, 64),
) -> Image.Image:
    """
    Process image using the Lightning UNet model.

    Args:
        image_path: Path to input image
        checkpoint_path: Path to model checkpoint
        device: Device to run inference on
        patch_size: Size of patches to process

    Returns:
        Processed image with overlay
    """
    # Load model
    model = load_lightning_model(checkpoint_path, device)

    # Load and preprocess image
    image = Image.open(image_path)
    width, height = image.size

    # Create output image
    output_image = Image.new("L", image.size, (0))
    output_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

    # Process image patch by patch
    for y in range(0, height, patch_size[1]):
        for x in range(0, width, patch_size[0]):
            # Extract patch
            patch = image.crop(
                (x, y, min(x + patch_size[0], width), min(y + patch_size[1], height))
            )

            # Pad if necessary
            if patch.size != patch_size:
                new_patch = Image.new(image.mode, patch_size, 0)
                new_patch.paste(patch, (0, 0))
                patch = new_patch

            # Process patch
            segmentation = process_patch(patch, model, device)

            # Create overlay
            overlay = create_colored_overlay(segmentation)

            # Convert original patch to RGBA, preserving bit depth for display
            overlay_img = Image.fromarray(overlay, "RGBA")

            # Paste into final image
            output_image.paste(patch, (x, y))
            output_overlay.paste(overlay_img, (x, y))
    output_image = np.array(output_image)
    output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
    output_image = (output_image * 255).astype(np.uint8)

    # Convert to RGBA
    output_image_rgba = Image.fromarray(output_image).convert("RGBA")

    # Ensure both images have the same size and mode
    if output_image_rgba.size != output_overlay.size:
        output_overlay = output_overlay.resize(output_image_rgba.size)

    # Composite the images
    return Image.alpha_composite(output_image_rgba, output_overlay)


# Add Neptune logging capability
def process_image_with_logging(
    image_path: str,
    checkpoint_path: str,
    neptune_run,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Image.Image:
    """Process image and log results to Neptune."""
    result = process_image(image_path, checkpoint_path, device)

    # Log the result to Neptune
    neptune_run["results/processed_image"].log(File.as_image(result))

    # Log original image for comparison
    original = Image.open(image_path)
    neptune_run["results/original_image"].log(File.as_image(original))

    return result


# Example usage
if __name__ == "__main__":
    import os
    from glob import glob
    from tqdm import tqdm
    import random
    from collections import defaultdict

    # Paths
    IMAGE_PATH = "/mnt/d/hyper-scope/data/interim/worms/imgs/*.tif"
    CHECKPOINT_PATH = (
        "/mnt/d/hyper-scope/models/checkpoints/unet-UN-427-epoch=07-val_dice=0.96.ckpt"
    )

    # Get all image paths
    all_images = glob(IMAGE_PATH)
    
    # Group images by prefix
    prefix_groups = defaultdict(list)
    for img_path in all_images:
        # Extract filename from path
        filename = os.path.basename(img_path)
        # Get prefix (everything before the first underscore)
        prefix = filename.split('_')[0]
        prefix_groups[prefix].append(img_path)
    
    # Select 5 random images from each prefix group
    selected_images = []
    for prefix, images in prefix_groups.items():
        # If there are less than 5 images, take all of them
        n_samples = min(5, len(images))
        selected_images.extend(random.sample(images, n_samples))
    
    # Initialize Neptune run
    run = neptune.init_run(
        api_token=os.environ.get("NEPTUNE_API_TOKEN"),
        project="richbai90/unet2",
        tags=["testing", "segmentation", "unet", "worms"],
    )

    try:
        # Log the number of images per prefix
        for prefix, images in prefix_groups.items():
            run[f"data/prefix_{prefix}_total_images"] = len(images)
            run[f"data/prefix_{prefix}_selected_images"] = min(5, len(images))

        # Process selected images
        for img in tqdm(selected_images):
            prefix = os.path.basename(img).split('_')[0]
            # Log images with their prefix in Neptune
            process_image_with_logging(
                image_path=img, 
                checkpoint_path=CHECKPOINT_PATH, 
                neptune_run=run
            )
    finally:
        run.stop()
