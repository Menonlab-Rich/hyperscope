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
from skimage.util import view_as_windows # We'll use this for efficient patching

# --- Assume these are in utils.py ---
# You must import your actual preprocessing functions
try:
    from utils import calc_crops, find_regions_and_generate_mask
except ImportError:
    print("FATAL: Could not import preprocessing functions from 'utils.py'")
    # Define dummy functions so the script can be read, but it will not work correctly.
    def calc_crops(img, *args): return 0, img.shape[0], 0, img.shape[1]
    def find_regions_and_generate_mask(img, **kwargs): return np.ones_like(img, dtype=np.uint8)
# ---

load_dotenv()

# --- Functions from your script (mostly unchanged) ---

def load_lightning_model(
    checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> pl.LightningModule:
    """Loads the PyTorch Lightning UNet model from checkpoint."""
    model = UNetLightning.load_from_checkpoint(checkpoint_path, map_location=device, n_classes=3)
    model = model.to(device)
    model.eval()
    return model

def process_patch(
    patch: np.ndarray, # Expects numpy array now
    model: pl.LightningModule,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """Process a single numpy patch through the model."""
    # The transform should handle numpy to tensor conversion
    patch_tensor = get_train_transform()["input"](
        image=patch, mask=np.zeros_like(patch).astype(np.uint8)
    )["image"].unsqueeze(0)
    patch_tensor = patch_tensor.to(device)

    with torch.no_grad():
        output = model(patch_tensor)
        output = torch.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)

    return output.squeeze(0).cpu().numpy() # Return numpy array on cpu

def create_colored_overlay(segmentation_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create a colored overlay from the segmentation map."""
    seg_map = segmentation_map
    overlay = np.zeros((*seg_map.shape, 4), dtype=np.uint8)
    overlay[seg_map == 1] = [255, 0, 0, int(255 * alpha)]  # Red for class 1
    overlay[seg_map == 2] = [0, 255, 0, int(255 * alpha)]  # Green for class 2
    # Note: Your model has n_classes=3, so class 0 is background
    return overlay


# --- REVISED AND CORRECTED IMAGE PROCESSING FUNCTION ---

def process_image_corrected(
    image_path: str,
    model: pl.LightningModule,
    device: str,
    # --- ADD PREPROCESSING PARAMS ---
    # These MUST match the values used during training!
    crop_args: dict,
    mask_gen_args: dict,
    patch_size: Tuple[int, int] = (128, 128),
    overlap: int = 32 # Add overlap for smoother stitching
) -> Image.Image:
    """
    Process an image using a pipeline that MIRRORS the training preprocessing.
    """
    # 1. Load Image
    image = np.array(Image.open(image_path))
    original_image_shape = image.shape[:2]

    # 2. Apply Initial Crop (from training)
    crop_top, crop_bottom, crop_left, crop_right = calc_crops(image, **crop_args)
    cropped_image = image[crop_top:crop_bottom, crop_left:crop_right]

    # 3. Generate ROI Mask (from training)
    # This mask identifies the regions the model was trained to look at.
    roi_mask = find_regions_and_generate_mask(cropped_image, **mask_gen_args)

    # 4. Standardize and Zero Background (from training)
    # Perform standardization BEFORE zeroing the background.
    processed_image = cropped_image.astype(np.float32)
    processed_image[roi_mask == 0] = 0 # Zero out background
    mean = processed_image[roi_mask == 1].mean()
    std = processed_image[roi_mask == 1].std()
    processed_image = (processed_image - mean) / (std + 1e-8) # Add epsilon for stability

    # 5. Patch and Reconstruct with Overlap
    step = patch_size[0] - overlap
    h, w = processed_image.shape
    
    # Pad the image to ensure patches fit perfectly at the edges
    pad_h = (step - (h - patch_size[0]) % step) % step
    pad_w = (step - (w - patch_size[1]) % step) % step
    padded_image = np.pad(processed_image, ((0, pad_h), (0, pad_w)), mode='constant')
    
    # These will store the final results and handle overlap averaging
    reconstructed_mask = np.zeros(padded_image.shape, dtype=np.float32)
    overlap_counts = np.zeros(padded_image.shape, dtype=np.float32)

    # Use sliding windows for efficient patching
    patches = view_as_windows(padded_image, patch_size, step=step)
    
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j]
            
            # --- Inference ---
            # Only process non-empty patches if desired, for speed.
            if patch.max() == 0:
                continue # Skip totally black patches

            segmentation_patch = process_patch(patch, model, device)

            # --- Stitching ---
            y_start, x_start = i * step, j * step
            y_end, x_end = y_start + patch_size[0], x_start + patch_size[1]
            
            reconstructed_mask[y_start:y_end, x_start:x_end] += segmentation_patch
            overlap_counts[y_start:y_end, x_start:x_end] += 1

    # Average the predictions in the overlapping regions
    overlap_counts[overlap_counts == 0] = 1 # Avoid division by zero
    final_mask_padded = (reconstructed_mask / overlap_counts).round().astype(np.uint8)

    # Un-pad to match original cropped image size
    final_mask = final_mask_padded[:h, :w]

    # 6. Create Visualization
    # Place the final mask back into the full-size image context
    full_sized_seg_map = np.zeros(original_image_shape, dtype=np.uint8)
    full_sized_seg_map[crop_top:crop_bottom, crop_left:crop_right] = final_mask
    
    overlay_rgba = create_colored_overlay(full_sized_seg_map)

    # Normalize original image for nice viewing
    img_view = (image - image.min()) / (image.max() - image.min() + 1e-8)
    img_view = (img_view * 255).astype(np.uint8)
    
    base_image_rgba = Image.fromarray(img_view).convert("RGBA")
    overlay_img = Image.fromarray(overlay_rgba, "RGBA")

    return Image.alpha_composite(base_image_rgba, overlay_img)


# Example usage
if __name__ == "__main__":
    import os
    from tqdm import tqdm

    # --- MUST MATCH TRAINING PARAMETERS ---
    CROP_KWARGS = {
        "crop_top_amt": 150, "crop_bottom_amt": 300, "crop_left_amt": 150, "crop_right_amt": 0
    }
    MASK_GEN_KWARGS = {
        "window_size": (16, 16), "min_area": 1000, "threshold": 1.5
    }
    PATCH_KWARGS = {
        "patch_size": (64, 64), "overlap": 32 # 50% overlap
    }
    # ---

    IMAGE_PATH = "/mnt/d/rich/hyper-scope/worm_nomod.tif" # Path to a single test image
    CHECKPOINT_PATH = "/mnt/d/rich/hyper-scope/models/checkpoints/unet-UN-507-epoch=11-val_dice=0.55.ckpt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model = load_lightning_model(CHECKPOINT_PATH, device=DEVICE)

    # --- For Neptune Logging ---
    # run = neptune.init_run(...)

    print(f"Processing image: {IMAGE_PATH}")
    result_image = process_image_corrected(
        image_path=IMAGE_PATH,
        model=model,
        device=DEVICE,
        crop_args=CROP_KWARGS,
        mask_gen_args=MASK_GEN_KWARGS,
        **PATCH_KWARGS
    )
    
    # Save or display the result
    result_image.save("result_overlay.png")
    print("Saved result to result_overlay.png")

    # --- Log to Neptune ---
    # if run:
    #     run["results/final_overlay"].upload(File.as_image(result_image))
    #     run.stop()
