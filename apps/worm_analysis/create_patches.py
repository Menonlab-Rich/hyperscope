import argparse
import logging
import sys
import time
from pathlib import Path
import h5py
import numpy as np
from skimage.util import view_as_windows

# Assuming 'utils.py' with 'calc_crops' and 'find_regions_and_generate_mask' exists
# If not, you'll need to define those functions here or in a separate file.
try:
    from utils import calc_crops, find_regions_and_generate_mask
except ImportError:
    print("Error: Could not import 'calc_crops' and 'find_regions_and_generate_mask' from 'utils'.")
    print("Please ensure 'utils.py' is in the same directory or in the PYTHONPATH.")
    # As a placeholder, let's define dummy functions if utils is not available
    def calc_crops(image, *args):
        return 0, image.shape[0], 0, image.shape[1]
    def find_regions_and_generate_mask(image, **kwargs):
        # This dummy function will create a mask of all ones.
        # Replace this with your actual mask generation logic.
        return np.ones(image.shape[:2], dtype=np.uint8)

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_patches_from_region(image, mask, patch_shape, step_h, step_w, min_mask_area_threshold):
    """
    Extracts patches from an image and mask.
    Filters patches based on the amount of mask content.
    """
    patches = []

    # Handle image channels if present
    if image.ndim == 3: # (H, W, C)
        img_h, img_w, img_c = image.shape
    else: # (H, W) for grayscale
        img_h, img_w = image.shape
        img_c = 1 # Treat as 1 channel

    # Pad the image and mask to ensure we can extract full patches up to the edge
    pad_h = (patch_shape[0] - (img_h % step_h)) % step_h
    pad_w = (patch_shape[1] - (img_w % step_w)) % step_w

    if image.ndim == 2:
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        padded_image_for_view = padded_image[:, :, np.newaxis] # Add channel dim for view_as_windows
    else:
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
        padded_image_for_view = padded_image

    padded_mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    # Use view_as_windows to create sliding window views
    image_views = view_as_windows(padded_image_for_view, window_shape=(patch_shape[0], patch_shape[1], img_c), step=(step_h, step_w, img_c))
    mask_views = view_as_windows(padded_mask, window_shape=patch_shape, step=(step_h, step_w))

    # Reshape to iterate over patches easily
    image_patches = image_views.reshape(-1, patch_shape[0], patch_shape[1], img_c)
    mask_patches = mask_views.reshape(-1, patch_shape[0], patch_shape[1])

    for img_patch, mask_patch in zip(image_patches, mask_patches):
        # Remove the added channel dimension if the original was grayscale
        if image.ndim == 2:
            img_patch = img_patch.squeeze(axis=-1)

        # Filter patches based on the percentage of the mask present
        if mask_patch.sum() > 0:
            patches.append((img_patch, mask_patch))

    return patches


def main(args):
    """Main function to orchestrate the serial processing pipeline."""
    for path in args.h5_file_path:
        h5_file_path = Path(path)
        if not h5_file_path.is_file():
            logging.error(f"Input file not found: {h5_file_path}")
            sys.exit(1)

        # Create separate output directories for image and mask patches
        image_output_dir = Path(args.output_dir) / "images"
        mask_output_dir = Path(args.output_dir) / "masks"
        image_output_dir.mkdir(parents=True, exist_ok=True)
        mask_output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Image patches will be saved in: {image_output_dir}")
        logging.info(f"Mask patches will be saved in: {mask_output_dir}")

        total_patches_saved = 0
        start_time = time.time()

        try:
            with h5py.File(h5_file_path, "r") as hf:
                if "images" not in hf or "masks" not in hf:
                    logging.warn("'images' dataset not found in H5 file.")
                    continue
                    
                
                images_ds = hf["images"]
                num_images = images_ds.shape[0]
                logging.info(f"Found {num_images} images in '{h5_file_path.name}'. Starting serial processing...")

                # --- Loop through each image in the HDF5 file ---
                for i in range(num_images):
                    logging.info(f"Processing image {i + 1}/{num_images}...")
                    
                    image_data = images_ds[i]

                    # --- 1. Generate the mask ---
                    crop_top, crop_bottom, crop_left, crop_right = calc_crops(image_data.squeeze(), args.crop_top, args.crop_bottom, args.crop_left, args.crop_right)
                    cropped_image_data = image_data[crop_top:crop_bottom, crop_left:crop_right]
                    
                    temp_mask = find_regions_and_generate_mask(
                        cropped_image_data.squeeze(),
                        window_size=(args.window_height, args.window_width),
                        min_area=args.min_area,
                        threshold=args.threshold
                    )


                    if 'nervecord' in h5_file_path.stem or 'mcherry' in h5_file_path.stem:
                        label_value = 2
                    else:
                        label_value = 1
                    
                    full_mask = np.zeros(image_data.shape[:2], dtype=np.uint8)
                    full_mask[crop_top:crop_bottom, crop_left:crop_right] = temp_mask
                    image_data[~full_mask] = 0
                    image_data = (image_data - image_data.mean())/image_data.std()
                    labeled_mask = (full_mask * label_value).astype(np.uint8)

                    if np.sum(labeled_mask) == 0:
                        logging.warning(f"Skipping image {i}: Generated mask is empty.")
                        continue

                    # --- 2. Extract patches ---
                    patch_shape = (args.patch_height, args.patch_width)
                    step_h = args.patch_height - args.overlap_height
                    step_w = args.patch_width - args.overlap_width

                    extracted_patches = extract_patches_from_region(
                        image_data,
                        labeled_mask,
                        patch_shape,
                        step_h,
                        step_w,
                        args.min_mask_perc
                    )

                    if not extracted_patches:
                        logging.warning(f"No valid patches found for image {i} after filtering.")
                        continue

                    # --- 3. Save patches to disk ---
                    for j, (img_patch, mask_patch) in enumerate(extracted_patches):
                        # Define a unique name for each patch based on the source image index and patch number
                        patch_filename = f"image_{i:04d}_patch_{j:04d}.npy"
                        
                        # Save the image patch
                        np.save(image_output_dir / patch_filename, img_patch)
                        
                        # Save the corresponding mask patch
                        np.save(mask_output_dir / patch_filename, mask_patch)

                    total_patches_saved += len(extracted_patches)
                    logging.info(f"Saved {len(extracted_patches)} patches for image {i}.")

        except Exception as e:
            logging.error(f"An error occurred during processing: {e}", exc_info=True)
            sys.exit(1)
            
        end_time = time.time()
        logging.info("--- Processing Complete ---")
        logging.info(f"Successfully saved a total of {total_patches_saved} image/mask patch pairs.")
        logging.info(f"Total time taken: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Serially generate masks and patches from an H5 file and save them as NumPy files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- File Arguments ---
    parser.add_argument("h5_file_path", type=str, help="Path to the input H5 file.", nargs='+')
    parser.add_argument("--output_dir", type=str, default="output_patches", help="Directory to save the output image and mask folders.")
    
    # --- Pre-processing Arguments ---
    parser.add_argument("--crop_top", type=int, default=0, help="Top crop amount.")
    parser.add_argument("--crop_bottom", type=int, default=0, help="Bottom crop amount.")
    parser.add_argument("--crop_left", type=int, default=0, help="Left crop amount.")
    parser.add_argument("--crop_right", type=int, default=0, help="Right crop amount.")

    # --- Region Detection Arguments ---
    parser.add_argument("--window_height", type=int, default=16, help="Height of the sliding window for region detection.")
    parser.add_argument("--window_width", type=int, default=16, help="Width of the sliding window for region detection.")
    parser.add_argument("--min_area", type=int, default=1000, help="Minimum area of a detected region to be kept.")
    parser.add_argument("--threshold", type=float, default=1.5, help="Threshold for the region detection algorithm.")

    # --- Mask Arguments ---
    parser.add_argument("--label_value", type=int, default=1, help="Value (1-255) to use for the generated mask.")

    # --- Patch Extraction Arguments ---
    parser.add_argument("--patch_height", type=int, default=128, help="Height of the extracted patches.")
    parser.add_argument("--patch_width", type=int, default=128, help="Width of the extracted patches.")
    parser.add_argument("--overlap_height", type=int, default=0, help="Overlap in pixels between vertical patches.")
    parser.add_argument("--overlap_width", type=int, default=0, help="Overlap in pixels between horizontal patches.")
    parser.add_argument("--min_mask_perc", type=float, default=0.01, help="Minimum percentage of mask pixels required to keep a patch (e.g., 0.5 for 50%).")

    main(parser.parse_args())
