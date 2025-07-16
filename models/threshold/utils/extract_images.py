import argparse
import logging
import sys
from pathlib import Path
import h5py
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Assuming 'utils.py' with these functions exists in the same directory
try:
    from utils import calc_crops, find_regions_and_generate_mask
except ImportError:
    print("Error: Could not import 'calc_crops' and 'find_regions_and_generate_mask' from 'utils'.")
    sys.exit(1)

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_image_worker(image_index, args, h5_file_path_str):
    """
    Worker function to process a single image.
    It crops the image, checks for a region of interest, and normalizes.
    Returns the processed image and its original index if valid, otherwise None.
    """
    # Each worker opens the H5 file to avoid conflicts.
    with h5py.File(h5_file_path_str, "r") as hf_in:
        image_data = hf_in["images"][image_index]

    # --- 1. Crop the image ---
    crop_top, crop_bottom, crop_left, crop_right = calc_crops(
        image_data.squeeze(), args.crop_top, args.crop_bottom, args.crop_left, args.crop_right
    )
    cropped_image = image_data[crop_top:crop_bottom, crop_left:crop_right]

    # --- 2. Determine if a Region of Interest (ROI) exists ---
    roi_mask = find_regions_and_generate_mask(
        cropped_image.squeeze(),
        window_size=(args.window_height, args.window_width),
        min_area=args.min_area,
        threshold=args.threshold,
        n_clusters=2
    )

    # If the mask is empty, the image is likely noise. Skip it.
    if not np.any(roi_mask):
        return None

    # --- 3. Normalize the Cropped Image ---
    # The mask is NOT applied; we just normalize the entire cropped region.
    normalized_image = (cropped_image.astype(np.float32) - cropped_image.min()) / (cropped_image.max() - cropped_image.min() + 1e-8)

    # Return the normalized image and its original index for unique file naming
    return (image_index, normalized_image)


def main(args):
    """Main function to orchestrate the parallel processing pipeline."""
    for path in args.h5_file_path:
        h5_file_path = Path(path)
        output_dir = Path(args.output_dir) / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"--- Starting processing for {h5_file_path.name} ---")

        with h5py.File(h5_file_path, "r") as hf:
            num_images = hf["images"].shape[0]

        # --- Setup Parallel Processing ---
        num_workers = max(1, cpu_count() - 1)
        worker_func = partial(process_image_worker, args=args, h5_file_path_str=str(h5_file_path))
        
        total_images_saved = 0
        with Pool(processes=num_workers) as pool:
            # imap_unordered is great for performance here
            results_iterator = pool.imap_unordered(worker_func, range(num_images))
            
            for result in tqdm(results_iterator, total=num_images, desc="Processing images"):
                if result is None:
                    continue
                
                original_image_idx, processed_image = result
                
                # --- Save the final .npy file ---
                output_filename = output_dir / f"image_{h5_file_path.stem}_{original_image_idx:04d}.npy"
                np.save(output_filename, processed_image.astype(np.float32))
                total_images_saved += 1

        logging.info(f"--- Done ---")
        logging.info(f"Successfully saved {total_images_saved} valid images to {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quickly generate clean, cropped images as .npy files using multiprocessing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Final, Simplified Argument Set ---
    parser.add_argument("h5_file_path", type=str, nargs='+', help="Path to one or more input H5 files.")
    parser.add_argument("--output_dir", type=str, default="output_images", help="Directory to save the output 'images' subfolder.")
    parser.add_argument("--crop_top", type=int, default=0)
    parser.add_argument("--crop_bottom", type=int, default=0)
    parser.add_argument("--crop_left", type=int, default=0)
    parser.add_argument("--crop_right", type=int, default=0)
    parser.add_argument("--window_height", type=int, default=16, help="Height of the sliding window for ROI detection.")
    parser.add_argument("--window_width", type=int, default=16, help="Width of the sliding window for ROI detection.")
    parser.add_argument("--min_area", type=int, default=1000, help="Minimum pixel area to be considered a valid region.")
    parser.add_argument("--threshold", type=float, default=1.5, help="Threshold for ROI detection.")

    main(parser.parse_args())
