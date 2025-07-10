
# calculate_stats.py

import argparse
import logging
import sys
from pathlib import Path
import h5py
import numpy as np

try:
    from utils import calc_crops, find_regions_and_generate_mask
except ImportError:
    print("Error: Could not import 'calc_crops' and 'find_regions_and_generate_mask' from 'utils'.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_dataset_statistics(args):
    """
    Calculates the mean and standard deviation of pixel values within the ROI
    across all images in the dataset.
    """
    running_sum = 0.0
    running_sum_of_squares = 0.0
    total_pixel_count = 0

    logging.info("Starting dataset statistics calculation...")

    for path in args.h5_file_path:
        h5_file_path = Path(path)
        if not h5_file_path.is_file():
            logging.error(f"Input file not found: {h5_file_path}")
            continue

        logging.info(f"Processing file: {h5_file_path.name}")
        with h5py.File(h5_file_path, "r") as hf:
            if "images" not in hf:
                logging.warning(f"'images' dataset not found in {h5_file_path.name}. Skipping.")
                continue
            
            images_ds = hf["images"]
            for i in range(images_ds.shape[0]):
                image_data = images_ds[i]

                # 1. Crop the image to the same region as in training
                crop_top, crop_bottom, crop_left, crop_right = calc_crops(
                    image_data.squeeze(), args.crop_top, args.crop_bottom, args.crop_left, args.crop_right
                )
                cropped_image = image_data[crop_top:crop_bottom, crop_left:crop_right]

                # 2. Generate the ROI mask to identify valid pixels
                roi_mask = find_regions_and_generate_mask(
                    cropped_image.squeeze(),
                    window_size=(args.window_height, args.window_width),
                    min_area=args.min_area,
                    threshold=args.threshold
                ).astype(bool)

                # 3. Extract only the pixels within the ROI
                roi_pixels = cropped_image[roi_mask]

                if roi_pixels.size == 0:
                    continue

                # 4. Update running statistics using float64 for precision
                running_sum += np.sum(roi_pixels, dtype=np.float64)
                running_sum_of_squares += np.sum(np.square(roi_pixels, dtype=np.float64))
                total_pixel_count += roi_pixels.size
    
    if total_pixel_count == 0:
        logging.error("No valid pixels found in the dataset. Cannot calculate statistics.")
        sys.exit(1)

    # 5. Calculate final mean and std
    mean = running_sum / total_pixel_count
    # Variance = E[X^2] - (E[X])^2
    variance = (running_sum_of_squares / total_pixel_count) - (mean ** 2)
    std = np.sqrt(variance)

    logging.info("--- Statistics Calculation Complete ---")
    logging.info(f"Dataset Mean: {mean}")
    logging.info(f"Dataset Std Dev: {std}")
    logging.info(f"Total valid pixels counted: {total_pixel_count}")

    # 6. Save statistics to a file
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, mean=mean, std=std)
    logging.info(f"Statistics saved to '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate and save dataset-level normalization statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Input/Output Arguments ---
    parser.add_argument("h5_file_path", type=str, help="Path to the input H5 file(s).", nargs='+')
    parser.add_argument("--output_file", type=str, default="stats/dataset_stats.npz", help="Path to save the output .npz file.")
    
    # --- Arguments MUST MATCH your patch generation script ---
    parser.add_argument("--crop_top", type=int, default=0)
    parser.add_argument("--crop_bottom", type=int, default=0)
    parser.add_argument("--crop_left", type=int, default=0)
    parser.add_argument("--crop_right", type=int, default=0)
    parser.add_argument("--window_height", type=int, default=16)
    parser.add_argument("--window_width", type=int, default=16)
    parser.add_argument("--min_area", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=1.5)
    
    args = parser.parse_args()
    calculate_dataset_statistics(args)
