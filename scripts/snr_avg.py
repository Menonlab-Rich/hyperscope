import tifffile
import numpy as np
import cv2
import argparse
import os
import sys
import h5py # Import the HDF5 library

def calculate_average_snr(file_path: str):
    """
    Calculates the average Signal-to-Noise Ratio (SNR) for a single image file.

    This function can handle both multipage TIFF (.tif, .tiff) files and HDF5 (.h5) files.
    For H5 files, it iterates slice-by-slice to conserve memory.
    It expects the data in H5 files to be in a dataset named 'images'.

    The SNR for each page/slice is calculated as mean / std_dev on the original, raw pixel data.

    Args:
        file_path (str): The full path to the image file.

    Returns:
        float or None: The average SNR for the file, or None if the file is skipped or an error occurs.
    """
    # 1. Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'", file=sys.stderr)
        return None

    try:
        file_extension = os.path.splitext(file_path)[1].lower()

        # --- Process based on file type ---
        if file_extension in ['.tif', '.tiff']:
            source_type = "TIFF"
            # TIFFs are typically loaded into memory by tifffile
            images = tifffile.imread(file_path)
            print(f"Successfully loaded '{os.path.basename(file_path)}' ({source_type})")

            # Handle single-page images by adding a new axis
            if images.ndim == 2:
                images = images[np.newaxis, ...]
            
            num_pages = images.shape[0]
            bit_depth = images.dtype.itemsize * 8
            print(f"Found {num_pages} pages with a bit depth of {bit_depth}-bit.")

            snr_values = []
            # Loop through each in-memory page
            for i in range(num_pages):
                page_data = images[i]
                mean_val = np.mean(page_data)
                std_val = np.std(page_data)
                snr = mean_val / std_val if std_val > 0 else 0.0
                snr_values.append(snr)
                if num_pages < 20:
                    print(f"  - Processing page {i + 1}/{num_pages}... SNR: {snr:.4f}")
            
            average_snr = np.mean(snr_values) if snr_values else 0.0

        elif file_extension == '.h5':
            source_type = "HDF5"
            with h5py.File(file_path, 'r') as f:
                if 'images' not in f:
                    print(f"Info: Skipping HDF5 file because 'images' dataset was not found.")
                    return None
                
                # Get a reference to the dataset without loading it into memory
                h5_dataset = f['images']
                print(f"Successfully opened '{os.path.basename(file_path)}' ({source_type})")
                
                num_pages = h5_dataset.shape[0]
                bit_depth = h5_dataset.dtype.itemsize * 8
                print(f"Found {num_pages} slices with a bit depth of {bit_depth}-bit.")

                total_snr = 0.0
                # Iterate slice by slice to save memory
                for i in range(num_pages):
                    # Load only one slice at a time
                    page_data = h5_dataset[i]
                    mean_val = np.mean(page_data)
                    std_val = np.std(page_data)
                    snr = mean_val / std_val if std_val > 0 else 0.0
                    total_snr += snr # Sum SNRs as we go
                    # Reduce print verbosity for very large files
                    if num_pages < 20:
                        print(f"  - Processing slice {i + 1}/{num_pages}... SNR: {snr:.4f}")

                average_snr = total_snr / num_pages if num_pages > 0 else 0.0
        else:
            print(f"Warning: Unsupported file type '{file_extension}'. Skipping file.")
            return None

        print(f"  -> Average SNR for this file: {average_snr:.4f}")
        return average_snr

    except Exception as e:
        print(f"An error occurred while processing '{file_path}': {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(
        description="Calculate the average Signal-to-Noise Ratio (SNR) across all pages of multipage TIFF or H5 files."
    )
    # Use your updated nargs="+" to accept one or more file paths
    parser.add_argument("file_paths", help="The path(s) to the input image file(s) (.tif, .tiff, .h5).", nargs="+")

    args = parser.parse_args()

    # --- Process all provided files and collect valid results ---
    # We will collect SNR values from all files that are processed successfully
    all_snr_results = []
    for path in args.file_paths:
        print(f"\n--- Processing file: {os.path.basename(path)} ---")
        file_avg_snr = calculate_average_snr(path)
        # Only add the result if the function didn't return None (i.e., it wasn't skipped)
        if file_avg_snr is not None:
            all_snr_results.append(file_avg_snr)

    # --- Calculate and print the final overall average ---
    if all_snr_results:
        # Calculate the mean of the per-file averages and convert to dB
        overall_avg_snr = 20 * np.log10(np.mean(all_snr_results))
        print("\n" + "="*45)
        print(f"Overall Average SNR across {len(all_snr_results)} valid file(s): {overall_avg_snr:.4f} dB")
        print("="*45)
    else:
        print("\nNo valid image files were processed.")
