import h5py
import numpy as np
import os
import argparse

def extract_images_from_hdf5(hdf5_file_path, dataset_name, output_folder):
    """
    Extracts images from an HDF5 file and saves them as individual .npy files.

    Args:
        hdf5_file_path (str): Path to the input HDF5 file.
        dataset_name (str): Name of the dataset containing the images (e.g., 'images').
        output_folder (str): Path to the folder where .npy files will be saved.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder '{output_folder}' ensured.")

    try:
        if not os.path.exists(hdf5_file_path):
            print(f"Error: HDF5 file '{hdf5_file_path}' not found.")
            return

        with h5py.File(hdf5_file_path, 'r') as f:
            if dataset_name not in f:
                print(f"Error: Dataset '{dataset_name}' not found in '{hdf5_file_path}'")
                print(f"Available datasets: {list(f.keys())}")
                return

            images_dataset = f[dataset_name]

            # Verify the dimensions (n, h, w)
            if images_dataset.ndim != 3:
                print(f"Warning: Dataset '{dataset_name}' does not have 3 dimensions (n, h, w). "
                      f"It has {images_dataset.ndim} dimensions and shape {images_dataset.shape}. "
                      "Proceeding, assuming the first dimension is the number of images.")

            n = images_dataset.shape[0]

            print(f"Found {n} images in dataset '{dataset_name}' from '{hdf5_file_path}'.")
            print(f"Dataset shape: {images_dataset.shape}")

            for i in range(n):
                # Extract each image
                # This slices out the i-th image (h, w) assuming (n, h, w) structure
                image = images_dataset[i, :, :]
                
                # Define the output file path
                # Pad with zeros for consistent sorting (e.g., image_00001.npy, image_00010.npy)
                output_file_path = os.path.join(output_folder, f"image_{i:05d}.npy") 

                # Save the image as a .npy file
                np.save(output_file_path, image)
                
                if (i + 1) % 100 == 0 or (i + 1) == n:
                    print(f"Extracted {i + 1}/{n} images...", end='\r')

            print(f"\nSuccessfully extracted all {n} images to '{output_folder}'.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract images from an HDF5 dataset and save them as individual .npy files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    parser.add_argument(
        '--hdf5_file', '-f',
        type=str,
        required=True,
        help="Path to the input HDF5 file containing the image dataset."
    )
    parser.add_argument(
        '--dataset_name', '-d',
        type=str,
        default='images', # Common default, but users can change it
        help="Name of the dataset within the HDF5 file that holds the images (e.g., 'images', '/data')."
    )
    parser.add_argument(
        '--output_folder', '-o',
        type=str,
        default='extracted_images', # Default output folder
        help="Path to the folder where the extracted .npy files will be saved."
    )
    parser.add_argument(
        '--create_dummy', '-c',
        action='store_true',
        help="Create a dummy HDF5 file for testing purposes if it doesn't exist. "
             "If this flag is used, --hdf5_file will be used as the path for the dummy file."
    )
    parser.add_argument(
        '--num_dummy_images', type=int, default=10,
        help="Number of images to create in the dummy HDF5 file (if --create_dummy is used)."
    )
    parser.add_argument(
        '--dummy_height', type=int, default=32,
        help="Height of images in the dummy HDF5 file."
    )
    parser.add_argument(
        '--dummy_width', type=int, default=32,
        help="Width of images in the dummy HDF5 file."
    )

    args = parser.parse_args()

    # --- Handle Dummy File Creation ---
    if args.create_dummy:
        print(f"\nAttempting to create a dummy HDF5 file at '{args.hdf5_file}' for demonstration...")
        try:
            if os.path.exists(args.hdf5_file):
                print(f"Dummy file '{args.hdf5_file}' already exists. Skipping creation.")
            else:
                dummy_data = np.random.randint(
                    0, 256, 
                    size=(args.num_dummy_images, args.dummy_height, args.dummy_width), 
                    dtype=np.uint8
                )
                with h5py.File(args.hdf5_file, 'w') as f:
                    f.create_dataset(args.dataset_name, data=dummy_data)
                print(f"Dummy HDF5 file '{args.hdf5_file}' with {args.num_dummy_images} images (shape {args.dummy_height}x{args.dummy_width}) created successfully.")
        except Exception as e:
            print(f"Error creating dummy file: {e}")
            exit(1) # Exit if dummy file creation fails

    # --- Run the extraction ---
    extract_images_from_hdf5(args.hdf5_file, args.dataset_name, args.output_folder)
