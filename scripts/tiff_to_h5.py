import h5py
import tifffile
import numpy as np
from pathlib import Path
from hyperscope import config
import os

def export_tifs_to_h5(h5_file_path: str, tif_dir: Path, tif_glob_pattern: str, dataset_name: str = 'images'):
    """
    Exports images from multiple TIFF files matching a glob pattern into an HDF5 dataset.

    Args:
        h5_file_path (str): The path to the HDF5 file.
        tif_glob_pattern (str): A glob pattern to find TIFF files (e.g., 'mkate_*.tif').
        dataset_name (str): The name of the dataset within the HDF5 file where images will be stored.
                            If the dataset doesn't exist, it will be created based on the first image's shape and dtype.
                            If it exists, new images will be appended, assuming compatible shape/dtype.
    """
    h5_path = Path(h5_file_path)
    tif_files = sorted(list(tif_dir.glob(tif_glob_pattern))) # Look in current directory

    if not tif_files:
        print(f"No TIFF files found matching pattern: {tif_glob_pattern} in {Path('.').resolve()}")
        return

    print(f"Found {len(tif_files)} TIFF files to process.")

    # Determine the shape and dtype from the first TIFF image to create/check H5 dataset
    try:
        first_tif_image = tifffile.imread(tif_files[0])
        # tifffile.imread can return a single image (2D or 3D) or a stack (3D or 4D).
        # We need to account for multi-page TIFFs as individual images,
        # so let's ensure the first dimension is treated as a single image if it's 2D.
        if first_tif_image.ndim == 2: # Single 2D image
            img_shape_per_entry = first_tif_image.shape
        elif first_tif_image.ndim == 3: # Multi-page grayscale or single 3-channel
            # Assume (height, width, channels) or (pages, height, width)
            # If it's (H, W, C), the "image" has 3 dimensions.
            # If it's (P, H, W), where P is number of pages, treat each page as an image.
            # We need to decide how to stack them.
            # For simplicity, let's assume each *file* contributes one or more "frames"
            # and we want to store each frame as a separate entry in the HDF5 dataset's first dimension.
            # If the TIFF is (P, H, W) and you want (H, W) images, then first_tif_image[0] is (H,W).
            # If the TIFF is (H, W, C), the image is (H,W,C).
            # The most common case for "images" dataset is (N, H, W, C) or (N, H, W).
            # Let's verify shape before resizing dataset.

            # For now, let's assume if it's 3D, it's (pages, height, width)
            # and each page becomes an 'image' entry in H5.
            # If it's (Height, Width, Channels), then it's a single image.
            # A more robust check might be needed for channel dimension.
            if first_tif_image.shape[0] < first_tif_image.shape[1] and \
               first_tif_image.shape[0] < first_tif_image.shape[2]: # Likely (pages, H, W)
                img_shape_per_entry = first_tif_image.shape[1:]
            else: # Likely (H, W, C) or (H, W) already
                img_shape_per_entry = first_tif_image.shape
        else: # Handle 4D or more, assuming it's (pages, height, width, channels)
            img_shape_per_entry = first_tif_image.shape[1:]

        img_dtype = first_tif_image.dtype
        print(f"Detected image shape per entry: {img_shape_per_entry}, dtype: {img_dtype}")

    except Exception as e:
        print(f"Error reading first TIFF file {tif_files[0]}: {e}")
        return

    current_h5_num_images = 0
    with h5py.File(h5_path, 'a') as f: # Open in append mode
        if dataset_name not in f:
            print(f"Creating new dataset '/{dataset_name}' in '{h5_file_path}'")
            # Create a resizable dataset with maxshape=(None, ...)
            f.create_dataset(
                dataset_name,
                (0,) + img_shape_per_entry, # Start with 0 images, then the image dimensions
                maxshape=(None,) + img_shape_per_entry, # Allow first dimension (number of images) to be unlimited
                dtype=img_dtype,
                chunks=True # Enable chunking for efficient appending
            )
        else:
            # Check if existing dataset is compatible
            existing_dset = f[dataset_name]
            if existing_dset.dtype != img_dtype or existing_dset.shape[1:] != img_shape_per_entry:
                print(f"Warning: Existing dataset '{dataset_name}' has incompatible shape or dtype.")
                print(f"Existing: shape={existing_dset.shape}, dtype={existing_dset.dtype}")
                print(f"New images: shape={img_shape_per_entry}, dtype={img_dtype}")
                print("Consider deleting the existing dataset or changing 'dataset_name' to avoid issues.")
                # You might want to raise an error or ask for user confirmation here
                # For now, we'll continue but data might not be compatible.
            current_h5_num_images = existing_dset.shape[0]
            print(f"Existing dataset '/{dataset_name}' found with {current_h5_num_images} images.")

        dataset = f[dataset_name]

        total_new_images_to_add = 0
        for tif_file in tif_files:
            try:
                images_from_tif = tifffile.imread(tif_file)
                if images_from_tif.ndim == len(img_shape_per_entry): # Single image (e.g., (H,W,C))
                    images_to_add = np.expand_dims(images_from_tif, axis=0) # Make it (1, H, W, C)
                elif images_from_tif.ndim == len(img_shape_per_entry) + 1: # Multi-page (e.g., (P, H, W, C))
                    images_to_add = images_from_tif
                else:
                    print(f"Skipping {tif_file}: Image dimensions ({images_from_tif.shape}) do not match expected ({img_shape_per_entry}).")
                    continue

                num_images_in_tif = images_to_add.shape[0]
                total_new_images_to_add += num_images_in_tif

                # Resize the HDF5 dataset to accommodate new images
                dataset.resize(current_h5_num_images + num_images_in_tif, axis=0)
                
                # Append the new images
                dataset[current_h5_num_images : current_h5_num_images + num_images_in_tif] = images_to_add
                
                current_h5_num_images += num_images_in_tif
                print(f"Successfully exported {num_images_in_tif} image(s) from '{tif_file.name}'. Total images in H5: {current_h5_num_images}")

            except Exception as e:
                print(f"Error processing TIFF file '{tif_file.name}': {e}")
                import traceback
                print(traceback.format_exc())
    print(f"Export complete. Total images in '{h5_file_path}' dataset '/{dataset_name}': {current_h5_num_images}")


if __name__ == "__main__":
    # --- Create dummy TIFF files for testing ---
    # You can skip this part if you already have your mkate_*.tif files
    # Create a dummy H5 file if it doesn't exist, just for the "images" dataset
    dummy_h5_path = "mkate_2.h5"
    if not Path(dummy_h5_path).exists():
        print(f"Creating dummy '{dummy_h5_path}' for testing...")
        with h5py.File(dummy_h5_path, 'w') as f:
            # Create a placeholder dataset if you don't want to start from scratch.
            # This is primarily for the *masks* part of your original app.
            # The script above will create /images if it doesn't exist.
            pass

    # Create some dummy multi-page TIFFs
    for i in range(3):
        # Create a multi-page TIFF (e.g., 5 pages, 100x100 pixels, 3 channels)
        dummy_image_data = np.random.randint(0, 256, size=(5, 100, 100, 3), dtype=np.uint8)
        tif_filename = f"mkate_{i+1}.tif"
        tifffile.imwrite(tif_filename, dummy_image_data)
        print(f"Created dummy TIFF: {tif_filename} (5 pages, 100x100x3)")

    # Create a dummy single-page TIFF (e.g., 200x200 pixels, grayscale)
    single_page_data = np.random.randint(0, 256, size=(200, 200), dtype=np.uint8)
    tifffile.imwrite("mkate_single.tif", single_page_data)
    print("Created dummy TIFF: mkate_single.tif (1 page, 200x200)")
    # --- End of dummy file creation ---


    # --- Call the export function ---
    data_dir = config.RAW_DATA_DIR / 'worms'
    h5_file = data_dir / "mkate_2.h5"
    tif_pattern = "mkate_*.tif" # Matches mkate_1.tif, mkate_2.tif, etc.

    # Check the initial state of the H5 file
    if Path(h5_file).exists():
        with h5py.File(h5_file, 'r') as f_check:
            if 'images' in f_check:
                print(f"\nInitial state of '{h5_file}': /images dataset has shape {f_check['images'].shape}")
            else:
                print(f"\nInitial state of '{h5_file}': /images dataset does not exist.")
    else:
        print(f"\n'{h5_file}' does not exist yet. It will be created.")

    export_tifs_to_h5(h5_file, data_dir, tif_pattern, dataset_name='images')

    # Verify the contents after export
    if Path(h5_file).exists():
        with h5py.File(h5_file, 'r') as f_verify:
            if 'images' in f_verify:
                print(f"\nFinal state of '{h5_file}': /images dataset has shape {f_verify['images'].shape}")
                # You can also inspect the data if needed:
                # print(f_verify['images'][0].shape)
            else:
                print(f"\nError: /images dataset not found in '{h5_file}' after export.")
