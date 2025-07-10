import h5py
import numpy as np
from hyperscope import config
from pathlib import Path

def main(file_path: Path, chunk_size: int = 100):
    """
    Checks if any value in the 'masks' dataset of an HDF5 file is greater than 0,
    processing the data in chunks to manage memory.

    Args:
        file_path (Path): The path to the HDF5 file.
        chunk_size (int): The number of masks to load into memory at a time.
                          Adjust this based on your available RAM and mask size.
    """
    file_path = Path(file_path) # Ensure it's a Path object

    if not file_path.exists():
        print(f"Error: H5 file not found at {file_path}")
        return

    with h5py.File(file_path, 'r') as f:
        if 'masks' not in f:
            print(f"Error: 'masks' dataset not found in {file_path}")
            return

        masks_dataset = f['masks']
        num_masks = masks_dataset.shape[0]

        print(f"Checking {num_masks} masks in chunks of {chunk_size}...")

        found_non_empty_mask = False
        max_mask_value = 0 # To track the global max value

        # Iterate over the dataset in chunks
        for i in range(0, num_masks, chunk_size):
            # Calculate the end index for the current chunk
            end_index = min(i + chunk_size, num_masks)

            # Load only the current chunk into memory
            # Slicing the h5py dataset directly returns a NumPy array
            current_chunk: np.ndarray = masks_dataset[i:end_index]

            # Perform the 'any' check on the current chunk
            if (current_chunk > 0).any():
                break
                # No need to break immediately if you also want the global max
                # If you only care about "any non-empty", you can break here:
                # break

            # Update the maximum value found so far
            current_chunk_max = current_chunk.max()
            if current_chunk_max > max_mask_value:
                max_mask_value = current_chunk_max

            print(f"Processed masks {i} to {end_index-1}. Current max: {max_mask_value}", end='\r')


        print("\nFinished checking masks.")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser('Verify Masks')
    parser.add_argument('file_path', help='path to the hdf file')

    args = parser.parse_args()
    main(args.file_path)

