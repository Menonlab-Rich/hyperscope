import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

def main(tif_dir, npz_dir, dry_run=False):
    changes = {"removed": [], "resized": []}
    # Get all npz files
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    print(f'found {len(npz_files)} to process') 
    if dry_run:
        print("Running in dry run mode")
    for npz_file in tqdm(npz_files, desc='Processing Files', unit='file'):
        # Get corresponding tif filename
        base_name = os.path.splitext(npz_file)[0]
        tif_file = base_name + '.tif'
        
        # Check if corresponding tif exists
        if not os.path.exists(os.path.join(tif_dir, tif_file)):
            print(f"No matching TIF file for {npz_file}")
            continue
            
        # Load files
        npz_path = os.path.join(npz_dir, npz_file)
        tif_path = os.path.join(tif_dir, tif_file)
        
        # Load NPZ and TIF
        npz_data = np.load(npz_path)
        mask = npz_data['arr_0']  # Adjust key if needed
        image = cv2.imread(tif_path, -1)  # -1 to preserve original format
        
        # Resize mask to match image size
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                                interpolation=cv2.INTER_AREA)
        
        # Trim 100 pixels from top and bottom
        image[:100, :] = 0
        image[-100:, :] = 0
        mask_resized[:100, :] = 0
        mask_resized[-100:, :] = 0
        
        # Check number of non-zero pixels
        if np.count_nonzero(mask_resized) <= 5:
            if dry_run:
                changes['removed'].append(tif_path)
                continue
            # Delete both files
            os.remove(npz_path)
            os.remove(tif_path)
            print(f"Deleted {npz_file} and {tif_file} due to insufficient non-zero pixels")
        else:
            # Save processed files
            if dry_run:
                if mask.shape != image.shape:
                    changes['resized'] = npz_path
                continue
            np.savez_compressed(npz_path, mask_resized)
            cv2.imwrite(tif_path, image)
            print(f"Processed and saved {npz_file} and {tif_file}")

    if dry_run:
        print(f'removed: {len(changes["removed"])}, resized: {len(changes["resized"])}')

# Example usage
if __name__ == "__main__":
	raise ValueError("This script should not be run directly")
