import numpy as np
import tifffile as tiff
import os
import multiprocessing
from pathlib import Path
from hyperscope.helpers.concurrency import ConcurrentTqdm
from warnings import warn
from itertools import islice
from traceback import print_exception
from tqdm import tqdm
from glob import glob
from hyperscope.config import logger


def process_file(img, mask, unique_id, patch_size):
    '''
    Process a single image/mask pair, cropping and splitting them into patches
    of the specified size. Patches with too few non-zero pixels are skipped.
    '''

    # Get bounding box for non-zero region in the mask
    nz = np.nonzero(mask)
    if len(nz[0]) == 0:
        return
    min_y, max_y = np.min(nz[0]), np.max(nz[0])
    min_x, max_x = np.min(nz[1]), np.max(nz[1])

    # Crop to the smallest area that can encompass the full non-zero area
    crop_height = (max_y - min_y + 1)
    crop_width = (max_x - min_x + 1)

    # Adjust crop to be divisible by patch size
    crop_height = ((crop_height + patch_size - 1) // patch_size) * patch_size
    crop_width = ((crop_width + patch_size - 1) // patch_size) * patch_size

    # Update min, max values to reflect the new crop
    min_y = max(0, min_y - (min_y % patch_size))
    min_x = max(0, min_x - (min_x % patch_size))
    max_y = min(min_y + crop_height, img.shape[0])
    max_x = min(min_x + crop_width, img.shape[1])

    # Ensure the crop fits within the original image/mask dimensions
    crop_height = max_y - min_y
    crop_width = max_x - min_x

    # Crop mask and image
    cropped_mask = mask[min_y:max_y, min_x:max_x]
    cropped_img = img[min_y:max_y, min_x:max_x]

    # Adjust crop if it is still not divisible by patch size
    if cropped_mask.shape[0] % patch_size != 0:
        extra = cropped_mask.shape[0] % patch_size
        cropped_mask = cropped_mask[:-extra, :]
        cropped_img = cropped_img[:-extra, :]
        max_y -= extra

    if cropped_mask.shape[1] % patch_size != 0:
        extra = cropped_mask.shape[1] % patch_size
        cropped_mask = cropped_mask[:, :-extra]
        cropped_img = cropped_img[:, :-extra]
        max_x -= extra

    # Generate patch coordinates using cropped image indices and adjust to original image coordinates
    patch_coords = [
        (x + min_x, y + min_y, x + min_x + patch_size, y + min_y + patch_size)
        for y in range(0, cropped_mask.shape[0], patch_size)
        for x in range(0, cropped_mask.shape[1], patch_size)
    ]

    # Split into patches
    mask_patches = [
        cropped_mask[y:y + patch_size, x:x + patch_size]
        for y in range(0, cropped_mask.shape[0], patch_size)
        for x in range(0, cropped_mask.shape[1], patch_size)
    ]

    img_patches = [
        cropped_img[y:y + patch_size, x:x + patch_size]
        for y in range(0, cropped_img.shape[0], patch_size)
        for x in range(0, cropped_img.shape[1], patch_size)
    ]


    threshold = .2

    # Save patches
    for i, (mask_patch, img_patch, coords) in enumerate(
            zip(mask_patches, img_patches, patch_coords)):
        # Skip patches with too few non-zero pixels
        if np.count_nonzero(mask_patch) < threshold * patch_size ** 2:
            continue
        yield unique_id, i, img_patch, mask_patch, coords


def process_file_pair(batch, img_folder, mask_folder, output_folder):
    '''
    Process a batch of image/mask pairs, splitting them into patches of
    different sizes and saving them to disk. The patches are saved in
    subdirectories of the output folder corresponding to their size.
    '''
    powers_of_two = [2 ** i for i in range(5, 8)]  # [32, 64, 128]
    pairs_to_save = []
    loaded_pairs = {}
    for mask_name, img_name in batch:
        img_path = img_folder / (img_name + ".tif")
        mask_path = mask_folder / (mask_name + ".npz")
        try:
            img = tiff.memmap(img_path)
        except FileNotFoundError:
            # try with tiff extension
            img_path = img_folder / (img_name + ".tiff")
            img = tiff.memmap(img_path)
        mask = np.load(mask_path, mmap_mode='r')['mask']
        loaded_pairs[img_name] = (img, mask)

    for img_name, (img, mask) in loaded_pairs.items():
        for patch_size in powers_of_two:
            for uid, idx, _img, _mask, coords in process_file(img, mask,
                                                      img_name, patch_size):
                pairs_to_save.append((_img, _mask, uid, idx, patch_size, coords))

    # Save the pairs
    # create a csv file to store the patch coordinates
    csv_file = output_folder / "patch_coordinates.csv"
    with open(csv_file, 'w') as f:
        f.write("uid_idx,patch_size,x0,y0,x1,y1\n")
        for img, mask, uid, idx, patch_size, (x0, y0, x1, y1) in pairs_to_save:
            f.write(f"{uid}_{idx},{patch_size},{x0},{y0},{x1},{y1}\n")
            img_output_path = output_folder / "imgs" / \
                f'{patch_size}x{patch_size}' / f"{uid}_{idx}.tif"
            mask_output_path = output_folder / "masks" / \
                f'{patch_size}x{patch_size}' / f"{uid}_{idx}.npz"
            np.savez_compressed(mask_output_path, mask=mask)
            tiff.imwrite(img_output_path, img.astype(np.uint16))


def into_batches(iterable, batch_size):
    '''
    Split an iterable into batches of a specified size.
    '''
    iterator = iter(iterable)
    while True:
        first = list(islice(iterator, 1))  # Get the first element (if it exists)
        if not first:
            break  # If there's no first element, exit the loop
        rest = list(islice(iterator, batch_size - 1))  # Get the remaining elements for the batch
        yield first + rest  # Combine the first element with the rest of the batch


def files_match(mask_file, img_file):
    '''
    Check if the mask and image files match by comparing their names.
    Composite mask files are matched with composite image files by correcting
    the name of the mask file to match the image file.
    '''
    
    return prepare_name_for_comparison(mask_file) == prepare_name_for_comparison(img_file)


def match_files(mask_files, img_files):
    '''
    Match mask and image files by comparing their names. If a mask file does
    not have a matching image file, it is removed from the list. If an image
    file does not have a matching mask file, it is removed from the list.
    '''
    mismatched = list(map(
        lambda x: x,
        filter(lambda x: not files_match(*x),
               zip(mask_files, img_files))))
    for m in mismatched:
        # check if it is the mask or the image that is missing
        prepared_names = list(map(prepare_name_for_comparison, m))
        if not any(map(lambda x: x in prepared_names, mask_files)):
            print(f"Mask file {m[0]} does not have a matching image file")
            mask_files.remove(m[0])
            continue
        if not any(map(lambda x: x in prepared_names, img_files)):
            print(f"Image file {m[1]} does not have a matching mask file")
            img_files.remove(m[1])

    print(f"Identified and corrected {len(mismatched)} mismatched files")
    return mask_files, img_files


def prepare_name_for_comparison(name):
    '''
    Prepare a file name for comparison by removing the extension and
    replacing "composite_mask" with "composite_image".
    '''
    if name.startswith("composite_mask"):
        name = name.replace("composite_mask", "composite_image")

    # remove the extension
    name = name.split('.')[0]
    return name


def prepare_directory_structure(output_dir):
    '''
    Create the directory structure for the output files.
    Create subdirectories for images and masks of different sizes.
    '''
    # Create output directories
    output_masks_dir = output_dir / "masks"
    output_imgs_dir = output_dir / "imgs"
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    output_imgs_dir.mkdir(parents=True, exist_ok=True)
    for patch_size in [32, 64, 128]:
        (output_masks_dir / f"{patch_size}x{patch_size}").mkdir(
            parents=True, exist_ok=True)
        (output_imgs_dir / f"{patch_size}x{patch_size}").mkdir(
            parents=True, exist_ok=True)


def make_assertions(mask_files, img_files):
    '''
    Verify that the mask and image files are appropriately aligned and that
    there are no missing files.
    '''
    assert all(map(lambda x: files_match(*x), zip(mask_files, img_files))
               ), "Some mask/image pairs do not match"
    assert len(mask_files) == len(
        img_files), "Number of mask files does not match number of image files"
    assert len(mask_files) > 0, "No mask files found"
    assert len(img_files) > 0, "No image files found"


def main(mask_dir, img_dir, output_dir, mp=True, batch_size=25):
    # Convert to Path objects for consistency
    output_dir = Path(output_dir)
    mask_dir = Path(mask_dir)
    img_dir = Path(img_dir)
    mask_files = []
    img_files = []


    # Create the output directory structure
    prepare_directory_structure(output_dir)

    if not mask_dir.exists() or not img_dir.exists():
        # try to glob the directory
        logger.info("Globbing directories")
        globbed_mask = glob(str(mask_dir))
        globbed_img = glob(str(img_dir))
        if not globbed_mask or not globbed_img:
            raise ValueError("No files found in the specified directories")
        
        mask_dir = Path(globbed_mask[0]).parent
        img_dir = Path(globbed_img[0]).parent
        
        mask_files = [Path(f).stem for f in globbed_mask]
        img_files = [Path(f).stem for f in globbed_img]
        
    else:
        mask_files = [Path(f).stem for f in os.listdir(
            mask_dir) if f.endswith('.npz')]
        img_files = [Path(f).stem for f in os.listdir(img_dir) if (f.endswith('.tif') or f.endswith('.tiff'))]
    
    # Create args list for multiprocessing
    mask_files = sorted(mask_files)
    img_files = sorted(img_files)

    # Check for and correct mismatched files
    mask_files, img_files = match_files(mask_files, img_files)
    make_assertions(mask_files, img_files)
    iterables = zip(mask_files, img_files)
    
    if not mp:
        # Single process
        for batch in into_batches(iterables, batch_size=batch_size):
            process_file_pair(batch, img_dir, mask_dir, output_dir)
        return

    batches = list(into_batches(iterables, batch_size=batch_size))
    max_workers = multiprocessing.cpu_count()  # Use all the available cores
    with multiprocessing.Pool(max_workers) as pool:
        tasks = [pool.apply_async(
            process_file_pair,
            args=(batch, img_dir, mask_dir, output_dir))
            for batch in batches]

        for ok, res in ConcurrentTqdm(
                tasks, total=len(tasks),
                desc="Processing files"):
            if not ok:
                print_exception(type(res), res, res.__traceback__)


if __name__ == "__main__":
    raise NotImplementedError("This script is not meant to be run directly.")
