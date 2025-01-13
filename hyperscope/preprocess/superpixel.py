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
from cv2 import imread, IMREAD_UNCHANGED


def process_file(img, mask, unique_id, patch_size):
    """
    Process a single image/mask pair, splitting them into patches
    of the specified size across the entire image.
    """
    if img.shape != mask.shape:
        raise ValueError("Image and mask must have the same dimensions")

    # Get full image dimensions
    height, width = img.shape[0], img.shape[1]

    # Calculate padding needed to make dimensions divisible by patch_size
    pad_height = (patch_size - height % patch_size) % patch_size
    pad_width = (patch_size - width % patch_size) % patch_size

    # Pad image and mask if necessary
    if pad_height > 0 or pad_width > 0:
        # Pad with zeros
        padded_img = np.pad(img, ((0, pad_height), (0, pad_width)), mode="constant")
        padded_mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode="constant")
    else:
        padded_img = img
        padded_mask = mask

    # Calculate number of patches in each dimension
    n_patches_height = padded_img.shape[0] // patch_size
    n_patches_width = padded_img.shape[1] // patch_size

    patch_index = 0
    # Iterate over the entire image
    for y in range(n_patches_height):
        y_start = y * patch_size
        y_end = y_start + patch_size

        for x in range(n_patches_width):
            x_start = x * patch_size
            x_end = x_start + patch_size

            # Extract patches
            img_patch = padded_img[y_start:y_end, x_start:x_end]
            mask_patch = padded_mask[y_start:y_end, x_start:x_end]

            # Calculate coordinates in original image
            orig_x_start = min(x_start, width)
            orig_y_start = min(y_start, height)
            orig_x_end = min(x_end, width)
            orig_y_end = min(y_end, height)

            coords = (orig_x_start, orig_y_start, orig_x_end, orig_y_end)

            # Yield all patches regardless of content
            yield unique_id, patch_index, img_patch, mask_patch, coords
            patch_index += 1


def process_file_pair(batch, img_folder, mask_folder, output_folder, patch_sizes=[64], save_coords=False):
    """
    Process a batch of image/mask pairs, splitting them into patches of
    different sizes and saving them to disk. The patches are saved in
    subdirectories of the output folder corresponding to their size.
    """
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
        except ValueError:
            try:
                img = imread(img_path, IMREAD_UNCHANGED)
            except Exception as e:
                logger.error(f"Could not open file {img_path}: {e}")
                exit(1)
        mask = np.load(mask_path, mmap_mode="r")["mask"]
        loaded_pairs[img_name] = (img, mask)

    for img_name, (img, mask) in loaded_pairs.items():
        for patch_size in patch_sizes:
            for uid, idx, _img, _mask, coords in process_file(img, mask, img_name, patch_size):
                pairs_to_save.append((_img, _mask, uid, idx, patch_size, coords))
                

    # Save the pairs
    # create a csv file to store the patch coordinates
    if save_coords:
        csv_file = output_folder / "patch_coordinates.csv"
        with open(csv_file, "w") as f:
            f.write("uid_idx,patch_size,x0,y0,x1,y1\n")
            for img, mask, uid, idx, patch_size, (x0, y0, x1, y1) in pairs_to_save:
                f.write(f"{uid}_{idx},{patch_size},{x0},{y0},{x1},{y1}\n")
                img_output_path = (
                    output_folder / "imgs" / f"{patch_size}x{patch_size}" / f"{uid}_{idx}.tif"
                )
                mask_output_path = (
                    output_folder / "masks" / f"{patch_size}x{patch_size}" / f"{uid}_{idx}.npz"
                )
                np.savez_compressed(mask_output_path, mask=mask)
                tiff.imwrite(img_output_path, img.astype(np.uint16))
    else:
        for img, mask, uid, idx, patch_size, _ in pairs_to_save:
            img_output_path = (
                output_folder / "imgs" / f"{patch_size}x{patch_size}" / f"{uid}_{idx}.tif"
            )
            mask_output_path = (
                output_folder / "masks" / f"{patch_size}x{patch_size}" / f"{uid}_{idx}.npz"
            )
            np.savez_compressed(mask_output_path, mask=mask)
            tiff.imwrite(img_output_path, img.astype(np.uint16))
        


def into_batches(iterable, batch_size):
    """
    Split an iterable into batches of a specified size.
    """
    iterator = iter(iterable)
    while True:
        first = list(islice(iterator, 1))  # Get the first element (if it exists)
        if not first:
            break  # If there's no first element, exit the loop
        rest = list(islice(iterator, batch_size - 1))  # Get the remaining elements for the batch
        yield first + rest  # Combine the first element with the rest of the batch


def files_match(mask_file, img_file):
    """
    Check if the mask and image files match by comparing their names.
    Composite mask files are matched with composite image files by correcting
    the name of the mask file to match the image file.
    """

    return prepare_name_for_comparison(mask_file) == prepare_name_for_comparison(img_file)


def match_files(mask_files, img_files):
    """
    Match mask and image files by comparing their names. If a mask file does
    not have a matching image file, it is removed from the list. If an image
    file does not have a matching mask file, it is removed from the list.
    """
    mismatched = list(
        map(lambda x: x, filter(lambda x: not files_match(*x), zip(mask_files, img_files)))
    )
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
    """
    Prepare a file name for comparison by removing the extension and
    replacing "composite_mask" with "composite_image".
    """
    if name.startswith("composite_mask"):
        name = name.replace("composite_mask", "composite_image")

    # remove the extension
    name = name.split(".")[0]
    return name


def prepare_directory_structure(output_dir):
    """
    Create the directory structure for the output files.
    Create subdirectories for images and masks of different sizes.
    """
    # Create output directories
    output_masks_dir = output_dir / "masks"
    output_imgs_dir = output_dir / "imgs"
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    output_imgs_dir.mkdir(parents=True, exist_ok=True)
    for patch_size in [32, 64, 128]:
        (output_masks_dir / f"{patch_size}x{patch_size}").mkdir(parents=True, exist_ok=True)
        (output_imgs_dir / f"{patch_size}x{patch_size}").mkdir(parents=True, exist_ok=True)


def make_assertions(mask_files, img_files):
    """
    Verify that the mask and image files are appropriately aligned and that
    there are no missing files.
    """
    assert all(
        map(lambda x: files_match(*x), zip(mask_files, img_files))
    ), "Some mask/image pairs do not match"
    assert len(mask_files) == len(
        img_files
    ), "Number of mask files does not match number of image files"
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
        mask_files = [Path(f).stem for f in os.listdir(mask_dir) if f.endswith(".npz")]
        img_files = [
            Path(f).stem
            for f in os.listdir(img_dir)
            if (f.endswith(".tif") or f.endswith(".tiff"))
        ]

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
        tasks = [
            pool.apply_async(process_file_pair, args=(batch, img_dir, mask_dir, output_dir))
            for batch in batches
        ]

        for ok, res in ConcurrentTqdm(tasks, total=len(tasks), desc="Processing files"):
            if not ok:
                print_exception(type(res), res, res.__traceback__)


if __name__ == "__main__":
    raise NotImplementedError("This script is not meant to be run directly.")
