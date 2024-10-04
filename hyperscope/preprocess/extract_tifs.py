import os
from tqdm import tqdm
import glob
from PIL import Image, ImageSequence
import numpy as np
import re
import cv2
from typing import List, Union, Tuple, Optional
from hyperscope.helpers.func import flatten
from hyperscope.helpers.mp import wait_for_memory
import scipy.ndimage
from time import sleep

# Monkey patch the Image.open function to support loading numpy arrays


def new_image_open_generator(old_open=Image.open):
    Image.old_open = old_open  # Save the original open function for later use

    def new_image_open(path, *args, **kwargs):
        if path.endswith('.npy'):
            arr = np.load(path, allow_pickle=False)
            if arr.dtype == np.uint8:
                return Image.fromarray(arr)
            elif arr.dtype == np.uint16:
                return Image.fromarray(arr, mode='I;16')
            else:
                raise ValueError(f"Unsupported dtype: {arr.dtype}")
        return Image.old_open(path, *args, **kwargs)

    return new_image_open


Image.open = new_image_open_generator(Image.open)


def expand_brace_pattern(pattern):
    brace_pattern = r'\{(.*?)\}'
    match = re.search(brace_pattern, pattern)
    if not match:
        return [pattern]

    parts = match.group(1).split(',')
    base_pattern = pattern.replace(match.group(0), '{}')
    expanded_patterns = [base_pattern.format(part) for part in parts]

    return expanded_patterns


def block_view(A, block_shape):
    shape = (
        A.shape[0] // block_shape[0],
        A.shape[1] // block_shape[1]) + block_shape
    strides = (
        block_shape[0] * A.strides[0],
        block_shape[1] * A.strides[1]) + A.strides
    return np.lib.stride_tricks.as_strided(A, shape=shape, strides=strides)


def calculate_snr(image, patch_size):
    blocks = block_view(image, (patch_size, patch_size))
    # Flatten the blocks to calculate the mean and std efficiently
    flattened_blocks = blocks.reshape(-1, patch_size, patch_size)
    means = np.mean(flattened_blocks, axis=(1, 2))
    stds = np.std(flattened_blocks, axis=(
        1, 2)) + 1e-5  # Avoid division by zero
    # Calculate the SNR values for each block without the need for a loop
    snr_values = means / stds
    # Reshape the SNR values to match the blocked image shape
    snr_map = snr_values.reshape(blocks.shape[0], blocks.shape[1])
    return snr_map


def smooth_snr_map(snr_map, kernel_size):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    smoothed_snr_map = scipy.ndimage.convolve(
        snr_map, kernel, mode='constant', cval=0.0)
    return smoothed_snr_map


def gen_mask(image, threshold):
    mask = np.zeros_like(image)
    mask[image > threshold] = 1
    return mask


def expand_mask(mask, patch_size):
    expanded_mask = np.kron(mask, np.ones((patch_size, patch_size)))
    return expanded_mask


def process_image(image, patch_size=32):
    image = np.array(image)
    dtype = image.dtype
    dtype_max = np.iinfo(dtype).max  # Get the maximum value for the image dtype

    # Normalize the image
    normed = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-5)

    snr_img = smooth_snr_map(
        calculate_snr(normed, patch_size),
        5)  # Calculate the SNR map
    mean, std = np.mean(snr_img), np.std(snr_img)
    # if the snr doesn't vary much, the image is likely noise
    if std < .06 and not np.isclose(std, 0.06, atol=0.01):
        return None

    # Generate the mask based on the SNR map to keep only the relevant regions
    # first convert the image to 8-bit range

    mask = gen_mask(snr_img, mean + 2 * std)  # Generate the mask

    if np.sum(mask) < mask.size * 0.001:
        return None

    # Expand the mask to the original image size
    mask = expand_mask(mask, patch_size)

    # Pad the mask if necessary to match the image size
    pad_height = (image.shape[0] - mask.shape[0] %
                  image.shape[0]) % image.shape[0]
    pad_width = (image.shape[1] - mask.shape[1] %
                 image.shape[1]) % image.shape[1]
    mask = np.pad(mask, ((0, pad_height), (0, pad_width)),
                  mode='constant', constant_values=0)

    return mask


def prepare_mask(mask, category):
    mask = mask.astype(np.uint8)
    mask *= category
    return mask


def extract_multipage_tiff(output_dir: str, lock,
                           create_dir: bool = False,
                           crop: Union[Tuple[int],
                                       int, None] = None,
                           paths: List[str] = None,
                           ):
    if create_dir:
        os.makedirs(os.path.join(output_dir, "imgs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    if lock:
        with lock:
            wait_for_memory(threshold_mb=1024, check_interval=10)

    images = [Image.open(path) for path in paths]
    image_with_path = zip(images, paths)

    for img, path in image_with_path:
        base_filename = os.path.basename(path)
        name, _ = os.path.splitext(base_filename)
        category = 1 if name.startswith("625") else 2
        # numpy arrays only have one page
        if base_filename.endswith('.npy'):
            # Uses the monkey patched Image.open function
            img_output_path = os.path.join(
                output_dir, "imgs", f"{name}.tif")
            mask_output_path = os.path.join(
                output_dir, "masks", f"{name}.npz")
            if os.path.exists(img_output_path) and os.path.exists(
                    mask_output_path):
                # continue
                pass
            img = Image.open(path)
            if crop:
                if len(crop) == 2:
                    # Crop the image symmetrically on all sides
                    crop = (
                        crop[0],
                        crop[1],
                        img.width - crop[0],
                        img.height - crop[1])
                if (-1 in crop):
                    # get the position of the -1
                    idx = crop.index(-1)
                    if idx == 0:
                        raise ValueError(
                            "Invalid crop value. -1 cannot be used for the left side.")
                    if idx == 1:
                        raise ValueError(
                            "Invalid crop value. -1 cannot be used for the upper side.")
                    if idx == 2:
                        crop = (crop[0], crop[1], img.width, crop[3])
                    if idx == 3:
                        crop = (crop[0], crop[1], crop[2], img.height)
                img = img.crop(crop)
            mask = process_image(img, 8)
            if mask is not None:
                # Prepare the mask for saving
                img = np.array(img).astype(np.uint16)
                # normalize the image to the full range of uint16
                img = cv2.normalize(
                    img, None, alpha=0, beta=int(2 ** 16 - 1),
                    norm_type=cv2.NORM_MINMAX)
                img = Image.fromarray(img)
                mask = prepare_mask(mask, category)
                np.savez(mask_output_path, mask=mask)
                img.save(img_output_path)

        else:
            for i, page in enumerate(ImageSequence.Iterator(img)):
                output_filename = f"{name}_page_{i+1}.tif"
                img_output_path = os.path.join(
                    output_dir, "imgs", output_filename)
                mask_output_path = os.path.join(
                    output_dir, "masks", f"{name}_page_{i+1}.npz")
                if os.path.exists(img_output_path) and os.path.exists(
                        mask_output_path):
                    continue
                if crop:
                    if len(crop) == 2:
                        # Crop the image symmetrically on all sides
                        crop = (
                            crop[0],
                            crop[1],
                            img.width - crop[0],
                            img.height - crop[1])
                    if (-1 in crop):
                        # get the position of the -1
                        idx = crop.index(-1)
                        if idx == 0:
                            raise ValueError(
                                "Invalid crop value. -1 cannot be used for the left side.")
                        if idx == 1:
                            raise ValueError(
                                "Invalid crop value. -1 cannot be used for the upper side.")
                        if idx == 2:
                            crop = (crop[0], crop[1], img.width, crop[3])
                        if idx == 3:
                            crop = (crop[0], crop[1], crop[2], img.height)
                    img = page.crop(crop)
                mask = process_image(img, 8)
                if mask is not None:
                    # Prepare the mask for saving
                    img = np.array(img).astype(np.uint16)
                    # normalize the image to the full range of uint16
                    img = cv2.normalize(
                        img, None, alpha=0, beta=int(2 ** 16 - 1),
                        norm_type=cv2.NORM_MINMAX)
                    img = Image.fromarray(img)
                    mask = prepare_mask(mask, category)
                    np.savez(mask_output_path, mask=mask)
                    img.save(img_output_path)

def verify_extracted_outputs(output_dir):
    files = glob.glob(os.path.join(output_dir, "*.tif"))
    for file in tqdm(files):
        try:
            img = Image.open(file)
            img.verify()
        except Exception as e:
            print(f"Error verifying {file}: {e}")
            os.remove(file)


def main(
    input_patterns: List[str],
    output: str,
    normalize: bool = False,
    create_dir: bool = False,
    crop: Optional[List[int]] = None,
    noiseframe: Optional[str] = None,
    verify: bool = False,
    no_confirm: bool = False,
    multi_process: bool = True,
    batch_size: int = 100,
    n_batches: int = -1,
    max_images: Optional[int] = None,
    df_threshold: float = 0.0,
    darkframe: Optional[str] = None,
):
    # If either is false, disable multiprocessing
    multi_process = args.multi_process
    if not args.no_confirm:
        print("This script will extract pages from multipage TIFF files.")
        print("It may take a long time to run.")
        print("Do you want to continue? (y/n)")
        response = input()
        if response.lower() != "y":
            print("Exiting...")
            exit(0)

    manager = None
    lock = None
    event = None
    pool = None
    df_path = None

    if args.crop and (len(args.crop) != 2 and len(args.crop) != 4):
        raise ValueError("Crop argument must have 2, or 4 values.")

    if args.max_images < 0:
        raise ValueError("max_images must be greater than or equal to 0.")
    args.max_images = None if args.max_images == 0 else args.max_images
    all_paths = flatten([glob.glob(input_pattern)
                        for input_pattern in args.input])
    # reverse sort the paths
    all_paths.sort(reverse=True)
    total_images = len(all_paths)
    last_index = 0
    futures = []
    if multi_process:
        import multiprocessing as mp
        pool = mp.Pool(mp.cpu_count())
        manager = mp.Manager()
        lock = manager.Lock()
        event = manager.Event()
        df_path = manager.Value(str, args.darkframe)
    for i in range(0, total_images, args.batch_size):
        # Check if we have processed the required number of batches
        if args.n_batches > 0 and last_index // batch_size >= args.n_batches:
            break
        print(f"Indexes: {i}:{i + args.batch_size}")
        paths = all_paths[i:i + args.batch_size]
        if multi_process:
            future = pool.apply_async(
                extract_multipage_tiff,
                (args.output, lock,),
                dict(
                    create_dir=args.create_dir,
                    crop=args.crop,
                    paths=paths)
            )
            futures.append(future)
        else:
            extract_multipage_tiff(
                args.output, lock, event, df_path,
                normalize=args.normalize, create_dir=args.create_dir,
                crop=args.crop, noiseframe_path=args.noiseframe,
                df_threshold=args.df_threshold, max_images=args.max_images,
                paths=paths)
        last_index += batch_size

    # Manually update tqdm as futures complete
    if multi_process:
        with tqdm(total=len(futures), desc="Processing files", unit="batch") as pbar:
            while futures:
                for future in futures[:]:
                    if future.ready():
                        # Check if the future has an exception
                        try:
                            future.get()
                        except Exception as e:
                            logger.warn(f"Error processing batch: {e}")
                        futures.remove(future)
                        pbar.update(1)
                sleep(0.001)  # Sleep for a short while to prevent high CPU usage

    if args.verify:
        print("Verifying the output images.")
        verify_extracted_outputs(args.output)
        print("Verification complete.")


if __name__ == "__main__":
    raise NotImplementedError("This script is not meant to be run directly.")
