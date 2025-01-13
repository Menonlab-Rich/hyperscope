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
from time import sleep
from loguru import logger
from pathlib import Path
from scipy import ndimage
from skimage.filters import threshold_multiotsu
import pywt
import inspect
import traceback


def format_error_message(e: Exception) -> str:
    """Format error message with stack trace information."""
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    filename = caller_frame.f_code.co_filename
    line_number = caller_frame.f_lineno
    
    error_details = {
        'error_type': type(e).__name__,
        'error_message': str(e),
        'filename': filename,
        'line_number': line_number,
        'traceback': ''.join(traceback.format_tb(e.__traceback__))
    }
    
    return (f"Error processing batch:\n"
            f"Location: {error_details['filename']}:{error_details['line_number']}\n"
            f"Error Type: {error_details['error_type']}\n"
            f"Message: {error_details['error_message']}\n"
            f"Traceback:\n{error_details['traceback']}")


# Monkey patch the Image.open function to support loading numpy arrays


def new_image_open_generator(old_open=Image.open):
    Image.old_open = old_open  # Save the original open function for later use

    def new_image_open(path, *args, **kwargs):
        if path.endswith(".npy"):
            arr = np.load(path, allow_pickle=False)
            if arr.dtype == np.uint8:
                return Image.fromarray(arr)
            elif arr.dtype == np.uint16:
                return Image.fromarray(arr, mode="I;16")
            else:
                raise ValueError(f"Unsupported dtype: {arr.dtype}")
        return Image.old_open(path, *args, **kwargs)

    return new_image_open


Image.open = new_image_open_generator(Image.open)


def expand_brace_pattern(pattern):
    brace_pattern = r"\{(.*?)\}"
    match = re.search(brace_pattern, pattern)
    if not match:
        return [pattern]

    parts = match.group(1).split(",")
    base_pattern = pattern.replace(match.group(0), "{}")
    expanded_patterns = [base_pattern.format(part) for part in parts]

    return expanded_patterns


def block_view(A, block_shape):
    shape = (A.shape[0] // block_shape[0], A.shape[1] // block_shape[1]) + block_shape
    strides = (block_shape[0] * A.strides[0], block_shape[1] * A.strides[1]) + A.strides
    return np.lib.stride_tricks.as_strided(A, shape=shape, strides=strides)


def process_image(image):
    image = np.array(image)
    thresholds = threshold_multiotsu(image)
    return np.digitize(image, bins=thresholds).astype(np.uint8)


def prepare_mask(mask, category):
    mask[mask == 2] += category - 1
    return mask


def analyze_horizontal_components(image: np.ndarray, wavelet: str = 'db2', level: int = 1) -> tuple:
    """
    Analyze horizontal components of an image using Daubechies wavelet transform.
    
    Args:
        image: Input image as numpy array (uint16)
        wavelet: Wavelet to use (default: 'db1')
        level: Decomposition level (default: 1)
    
    Returns:
        tuple: Coefficients and horizontal detail coefficients
    """
    # Convert to float64 for wavelet transform
    image_float = np.array(image).astype(np.float64)
    
    # Perform 2D wavelet transform
    coeffs = pywt.wavedec2(image_float, wavelet, level=level)
    
    # Get horizontal details at each level
    horizontal_details = [detail[1] for detail in coeffs[1:]]
    
    return horizontal_details


def extract_multipage_tiff(
    output_dir: str,
    lock,
    create_dir: bool = False,
    crop: Union[Tuple[int], int, None] = None,
    paths: List[str] = None,
):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
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
        category = 1 if name.startswith("mkate") else 2
        # numpy arrays only have one page
        if base_filename.endswith(".npy"):
            # Uses the monkey patched Image.open function
            img_output_path = os.path.join(output_dir, "imgs", f"{name}.tif")
            mask_output_path = os.path.join(output_dir, "masks", f"{name}.npz")
            if os.path.exists(img_output_path) and os.path.exists(mask_output_path):
                # continue
                pass
            img = Image.open(path)
            if crop:
                if len(crop) == 2:
                    # Crop the image symmetrically on all sides
                    crop = (crop[0], crop[1], img.width - crop[0], img.height - crop[1])
                if -1 in crop:
                    # get the position of the -1
                    idx = crop.index(-1)
                    if idx == 0:
                        raise ValueError(
                            "Invalid crop value. -1 cannot be used for the left side."
                        )
                    if idx == 1:
                        raise ValueError(
                            "Invalid crop value. -1 cannot be used for the upper side."
                        )
                    if idx == 2:
                        crop = (crop[0], crop[1], img.width, crop[3])
                    if idx == 3:
                        crop = (crop[0], crop[1], crop[2], img.height)
                img = img.crop(crop)
            hvar = analyze_horizontal_components(img)[0].var()
            img = clahe.apply(np.array(img).astype(np.uint16)).astype(np.uint16)
            mask = process_image(img) if hvar >= 3 else np.zeros(img.shape, dtype=np.uint8)
            mask = prepare_mask(mask, category)
            np.savez(mask_output_path, mask=mask)
            img = cv2.normalize(
                img, None, alpha=0, beta=int(2**16 - 1), norm_type=cv2.NORM_MINMAX
            )
            cv2.imwrite(img_output_path, img)

        else:
            for i, page in enumerate(ImageSequence.Iterator(img)):
                output_filename = f"{name}_page_{i+1}.tif"
                img_output_path = os.path.join(output_dir, "imgs", output_filename)
                mask_output_path = os.path.join(output_dir, "masks", f"{name}_page_{i+1}.npz")
                if os.path.exists(img_output_path) and os.path.exists(mask_output_path):
                    continue
                if crop:
                    if len(crop) == 2:
                        # Crop the image symmetrically on all sides
                        crop = (crop[0], crop[1], img.width - crop[0], img.height - crop[1])
                    if -1 in crop:
                        # get the position of the -1
                        idx = crop.index(-1)
                        if idx == 0:
                            raise ValueError(
                                "Invalid crop value. -1 cannot be used for the left side."
                            )
                        if idx == 1:
                            raise ValueError(
                                "Invalid crop value. -1 cannot be used for the upper side."
                            )
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
                        img, None, alpha=0, beta=int(2**16 - 1), norm_type=cv2.NORM_MINMAX
                    )
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
    create_dir: bool = True,
    crop: Optional[List[int]] = None,
    noiseframe: Optional[str] = None,
    verify: bool = False,
    no_confirm: bool = False,
    multi_process: bool = True,
    batch_size: int = 100,
    n_batches: int = -1,
    max_images: Optional[int] = 0,
    df_threshold: float = 0.0,
    darkframe: Optional[str] = None,
):
    # If either is false, disable multiprocessing
    if not no_confirm:
        print("This script will extract pages from multipage TIFF files.")
        print("It may take a long time to run.")
        print("Do you want to continue? (y/n)")
        response = input()
        if response.lower() != "y":
            print("Exiting...")
            exit(0)

    output = Path(output)
    if create_dir:
        output.mkdir(exist_ok=True)
        (output / "masks").mkdir(exist_ok=True)
        (output / "imgs").mkdir(exist_ok=True)
    manager = None
    lock = None
    event = None
    pool = None
    df_path = None

    if crop and (len(crop) != 2 and len(crop) != 4):
        raise ValueError("Crop argument must have 2, or 4 values.")

    if max_images < 0:
        raise ValueError("max_images must be greater than or equal to 0.")
    max_images = None if max_images == 0 else max_images
    all_paths = flatten([glob.glob(input_pattern) for input_pattern in input_patterns])
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
        df_path = manager.Value(str, darkframe)
    for i in range(0, total_images, batch_size):
        # Check if we have processed the required number of batches
        if n_batches > 0 and last_index // batch_size >= n_batches:
            break
        print(f"Indexes: {i}:{i + batch_size}")
        paths = all_paths[i : i + batch_size]
        if multi_process:
            future = pool.apply_async(
                extract_multipage_tiff,
                (
                    output,
                    lock,
                ),
                dict(create_dir=create_dir, crop=crop, paths=paths),
            )
            futures.append(future)
        else:
            extract_multipage_tiff(
                output,
                lock,
                event,
                df_path,
                normalize=normalize,
                create_dir=create_dir,
                crop=crop,
                noiseframe_path=noiseframe,
                df_threshold=df_threshold,
                max_images=max_images,
                paths=paths,
            )
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
                            logger.warning(format_error_message(e))
                        futures.remove(future)
                        pbar.update(1)
                sleep(0.001)  # Sleep for a short while to prevent high CPU usage

    if verify:
        print("Verifying the output images.")
        verify_extracted_outputs(output)
        print("Verification complete.")


if __name__ == "__main__":
    raise NotImplementedError("This script is not meant to be run directly.")
