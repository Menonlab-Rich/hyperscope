import os
import random
import multiprocessing
from itertools import islice
import numpy as np
import tifffile as tiff
import bisect
from concurrency.concurrent_tqdm import ConcurrentTqdm
import traceback
from pathlib import Path


def process_batch(batch, image_dir, mask_dir, output_dir):
    output_img_dir = os.path.join(output_dir, "imgs")
    output_mask_dir = os.path.join(output_dir, "masks")
    for i, (img_path_625, mask_path_625, img_path_605, mask_path_605, batch_id) in enumerate(
        batch
    ):
        # Load the images and masks using memory mapping
        img_625 = tiff.memmap(os.path.join(image_dir, img_path_625), mode="r")
        mask_625 = np.load(os.path.join(mask_dir, mask_path_625), mmap_mode="r")["mask"]
        img_605 = tiff.memmap(os.path.join(image_dir, img_path_605), mode="r")
        mask_605 = np.load(os.path.join(mask_dir, mask_path_605), mmap_mode="r")["mask"]

        # Create new composite image and mask
        composite_image = np.zeros_like(img_625)
        composite_mask = np.zeros_like(mask_625)

        # Identify regions where the masks have non-zero values
        mask_625_indices = mask_625 > 0
        mask_605_indices = mask_605 > 0

        # Identify overlapping regions
        overlap_indices = mask_625_indices & mask_605_indices
        non_overlap_625 = mask_625_indices & ~overlap_indices
        non_overlap_605 = mask_605_indices & ~overlap_indices

        # Combine the masks and images, resolving overlaps randomly
        if random.choice([True, False]):
            composite_mask[overlap_indices] = 1
            composite_image[overlap_indices] = img_625[overlap_indices]
        else:
            composite_mask[overlap_indices] = 2
            composite_image[overlap_indices] = img_605[overlap_indices]

        # Handle non-overlapping regions
        composite_mask[non_overlap_625] = 1
        composite_mask[non_overlap_605] = 2

        composite_image[non_overlap_625] = img_625[non_overlap_625]
        composite_image[non_overlap_605] = img_605[non_overlap_605]

        # Save the composite image and mask
        epoch = batch_id + i  # Use the batch index and local index to calculate the global index
        composite_image_path = os.path.join(output_img_dir, f"composite_image_{epoch}.tiff")
        composite_mask_path = os.path.join(output_mask_dir, f"composite_mask_{epoch}.npz")

        tiff.imsave(composite_image_path, composite_image)
        np.savez_compressed(composite_mask_path, mask=composite_mask)


def split_into_batches(iterable, batch_size):
    iterator = iter(iterable)
    for first in iterator:
        yield list(islice(iterator, batch_size - 1))


def strip_extension(path):
    p = Path(path)
    return p.stem


def main():
    image_dir = r"D:\CZI_scope\code\preprocess\training_set\imgs"
    mask_dir = r"D:\CZI_scope\code\preprocess\training_set\masks"
    output_dir = r"D:\CZI_scope\code\preprocess\training_set"

    # Ensure output directory exists along with subdirectories for images and masks
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "imgs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    # Create the sorted set of all image paths in the image directory
    all_image_paths = sorted(os.listdir(image_dir))

    # Search for image paths that begin with '625' using a balanced binary search
    idx_625 = bisect.bisect_left(all_image_paths, "625")

    paths_625 = all_image_paths[idx_625:]
    paths_605 = all_image_paths[:idx_625]

    # Set the parameters for the composite image generation
    num_images = 10000  # Number of composite images to generate
    num_samples = 100  # Number of unique images and masks to sample from each category
    batch_size = 500  # Number of images per batch

    # Randomly sample from the paths
    sampled_625 = random.sample(paths_625, num_samples)
    sampled_605 = random.sample(paths_605, num_samples)

    # Pair the images and masks
    pairs = [
        (
            sampled_625[i % num_samples],
            f"{strip_extension(sampled_625[i % num_samples])}.npz",
            sampled_605[i % num_samples],
            f"{strip_extension(sampled_605[i % num_samples])}.npz",
            i,
        )
        for i in range(num_images)
    ]

    # Split pairs into batches
    batches = list(split_into_batches(pairs, batch_size))

    # Use multiprocessing to process batches concurrently with ConcurrentTqdm progress bar
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        tasks = [
            pool.apply_async(process_batch, (batch, image_dir, mask_dir, output_dir))
            for batch in batches
        ]

        # Use ConcurrentTqdm to track the progress of the tasks
        for ok, o in ConcurrentTqdm(tasks, total=len(tasks)):
            if not ok:
                # o is an exception so print the stack trace
                traceback.print_exception(type(o), o, o.__traceback__)


if __name__ == "__main__":
    main()
