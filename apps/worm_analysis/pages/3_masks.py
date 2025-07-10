import os
import traceback
from pathlib import Path
from typing import Tuple

from matplotlib import pyplot as plt
import cv2
import h5py
import numpy as np
import streamlit as st
from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm, ColorPalette
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.util import view_as_windows

from utils import calc_crops, find_regions_and_generate_mask


def draw_grid(image_8bit_bgr, mask, patch_shape, mask_perc, step_h, step_w, highlight):
    """
    Draws a grid on the image and highlights patches that meet the mask criteria.
    """
    # Convert the BGR image to BGRA to handle transparency for the overlay
    img_with_overlay = cv2.cvtColor(image_8bit_bgr, cv2.COLOR_BGR2BGRA)
    overlay = img_with_overlay.copy()  # Create a transparent layer for drawing highlights

    img_h, img_w, _ = img_with_overlay.shape
    patch_h, patch_w = patch_shape

    # Loop through the image by steps to find the top-left corner of each patch
    for y in range(0, img_h - patch_h + 1, step_h):
        for x in range(0, img_w - patch_w + 1, step_w):
            # Define the full patch region
            y1, x1 = y + patch_h, x + patch_w
            
            # Get the corresponding mask patch
            mask_patch = mask[y:y1, x:x1]
            
            # The mask values are labels (1, 2, etc.). To get a percentage, we count non-zero pixels.
            num_mask_pixels = np.count_nonzero(mask_patch)
            total_patch_pixels = patch_h * patch_w
            
            # Check if the percentage of mask pixels is above the threshold
            if (num_mask_pixels / total_patch_pixels) > mask_perc:
                # If it is, draw a semi-transparent red rectangle on the overlay layer
                if highlight:
                    cv2.rectangle(overlay, (x, y), (x1, y1), (0, 0, 255, 32), -1)  # BGRA color, 64 is ~25% alpha

    # Blend the overlay with the original image
    alpha = 0.5  # Transparency factor
    cv2.addWeighted(overlay, alpha, img_with_overlay, 1 - alpha, 0, img_with_overlay)

    # Draw the green grid lines on top of the blended image
    if step_h > 0 and step_w > 0:
        for y_grid in range(0, img_h, step_h):
            cv2.line(img_with_overlay, (0, y_grid), (img_w, y_grid), (0, 255, 0, 255), 1)
        for x_grid in range(0, img_w, step_w):
            cv2.line(img_with_overlay, (x_grid, 0), (x_grid, img_h), (0, 255, 0, 255), 1)

    return img_with_overlay

def process_events(evts: np.ndarray, shape: Tuple[int, int], acc_time_us: int):
    gen = OnDemandFrameGenerationAlgorithm(shape[1], shape[0], accumulation_time_us=acc_time_us, palette=ColorPalette.Dark)
    ts = evts['t'][-1]
    im = np.zeros(shape=(*shape,3), dtype=np.uint8)
    if ts >= acc_time_us:
        gen.process_events(evts)
        gen.generate(acc_time_us, im)
    print(f'min: {im.min()}, max: {im.max()}')
    return im

def files_exist(*files):
    for f in files:
        if not os.path.exists(f) or not os.path.isfile(f):
            return False
    return True

# --- Streamlit App ---

st.set_page_config(layout="wide")

st.title("Image Patching Visualization Tool")

# --- Sidebar for Parameters ---
st.sidebar.header("Processing Parameters")

# --- File Uploaders ---
st.sidebar.subheader("1. File Selection")
h5_file = st.sidebar.text_input("Path to H5 File")
stats_file = st.sidebar.text_input("Path to stats file")
events_file = st.sidebar.text_input("Path to event data")


# --- Parameter Sliders ---
st.sidebar.subheader("2. Adjust Parameters")
crop_top = st.sidebar.slider("Crop Top", 0, 500, 0)
crop_bottom = st.sidebar.slider("Crop Bottom", 0, 500, 0)
crop_left = st.sidebar.slider("Crop Left", 0, 500, 0)
crop_right = st.sidebar.slider("Crop Right", 0, 500, 0)

window_height = st.sidebar.slider("Window Height (for mask)", 1, 100, 16, step=2)
window_width = st.sidebar.slider("Window Width (for mask)", 1, 100, 16, step=2)
min_area = st.sidebar.slider("Min Area (for mask)", 100, 10000, 1000)
max_area = st.sidebar.slider("Max Area (for mask)", 100, int(10e6), 500000)
threshold = st.sidebar.slider("Threshold (for mask)", 0.1, 5.0, 1.5, 0.1)
n_clusters = st.sidebar.slider("Number of K-Means Clusters", 2, 10, 3, help="Number of classes to find within the detected regions.")


patch_height = st.sidebar.slider("Patch Height", 32, 512, 128, step=16)
patch_width = st.sidebar.slider("Patch Width", 32, 512, 128, step=16)
overlap_height = st.sidebar.slider("Overlap Height", 0, patch_height - 1, 0)
overlap_width = st.sidebar.slider("Overlap Width", 0, patch_width - 1, 0)

mask_perc = st.sidebar.slider("Min Mask %", 0.0, 1.0, 0.01, 0.01, help="Patches with a mask area percentage below this will be discarded.")

# -- Highlight
do_highlight = st.sidebar.checkbox("Highlight", True)



# --- Main App Logic ---
if h5_file and stats_file and events_file and files_exist(h5_file, stats_file, events_file):
    try:
        # Load stats
        stats = np.load(stats_file)
        dataset_mean = stats['mean']
        dataset_std = stats['std']
        st.sidebar.success(f"Loaded Stats: Mean={dataset_mean:.2f}, Std={dataset_std:.2f}")

        with h5py.File(h5_file, "r") as hf:
            acc_time = 50_000
            images_ds = hf["images"]
            num_images = images_ds.shape[0]

            # --- Image Selection ---
            st.sidebar.subheader("3. Select Image")
            img_index = st.sidebar.selectbox(f"Select Image (0 to {num_images - 1})", range(num_images))
            iterator = EventsIterator(events_file, start_ts=img_index * acc_time, delta_t=acc_time)
            evts = next(iter(iterator))
            print(len(evts))

            # Load the selected image
            image_data = images_ds[img_index]
            events_img = process_events(evts, iterator.get_size(), acc_time)

            # --- Processing Pipeline ---
            img_h_orig, img_w_orig = image_data.shape[:2]
            
            # 1. Cropping
            crop_t, crop_b, crop_l, crop_r = calc_crops(image_data, crop_top, crop_bottom, crop_left, crop_right)
            
            if crop_b <= crop_t or crop_r <= crop_l:
                st.error("Invalid crop settings. Top crop is greater than bottom crop or left is greater than right.")
            else:
                cropped_image_data = image_data[crop_t:crop_b, crop_l:crop_r]

                # 2. Mask Generation
                temp_mask = find_regions_and_generate_mask(
                    cropped_image_data.squeeze(),
                    window_size=(window_height, window_width),
                    min_area=min_area,
                    max_area=max_area,
                    threshold=threshold,
                    n_clusters=n_clusters # Pass the new parameter
                )

                # 3. Normalization and Image Preparation
                normalized_cropped_image = (cropped_image_data.astype(np.float32) - dataset_mean) / dataset_std
                
                # Create the final image to be patched
                image_to_patch = np.zeros_like(image_data, dtype=np.float32)
                image_to_patch[crop_t:crop_b, crop_l:crop_r] = normalized_cropped_image

                # Create the labeled mask
                full_mask = np.zeros(image_data.shape[:2], dtype=np.uint8)
                full_mask[crop_t:crop_b, crop_l:crop_r] = temp_mask
                
                # Assign label value (simplified from your script)
                label_value = 1 # We use k-means labels now, so this is less relevant
                labeled_mask = (full_mask * label_value).astype(np.uint8)
                
                # 4. Padding for Patching
                patch_shape = (patch_height, patch_width)
                step_h = patch_height - overlap_height
                step_w = patch_width - overlap_width

                img_h, img_w = image_to_patch.shape
                pad_h = (step_h - (img_h - patch_shape[0]) % step_h) % step_h if step_h > 0 else 0
                pad_w = (step_w - (img_w - patch_shape[1]) % step_w) % step_w if step_w > 0 else 0


                padded_image = np.pad(image_to_patch, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                padded_mask = np.pad(labeled_mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

                # --- Visualization ---
                st.header("Visualization")
                
                # Normalize for display
                display_image_norm = cv2.normalize(padded_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                display_image_bgr = cv2.cvtColor(display_image_norm, cv2.COLOR_GRAY2BGR)

                # Normalize mask for display and apply colormap
                display_mask_norm = cv2.normalize(padded_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                display_mask_color = cv2.applyColorMap(display_mask_norm, cv2.COLORMAP_JET)
                # Ensure background (label 0) is black
                display_mask_color[padded_mask == 0] = [0, 0, 0]


                # Draw grids
                img_with_grid = draw_grid(display_image_bgr, padded_mask, patch_shape, mask_perc, step_h, step_w, do_highlight)
                mask_with_grid = draw_grid(display_mask_color, padded_mask, patch_shape, mask_perc, step_h, step_w, do_highlight)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Image for Model")
                    st.image(img_with_grid, caption="Padded, Normalized Image with Patch Grid", use_column_width=True, channels="BGRA")

                with col2:
                    st.subheader("Mask for Model")
                    st.image(mask_with_grid, caption="Padded, Labeled Mask with Patch Grid (K-Means Clusters)", use_column_width=True, channels="BGRA")
                with col3:
                    st.subheader("Image Representation")
                    fig = plt.figure()
                    plt.imshow(events_img)
                    st.pyplot(fig)

    except Exception as e:
        err = traceback.format_exc()
        st.error(f"An error occurred: {err}")
else:
    st.info("Please provide paths to an H5 file and a statistics file to begin.")
