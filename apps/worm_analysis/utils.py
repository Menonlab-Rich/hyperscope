import numpy as np
from skimage.filters import threshold_multiotsu
import cv2 as cv
from sklearn.cluster import KMeans


def calc_crops(img, crop_top_amt, crop_bottom_amt, crop_left_amt, crop_right_amt):
    h, w, *_ = img.shape
    return crop_top_amt, h - crop_bottom_amt, crop_left_amt, w - crop_right_amt

def compute_windowed_std_dev_vectorized(image, window_size):
    image = image.astype(np.float32)
    integral = cv.integral(image)
    integral_squared = cv.integral(image * image)
    win_h, win_w = window_size
    h, w = image.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    top_left_y = np.maximum(0, y - win_h // 2)
    top_left_x = np.maximum(0, x - win_w // 2)
    bottom_right_y = np.minimum(h, y + win_h // 2 + 1)
    bottom_right_x = np.minimum(w, x + win_w // 2 + 1)
    areas = (bottom_right_y - top_left_y) * (bottom_right_x - top_left_x)
    areas = np.where(areas == 0, 1, areas)
    sums = (
        integral[bottom_right_y, bottom_right_x]
        - integral[bottom_right_y, top_left_x]
        - integral[top_left_y, bottom_right_x]
        + integral[top_left_y, top_left_x]
    )
    sum_squares = (
        integral_squared[bottom_right_y, bottom_right_x]
        - integral_squared[bottom_right_y, top_left_x]
        - integral_squared[top_left_y, bottom_right_x]
        + integral_squared[top_left_y, top_left_x]
    )
    means = sums / areas
    variances = (sum_squares / areas) - (means * means)
    variances = np.maximum(0, variances)
    std_dev_image = np.sqrt(variances)
    max_std_dev = std_dev_image.max()
    if max_std_dev > 0:
        return std_dev_image / max_std_dev
    return std_dev_image

def find_regions_and_generate_mask(
    image,
    window_size=(16, 16),
    min_area=20,
    threshold=1.5,
    n_clusters=3, # New parameter for K-Means
    max_area=5000
):
    if image is None:
        return np.zeros((10, 10), dtype=np.uint8)

    if image.ndim == 3 and image.shape[2] > 1:
        num_channels = image.shape[2]
        if num_channels == 3:
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        elif num_channels == 4:
            image_gray = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
        else:
            image_gray = image[:, :, 0]
    elif image.ndim == 2:
        image_gray = image
    else:
        return np.zeros(image.shape[:2] if image.ndim >= 2 else (10, 10), dtype=np.uint8)

    image_float = image_gray.astype(np.float32)

    if image_float.max() > 0:
        image_norm = image_float / image_float.max()
    else:
        image_norm = image_float

    std_dev_img = compute_windowed_std_dev_vectorized(image_norm, window_size)
    std_dev_thresholded = std_dev_img.copy()
    # Thresholding using the mean of std_dev_img
    mean_std_dev = std_dev_img.mean()
    std_dev_thresholded[std_dev_img <= threshold * mean_std_dev] = 0

    # Initial binary mask from simple thresholding
    binary_img = (std_dev_thresholded > 0).astype(np.uint8)
    contours, _ = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    output_mask_shape = image_gray.shape[:2]
    mask = np.zeros(output_mask_shape, dtype=np.uint8)
    
    # Early exit if too many small contours are found, indicating noise
    if len(contours) > 500:
       return mask

    # First pass filtering contours
    for contour in contours:
        area = cv.contourArea(contour)
        if min_area <= area <= max_area:
            cv.drawContours(mask, [contour], 0, 1, -1)

    # --- K-Means Clustering and Label Re-ordering ---
    if mask.any() and n_clusters > 1:
        # Get the coordinates and feature values of the pixels to be clustered
        pixels_to_cluster_coords = np.where(mask == 1)
        features = std_dev_thresholded[pixels_to_cluster_coords].reshape(-1, 1)

        # Check if we have enough pixels to form clusters
        if len(features) >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(features)
            
            # --- Re-order labels based on cluster center intensity ---
            # The cluster with the lowest center value is the "outermost".
            
            # 1. Get the original k-means labels sorted by their center value (ascending).
            #    This gives the order: [label_of_outermost, label_of_next, ..., label_of_innermost]
            sorted_center_indices = np.argsort(kmeans.cluster_centers_.flatten())
            
            # 2. Create a lookup table to map from the original k-means label to the new, ordered label.
            #    The new labels will be 1, 2, 3... corresponding to the sorted centers.
            label_mapping = np.zeros(n_clusters, dtype=np.uint8)
            label_mapping[sorted_center_indices] = np.arange(0, n_clusters)
            
            # 3. Apply this mapping to the labels assigned to every pixel.
            new_labels = label_mapping[kmeans.labels_]
            
            # 4. Create the final mask and assign the re-ordered labels.
            clustered_mask = np.zeros(output_mask_shape, dtype=np.uint8)
            clustered_mask[pixels_to_cluster_coords] = new_labels
            
            return clustered_mask

    # Return the binary mask if no clustering was performed
    return mask


