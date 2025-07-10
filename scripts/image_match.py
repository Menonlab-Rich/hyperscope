import cv2
import numpy as np
import os
import csv
from joblib import Parallel, delayed
from tqdm import tqdm # For a nice progress bar
import argparse # Import argparse

def calculate_confidence(matches, kp1, kp2):
    """
    Calculates a confidence score based on the number of good matches
    relative to the total number of keypoints.

    Args:
        matches (list): List of DMatch objects representing good matches.
        kp1 (list): Keypoints from the first image.
        kp2 (list): Keypoints from the second image.

    Returns:
        float: A confidence score between 0 and 1.
    """
    if len(kp1) == 0 or len(kp2) == 0:
        return 0.0
    
    min_keypoints = min(len(kp1), len(kp2))
    if min_keypoints == 0:
        return 0.0
        
    confidence = len(matches) / min_keypoints
    return min(1.0, confidence) # Ensure confidence doesn't exceed 1.0

def _load_and_preprocess_image(image_path):
    """
    Loads an image (JPG/PNG or .npy) and preprocesses it to a
    grayscale, uint8 NumPy array suitable for OpenCV feature detection.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray or None: The preprocessed image array (grayscale, uint8)
                               or None if loading/processing fails.
    """
    img = None
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    elif image_path.lower().endswith(('.npy')):
        try:
            np_array = np.load(image_path)
            
            # Convert dtype to uint8 if not already
            if np_array.dtype != np.uint8:
                # Scale to 0-255 range if float, otherwise just convert
                if np.issubdtype(np_array.dtype, np.floating):
                    img = (np_array * 255).astype(np.uint8)
                elif np.issubdtype(np_array.dtype, np.integer):
                    img = np_array.astype(np.uint8)
                else:
                    # Unsupported dtype
                    return None
            else:
                img = np_array

            # Check shape and convert to grayscale if necessary
            if img.ndim == 2:
                # Already 2D (grayscale)
                pass 
            elif img.ndim == 3:
                # Assume 3-channel color (RGB/BGR), convert to grayscale
                # OpenCV uses BGR by default, but for grayscale conversion, order matters less
                # unless explicitly using color-specific operations later.
                if img.shape[2] == 3: # Common for color images
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                elif img.shape[2] == 4: # RGBA, drop alpha and convert
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                else:
                    # print(f"Warning: .npy file {image_path} has 3D shape but unexpected channel count {img.shape[2]}. Skipping.")
                    return None
            else:
                # print(f"Warning: .npy file {image_path} has unsupported dimensions ({img.ndim}). Skipping.")
                return None
                
        except Exception as e:
            # print(f"Error loading .npy file {image_path}: {e}")
            return None
    else:
        # print(f"Warning: Unsupported file type for {image_path}. Skipping.")
        return None

    if img is None:
        return None # Image loading failed
    
    # Final check: ensure it's a valid OpenCV image (uint8 grayscale)
    if img.ndim != 2 or img.dtype != np.uint8:
        # This case should ideally be handled by the logic above, but as a safeguard.
        return None 
        
    return img

def process_single_query_image(query_image_path, target_images_dir):
    """
    Finds the image in target_images_dir that most closely resembles the query_image
    and calculates a confidence score. This function is designed to be run in parallel.

    Args:
        query_image_path (str): Path to the single image from the first collection.
        target_images_dir (str): Path to the directory of images from the second collection.

    Returns:
        tuple: (query_image_path, best_match_image_path, confidence_score).
               Returns (query_image_path, "No Match Found", 0.0) or error messages.
    """
    img1 = _load_and_preprocess_image(query_image_path)
    if img1 is None:
        return query_image_path, "Error: Query image processing failed", 0.0

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)

    if des1 is None or len(kp1) < 10:
        return query_image_path, "Not enough keypoints in query image", 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    best_match_path = None
    highest_confidence = -1.0 
    
    # Get list of all image/npy files in target directory
    valid_target_files = [f for f in os.listdir(target_images_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.npy'))]

    for target_image_name in valid_target_files:
        target_image_path = os.path.join(target_images_dir, target_image_name)
        
        img2 = _load_and_preprocess_image(target_image_path)
        if img2 is None:
            continue # Skip problematic target images

        kp2, des2 = orb.detectAndCompute(img2, None)

        if des2 is None or len(kp2) < 10:
            continue

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        current_confidence = calculate_confidence(matches, kp1, kp2)

        if current_confidence > highest_confidence:
            highest_confidence = current_confidence
            best_match_path = target_image_path

    if best_match_path:
        return query_image_path, best_match_path, highest_confidence
    else:
        return query_image_path, "No Match Found", 0.0

# --- Helper function for dummy data creation ---
def create_dummy_data(collection_A_dir, collection_B_dir, num_a=10, num_b=5):
    """Creates dummy image data (JPG and .npy) for testing purposes."""
    print(f"Creating {num_a} dummy images in {collection_A_dir} and {num_b} in {collection_B_dir}...")
    os.makedirs(collection_A_dir, exist_ok=True)
    os.makedirs(collection_B_dir, exist_ok=True)

    # Create dummy JPGs for Collection A
    for i in range(num_a // 2):
        img_a = np.ones((100, 150), dtype=np.uint8) * 255
        cv2.rectangle(img_a, (20 + (i%5)*5, 20 + (i%5)*5), (80 + (i%5)*5, 80 + (i%5)*5), (i%5)*20, -1)
        cv2.imwrite(os.path.join(collection_A_dir, f"activity_A_jpg_{i+1}.jpg"), img_a)
    
    # Create dummy .npy (2D and 3D) for Collection A
    for i in range(num_a // 2, num_a):
        if i % 2 == 0: # 2D grayscale .npy
            np_img_a = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
            # Add a feature
            cv2.circle(np_img_a, (50 + (i%5)*3, 50 + (i%5)*3), 20, 0, -1)
            np.save(os.path.join(collection_A_dir, f"activity_A_npy2D_{i+1}.npy"), np_img_a)
        else: # 3D color .npy (float dtype for conversion test)
            np_img_a = np.random.rand(100, 150, 3).astype(np.float32) # Float for dtype conversion test
            # Add a feature (draw a green square in float image)
            np_img_a[20+(i%5)*2:60+(i%5)*2, 20+(i%5)*2:60+(i%5)*2, 1] = 0.8 # Green channel
            np.save(os.path.join(collection_A_dir, f"activity_A_npy3D_{i+1}.npy"), np_img_a)


    # Create dummy JPGs for Collection B
    for i in range(num_b // 2):
        img_b = np.ones((110 + (i%3)*5, 160 + (i%3)*5), dtype=np.uint8) * (255 - (i%3)*10)
        if i % 2 == 0: 
            cv2.rectangle(img_b, (25 + (i%3)*5, 25 + (i%3)*5), (85 + (i%3)*5, 85 + (i%3)*5), (i%3)*20 + 10, -1)
        else: 
            cv2.circle(img_b, (50 + (i%3)*10, 50 + (i%3)*10), 30, (i%3)*30, -1)
        cv2.imwrite(os.path.join(collection_B_dir, f"activity_B_jpg_{i+1}.jpg"), img_b)

    # Create dummy .npy (2D and 3D) for Collection B
    for i in range(num_b // 2, num_b):
        if i % 2 == 0: # 2D grayscale .npy
            np_img_b = np.random.randint(0, 256, (110, 160), dtype=np.int16) # int16 for dtype conversion test
            np_img_b[20+(i%3)*2:60+(i%3)*2, 20+(i%3)*2:60+(i%3)*2] = 50
            np.save(os.path.join(collection_B_dir, f"activity_B_npy2D_{i+1}.npy"), np_img_b)
        else: # 3D color .npy (uint16 for conversion test)
            np_img_b = np.random.randint(0, 256, (110, 160, 3), dtype=np.uint16) # uint16 for dtype conversion test
            np_img_b[20+(i%3)*2:60+(i%3)*2, 20+(i%3)*2:60+(i%3)*2, 0] = 200 # Blue channel
            np.save(os.path.join(collection_B_dir, f"activity_B_npy3D_{i+1}.npy"), np_img_b)

    print("Dummy image creation complete.")

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finds most resembling images between two collections using feature matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )
    parser.add_argument(
        '-a', '--collection_a_dir', 
        type=str, 
        default="collection_A",
        help="Path to the directory containing the first collection of images (query images)."
    )
    parser.add_argument(
        '-b', '--collection_b_dir', 
        type=str, 
        default="collection_B",
        help="Path to the directory containing the second collection of images (target images)."
    )
    parser.add_argument(
        '-o', '--output_csv', 
        type=str, 
        default="image_matches_joblib.csv",
        help="Path to the output CSV file where matches will be written."
    )
    parser.add_argument(
        '-j', '--num_jobs', 
        type=int, 
        default=-1, # -1 means use all available CPU cores
        help="Number of parallel processes to use. Set to -1 to use all available CPU cores."
    )
    parser.add_argument(
        '--create_dummy_data', 
        action='store_true', # This argument does not expect a value
        help="If set, creates dummy image data in collection_A and collection_B directories for testing."
    )

    args = parser.parse_args()

    collection_A_dir = args.collection_a_dir
    collection_B_dir = args.collection_b_dir
    output_csv_file = args.output_csv
    num_jobs = args.num_jobs

    # Handle dummy data creation
    if args.create_dummy_data:
        create_dummy_data(collection_A_dir, collection_B_dir)
    else:
        if not os.path.isdir(collection_A_dir):
            print(f"Error: Collection A directory '{collection_A_dir}' not found.")
            print("Please ensure the directory exists or use --create_dummy_data.")
            exit(1)
        if not os.path.isdir(collection_B_dir):
            print(f"Error: Collection B directory '{collection_B_dir}' not found.")
            print("Please ensure the directory exists or use --create_dummy_data.")
            exit(1)


    # Get all query image paths (now including .npy)
    query_image_paths = [os.path.join(collection_A_dir, f) for f in os.listdir(collection_A_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.npy'))]

    if not query_image_paths:
        print(f"No image files (JPG/PNG/NPY) found in '{collection_A_dir}'. Exiting.")
        exit(0)

    print(f"Found {len(query_image_paths)} images in '{collection_A_dir}'.")
    print(f"Searching for matches in '{collection_B_dir}'.")
    print(f"Using {num_jobs if num_jobs != -1 else os.cpu_count()} parallel jobs.")
    print(f"Results will be written to '{output_csv_file}'.")

    # --- Run parallel computation with Joblib ---
    # Wrap with try-except for joblib to catch potential issues during parallel execution
    try:
        results = Parallel(n_jobs=num_jobs, verbose=10)(
            delayed(process_single_query_image)(query_path, collection_B_dir)
            for query_path in tqdm(query_image_paths, desc="Processing images with Joblib")
        )
    except Exception as e:
        print(f"\nAn error occurred during parallel processing: {e}")
        print("This might indicate an issue with an image file, or a memory problem.")
        results = [] # Clear results to prevent writing partial/corrupted data if parallelization crashed

    # --- Write results to CSV ---
    with open(output_csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image1_path', 'image2_path', 'confidence'])

        for query_path, best_match_path, confidence_score in results:
            csv_writer.writerow([query_path, best_match_path, f"{confidence_score:.4f}"])
            # Provide more detailed console feedback for the user
            if "Error:" in best_match_path or "Not enough keypoints" in best_match_path:
                print(f"Skipped {os.path.basename(query_path)}: {best_match_path} (Confidence: {confidence_score:.4f})")
            elif best_match_path == "No Match Found":
                print(f"No good match found for {os.path.basename(query_path)} (Confidence: {confidence_score:.4f})")
            else:
                print(f"Matched {os.path.basename(query_path)} with {os.path.basename(best_match_path)} (Confidence: {confidence_score:.4f})")

    print(f"\nAll image matches written to {output_csv_file}")

    # --- Clean up dummy directories and images ONLY IF created by this script ---
    if args.create_dummy_data:
        print("Cleaning up dummy data...")
        # Get list of all files (including .npy)
        all_a_files = [f for f in os.listdir(collection_A_dir) if os.path.isfile(os.path.join(collection_A_dir, f))]
        for f in all_a_files:
            os.remove(os.path.join(collection_A_dir, f))
        os.rmdir(collection_A_dir)
        
        all_b_files = [f for f in os.listdir(collection_B_dir) if os.path.isfile(os.path.join(collection_B_dir, f))]
        for f in all_b_files:
            os.remove(os.path.join(collection_B_dir, f))
        os.rmdir(collection_B_dir)
        print("Dummy data cleaned.")
