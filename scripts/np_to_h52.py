import argparse
import fnmatch
import os
import h5py
import numpy as np
from tqdm import tqdm

# Assuming find_matching_files is defined as in your original script:
def find_matching_files(directory_path, filename_pattern):
    """
    Finds all files in a directory (and subdirectories) that match a given fnmatch-style pattern.
    """
    print("Finding Matching Files")
    matched_files = []
    for root, _, files in tqdm(os.walk(directory_path), desc="Scanning directories"):
        for basename in files:
            if fnmatch.fnmatch(basename, filename_pattern):
                filename = os.path.join(root, basename)
                matched_files.append(filename)
    return matched_files


def create_hdf5_from_numpy_files(
    directory_path, file_pattern, hdf5_filepath, dataset_name="images"
):
    if not os.path.isdir(directory_path):
        print(f"❌ Error: Directory not found: {directory_path}")
        return False

    print(f"🔍 Searching for files matching '{file_pattern}' in directory '{directory_path}'...")
    matching_files = find_matching_files(directory_path, file_pattern)

    if not matching_files:
        print("No files found matching the pattern.")
        return False

    print(f"Found {len(matching_files)} potential files.")

    # --- Pass 1: Collect metadata (paths, shapes, dtypes) from all 2D arrays ---
    print("\n--- Pass 1: Scanning files for metadata ---")
    collected_metadata = []  # List of dicts: {'path': str, 'shape': tuple, 'dtype': np.dtype}
    skipped_files_pass1_count = 0

    for filepath in tqdm(matching_files, desc="Scanning files for metadata"):
        try:
            data = None
            file_ext = os.path.splitext(filepath)[1].lower()

            if file_ext == ".npy":
                data = np.load(filepath, allow_pickle=False)
            elif file_ext == ".npz":
                with np.load(filepath, allow_pickle=False) as npz_file:
                    if npz_file.files:
                        data = npz_file[npz_file.files[0]]
                    else:
                        print(f"⚠️ Warning: NPZ file '{filepath}' is empty. Skipping.")
                        skipped_files_pass1_count += 1
                        continue
            elif file_ext in [".txt", ".csv", ".dat"]:
                delimiter = None
                if file_ext == ".csv":
                    delimiter = ","
                data = np.loadtxt(filepath, delimiter=delimiter)
            else:
                # print(f"Unsupported file type '{file_ext}' for '{filepath}'. Attempting generic load...") # Can be verbose
                try:
                    data = np.load(filepath, allow_pickle=False)
                except ValueError:
                    try:
                        # print(f"Generic np.load (allow_pickle=False) failed for '{filepath}'. Trying with allow_pickle=True (use with caution).") # Can be verbose
                        data = np.load(filepath, allow_pickle=True)
                    except Exception as e_pickle:
                        # print(f"❌ Error loading '{filepath}' even with allow_pickle=True: {e_pickle}. Skipping.") # Can be verbose
                        skipped_files_pass1_count += 1
                        continue
                except Exception as e_gen:
                    # print(f"❌ Error loading '{filepath}' with generic np.load: {e_gen}. Skipping.") # Can be verbose
                    skipped_files_pass1_count += 1
                    continue
            
            if data is None:
                skipped_files_pass1_count += 1
                continue

            if data.ndim == 2:
                collected_metadata.append({'path': filepath, 'shape': data.shape, 'dtype': data.dtype})
            else:
                print(f"⚠️ File '{filepath}' loaded, but data is not 2D (shape: {data.shape}). Skipping.")
                skipped_files_pass1_count += 1
            
            del data  # Explicitly release memory for the loaded array

        except Exception as e:
            print(f"❌ Error processing file '{filepath}' during Pass 1: {e}. Skipping.")
            skipped_files_pass1_count += 1
            continue
    
    print(f"Scanned {len(matching_files)} files. Collected metadata for {len(collected_metadata)} 2D arrays. Skipped {skipped_files_pass1_count} other files/errors in Pass 1.")

    if not collected_metadata:
        print("\nNo valid 2D NumPy arrays were found after metadata collection.")
        return False

    # --- Validate shapes and dtypes based on collected metadata ---
    print("\n--- Validating metadata ---")
    reference_meta = collected_metadata[0]
    reference_shape = reference_meta['shape']
    reference_dtype = reference_meta['dtype']

    print(f"Reference shape for dataset: {reference_shape} (from '{os.path.basename(reference_meta['path'])}')")
    print(f"Reference dtype for dataset: {reference_dtype} (from '{os.path.basename(reference_meta['path'])}')")

    final_valid_paths = []
    all_shapes_match = True
    has_dtype_mismatch = False

    for meta in collected_metadata:
        if meta['shape'] != reference_shape:
            all_shapes_match = False
            break 
        if meta['dtype'] != reference_dtype:
            has_dtype_mismatch = True # Flag if any dtype differs
        final_valid_paths.append(meta['path'])

    if not all_shapes_match:
        print("\n⚠️ Error: Not all collected 2D arrays have the same shape.")
        print("Shapes found:")
        # Iterate up to first 10 differing shapes to avoid overly long output
        shown_count = 0
        for i, meta_info in enumerate(collected_metadata):
            print(f"  Array {i} (from {os.path.basename(meta_info['path'])}): {meta_info['shape']}", end="")
            if meta_info['shape'] != reference_shape and shown_count < 10:
                print(" <-- MISMATCH")
                shown_count +=1
            else:
                print("")
        if any(meta['shape'] != reference_shape for meta in collected_metadata[10:] if shown_count >=10):
             print("  ... and more differing shapes.")
        print(f"Cannot stack arrays of different shapes into a single HDF5 dataset named '{dataset_name}'.")
        print("Please ensure all target files contain 2D arrays of identical dimensions.")
        return False
        
    if has_dtype_mismatch:
        print(f"\nℹ️ Info: Not all 2D arrays have the same dtype as the reference ({reference_dtype}). Data will be cast to {reference_dtype} when saving to HDF5.")

    num_arrays_to_save = len(final_valid_paths)
    if num_arrays_to_save == 0 : # Should be caught by 'if not collected_metadata' but as a safeguard
        print("No arrays passed all consistency checks.")
        return False

    print(f"All {num_arrays_to_save} arrays to be saved have the reference shape {reference_shape}.")
    print(f"Data will be stored with dtype {reference_dtype}.")

    # --- Pass 2: Load data again and write to HDF5 ---
    print(f"\n--- Pass 2: Writing {num_arrays_to_save} arrays to HDF5 ---")
    
    output_dir = os.path.dirname(hdf5_filepath)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"❌ Error creating output directory '{output_dir}': {e}")
            return False

    try:
        with h5py.File(hdf5_filepath, "w") as hf:
            dataset_shape = (num_arrays_to_save,) + reference_shape
            # Use chunking for better I/O performance with large datasets
            # A common chunking strategy is to make each chunk one full image/array.
            chunks_shape = (1,) + reference_shape 
            
            dataset = hf.create_dataset(
                dataset_name,
                shape=dataset_shape,
                dtype=reference_dtype,
                chunks=chunks_shape
            )
            print(f"💾 Created HDF5 dataset '{dataset_name}' with shape {dataset_shape}, dtype {reference_dtype}, and chunk shape {chunks_shape}.")

            for i, filepath in enumerate(tqdm(final_valid_paths, desc="Writing to HDF5")):
                data_to_write = None
                try:
                    file_ext = os.path.splitext(filepath)[1].lower()
                    # Reload data using the same logic as Pass 1
                    if file_ext == ".npy":
                        data_to_write = np.load(filepath, allow_pickle=False)
                    elif file_ext == ".npz":
                        with np.load(filepath, allow_pickle=False) as npz_file:
                            if npz_file.files: data_to_write = npz_file[npz_file.files[0]]
                    elif file_ext in [".txt", ".csv", ".dat"]:
                        delimiter = None
                        if file_ext == ".csv": delimiter = ","
                        data_to_write = np.loadtxt(filepath, delimiter=delimiter)
                    else:
                        try: data_to_write = np.load(filepath, allow_pickle=False)
                        except ValueError: data_to_write = np.load(filepath, allow_pickle=True)
                    
                    if data_to_write is not None:
                        if data_to_write.shape == reference_shape:
                            dataset[i] = data_to_write.astype(reference_dtype, copy=False) # Ensure dtype consistency
                        else:
                            print(f"❌ Error during Pass 2: Data from '{filepath}' (shape {data_to_write.shape}) does not match expected shape {reference_shape}. Array index {i} in HDF5 dataset may be incorrect or empty.")
                    else:
                        print(f"❌ Error during Pass 2: Could not reload data from '{filepath}'. Array index {i} in HDF5 dataset may be incorrect or empty.")
                    
                    del data_to_write # Free memory for this array

                except Exception as load_err:
                    print(f"❌ Error reloading or writing file '{filepath}' in Pass 2: {load_err}. Array index {i} in HDF5 dataset may be incorrect or empty.")

        with h5py.File(hdf5_filepath, "r") as hf_read:
            final_dataset_shape = hf_read[dataset_name].shape
            final_dataset_dtype = hf_read[dataset_name].dtype
        print(f"🎉 Successfully saved data to '{hdf5_filepath}' (Dataset: '{dataset_name}', Shape: {final_dataset_shape}, Dtype: {final_dataset_dtype})")
        return True

    except Exception as e:
        print(f"❌ Error during Pass 2 HDF5 saving stage: {e}")
        if os.path.exists(hdf5_filepath):
            print(f"Removing potentially corrupted HDF5 file: {hdf5_filepath}")
            try:
                os.remove(hdf5_filepath)
            except OSError as rm_err:
                print(f"Error removing file '{hdf5_filepath}': {rm_err}")
        return False


def setup_dummy_test_data(
    base_dir="test_data_dir_argparse", output_base_dir="test_output_argparse"
):
    """Creates dummy files and directories for testing the script."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created dummy data directory: {base_dir}")
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        print(f"Created dummy output directory: {output_base_dir}")

    # Create some 2D numpy files with the same shape
    np.save(os.path.join(base_dir, "img_01.npy"), np.random.rand(50, 75).astype(np.float32))
    np.save(os.path.join(base_dir, "img_02.npy"), np.random.rand(50, 75).astype(np.float32))

    # Create a 2D numpy file with a different shape
    np.save(os.path.join(base_dir, "img_03_diffshape.npy"), np.random.rand(60, 80).astype(np.float32))

    # Create a 3D numpy file (will be skipped)
    np.save(os.path.join(base_dir, "img_04_3d.npy"), np.random.rand(5, 50, 75).astype(np.float32))

    # Create a non-numpy file (will be skipped or cause error depending on loader)
    with open(os.path.join(base_dir, "img_05_text.txt"), "w") as f:
        f.write("this is not a numpy array")

    # Create a loadable text file (CSV)
    np.savetxt(os.path.join(base_dir, "img_06_data.csv"), np.random.rand(50, 75), delimiter=",")
    print("Dummy test data created.")


def main():
    parser = argparse.ArgumentParser(
        description="Search for NumPy-compatible files, filter 2D arrays, and save them to an HDF5 file.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("directory", help="Path to the directory to search for files.")
    parser.add_argument(
        "pattern", help="Glob-style pattern to match filenames (e.g., '*.npy', 'image_*.txt')."
    )
    parser.add_argument(
        "output_h5_file", help="Path to the output HDF5 file (e.g., 'output/data.h5')."
    )
    parser.add_argument(
        "--dataset_name",
        default="images",
        help="Name of the dataset within the HDF5 file (default: 'images').",
    )
    parser.add_argument(
        "--setup-test-data",
        action="store_true",
        help="If specified, creates a 'test_data_dir_argparse' and 'test_output_argparse' with dummy files for testing.",
    )

    args = parser.parse_args()

    if args.setup_test_data:
        setup_dummy_test_data()
        print(
            "\nTest data setup complete. You can now run the script pointing to 'test_data_dir_argparse'."
        )
        print("Example:")
        print(
            f'  python {os.path.basename(__file__)} test_data_dir_argparse "img_0[1-2].npy" test_output_argparse/same_shape.h5'
        )
        print(
            f'  python {os.path.basename(__file__)} test_data_dir_argparse "img_*.npy" test_output_argparse/diff_shape_error.h5' # This should error out
        )
        print(
            f'  python {os.path.basename(__file__)} test_data_dir_argparse "*.csv" test_output_argparse/csv_data.h5'
        )
        return 

    create_hdf5_from_numpy_files(
        directory_path=args.directory,
        file_pattern=args.pattern,
        hdf5_filepath=args.output_h5_file,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
