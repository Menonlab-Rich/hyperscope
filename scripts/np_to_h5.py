import os
import fnmatch  # For wildcard filename matching (e.g., *.txt, data_?.npy)
import numpy as np
import h5py
import argparse # For command-line argument parsing

def find_matching_files(directory_path, filename_pattern):
    """
    Finds all files in a directory (and subdirectories) that match a given fnmatch-style pattern.

    Args:
        directory_path (str): The directory to search in.
        filename_pattern (str): The fnmatch pattern for filenames (e.g., "*.npy").

    Returns:
        list: A list of full file paths for matching files.
    """
    matched_files = []
    for root, _, files in os.walk(directory_path):
        for basename in files:
            if fnmatch.fnmatch(basename, filename_pattern):
                filename = os.path.join(root, basename)
                matched_files.append(filename)
    return matched_files

def create_hdf5_from_numpy_files(directory_path, file_pattern, hdf5_filepath, dataset_name="images"):
    """
    Searches a directory for files matching a pattern, loads them with NumPy,
    filters for 2D arrays of the same shape, and saves them into an HDF5 dataset.

    Args:
        directory_path (str): The path to the directory to search.
        file_pattern (str): The fnmatch-style pattern for filenames (e.g., "*.npy", "data_*.txt").
        hdf5_filepath (str): The path to the HDF5 file to create/overwrite.
        dataset_name (str): The name for the dataset within the HDF5 file.
    """
    if not os.path.isdir(directory_path):
        print(f"❌ Error: Directory not found: {directory_path}")
        return False

    print(f"🔍 Searching for files matching '{file_pattern}' in directory '{directory_path}'...")
    matching_files = find_matching_files(directory_path, file_pattern)

    if not matching_files:
        print("No files found matching the pattern.")
        return False

    print(f"Found {len(matching_files)} potential files.")

    valid_2d_arrays = []
    processed_file_paths = [] # To keep track of files from which arrays were successfully loaded
    skipped_files_count = 0

    for filepath in matching_files:
        try:
            data = None
            file_ext = os.path.splitext(filepath)[1].lower()

            if file_ext == '.npy':
                data = np.load(filepath, allow_pickle=False)
            elif file_ext == '.npz':
                with np.load(filepath, allow_pickle=False) as npz_file:
                    if npz_file.files:
                        data = npz_file[npz_file.files[0]]
                    else:
                        print(f"⚠️ Warning: NPZ file '{filepath}' is empty. Skipping.")
                        skipped_files_count += 1
                        continue
            elif file_ext in ['.txt', '.csv', '.dat']:
                delimiter = None
                if file_ext == '.csv':
                    delimiter = ','
                data = np.loadtxt(filepath, delimiter=delimiter)
            else:
                print(f"Unsupported file type '{file_ext}' for '{filepath}' based on extension. Attempting generic load...")
                try:
                    data = np.load(filepath, allow_pickle=False)
                except ValueError:
                    try:
                        print(f"Generic np.load (allow_pickle=False) failed for '{filepath}'. Trying with allow_pickle=True (use with caution).")
                        data = np.load(filepath, allow_pickle=True)
                    except Exception as e_pickle:
                        print(f"❌ Error loading '{filepath}' even with allow_pickle=True: {e_pickle}. Skipping.")
                        skipped_files_count += 1
                        continue
                except Exception as e_gen:
                    print(f"❌ Error loading '{filepath}' with generic np.load: {e_gen}. Skipping.")
                    skipped_files_count += 1
                    continue
            
            if data is None:
                print(f"❓ File '{filepath}' was not loaded for an unknown reason. Skipping.")
                skipped_files_count += 1
                continue

            if data.ndim == 2:
                valid_2d_arrays.append(data)
                processed_file_paths.append(filepath) # Keep track of source for messages
                print(f"✅ Successfully loaded 2D array from '{filepath}' (shape: {data.shape})")
            else:
                print(f"⚠️ File '{filepath}' loaded, but data is not 2D (shape: {data.shape}). Skipping.")
                skipped_files_count += 1

        except Exception as e:
            print(f"❌ Error processing file '{filepath}': {e}. Skipping.")
            skipped_files_count += 1
            continue

    if not valid_2d_arrays:
        print(f"\nNo valid 2D NumPy arrays were collected from the {len(matching_files) - skipped_files_count} successfully opened files.")
        return False

    print(f"\nCollected {len(valid_2d_arrays)} 2D arrays.")

    first_shape = valid_2d_arrays[0].shape
    all_same_shape = True
    for i, arr in enumerate(valid_2d_arrays[1:], start=1):
        if arr.shape != first_shape:
            all_same_shape = False
            print("\n⚠️ Warning: Not all collected 2D arrays have the same shape.")
            print("Shapes found:")
            print(f"  Array 0 (from {processed_file_paths[0]}): {first_shape}")
            for j, va in enumerate(valid_2d_arrays):
                 print(f"  Array {j} (from {processed_file_paths[j]}): {va.shape}")
            print(f"Cannot stack arrays of different shapes into a single HDF5 dataset named '{dataset_name}'.")
            print("Please ensure all target files contain 2D arrays of identical dimensions or consider saving them as separate datasets.")
            return False
    
    try:
        stacked_arrays = np.array(valid_2d_arrays)
        print(f"\n💾 Attempting to save {stacked_arrays.shape[0]} arrays (each shape {first_shape}) into '{hdf5_filepath}' under dataset '{dataset_name}'...")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(hdf5_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        with h5py.File(hdf5_filepath, 'w') as hf:
            hf.create_dataset(dataset_name, data=stacked_arrays)
        print(f"🎉 Successfully saved data to '{hdf5_filepath}' (Dataset shape: {stacked_arrays.shape})")
        return True

    except Exception as e:
        print(f"❌ Error saving data to HDF5: {e}")
        return False

def setup_dummy_test_data(base_dir="test_data_dir_argparse", output_base_dir="test_output_argparse"):
    """Creates dummy files and directories for testing the script."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created dummy data directory: {base_dir}")
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        print(f"Created dummy output directory: {output_base_dir}")

    # Create some 2D numpy files with the same shape
    np.save(os.path.join(base_dir, "img_01.npy"), np.random.rand(50, 75))
    np.save(os.path.join(base_dir, "img_02.npy"), np.random.rand(50, 75))
    
    # Create a 2D numpy file with a different shape
    np.save(os.path.join(base_dir, "img_03_diffshape.npy"), np.random.rand(60, 80))

    # Create a 3D numpy file (will be skipped)
    np.save(os.path.join(base_dir, "img_04_3d.npy"), np.random.rand(5, 50, 75))

    # Create a non-numpy file (will be skipped or cause error depending on loader)
    with open(os.path.join(base_dir, "img_05_text.txt"), "w") as f:
        f.write("this is not a numpy array")
    
    # Create a loadable text file (CSV)
    np.savetxt(os.path.join(base_dir, "img_06_data.csv"), np.random.rand(50,75), delimiter=',')
    print("Dummy test data created.")


def main():
    parser = argparse.ArgumentParser(
        description="Search for NumPy-compatible files, filter 2D arrays, and save them to an HDF5 file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("directory", help="Path to the directory to search for files.")
    parser.add_argument("pattern", help="Glob-style pattern to match filenames (e.g., '*.npy', 'image_*.txt').")
    parser.add_argument("output_h5_file", help="Path to the output HDF5 file (e.g., 'output/data.h5').")
    parser.add_argument("--dataset_name", default="images",
                        help="Name of the dataset within the HDF5 file (default: 'images').")
    parser.add_argument("--setup-test-data", action="store_true",
                        help="If specified, creates a 'test_data_dir_argparse' and 'test_output_argparse' with dummy files for testing.")

    args = parser.parse_args()

    if args.setup_test_data:
        setup_dummy_test_data()
        print("\nTest data setup complete. You can now run the script pointing to 'test_data_dir_argparse'.")
        print("Example:")
        print(f"  python {os.path.basename(__file__)} test_data_dir_argparse \"img_0[1-2].npy\" test_output_argparse/same_shape.h5")
        print(f"  python {os.path.basename(__file__)} test_data_dir_argparse \"img_*.npy\" test_output_argparse/diff_shape.h5")
        print(f"  python {os.path.basename(__file__)} test_data_dir_argparse \"*.csv\" test_output_argparse/csv_data.h5")
        return # Exit after setting up test data

    create_hdf5_from_numpy_files(
        directory_path=args.directory,
        file_pattern=args.pattern,
        hdf5_filepath=args.output_h5_file,
        dataset_name=args.dataset_name
    )

if __name__ == '__main__':
    main()
