#!/bin/bash

# Purpose: Run the metavision_sparse_optical_flow.py script on multiple .raw files
#          located in a specified input directory, saving the output .npy files
#          to a specified output directory. This version runs the script for
#          multiple accumulation times (-a parameter) for each input file.

# --- Configuration ---
# Set the directory containing the input .raw files
# Adjust this path if your files are located elsewhere relative to the script's location
INPUT_DIR="../data/raw/metavision"

# Set the directory where the output .npy files should be saved
# Adjust this path if needed
OUTPUT_DIR="../data/interim"

# Python script name/path
PYTHON_SCRIPT="generic_tracking.py"

# Define the list of accumulation times (-a parameter) to iterate over
UPDATE_FREQUENCIES=(100) # Bash array

# --- Script Logic ---
# Create the output directory if it doesn't already exist
# The -p flag ensures no error if the directory exists and creates parent directories if needed
mkdir -p "$OUTPUT_DIR"

# Check if the input directory exists
if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Input directory '$INPUT_DIR' not found."
  exit 1
fi

echo "Starting processing..."
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Accumulation Times: ${UPDATE_FREQUENCIES[*]}" # Print the array elements
echo "=================================================="

# Find all files ending with .raw in the input directory
# Use find for more robust handling of filenames, though a simple glob is often sufficient
find "$INPUT_DIR" -maxdepth 1 -name 'recording_2025-04-29*.raw' -print0 | while IFS= read -r -d $'\0' file_path; do

  # Check if it's actually a file (and not a directory named *.raw)
  if [ -f "$file_path" ]; then
    # Extract the filename without the directory path and the .raw extension
    # e.g., ../data/raw/metavision/recording1.raw -> recording1
    base_filename=$(basename "$file_path" .raw)

    echo "--------------------------------------------------"
    echo "Processing File: $base_filename.raw"
    echo "Input file: $file_path"
    echo "--------------------------------------------------"

    # Loop through each accumulation time defined in the array
    for param_a in "${UPDATE_FREQUENCIES[@]}"; do

      # Construct the output filename including the accumulation time
      output_npy_filename="${OUTPUT_DIR}/tracking_${base_filename}_${param_a}hz.npy"
      if [ -f "$output_npy_filename" ]; then
	      echo "$output_npy_filename exists. Skipping"
	      continue
      fi
      echo "  Running with frequency: $param_a"
      echo "  Output file: $output_npy_filename"

      # Construct and execute the python command
      # Using quotes around variables to handle potential spaces or special characters in filenames
      ../.venv/bin/python "$PYTHON_SCRIPT" \
        -i "$file_path" \
	-a 30000 \
        --update-frequency "$param_a" \
        --out-np "$output_npy_filename" \
        --headless < /dev/tty

      # Check the exit status of the python script
      if [ $? -ne 0 ]; then
        echo "  Error: Python script failed for file '$base_filename.raw' with a=$param_a. Check script output."
        # Optional: uncomment the next line to stop the script on the first error
        # exit 1
      else
        echo "  Successfully processed '$base_filename.raw' with a=$param_a."
      fi
      echo # Add a blank line for readability between accumulation times

    done # End of accumulation time loop

    echo # Add a blank line for readability between files
  fi
done # End of file loop

echo "=================================================="
echo "Finished processing all .raw files in '$INPUT_DIR' for all specified accumulation times."
echo "=================================================="

exit 0

