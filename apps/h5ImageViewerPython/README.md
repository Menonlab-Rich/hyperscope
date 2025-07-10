# h5ImageViewer

A simple application for viewing image data stored in HDF5 files.

## Features

- Browse HDF5 file structure in a tree view
- View datasets as images where possible
- Automatically handles different image types (grayscale, RGB, RGBA)
- Normalizes data for display if necessary
- Shows dataset information (shape, dtype, min/max values)

## Requirements

- Python 3.6+
- h5py
- PySide6
- NumPy
- Pillow (PIL)

## Installation

Using pip:

```bash
pip install .
```

## Usage

After installation, you can run the application from the command line:

```bash
h5imageviewer
```

Or you can run it directly from the source directory:

```bash
python -m h5imageviewer
```

### Using the Application

1. Click the "Open HDF5 File" button to open an HDF5 file (.h5 or .hdf5)
2. Navigate the file structure in the tree view on the left
3. Click on a dataset to view it as an image (if it's displayable)
4. The application will show dataset information at the bottom of the image display

### Supported Image Types

The application can display datasets with the following characteristics:

- 2D arrays (displayed as grayscale images)
- 3D arrays with 1 channel (grayscale), 3 channels (RGB), or 4 channels (RGBA)
- Any numeric data type (will be normalized to uint8 for display if needed)

## License

[MIT](LICENSE)
