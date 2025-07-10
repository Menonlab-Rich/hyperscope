from setuptools import setup, find_packages

setup(
    name="h5imageviewer",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "h5imageviewer=h5imageviewer.main:main",
        ],
    },
    install_requires=[
        "h5py",
        "PySide6",
        "numpy<2",
        "Pillow",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A viewer for HDF5 files with image data",
    keywords="hdf5, image, viewer",
    python_requires=">=3.6",
)
