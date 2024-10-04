from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from hyperscope.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR
from hyperscope.preprocess import superpixel as pp_superpixel
from hyperscope.preprocess import extract_tifs as pp_extract_tifs
from typing import List, Optional

app = typer.Typer()


@app.command()
def main():
    logger.info("Processing dataset...")
    extract_tif(
        input=RAW_DATA_DIR / "images/*.tif",
        output= INTERIM_DATA_DIR / "extracted",
        normalize=True,
        create_dir=True,
        verify=True,
        no_confirm=True,
        multi_process=True,
        batch_size=10,
        n_batches=-1,
    )
    
    superpixel(
        image_dir=INTERIM_DATA_DIR / "extracted" / "imgs",
        mask_dir=INTERIM_DATA_DIR / "extracted" / "masks",
        output_dir=PROCESSED_DATA_DIR / "superpixel_images",
        multi_process=True,
    )

@app.command()
def extract_tif(
    input: List[str] = typer.Argument(..., help="Glob pattern to match TIFF files."),
    output: str = typer.Option(..., "--output", "-o", help="Directory where extracted pages will be saved."),
    normalize: bool = typer.Option(False, "--normalize", help="Normalize the extracted images."),
    create_dir: bool = typer.Option(False, "--create-dir", help="Create the output directory if it doesn't exist."),
    crop: Optional[List[int]] = typer.Option(
        None, "--crop", help="Remove the specified number of pixels from the edges (left, upper, right, lower)."
    ),
    verify: bool = typer.Option(False, "--verify", help="Verify the extracted images."),
    no_confirm: bool = typer.Option(False, "--no-confirm", help="Suppress the confirmation prompt."),
    multi_process: bool = typer.Option(
        True, "--multi-process", help="Use multiple processes to extract images."
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", help="Number of images to process in each batch."
    ),
    n_batches: int = typer.Option(-1, "--n-batches", help="Number of batches to process. -1 for all."),
):
    """
    Extract pages from TIFF files or NumPy arrays.
    """
    logger.info(f"Extracting images from: {input}")
    pp_extract_tifs.main(
        input=input,
        output=output,
        normalize=normalize,
        create_dir=create_dir,
        crop=crop,
        verify=verify,
        no_confirm=no_confirm,
        multi_process=multi_process,
        batch_size=batch_size,
        n_batches=n_batches,
    )
    


@app.command()
def preprocess_superpixel(
    image_dir: str = typer.Argument(..., help="Directory containing images to process."),
    mask_dir: str = typer.Argument(..., help="Directory containing masks for images."),
    output_dir: str = typer.Argument(..., help="Directory to save superpixel images."),
    multi_process: bool = typer.Option(False, "--multi-process", help="Use multiple processes."),
):
    """
    Apply superpixel segmentation to images.
    """
    logger.info(f"Applying superpixel segmentation to images in: {image_dir}")
    pp_superpixel.main(
        img_dir=image_dir,
        mask_dir=mask_dir,
        output_dir=output_dir,
        mp=multi_process,
    )

if __name__ == "__main__":
    app()
