import json
import typer
from loguru import logger

from hyperscope.preprocess import superpixel as pp_superpixel
from hyperscope.preprocess import extract_tifs as pp_extract_tifs
from typing import List, Optional, Dict

app = typer.Typer()


def parse_json_dict(value: str) -> Dict:
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"Invalid JSON format: {e}")


def parse_crop_dimensions(value: str) -> Optional[List[int]]:
    """Parse crop dimensions from string input."""
    if value is None:
        return None
    try:
        # Handle different input formats
        if "," in value:
            dims = [int(x.strip()) for x in value.split(",")]
        else:
            dims = [int(x) for x in value.split()]

        if len(dims) != 4:
            raise typer.BadParameter("Crop must have exactly 4 values: left, top, right, bottom")
        return dims
    except ValueError:
        raise typer.BadParameter("Crop values must be integers")


# '''
#     input_patterns: List[str],
#     output: str,
#     catmap: dict,
#     create_dir: bool = True,
#     crop: Optional[List[int]] = None,
#     verify: bool = False,
#     batch_size: int = 4,
#     darkframe: Optional[str] = None,
#     exclude: Optional[str] = None,
#     max_size: int = 1024,  # New parameter for maximum image dimension
# '''


@app.command()
def extract_tif(
    input_patterns: List[str] = typer.Argument(..., help="Input file patterns to process"),
    output: str = typer.Argument(..., help="Output directory path"),
    catmap: str = typer.Option(
        ..., "--catmap", help="Category mapping JSON dictionary", callback=parse_json_dict
    ),
    create_dir: bool = typer.Option(
        True, "--create-dir/--no-create-dir", help="Create output directory if it doesn't exist"
    ),
    crop: Optional[str] = typer.Option(
        None,
        "--crop",
        help="Crop dimensions [left, top, right, bottom]",
        callback=parse_crop_dimensions,
    ),
    verify: bool = typer.Option(
        False, "--verify/--no-verify", help="Verify input files before processing"
    ),
    batch_size: int = typer.Option(
        100, "--batch-size", help="Number of images to process in each batch"
    ),
    darkframe: Optional[str] = typer.Option(
        None, "--darkframe", help="Path to darkframe image for correction"
    ),
    exclude: Optional[str] = typer.Option(
        None, "--exclude", help="Pattern to exclude files from processing"
    ),
):
    """
    Extract pages from TIFF files or NumPy arrays.
    """

    pp_extract_tifs.main(
        input_patterns=input_patterns,
        output=output,
        catmap=catmap,
        create_dir=create_dir,
        crop=crop,
        verify=verify,
        batch_size=batch_size,
        darkframe=darkframe,
        exclude=exclude,
    )


@app.command()
def preprocess_superpixel(
    image_dir: str = typer.Argument(..., help="Directory containing images to process."),
    mask_dir: str = typer.Argument(..., help="Directory containing masks for images."),
    output_dir: str = typer.Argument(..., help="Directory to save superpixel images."),
    multi_process: bool = typer.Option(True, "--multi-process", help="Use multiple processes."),
    batch_size: int = typer.Option(25, "--batch-size", help="Size of batches if multiprocessing"),
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
        batch_size=batch_size,
    )


if __name__ == "__main__":
    app()
