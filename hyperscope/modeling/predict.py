import os
import typer
from loguru import logger
from pathlib import Path
from tqdm import tqdm

from hyperscope.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()





@app.command()
def main(
    model_path: Path = MODELS_DIR / "unet-UN-348-epoch=05-val_dice=0.89.ckpt",
    predictions_path: Path = PROCESSED_DATA_DIR / "predictions",
    input_dir: Path = PROCESSED_DATA_DIR / "superpixel_images",
    target_dir: Path = PROCESSED_DATA_DIR / "superpixel_masks",
):

    logger.info("Making predictions...")

    raise NotImplementedError("Implement the prediction logic here.")


if __name__ == "__main__":
    app()