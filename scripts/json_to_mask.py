import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from hyperscope import config
import shutil


def create_mask_from_json(json_path):
    # Read the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    img_path = Path(data["imagePath"])
    img_path = config.INTERIM_DATA_DIR / 'worms' / 'imgs' / img_path.name
    image_shape = np.array(Image.open(img_path)).shape
    # Create an empty mask
    mask = Image.new("L", (image_shape[1], image_shape[0]), 0)
    draw = ImageDraw.Draw(mask)

    # Process each shape in the JSON
    for shape in data["shapes"]:
        # Get points and convert from fractional to absolute coordinates
        points = shape["points"]
        # Convert points to integer coordinates
        points = [(int(x), int(y)) for x, y in points]

        # Draw the polygon
        draw.polygon(points, fill=1)

    # Convert to numpy array and then to PyTorch tensor
    mask_array = np.array(mask).astype(np.uint8)

    np.savez(
        config.INTERIM_DATA_DIR / "worms" / "labels" / "masks" / f"{img_path.name}.npz",
        mask_array,
    )

    shutil.copy2(img_path, config.INTERIM_DATA_DIR / 'worms' / 'labels' / 'imgs')    


if __name__ == '__main__':
    for f in (config.INTERIM_DATA_DIR / 'worms' / 'labels').glob('*.json'):
        create_mask_from_json(f)
