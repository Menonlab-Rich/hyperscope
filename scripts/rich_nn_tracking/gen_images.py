from pathlib import Path

import imageio.v3 as iio
import numpy as np
from joblib import Parallel, delayed
from metavision_core.event_io import EventsIterator
from tqdm import tqdm
from metavision_sdk_core import (
    MostRecentTimestampBuffer,
    TimeSurfaceProducerAlgorithmMergePolarities,
)


def save_image(output_path, image_data):
    """Helper function to save an image using imageio."""
    try:
        iio.imwrite(output_path, image_data)
        # print(f"Saved {output_path}") # Optional: uncomment for progress
    except Exception as e:
        print(f"Error saving {output_path}: {e}")


to_save = []


def process(args):
    evt_iter = EventsIterator(args.input_path, delta_t=args.dt)
    height, width = evt_iter.get_size()
    last_t = 0
    ts = MostRecentTimestampBuffer(rows=height, cols=width, channels=1)
    producer = TimeSurfaceProducerAlgorithmMergePolarities(width=width, height=height)
    out_dir = Path(args.out_dir)
    out_filename = Path(args.input_path).stem

    # Make sure that outdir exists
    out_dir.mkdir(exist_ok=True, parents=True)
    to_save = []

    def cb_ts(t, data):
        nonlocal last_t
        nonlocal ts
        last_t = t
        ts = data

    producer.set_output_callback(cb_ts)
    img = np.empty((height, width), dtype=np.uint8)

    for evt in tqdm(evt_iter, desc='Iterating Events', position=0):
        producer.process_events(evt)
        ts.generate_img_time_surface(last_t, args.dt, img)
        filename = f"{out_filename}_ts_{last_t}.jpg"
        to_save.append(delayed(save_image)(out_dir / filename, img.copy()))

        if len(to_save) >= 12000:
            Parallel(n_jobs=-1)(tqdm(to_save, unit='image', desc='saving images', position=1))
            to_save = []

    Parallel(n_jobs=-1)(to_save)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "timesurface processor", description="Generate time surface images from events"
    )
    parser.add_argument("input_path", type=str, help="The path of the raw or hdf file.")
    parser.add_argument("out_dir", type=str, help="The location to store the output images")
    parser.add_argument("--dt", type=int, default=1000, help="Accumulation time in microseconds")

    args = parser.parse_args()
    process(args)
