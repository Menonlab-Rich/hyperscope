from collections import defaultdict
from metavision_core.event_io import RawReader
from hyperscope import config
from metavision_ml.preprocessing import timesurface
from metavision_sdk_core import RoiFilterAlgorithm, FlipXAlgorithm, PolarityFilterAlgorithm
from metavision_sdk_base import EventCDBuffer
from metavision_sdk_cv import RotateEventsAlgorithm, SpatioTemporalContrastAlgorithm, SparseOpticalFlowAlgorithm, AntiFlickerAlgorithm, TrailFilterAlgorithm
from math import pi
import gc

raw_file_path = str(config.DATA_DIR / "raw" / "metavision" / "recording_2024-07-30_13-51-19.raw")
reader = RawReader(raw_file_path)
height, width = reader.get_size()

class MemmapResource:
    def __init__(self, filename, shape, mode="r+", dtype="float32"):
        """
        A context manager for working with memory-mapped numpy arrays.

        Parameters:
        - filename: Path to the memory-mapped file.
        - shape: Shape of the memory-mapped array.
        - mode: File access mode ('r+', 'w+', etc.).
        - dtype: Data type of the array.
        """
        self.array = None
        self.filename = filename
        self.shape = shape
        self.mode = mode
        self.dtype = dtype
        self.file = None

    def __enter__(self):
        mode = self.mode + "+" if self.mode in ("r", "w") else self.mode
        if self.mode in ("w+", "w"):
            self.file = open(self.filename, "wb+")
        elif self.mode in ("r", "r+"):
            self.file = open(self.filename, "rb+")
        else:
            raise ValueError(f"Unsupported mode {self.mode}")

        # Create the memory-mapped array
        self.array = np.memmap(self.file, dtype=self.dtype, mode=mode, shape=self.shape)
        return self.array

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flush changes and close the file
        if self.array is not None:
            self.array.flush()
            del self.array  # Ensure the array is deleted to release the memory map
            # call the garbage collector to release the memory map
            gc.collect()

        if self.file is not None:
            self.file.close()
            self.file = None


x1, y1, x2, y2 = 265, 100, 430, 400

polarity_filter = PolarityFilterAlgorithm(polarity=1)  # only positive events
trail_filter_algorithm = TrailFilterAlgorithm(height, width, 10)
anti_flicker_filter = AntiFlickerAlgorithm(height, width)
rot_algo = RotateEventsAlgorithm(height - 1, width - 1, -pi / 2)
flip_x_algo = FlipXAlgorithm(width - 1)
roi_filter = RoiFilterAlgorithm(x1, y1, x2, y2, output_relative_coordinates=True)


polarity_buf = polarity_filter.get_empty_output_buffer()
rot_buf = rot_algo.get_empty_output_buffer()
flip_x_buf = flip_x_algo.get_empty_output_buffer()
roi_buf = roi_filter.get_empty_output_buffer()
events_buf = trail_filter_algorithm.get_empty_output_buffer()
events_buf_2 = anti_flicker_filter.get_empty_output_buffer()


def filter_events(evs):
    polarity_filter.process_events(evs, polarity_buf)
    rot_algo.process_events(polarity_buf, rot_buf)
    flip_x_algo.process_events(rot_buf, flip_x_buf)
    trail_filter_algorithm.process_events(flip_x_buf, events_buf)
    anti_flicker_filter.process_events(events_buf, roi_buf)
    roi_filter.process_events(roi_buf, events_buf_2)




volume_file = "volume.dat"
start_time = int(25e6)
duration = int(2e6)  # capture 2s
stop_time = start_time + duration
accumulation_time = 10  # 10us
dt = 100_000  # batch length in us
exposure_time = 10_000 # exposure time of cmos in us
bins = duration // accumulation_time  # how many 10us bins in total duration
batches = duration // dt  # how many batches we need to process the total duration

from tqdm import tqdm

# Seek to
reader.reset()
reader.seek_time(start_time)
reader.clear_ext_trigger_events()  # not interested in earlier triggers

h = y2 - y1
w = x2 - x1
t_0 = None

buffer = []
pivot_ts = None

while not reader.is_done() and reader.current_time < stop_time:
    reader.load_delta_t(dt)

ext_evts = reader.get_ext_trigger_events()
ext_evts = ext_evts[ext_evts['p'] == 1]
reader.reset()
reader.seek_time(start_time)

ordered_pairs = []

import numpy as np

v = np.zeros((exposure_time // accumulation_time, 1, h + 1, w + 1), dtype=np.float32)
for e in tqdm(ext_evts, desc='Aligning CMOS Exposures'):
    if reader.is_done():
        break
    reader.seek_time(e['t'] - exposure_time // 2)
    evts = reader.load_delta_t((e['t'] + exposure_time // 2) - reader.current_time)
    filter_events(evts)
    evts = events_buf_2.numpy(copy=True)
    evts['t'] -= evts['t'][0]
    evts['p'] = 0
    timesurface(evts, v, exposure_time)
    ordered_pairs.append((e, v))

# import matplotlib.pyplot as plt
# import mpl_toolkits
# from mpl_toolkits.mplot3d import Axes3D

# # Assuming 'ordered_pairs' is your list of (ext_evt, evts) tuples

# v = np.zeros((exposure_time // accumulation_time, 2, h + 1, w + 1), dtype=np.float32)
# evts = ordered_pairs[0][1]
# print(evts.dtype)
#timesurface(evts, v, exposure_time)
# for ext_evt, evts in tqdm(ordered_pairs, desc="Plotting events"):
#     fig = plt.figure(figsize=(12, 8))
#     # ax = fig.add_subplot(111, projection='3d')
#     # Plot events
#     timesurface(evts, v, exposure_time)
#     # Plot plane for external event
#     # xx, yy = np.meshgrid(range(w), range(h))
#     # for t in tqdm(ext_evt['t'], desc='Plotting Surfaces'):
#     #     zz = np.full_like(xx, t)
#     #     ax.plot_surface(xx, yy, zz)

#     plt.imshow(v)

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     plt.show()
#     plt.close()