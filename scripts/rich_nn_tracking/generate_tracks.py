from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PolarityFilterAlgorithm
from rsklearn import clustering
from sort import Sort
from tqdm import tqdm
import numpy as np
import joblib
import h5py  # Added for HDF5 functionality
import os  # Added to help with output filename


def main(args: object):
    input_file = args.input  # Renamed to avoid conflict
    dt = args.delta_t
    start_ts = args.start_ts
    eps = args.eps
    min_samples = args.min_samples
    max_duration = args.max_duration if args.max_duration > 0 else None
    sort_max_age = args.sort_max_age
    sort_min_hits = args.sort_min_hits
    sort_iou_thresh = args.sort_iou_thresh
    output_hdf5_file = args.output_file  # Get output file from args

    # Data collectors for HDF5 saving
    events_for_hdf5 = {}  # This will store the same as your 'events' dict
    all_tracks_list = []  # To store all tracked objects with their window timestamp

    polarity_filter_alg = PolarityFilterAlgorithm(1)  # keep positive events
    filtered = PolarityFilterAlgorithm.get_empty_output_buffer()

    dbscan = clustering.DBScan(
        eps=eps, min_samples=min_samples, metric=""
    )  # Assuming "" is a valid metric or handled by rsklearn
    mot_tracker = Sort(
        max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_iou_thresh
    )

    print(f"Processing event file: {input_file}")
    evts_iter = EventsIterator(
        input_file, start_ts=start_ts, delta_t=dt, max_duration=max_duration
    )

    n_windows = 0
    for _ in evts_iter:
        n_windows += 1
    
    evts_iter = EventsIterator(
        input_file, start_ts=start_ts, delta_t=dt, max_duration=max_duration
    )


    loop_iterator = tqdm(evts_iter, desc="Processing Windows", total=n_windows, unit='Window')  # Default iterator

    for evts in loop_iterator:
        current_window_end_ts = -1  # Initialize timestamp for the window

        polarity_filter_alg.process_events(evts, filtered)

        filtered_numpy = filtered.numpy()  # Get numpy array once
        if not len(filtered_numpy):
            # Still need to update tracker with empty detections
            # This assumes your Sort.update can handle 0-row, 6-column arrays
            tracked_objects = mot_tracker.update(np.empty((0, 6)))
            if (
                len(tracked_objects) > 0
            ):  # Should not happen if dets are empty, but for safety
                # Unlikely to have tracks if no detections, but if Sort could output predictively
                # For now, we only add tracks if they came from non-empty detections below
                pass
            continue

        current_window_end_ts = filtered_numpy[-1][
            "t"
        ]  # Timestamp of the last event in the batch

        data = [(evt["x"], evt["y"], evt["t"]) for evt in filtered_numpy]
        data = np.array(data, dtype=np.float32)
        labels = dbscan.fit(data)
        unique_labels = np.unique(labels)

        processed_detections = []
        for label in unique_labels:
            if label == -1:
                continue

            points = data[labels == label]
            # Use a simpler way to get a unique ID if joblib.hash is complex or slow repeatedly
            # For HDF5 dataset names, string conversion of the label might also work if locally unique
            # However, your hash method ensures wider uniqueness if points can be identical across windows but are different clusters
            uniq_pts_id_str = str(
                int(joblib.hash(points), 16)
            )  # Convert to string for HDF5 dataset names

            events_for_hdf5[uniq_pts_id_str] = points  # Store events for HDF5

            xy = points[:, :2]
            x1 = np.min(xy[:, 0])
            y1 = np.min(xy[:, 1])
            x2 = np.max(xy[:, 0])
            y2 = np.max(xy[:, 1])

            confidence = float(
                points.shape[0] * points.shape[1]
            )  # Number of elements (points * features_per_point)
            # Original: points.size, which is N_points * 3 (x,y,t)
            # A more common confidence might be just N_points: float(points.shape[0])

            # The original_id for SORT should be a number if possible, but string keys for HDF5 events are fine.
            # We'll use the numerical hash for SORT original_id, and string for HDF5 key
            uniq_pts_id_num = int(joblib.hash(points), 16)

            if x1 < x2 and y1 < y2:
                processed_detections.append(
                    [x1, y1, x2, y2, confidence, uniq_pts_id_num]
                )
            elif xy.shape[0] > 0:  # handles single point or collinear points
                processed_detections.append(
                    [x1, y1, x1 + 1, y1 + 1, confidence, uniq_pts_id_num]
                )

        detections_to_sort = np.empty((0, 6))  # Default to 6 columns for SORT
        if len(processed_detections) > 0:
            detections_to_sort = np.array(processed_detections) 
            tracked_objects = mot_tracker.update(detections_to_sort)
        else:
            tracked_objects = mot_tracker.update(
                detections_to_sort
            )  # Pass empty (0,6) array

        if len(tracked_objects) > 0:
            # print(f"Window ending at {current_window_end_ts}, Tracks: {tracked_objects}")
            # Add timestamp to each track row for saving
            timestamp_col = np.full(
                (tracked_objects.shape[0], 1), current_window_end_ts, dtype=np.int64
            )
            tracks_with_ts = np.hstack((timestamp_col, tracked_objects))
            all_tracks_list.append(tracks_with_ts)

    # --- Finished processing all event windows ---
    print(f"Finished processing. Saving data to HDF5 file: {output_hdf5_file}")

    with h5py.File(output_hdf5_file, "w") as hf:
        # Save Tracks
        if len(all_tracks_list) > 0:
            final_tracks_array = np.concatenate(all_tracks_list, axis=0)
            tracks_dset = hf.create_dataset(
                "tracks", data=final_tracks_array, compression="gzip"
            )
            tracks_dset.attrs["columns"] = [
                "window_end_ts",
                "x1",
                "y1",
                "x2",
                "y2",
                "track_id",
                "original_event_cluster_id",
            ]
            tracks_dset.attrs["description"] = (
                "Tracked object data. 'original_event_cluster_id' links to datasets in '/event_clusters'."
            )
            print(
                f"Saved {final_tracks_array.shape[0]} track segments to 'tracks' dataset."
            )
        else:
            hf.create_dataset(
                "tracks", data=np.empty((0, 7))
            )  # Create empty if no tracks, with 7 columns
            tracks_dset = hf["tracks"]
            tracks_dset.attrs["columns"] = [
                "window_end_ts",
                "x1",
                "y1",
                "x2",
                "y2",
                "track_id",
                "original_event_cluster_id",
            ]
            tracks_dset.attrs["description"] = "No tracks were generated or recorded."
            print("No tracks to save.")

        # Save Event Clusters
        event_clusters_group = hf.create_group("event_clusters")
        if len(events_for_hdf5) > 0:
            print(f"Saving {len(events_for_hdf5)} event clusters...")
            for uniq_id_str, points_array in events_for_hdf5.items():
                cluster_dset = event_clusters_group.create_dataset(
                    uniq_id_str, data=points_array, compression="gzip"
                )
                cluster_dset.attrs["columns"] = ["x", "y", "t"]
                cluster_dset.attrs["description"] = (
                    f"Events forming cluster with original_id {uniq_id_str}."
                )
            print(f"Saved event clusters to '/event_clusters' group.")
        else:
            event_clusters_group.attrs["description"] = "No event clusters were stored."
            print("No event clusters to save.")

    print("HDF5 saving complete.")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="EventTrackerSaver",
    )  # Renamed prog
    parser.add_argument("input", type=str, help="Input Raw File")
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="Output HDF5 file name",
        default="tracked_event_data.h5",
    )
    parser.add_argument("--start-ts", type=int, help="Start timestamp in us", default=0)
    parser.add_argument(
        "--max-duration", type=int, help="Maximum duration in us to process", default=0
    )
    parser.add_argument(
        "--delta-t", type=int, help="Length of the time window in us", default=20000
    )  # Adjusted from 100us to 20ms
    parser.add_argument(
        "--eps",
        type=float,
        help="DBSCAN: Maximum distance between events to consider part of the same object",
        default=5.0,
    )  # Adjusted
    parser.add_argument(
        "--min-samples",
        type=int,
        help="DBSCAN: Mininmum number of samples to consider cluster as non-noise",
        default=10,
    )  # Adjusted
    parser.add_argument(
        "--sort-max-age",
        type=int,
        default=5,
        help="SORT: Max age of a track (in #windows)",
    )
    parser.add_argument(
        "--sort-min-hits",
        type=int,
        default=3,
        help="SORT: Minimum number of hits for track to become active",
    )
    parser.add_argument(
        "--sort-iou-thresh",
        type=float,
        default=0.1,
        help="SORT: Intersection over union threshold for track association",
    )

    args = parser.parse_args()

    # Ensure output directory exists if a path is specified
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    main(args)
