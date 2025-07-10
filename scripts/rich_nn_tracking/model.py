from enum import unique
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PolarityFilterAlgorithm
from rsklearn import clustering
from sort import Sort
import numpy as np
import joblib


def main(args: object):
    input = args.input
    dt = args.delta_t
    start_ts = args.start_ts
    eps = args.eps
    min_samples = args.min_samples
    max_duration = args.max_duration if args.max_duration > 0 else None
    sort_max_age = args.sort_max_age
    sort_min_hits = args.sort_min_hits
    sort_iou_thresh = args.sort_iou_thresh

    # We might want to keep the events for later?
    events = {}


    polarity_filter_alg = PolarityFilterAlgorithm(1) # keep positive events
    filtered = PolarityFilterAlgorithm.get_empty_output_buffer()

    dbscan = clustering.DBScan(eps=eps, min_samples=min_samples, metric="")
    mot_tracker = Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_iou_thresh)
    evts_iter = EventsIterator(input, start_ts=start_ts, delta_t=dt, max_duration=max_duration)
    for evts in evts_iter:
        polarity_filter_alg.process_events(evts, filtered)
        if not len(filtered.numpy()):
            continue
        filtered.numpy()[-1]['t']
        data = [(evt['x'], evt['y'], evt['t']) for evt in filtered.numpy()]
        data = np.array(data, dtype=np.float32)
        labels = dbscan.fit(data)
        unique_labels = np.unique(labels)
        # Resets every window
        processed_detections = []
        for label in unique_labels:
            if label == -1:
                continue
            points = data[labels == label]
            uniq_pts_id = int(joblib.hash(points), 16) # convert md5 hash to number to serve as id
            events[uniq_pts_id] = points
            # isolate the xy values (throw away t)
            xy = points[:, :2]
            x1 = np.min(xy[:, 0])
            y1 = np.min(xy[:, 1])
            x2 = np.max(xy[:, 0])
            y2 = np.max(xy[:, 1])

            confidence = float(points.size) # confidence grows with number of points
            if x1 < x2 and y1 < y2:
                processed_detections.append([x1, y1, x2, y2, confidence, uniq_pts_id])
            else:
                processed_detections.append([x1, y1, x1 + 1, y1 + 1, confidence, uniq_pts_id]) # if values are equal create a 1x1 area
            if len(processed_detections) > 0:
                detections = np.array(processed_detections)  
                tracked_objects = mot_tracker.update(detections)
            else:
                tracked_objects = mot_tracker.update()

            if len(tracked_objects):
                print(tracked_objects)




if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(prog="GenerateNodes",)
    parser.add_argument('input', type=str, help='Input Raw File')
    parser.add_argument('--start-ts', type=int, help='Start timestamp in us', default=0)
    parser.add_argument('--max-duration', type=int, help='Maximum duration in us', default=0)
    parser.add_argument('--delta-t', type=int, help='Length of the time window in us', default=100)
    parser.add_argument('--eps', type=float, help='Maximum distance between events in pixels to consider part of the same object', default=1000)
    parser.add_argument('--min-samples', type=int, help='Mininmum number of samples to consider cluster as non-noise', default=30)
    parser.add_argument('--sort-max-age', type=int, default=5, help='Max age of a track')
    parser.add_argument('--sort-min-hits', type=int, default=3, help='Minimum number of hits for track to become active')
    parser.add_argument('--sort-iou-thresh', type=float, default=0.1, help='Intersection over union for the track')
    args = parser.parse_args()
    main(args)

