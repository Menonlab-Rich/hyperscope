# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Simple script to track general objects, with optional headless mode and NumPy output.
Use '--headless' to disable the display window.
Use '--out-npy path/to/output.npy' to save tracking results.
You can use it, for example, with the reference file traffic_monitoring.raw.
"""

import numpy as np
import argparse
from contextlib import nullcontext # Used for conditional 'with' statement

# --- Always needed imports ---
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_analytics import TrackingAlgorithm, TrackingConfig
from metavision_sdk_core import RollingEventBufferConfig, RollingEventCDBuffer
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, TrailFilterAlgorithm

# --- Optional imports (for UI/Progress bar) ---
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None # tqdm is optional

# UI modules will be imported later if not args.headless

# --- Argument Parsing ---
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generic Tracking sample with optional headless mode.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Base options
    base_options = parser.add_argument_group('Base options')
    base_options.add_argument(
        '-i', '--input-event-file', dest='event_file_path', default="",
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used. "
             "If it's a camera serial number, it will try to open that camera instead.")
    base_options.add_argument('--process-from', dest='process_from', type=int, default=0,
                              help='Start time to process events (in us).')
    base_options.add_argument('--process-to', dest='process_to', type=int, default=None,
                              help='End time to process events (in us).')
    base_options.add_argument('--headless', action='store_true',
                              help='Run without displaying the graphical window.')

    # Tracking options
    tracking_options = parser.add_argument_group('Tracking options')
    tracking_options.add_argument('--update-frequency', dest='update_frequency', type=float,
                                   default=200., help="Tracker's update frequency, in Hz.")
    tracking_options.add_argument('-a', '--acc-time', dest='accumulation_time_us', type=int, default=10000,
                                  help='Duration of the time slice to store in the rolling event buffer at each tracking step (in us)')

    # Min/Max size options
    minmax_size_options = parser.add_argument_group('Min/Max size options')
    minmax_size_options.add_argument('--min-size', dest='min_size', type=int,
                                     default=10, help='Minimal size of an object to track (in pixels).')
    minmax_size_options.add_argument('--max-size', dest='max_size', type=int,
                                     default=300, help='Maximal size of an object to track (in pixels).')

    # Filtering options
    filter_options = parser.add_argument_group('Filtering options')
    filter_options.add_argument(
        '--activity-time-ths', dest='activity_time_ths', type=int, default=10000,
        help='Length of the time window for activity filtering (in us, disabled if equal to 0).')
    filter_options.add_argument('--activity-trail-ths', dest='activity_trail_ths', type=int, default=1000,
                                help='Length of the time window for trail filtering (in us, disabled if equal to 0).')

    # Outcome Options
    outcome_options = parser.add_argument_group('Outcome options')
    outcome_options.add_argument(
        '-o', '--out-video', dest='out_video', type=str, default="",
        help="[UI Mode Only] Path to an output AVI file to save the resulting video. Requires --headless=False.")
    outcome_options.add_argument(
        '--out-npy', dest='out_npy', type=str, default="",
        help="Path to an output NumPy file (.npy) to save the tracking results (timestamp, id, bbox).")

    # Replay Option
    replay_options = parser.add_argument_group('Replay options')
    replay_options.add_argument(
        '-f', '--replay_factor', type=float, default=1,
        help="[UI Mode Only] Replay Factor. If greater than 1.0 we replay with slow-motion, otherwise this is a speed-up over real-time. Ignored in headless mode.")

    args = parser.parse_args()

    if args.headless and args.out_video:
        print("Warning: --out-video is specified but ignored in --headless mode.")
        args.out_video = ""
    if args.headless and args.replay_factor != 1:
        print("Warning: --replay_factor is specified but ignored in --headless mode.")
        args.replay_factor = 1

    if args.out_npy and not args.out_npy.lower().endswith('.npy'):
        args.out_npy += '.npy'
        print(f"Appending .npy extension. NumPy output file will be: {args.out_npy}")

    if args.process_to and args.process_from > args.process_to:
        print(f"Error: The processing time interval is not valid. [{args.process_from,}, {args.process_to}]")
        exit(1)

    return args

# --- Core Processing Logic ---
def processing_loop(mv_iterator, height, width,
                    activity_noise_filter, trail_filter,
                    tracking_algo, rolling_buffer,
                    args, window=None, video_writer=None, output_img=None, draw_tracking_results_func=None, base_frame_gen_algo=None):
    """
    Processes events, performs tracking, optionally displays/saves video, and collects data for NumPy output.
    UI-related arguments (window, video_writer, output_img, draw_tracking_results_func, base_frame_gen_algo)
    should be None if running in headless mode.
    """
    events_buf_filter1 = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
    events_buf_filter2 = TrailFilterAlgorithm.get_empty_output_buffer()
    tracking_results = tracking_algo.get_empty_output_buffer()

    all_tracked_objects = []
    total_time_us = mv_iterator.max_duration
    pbar = None
    if args.headless and tqdm and total_time_us and not is_live_camera(args.event_file_path):
        pbar = tqdm(total=total_time_us, unit='us', desc="Processing events")
    elif not args.headless and total_time_us and not is_live_camera(args.event_file_path):
         # Use tqdm even in UI mode if available, for console progress
         if tqdm:
              pbar = tqdm(total=total_time_us, unit='us', desc="Processing events")

    last_ts = args.process_from

    for evs in mv_iterator:
        # --- Progress Update ---
        current_last_ts = evs['t'][-1] if len(evs) > 0 else last_ts
        if pbar:
            pbar.update(current_last_ts - last_ts)
            last_ts = current_last_ts
        elif not args.headless and window is None and len(evs) == 0 and mv_iterator.is_done():
            # Handle loop end condition for live camera in UI mode without pbar
            pass

        # --- Filtering ---
        events_to_process = evs
        if activity_noise_filter:
            activity_noise_filter.process_events(events_to_process, events_buf_filter1)
            events_to_process = events_buf_filter1
        if trail_filter:
            if activity_noise_filter:
                trail_filter.process_events_(events_to_process)
            else:
                trail_filter.process_events(events_to_process, events_buf_filter2)
                events_to_process = events_buf_filter2

        # Convert to NumPy if needed
        if not isinstance(events_to_process, np.ndarray):
             events_np = events_to_process.numpy()
        else:
             events_np = events_to_process

        # --- Tracking ---
        if len(events_np) > 0:
            buffer_last_ts = events_np['t'][-1]
            rolling_buffer.insert_events(events_np)
            tracking_algo.process_events(rolling_buffer, tracking_results)

            # --- NumPy Data Collection (Always) ---
            for track in tracking_results.numpy():
                if len(track) == 0:
                    continue
                all_tracked_objects.append(track)

            # --- UI Operations (Conditional) ---
            if window and base_frame_gen_algo and draw_tracking_results_func:
                 base_frame_gen_algo.generate_frame(rolling_buffer, output_img)
                 draw_tracking_results_func(buffer_last_ts, tracking_results, output_img)
                 window.show_async(output_img)
                 if video_writer:
                      video_writer.write(output_img)

        # --- UI Event Handling (Conditional) ---
        if window:
             from metavision_sdk_ui import EventLoop # Import here to avoid top-level dependency when headless
             EventLoop.poll_and_dispatch()
             if window.should_close():
                  break

    if pbar:
        if pbar.n < total_time_us:
             pbar.update(total_time_us - pbar.n) # Ensure bar completes
        pbar.close()

    return all_tracked_objects


# --- Main Execution ---
def main():
    """ Main """
    args = parse_args()

    # --- Conditional UI / Module Imports ---
    window_context = None
    cv2 = None
    BaseFrameGenerationAlgorithm = None
    draw_tracking_results = None
    EventLoop = None
    BaseWindow = None
    MTWindow = None
    UIAction = None
    UIKeyEvent = None

    if not args.headless:
        try:
            import cv2
            from metavision_sdk_core import BaseFrameGenerationAlgorithm
            from metavision_sdk_analytics import draw_tracking_results
            from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
            print("Running in UI mode.")
        except ImportError as e:
            print(f"Error importing UI components: {e}")
            print("Please install Metavision SDK UI components and OpenCV (cv2) to run with display.")
            print("Alternatively, run with the --headless flag.")
            exit(1)
    else:
        print("Running in headless mode.")


    # Rolling event buffer (always needed)
    buffer_config = RollingEventBufferConfig.make_n_us(args.accumulation_time_us)
    rolling_buffer = RollingEventCDBuffer(buffer_config)

    # Events iterator (always needed)
    delta_t = int(1e6 / args.update_frequency) if args.update_frequency > 0 else 10000
    try:
        mv_iterator = EventsIterator(input_path=args.event_file_path, start_ts=args.process_from,
                                     max_duration=args.process_to - args.process_from if args.process_to else None,
                                     delta_t=delta_t, mode="delta_t", relative_timestamps=False)
        height, width = mv_iterator.get_size()
    except Exception as e:
        print(f"Error opening input source '{args.event_file_path}': {e}")
        exit(1)

    # Apply replay factor ONLY if in UI mode and not live camera
    if not args.headless and args.replay_factor != 1 and not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)
        print(f"Applying replay factor: {args.replay_factor}")

    # Filters (always needed)
    activity_noise_filter = ActivityNoiseFilterAlgorithm(width, height, args.activity_time_ths) if args.activity_time_ths > 0 else None
    trail_filter = TrailFilterAlgorithm(width, height, args.activity_trail_ths) if args.activity_trail_ths > 0 else None

    # Tracking Algorithm (always needed)
    tracking_config = TrackingConfig()
    tracking_algo = TrackingAlgorithm(sensor_width=width, sensor_height=height, tracking_config=tracking_config)
    tracking_algo.min_size = args.min_size
    tracking_algo.max_size = args.max_size

    # --- Prepare for loop (conditional UI setup) ---
    all_tracked_objects = []
    try:
        if not args.headless:
            # Setup UI specific components
            output_img = np.zeros((height, width, 3), np.uint8)
            video_writer = None
            if args.out_video:
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                video_name = args.out_video if args.out_video.lower().endswith(".avi") else args.out_video + ".avi"
                try:
                    video_writer = cv2.VideoWriter(video_name, fourcc, args.update_frequency, (width, height)) # Use update_frequency for frame rate approx
                    if not video_writer.isOpened():
                         raise IOError(f"Could not open video writer for {video_name}")
                    print(f"Will write video output to {video_name}")
                except Exception as e:
                    print(f"Warning: Could not initialize video writer: {e}. Video output disabled.")
                    video_writer = None # Ensure it's None if failed

            with MTWindow(title="Generic Tracking", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
                window.show_async(output_img) # Show initial empty frame

                def keyboard_cb(key, scancode, action, mods):
                    SIZE_STEP = 10
                    if action != UIAction.RELEASE: return
                    if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                        window.set_close_flag()
                    elif key == UIKeyEvent.KEY_A: # Inc min size
                        new_min = tracking_algo.min_size + SIZE_STEP
                        if new_min <= tracking_algo.max_size:
                             tracking_algo.min_size = new_min
                             print(f"Increased min size to {tracking_algo.min_size}")
                    elif key == UIKeyEvent.KEY_B: # Dec min size
                         new_min = tracking_algo.min_size - SIZE_STEP
                         if new_min >= 0:
                              tracking_algo.min_size = new_min
                              print(f"Decreased min size to {tracking_algo.min_size}")
                    elif key == UIKeyEvent.KEY_C: # Inc max size
                         tracking_algo.max_size += SIZE_STEP
                         print(f"Increased max size to {tracking_algo.max_size}")
                    elif key == UIKeyEvent.KEY_D: # Dec max size
                         new_max = tracking_algo.max_size - SIZE_STEP
                         if new_max >= tracking_algo.min_size:
                              tracking_algo.max_size = new_max
                              print(f"Decreased max size to {tracking_algo.max_size}")

                window.set_keyboard_callback(keyboard_cb)
                print("Press 'q' or Escape to leave.")
                print("Press 'a'/'b' to inc/dec minimum object size.")
                print("Press 'c'/'d' to inc/dec maximum object size.")

                # Run processing loop with UI components
                all_tracked_objects = processing_loop(
                    mv_iterator, height, width,
                    activity_noise_filter, trail_filter,
                    tracking_algo, rolling_buffer, args,
                    window=window, video_writer=video_writer, output_img=output_img,
                    draw_tracking_results_func=draw_tracking_results, base_frame_gen_algo=BaseFrameGenerationAlgorithm
                )

            # Video writer release happens after 'with MTWindow' finishes
            if video_writer:
                print("Releasing video writer...")
                video_writer.release()
                if args.out_video: # Only print if it was requested and successfully opened
                    print(f"Video saved to {video_name}")

        else: # Headless mode
            # Run processing loop without UI components
            all_tracked_objects = processing_loop(
                mv_iterator, height, width,
                activity_noise_filter, trail_filter,
                tracking_algo, rolling_buffer, args,
                window=None, video_writer=None, output_img=None,
                draw_tracking_results_func=None, base_frame_gen_algo=None
            )

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user (Ctrl+C).")
    finally:
        # --- NumPy Saving (Always, if requested and data exists) ---
        if args.out_npy:
            if all_tracked_objects:
                print(f"\nSaving {len(all_tracked_objects)} tracking results to {args.out_npy}...")
                try:
                    tracked_data_np = np.array(all_tracked_objects) 
                    np.save(args.out_npy, tracked_data_np)
                    print(f"Tracking results successfully saved.")
                except Exception as e:
                    print(f"Error saving NumPy file '{args.out_npy}': {e}")
            else:
                print(f"\nNo tracked objects recorded. NumPy file '{args.out_npy}' not saved.")
        else:
             print("\nProcessing finished.")


if __name__ == "__main__":
    main()
