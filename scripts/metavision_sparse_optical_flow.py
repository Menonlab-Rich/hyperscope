# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Code sample showing how to use Metavision SDK to display results of sparse optical flow.
MODIFIED FOR HEADLESS OPERATION.
"""

import numpy as np
import os
import h5py
from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
from metavision_sdk_cv import SparseOpticalFlowAlgorithm, SparseOpticalFlowConfigPreset, SparseFlowFrameGeneratorAlgorithm, SpatioTemporalContrastAlgorithm
# Import UI elements conditionally later if needed
# from metavision_sdk_ui import EventLoop, BaseWindow, Window, UIAction, UIKeyEvent
from skvideo.io import FFmpegWriter
import argparse # Moved import here for cleaner structure


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision Sparse Optical Flow sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Input Group ---
    input_group = parser.add_argument_group(
        "Input", "Arguments related to input sequence.")
    input_group.add_argument(
        '-i', '--input-event-file', dest='event_file_path', default="",
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    input_group.add_argument(
        '-r', '--replay_factor', type=float, default=1,
        help="Replay Factor. If greater than 1.0 we replay with slow-motion, otherwise this is a speed-up over real-time.")
    input_group.add_argument(
        '--dt-step', type=int, default=33333, dest="dt_step",
        help="Time processing step (in us), used as iteration delta_t period, visualization framerate and accumulation time.")

    # --- Noise Filtering Group ---
    noise_filtering_group = parser.add_argument_group(
        "Noise Filtering", "Arguments related to STC noise filtering.")
    noise_filtering_group.add_argument(
        "--disable-stc", dest="disable_stc", action="store_true",
        help="Disable STC noise filtering. All other options related to noise filtering are discarded.")
    noise_filtering_group.add_argument("--stc-filter-thr", dest="stc_filter_thr", type=int, default=40000,
                                       help="Length of the time window for filtering (in us).")
    noise_filtering_group.add_argument(
        "--disable-stc-cut-trail", dest="stc_cut_trail", default=True, action="store_false",
        help="When stc cut trail is enabled, after an event goes through, it removes all events until change of polarity.")

    # --- Output Flow Group ---
    output_flow_group = parser.add_argument_group(
        "Output flow", "Arguments related to output optical flow.")
    output_flow_group.add_argument(
        "--output-sparse-npy-filename", dest="output_sparse_npy_filename",
        help="If provided, the predictions will be saved as numpy structured array of EventOpticalFlow. In this "
        "format, the flow vx and vy are expressed in pixels per second.")
    output_flow_group.add_argument(
        "--output-dense-h5-filename", dest="output_dense_h5_filename",
        help="If provided, the predictions will be saved as a sequence of dense flow in HDF5 data. The flows are "
        "averaged pixelwise over timeslices of --dt-step. The dense flow is expressed in terms of "
        "pixels per timeslice (of duration dt-step), not in pixels per second.")
    output_flow_group.add_argument(
        '-o', '--out-video', dest='out_video', type=str, default="",
        help="Path to an output AVI file to save the resulting video.")
    output_flow_group.add_argument(
        '--fps', dest='fps', type=int, default=30,
        help="replay fps of output video")

    # --- Display/Headless Group ---
    display_group = parser.add_argument_group(
        "Display", "Arguments related to visualization.")
    display_group.add_argument(
        '--headless', action='store_true',
        help="Run the script without displaying a GUI window. Necessary if no display server is available.")

    args = parser.parse_args()

    # --- Argument Validation ---
    if not args.headless and not args.out_video and not args.output_sparse_npy_filename and not args.output_dense_h5_filename:
         print("Warning: No output specified (video, npy, h5) and not running headless. GUI window will be shown.")
    elif args.headless and not args.out_video and not args.output_sparse_npy_filename and not args.output_dense_h5_filename:
        parser.error("Error: Running headless requires at least one output argument (--out-video, --output-sparse-npy-filename, or --output-dense-h5-filename).")


    if args.output_sparse_npy_filename:
        # Basic check, could be more robust (check dir exists and is writable)
        assert not os.path.exists(args.output_sparse_npy_filename), f"Output file {args.output_sparse_npy_filename} already exists."
    if args.output_dense_h5_filename:
        assert not os.path.exists(args.output_dense_h5_filename), f"Output file {args.output_dense_h5_filename} already exists."
    if args.out_video:
         video_name = args.out_video if args.out_video.lower().endswith(".avi") else args.out_video + ".avi"
         assert not os.path.exists(video_name), f"Output video file {video_name} already exists."


    return args


def main():
    """ Main """
    args = parse_args()

    # Import UI components only if needed
    if not args.headless:
        try:
            from metavision_sdk_ui import EventLoop, BaseWindow, Window, UIAction, UIKeyEvent
        except ImportError as e:
            print(f"Error importing Metavision UI components: {e}")
            print("Please ensure Metavision SDK with UI support is installed correctly, or run with the --headless flag.")
            return 1 # Indicate error

    # --- Initialize Metavision components ---
    print("Initializing...")
    # Events iterator on Camera or event file
    try:
        mv_iterator = EventsIterator(
            input_path=args.event_file_path, delta_t=args.dt_step)

        # Set ERC for live cameras if possible
        if hasattr(mv_iterator.reader, "device") and mv_iterator.reader.device:
            try:
                erc_module = mv_iterator.reader.device.get_i_erc_module()
                if erc_module:
                    print("Setting camera ERC to 20 Mev/s")
                    erc_module.set_cd_event_rate(20000000)
                    erc_module.enable(True)
            except Exception as e:
                print(f"Warning: Could not configure ERC: {e}")

    except Exception as e:
        print(f"Error: Could not initialize EventsIterator: {e}")
        print(f"Check input path '{args.event_file_path}' or camera connection.")
        return 1 # Indicate error

    # Add LiveReplay wrapper if needed
    if args.replay_factor != 1 and not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(
            mv_iterator, replay_factor=args.replay_factor)

    try:
        height, width = mv_iterator.get_size()  # Camera Geometry
    except Exception as e:
        print(f"Error: Could not get sensor geometry from EventsIterator: {e}")
        return 1 # Indicate error

    print(f"Sensor geometry: {width}x{height}")

    # Event Frame Generator (needed for visualization/video output)
    if not args.headless or args.out_video:
        event_frame_gen = OnDemandFrameGenerationAlgorithm(width, height, args.dt_step)
        output_img = np.zeros((height, width, 3), np.uint8) # Initialize buffer

    # Sparse Optical Flow Algorithm
    flow_algo = SparseOpticalFlowAlgorithm(width, height, SparseOpticalFlowConfigPreset.FastObjects)
    flow_buffer = SparseOpticalFlowAlgorithm.get_empty_output_buffer()

    # Flow Frame Generator (needed for visualization/video output)
    if not args.headless or args.out_video:
        flow_frame_gen = SparseFlowFrameGeneratorAlgorithm()

    # STC filter
    if not args.disable_stc:
        stc_filter = SpatioTemporalContrastAlgorithm(width, height, args.stc_filter_thr, args.stc_cut_trail)
        events_buf = SpatioTemporalContrastAlgorithm.get_empty_output_buffer()
    else:
        stc_filter = None # Explicitly set to None if disabled
        events_buf = None # Not used, but set for clarity

    # Data storage for output files
    all_flow_events = [] if args.output_sparse_npy_filename else None
    all_dense_flows = [] if args.output_dense_h5_filename else None
    all_dense_flows_start_ts = [] if args.output_dense_h5_filename else None
    all_dense_flows_end_ts = [] if args.output_dense_h5_filename else None

    # Video Writer
    writer = None
    if args.out_video:
        video_name = args.out_video if args.out_video.lower().endswith(".avi") else args.out_video + ".avi"
        try:
            writer = FFmpegWriter(video_name, inputdict={'-r': str(args.fps)}, outputdict={
                '-vcodec': 'libx264', # More common and efficient codec
                '-pix_fmt': 'yuv420p', # Necessary for broad compatibility
                '-crf': '23', # Constant Rate Factor (lower is better quality, 18-28 is common)
                '-preset': 'fast', # Encoding speed vs compression trade-off
                '-r': str(args.fps)
            })
            print(f"Output video will be saved to: {video_name}")
        except Exception as e:
            print(f"Error: Could not initialize FFmpegWriter for {video_name}: {e}")
            print("Ensure FFmpeg is installed and accessible by sk-video.")
            # Decide whether to continue without video or exit
            if not args.headless and not args.output_sparse_npy_filename and not args.output_dense_h5_filename:
                 print("No other outputs requested, exiting.")
                 return 1
            else:
                 print("Continuing without video output.")
                 args.out_video = "" # Disable video writing logic later

    # --- Setup Window (only if not headless) ---
    window = None
    if not args.headless:
        window = Window(title="Metavision Sparse Optical Flow", width=width, height=height, mode=BaseWindow.RenderMode.BGR)

        def keyboard_cb(key, scancode, action, mods):
            if window and (action == UIAction.PRESS or action == UIAction.REPEAT): # Handle press or repeat
                if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                    print("Q or ESC pressed, closing window.")
                    window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)
        print("GUI Window created. Press Q or ESC to exit.")

    # --- Processing Loop ---
    print("Starting processing loop...")
    total_events = 0
    event_count_in_period = 0
    last_ts = mv_iterator.get_current_time() if hasattr(mv_iterator, 'get_current_time') else 0 # Approximate start
    processing_ts = last_ts

    try:
        for evs in mv_iterator:
            if evs.size == 0:
                continue # Skip empty buffers

            current_time = mv_iterator.get_current_time() # More reliable timestamp
            processing_ts = current_time # Align processing TS with buffer end time
            total_events += evs.size
            event_count_in_period += evs.size

            # --- UI Handling (only if not headless) ---
            if window:
                EventLoop.poll_and_dispatch()
                if window.should_close():
                    print("Window close requested.")
                    break

            # --- STC Filtering ---
            if stc_filter:
                stc_filter.process_events(evs, events_buf)
                input_for_flow = events_buf # Use filtered events
            else:
                input_for_flow = evs # Use raw events

            if input_for_flow.size == 0 and (not args.headless or args.out_video):
                 # If no events after filtering, we still need to generate a blank frame
                 # for the video or display to keep timing consistent.
                 if event_frame_gen:
                     event_frame_gen.generate(processing_ts, output_img) # Generate empty frame
                 # No flow to calculate or draw if input is empty
            else:
                # --- Flow Calculation ---
                flow_algo.process_events(input_for_flow, flow_buffer)

                # --- Frame Generation (if needed for video or display) ---
                if not args.headless or args.out_video:
                    # Generate base event frame
                    event_frame_gen.process_events(input_for_flow) # Use the same events as flow input
                    event_frame_gen.generate(processing_ts, output_img)

                    # Draw flow on top
                    flow_frame_gen.add_flow_for_frame_update(flow_buffer)
                    flow_frame_gen.clear_ids() # Reset internal state
                    flow_frame_gen.update_frame_with_flow(output_img)

            # --- Data Aggregation for Files ---
            if all_flow_events is not None:
                # Ensure buffer is copied if needed later
                flow_numpy = flow_buffer.numpy()
                if flow_numpy.size > 0: # Avoid appending empty arrays
                     all_flow_events.append(flow_numpy.copy())

            if all_dense_flows is not None:
                all_dense_flows_start_ts.append(processing_ts - args.dt_step) # Approximate start
                all_dense_flows_end_ts.append(processing_ts) # End timestamp
                flow_np = flow_buffer.numpy() # Get numpy version (might be empty)
                if flow_np.size == 0:
                    all_dense_flows.append(np.zeros((2, height, width), dtype=np.float32))
                else:
                    # (Dense flow calculation remains the same as original)
                    xs, ys = flow_np["x"], flow_np["y"]
                    vx, vy = flow_np["vx"], flow_np["vy"]
                     # Filter out potential out-of-bounds coordinates (safer)
                    valid_idx = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
                    if not np.any(valid_idx):
                         all_dense_flows.append(np.zeros((2, height, width), dtype=np.float32))
                         continue # Skip if no valid events after filtering

                    xs, ys = xs[valid_idx], ys[valid_idx]
                    vx, vy = vx[valid_idx], vy[valid_idx]

                    coords = np.stack((ys, xs)) # Use filtered coords
                    abs_coords = np.ravel_multi_index(coords, (height, width))
                    # Weights: Use 1 for counts, vx/vy for flow sums
                    ones = np.ones(vx.size, dtype=np.float32)
                    counts = np.bincount(abs_coords, weights=ones, minlength=height*width).reshape(height, width)
                    flow_x = np.bincount(abs_coords, weights=vx, minlength=height*width).reshape(height, width)
                    flow_y = np.bincount(abs_coords, weights=vy, minlength=height*width).reshape(height, width)

                    # Avoid division by zero - use np.divide with where clause
                    valid_counts_mask = counts > 0
                    flow_x = np.divide(flow_x, counts, out=np.zeros_like(flow_x), where=valid_counts_mask)
                    flow_y = np.divide(flow_y, counts, out=np.zeros_like(flow_y), where=valid_counts_mask)

                    # flow expressed in pixels per delta_t
                    flow_x *= (args.dt_step * 1e-6)
                    flow_y *= (args.dt_step * 1e-6)
                    flow = np.stack((flow_x, flow_y)).astype(np.float32)
                    all_dense_flows.append(flow)


            # --- Update Display (if not headless) ---
            if window:
                window.show(output_img)

            # --- Write Video Frame (if enabled) ---
            if writer:
                try:
                    # FFmpegWriter usually expects RGB format (reverse BGR)
                    writer.writeFrame(output_img[..., ::-1])
                except Exception as e:
                    print(f"Warning: Failed to write video frame: {e}")
                    # Optionally disable further writing or break
                    # writer.close()
                    # writer = None


            # --- Progress reporting (optional) ---
            # if current_time - last_ts >= 1000000: # Report roughly every second
            #      rate = event_count_in_period / ((current_time - last_ts) * 1e-6)
            #      print(f"Processed up to T={current_time/1e6:.2f} s, Rate={rate/1e6:.2f} Mev/s")
            #      last_ts = current_time
            #      event_count_in_period = 0


    except StopIteration:
        print("Input iterator finished.")
    except KeyboardInterrupt:
        print("Processing interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"An unexpected error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    finally:
        # --- Cleanup ---
        print("Cleaning up resources...")
        if window:
            window.destroy()
        if writer:
            print("Closing video writer...")
            try:
                writer.close()
            except Exception as e:
                print(f"Warning: Error closing video writer: {e}")

    print(f"Processed a total of {total_events} events.")

    # --- Save Output Files ---
    if args.output_sparse_npy_filename and all_flow_events:
        print(f"Concatenating {len(all_flow_events)} sparse flow buffers...")
        try:
             if all_flow_events: # Ensure list is not empty
                 all_flow_events_np = np.concatenate(all_flow_events)
                 print(f"Saving {all_flow_events_np.size} sparse flow events to: {args.output_sparse_npy_filename}")
                 np.save(args.output_sparse_npy_filename, all_flow_events_np)
             else:
                  print(f"No sparse flow events generated, skipping save to {args.output_sparse_npy_filename}")
        except Exception as e:
            print(f"Error saving sparse NPY file: {e}")


    if args.output_dense_h5_filename and all_dense_flows:
        print(f"Saving {len(all_dense_flows)} dense flow frames to: {args.output_dense_h5_filename}")
        try:
            if all_dense_flows: # Ensure list is not empty
                flow_start_ts = np.array(all_dense_flows_start_ts)
                flow_end_ts = np.array(all_dense_flows_end_ts)
                flows = np.stack(all_dense_flows) # Stack along a new first dimension
                N = flow_start_ts.size
                assert flow_end_ts.size == N
                assert flows.shape == (N, 2, height, width)

                dirname = os.path.dirname(args.output_dense_h5_filename)
                if dirname and not os.path.isdir(dirname): # Check if dirname is not empty
                    print(f"Creating output directory: {dirname}")
                    os.makedirs(dirname)

                with h5py.File(args.output_dense_h5_filename, "w") as flow_h5:
                    flow_h5.create_dataset("flow_start_ts", data=flow_start_ts, compression="gzip")
                    flow_h5.create_dataset("flow_end_ts", data=flow_end_ts, compression="gzip")
                    flow_h5.create_dataset("flow", data=flows.astype(np.float32), compression="gzip")
                    # Add attributes
                    flow_h5["flow"].attrs["input_file_name"] = os.path.basename(args.event_file_path) if args.event_file_path else "live_camera"
                    flow_h5["flow"].attrs["checkpoint_path"] = "metavision_sparse_optical_flow" # Placeholder/identifier
                    flow_h5["flow"].attrs["event_input_height"] = height
                    flow_h5["flow"].attrs["event_input_width"] = width
                    flow_h5["flow"].attrs["delta_t_us"] = args.dt_step # Store dt_step in microseconds
            else:
                 print(f"No dense flow frames generated, skipping save to {args.output_dense_h5_filename}")

        except Exception as e:
            print(f"Error saving dense HDF5 file: {e}")

    print("Processing finished.")
    return 0 # Indicate success


if __name__ == "__main__":
    import sys
    sys.exit(main()) # Exit with the return code of main
