import traceback
import cv2
from pathlib import Path

import joblib
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import streamlit as st
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PolarityFilterAlgorithm
from metavision_sdk_cv import  ActivityNoiseFilterAlgorithm
from rsklearn import \
    clustering  # Assuming this is 'sklearn.cluster' or a custom Rust-backed one
from sort import \
    Sort  # Assuming this is the standard SORT algorithm or a compatible version
from stqdm import stqdm as tqdm  # For Streamlit progress bars

# --- Page Configuration ---
st.set_page_config(page_title="Event Track Generator", page_icon='👣', layout="wide")

# --- Application Title and Introduction ---
st.title("👣 Event-Based Vision: Track Generation Pipeline")

st.markdown("""
Welcome to the Event Track Generation application! This tool processes event-based vision data (`.raw` files)
to detect and track objects. It follows a three-step pipeline:

1.  **Initialize Processing Tools**: Sets up filters, clustering algorithms, and trackers based on your configuration.
2.  **Process Events & Generate Tracks**: Reads event data in time windows, filters events, clusters them to find objects, and tracks these objects over time.
3.  **Save Results**: Saves the generated tracks and event cluster details to Parquet files for further analysis.

Use the sidebar to configure the processing parameters before running the pipeline steps.
""")

with st.expander("ℹ️ About this App & How to Use"):
    st.markdown("""
    - **Configuration**: All processing parameters can be set in the sidebar. Adjust these before starting.
    - **Step-by-Step Execution**:
        - Click "Run Step 1" to initialize all necessary components.
        - Once Step 1 is complete, click "Run Step 2" to process the event data. This may take time depending on the file size and parameters.
        - After Step 2 finishes, click "Run Step 3" to save the results.
    - **Persistence**: Intermediate objects and data are stored in the session state, so you need to run the steps in order. Re-running Step 1 will reset the subsequent steps.
    - **Output**: The application will generate two Parquet files: one for tracks and one for the raw event data forming each cluster.
    - **Visualization**: After Step 2, a new section will appear allowing you to scrub through time slices and visualize clusters and tracks with their history.
    """)

# --- Initialize session state variables ---
# These will persist across script reruns
if 'activity_filter' not in st.session_state:
    st.session_state.activity_filter = None
if 'activity_filter_buffer' not in st.session_state:
    st.session_state.activity_filter_buffer = None
if 'polarity_filter_alg' not in st.session_state:
    st.session_state.polarity_filter_alg = None
if 'events_buffer' not in st.session_state: # Buffer for filtered events
    st.session_state.events_buffer = None
if 'dbscan_object' not in st.session_state: # DBSCAN clustering object
    st.session_state.dbscan_object = None
if 'mot_tracker' not in st.session_state: # SORT tracking object
    st.session_state.mot_tracker = None
if 'loop_iterator_object' not in st.session_state: # Iterator for event windows
    st.session_state.loop_iterator_object = None
if 'n_windows_for_progress' not in st.session_state:
    st.session_state.n_windows_for_progress = 0


# Data collectors
if 'events_for_parquet' not in st.session_state:
    st.session_state.events_for_parquet = {}
if 'all_tracks_list' not in st.session_state:
    st.session_state.all_tracks_list = []
if 'all_tracks_concatenated' not in st.session_state:
    st.session_state.all_tracks_concatenated = np.empty((0, 7))

# Flags to manage execution flow
if 'step1_completed' not in st.session_state:
    st.session_state.step1_completed = False
if 'step2_completed' not in st.session_state:
    st.session_state.step2_completed = False
if 'step3_completed' not in st.session_state:
    st.session_state.step3_completed = False

# New session state for visualization
if 'time_slices_data' not in st.session_state:
    st.session_state.time_slices_data = []
if 'sensor_height' not in st.session_state:
    st.session_state.sensor_height = None
if 'sensor_width' not in st.session_state:
    st.session_state.sensor_width = None
if 'max_data_x' not in st.session_state: # Fallback if sensor size not available
    st.session_state.max_data_x = 0
if 'max_data_y' not in st.session_state: # Fallback if sensor size not available
    st.session_state.max_data_y = 0


# --- Configuration Settings in Sidebar ---
st.sidebar.header("⚙️ Configuration Settings")

st.sidebar.markdown("### Input/Output")
default_raw_file = "data/raw/metavision/combo_nomod.raw"
default_output_dir = "data/interim/worms_output"

if 'config_input_file' not in st.session_state:
    st.session_state.config_input_file = default_raw_file
if 'config_output_dir' not in st.session_state:
    st.session_state.config_output_dir = default_output_dir

st.session_state.config_input_file = st.sidebar.text_input(
    "Input RAW File Path",
    value=st.session_state.config_input_file,
    help="Path to the Metavision .raw event file."
)
st.session_state.config_output_dir = st.sidebar.text_input(
    "Output Directory Path",
    value=st.session_state.config_output_dir,
    help="Directory where Parquet files will be saved."
)

st.sidebar.markdown("### Event Processing")
st.session_state.config_dt = st.sidebar.number_input(
    "Time Window (dt, μs)",
    min_value=10, value=10000, step=1000,
    help="Length of each time window for event accumulation (in microseconds)."
)
st.session_state.config_start_ts = st.sidebar.number_input(
    "Start Timestamp (μs)",
    min_value=0, value=0, step=1,
    help="Timestamp in the recording to begin processing."
)
st.session_state.config_max_duration = st.sidebar.number_input(
    "Max Processing Duration (μs)",
    min_value=0, value=0, step=1,
    help="Maximum duration to process. Enter 0 for no limit."
)

st.sidebar.markdown("### DBSCAN Clustering")
st.session_state.config_eps = st.sidebar.number_input(
    "DBSCAN Epsilon (eps)",
    min_value=1, value=10, step=1,
    help="The maximum distance between two samples for one to be considered as in the neighborhood of the other."
)
st.session_state.config_min_samples = st.sidebar.number_input(
    "DBSCAN Min Samples",
    min_value=1, value=3, step=1,
    help="The number of samples in a neighborhood for a point to be considered as a core point."
)

st.sidebar.markdown("### SORT Tracking")
st.session_state.config_sort_max_age = st.sidebar.number_input(
    "SORT Max Age (time windows)",
    min_value=1, value=5, step=1,
    help="Maximum number of frames a track can exist without a detection before being deleted."
)
st.session_state.config_sort_min_hits = st.sidebar.number_input(
    "SORT Min Hits",
    min_value=1, value=3, step=1,
    help="Minimum number of hits (detections) required to start a track."
)
st.session_state.config_sort_iou_thresh = st.sidebar.slider(
    "SORT IOU Threshold",
    min_value=0.0, max_value=1.0, value=0.1, step=0.01,
    help="Minimum Intersection Over Union (IOU) required to associate a detection with a track."
)

activity_threshold = st.sidebar.number_input('Activity Threshold μs', 0, 10_000_000)

# --- Helper function for colors ---
def get_distinct_colors(n):
    """Generates N distinct colors."""
    if n == 0:
        return []
    if n <= 20:
        cmap = plt.cm.get_cmap('tab20', n)
        colors = [mcolors.to_hex(cmap(i)) for i in range(n)]
    else:
        colors = [mcolors.to_hex(plt.cm.hsv(i / n)) for i in range(n)]
    return colors

# --- Step 1: Prepare to process the events ---
st.header("Step 1: Initialize Processing Tools")
st.markdown("This step prepares all the necessary algorithms and data iterators based on the current configuration.")

with st.container():
    def step1_prepare_processing():
        st.info("Running Step 1: Preparing processing tools...")
        try:
            input_file = Path(st.session_state.config_input_file)
            dt_us = st.session_state.config_dt
            start_ts_us = st.session_state.config_start_ts
            max_duration_us = st.session_state.config_max_duration if st.session_state.config_max_duration > 0 else None
            eps = st.session_state.config_eps
            min_samples = st.session_state.config_min_samples
            sort_max_age = st.session_state.config_sort_max_age
            sort_min_hits = st.session_state.config_sort_min_hits
            sort_iou_thresh = st.session_state.config_sort_iou_thresh

            st.session_state.polarity_filter_alg = PolarityFilterAlgorithm(1)
            st.session_state.events_buffer = PolarityFilterAlgorithm.get_empty_output_buffer()
            st.session_state.activity_filter = ActivityNoiseFilterAlgorithm(640,480,activity_threshold)
            st.session_state.activity_filter_buffer = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
            st.session_state.dbscan_object = clustering.DBScan(eps=eps, min_samples=min_samples, metric="euclidean")
            st.session_state.mot_tracker = Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_iou_thresh)

            if not input_file.exists():
                st.error(f"Input file not found: {input_file}")
                st.session_state.step1_completed = False
                return

            try:
                temp_iter_for_size = EventsIterator(str(input_file), delta_t=1)
                sensor_size = temp_iter_for_size.get_size()
                if sensor_size and len(sensor_size) == 2:
                    st.session_state.sensor_height, st.session_state.sensor_width = sensor_size[0], sensor_size[1]
                    st.write(f"Detected sensor dimensions: Width={st.session_state.sensor_width}, Height={st.session_state.sensor_height}")
                else:
                    st.session_state.sensor_height, st.session_state.sensor_width = None, None
                    st.write("Could not retrieve sensor dimensions automatically. Plot limits will be data-driven.")
                del temp_iter_for_size
            except Exception as e_size:
                st.warning(f"Could not get sensor size from EventsIterator: {e_size}. Plot limits will be data-driven.")
                st.session_state.sensor_height, st.session_state.sensor_width = None, None

            with st.spinner("Counting event windows for progress estimate..."):
                temp_evts_iter = EventsIterator(str(input_file), start_ts=start_ts_us, delta_t=dt_us, max_duration=max_duration_us)
                n_windows = sum(1 for _ in temp_evts_iter)
                del temp_evts_iter
            
            st.session_state.n_windows_for_progress = n_windows
            st.write(f"Found {n_windows} event windows to process.")

            final_evts_iter = EventsIterator(str(input_file), start_ts=start_ts_us, delta_t=dt_us, max_duration=max_duration_us)
            st.session_state.loop_iterator_object = tqdm(final_evts_iter, desc="Processing Windows", total=n_windows, unit="Window")

            st.session_state.events_for_parquet = {}
            st.session_state.all_tracks_list = []
            st.session_state.time_slices_data = []
            st.session_state.max_data_x = 0
            st.session_state.max_data_y = 0

            st.session_state.step1_completed = True
            st.session_state.step2_completed = False
            st.session_state.step3_completed = False
            st.success("Step 1 completed: Processing tools and iterator are ready.")

        except FileNotFoundError:
            st.error(f"Error: Input file not found at '{st.session_state.config_input_file}'.")
            st.session_state.step1_completed = False
        except Exception as e:
            st.error(f"Error during Step 1 initialization: {e}")
            st.text(traceback.format_exc())
            st.session_state.step1_completed = False

if st.button("Run Step 1: Initialize Tools & Iterator", key="run_step1"):
    step1_prepare_processing()

# --- Step 2: Process events to generate tracks ---
st.header("Step 2: Process Events and Generate Tracks")
st.markdown("This step iterates through the event data, applies filters, performs clustering, and tracks objects using the SORT algorithm.")

with st.container():
    def step2_process_events():
        if not st.session_state.get('step1_completed', False) or not st.session_state.loop_iterator_object:
            st.warning("Please run Step 1 first.")
            return

        st.info("Running Step 2: Processing events and generating tracks...")
        loop_iterator = st.session_state.loop_iterator_object
        polarity_filter_alg = st.session_state.polarity_filter_alg
        events_buffer = st.session_state.events_buffer
        activity_filter = st.session_state.activity_filter
        activity_buffer = st.session_state.activity_filter_buffer
        mot_tracker = st.session_state.mot_tracker
        dbscan = st.session_state.dbscan_object
        
        st.session_state.events_for_parquet = {}
        st.session_state.all_tracks_list = []
        st.session_state.time_slices_data = []
        current_max_x_in_run = 0
        current_max_y_in_run = 0

        try:
            for evts in loop_iterator:
                current_window_start_ts = -1
                current_window_end_ts = -1

                if len(evts) > 0:
                    current_window_start_ts = evts['t'][0]
                    current_window_end_ts = evts['t'][-1]
                
                polarity_filter_alg.process_events(evts, events_buffer)
                activity_filter.process_events(events_buffer, activity_buffer)
                filtered_numpy = activity_buffer.numpy()

                slice_data_for_viz = {
                    "timestamp_start": current_window_start_ts,
                    "timestamp_end": current_window_end_ts,
                    "raw_events_xy": filtered_numpy[['x', 'y']].copy() if len(filtered_numpy) > 0 else np.empty((0, 2), dtype=np.uint16),
                    "cluster_points_xyt": np.empty((0, 3), dtype=np.float32),
                    "cluster_labels": np.empty(0, dtype=int),
                    "detections_for_sort": np.empty((0, 6)),
                    "tracked_objects_slice": np.empty((0, 6))
                }

                if not len(filtered_numpy):
                    tracked_objects_output = mot_tracker.update(np.empty((0, 6)))
                    slice_data_for_viz["tracked_objects_slice"] = tracked_objects_output if tracked_objects_output.shape[0] > 0 else np.empty((0, 6))
                    st.session_state.time_slices_data.append(slice_data_for_viz)
                    continue

                if current_window_end_ts == -1: current_window_end_ts = filtered_numpy[-1]["t"]
                slice_data_for_viz["timestamp_end"] = current_window_end_ts

                if st.session_state.sensor_width is None and len(filtered_numpy['x']) > 0:
                    current_max_x_in_run = max(current_max_x_in_run, np.max(filtered_numpy['x']))
                if st.session_state.sensor_height is None and len(filtered_numpy['y']) > 0:
                    current_max_y_in_run = max(current_max_y_in_run, np.max(filtered_numpy['y']))

                data_for_dbscan = np.array([(evt["x"], evt["y"], evt["t"]) for evt in filtered_numpy], dtype=np.float32, order='C')
                slice_data_for_viz["cluster_points_xyt"] = data_for_dbscan.copy()

                labels = dbscan.fit(data_for_dbscan)
                slice_data_for_viz["cluster_labels"] = labels.copy()
                unique_labels = np.unique(labels)

                processed_detections_for_sort = []
                for label_val in unique_labels:
                    if label_val == -1: continue
                    points = data_for_dbscan[labels == label_val]
                    xy_points = points[:, :2].astype(np.float32)
                    if xy_points.shape[0] > 1:
                        rotated_rect = cv2.minAreaRect(xy_points)
                        box_points =    cv2.boxPoints(rotated_rect)
                        x1, y1 = np.min(box_points, axis=0)
                        x2, y2 = np.max(box_points, axis=0)
                        # large areas are not what we want
                        hash_str = joblib.hash(points)[:15]
                        original_event_cluster_id = int(hash_str, 16)
                        st.session_state.events_for_parquet[str(original_event_cluster_id)] = points

                        confidence = float(points.shape[0])

                        if x1 < x2 and y1 < y2:
                            processed_detections_for_sort.append([x1, y1, x2, y2, confidence, original_event_cluster_id])
                        elif xy_points.shape[0] > 0:
                            processed_detections_for_sort.append([x1, y1, x1 + 1, y1 + 1, confidence, original_event_cluster_id])
                
                detections_to_sort_np = np.array(processed_detections_for_sort, dtype=np.float64) if processed_detections_for_sort else np.empty((0, 6))
                slice_data_for_viz["detections_for_sort"] = detections_to_sort_np.copy()
                
                tracked_objects_output = mot_tracker.update(detections_to_sort_np)

                final_tracked_objects_for_slice_and_saving = np.empty((0, 6))
                if tracked_objects_output.shape[0] > 0:
                    if tracked_objects_output.shape[1] >= 5:
                        # Assuming 6th col is original_event_cluster_id or detection index.
                        # For simplicity, we'll ensure it is 6 columns for saving.
                        # A robust implementation would map detection index back to original_event_cluster_id.
                        if tracked_objects_output.shape[1] == 5: # x1,y1,x2,y2,track_id
                            placeholder_id_col = np.full((tracked_objects_output.shape[0], 1), -1, dtype=np.int64)
                            final_tracked_objects_for_slice_and_saving = np.hstack((tracked_objects_output, placeholder_id_col))
                        else: # Assume 6 columns are passed through
                            final_tracked_objects_for_slice_and_saving = tracked_objects_output.copy()
                
                slice_data_for_viz["tracked_objects_slice"] = final_tracked_objects_for_slice_and_saving.copy()

                if final_tracked_objects_for_slice_and_saving.shape[0] > 0:
                    timestamp_col = np.full((final_tracked_objects_for_slice_and_saving.shape[0], 1), current_window_end_ts, dtype=np.int64)
                    tracks_to_save = np.hstack((timestamp_col, final_tracked_objects_for_slice_and_saving))
                    st.session_state.all_tracks_list.append(tracks_to_save)
                
                st.session_state.time_slices_data.append(slice_data_for_viz)
            
            if st.session_state.sensor_width is None:
                st.session_state.max_data_x = current_max_x_in_run if current_max_x_in_run > 0 else 640
            if st.session_state.sensor_height is None:
                st.session_state.max_data_y = current_max_y_in_run if current_max_y_in_run > 0 else 480

            # Consolidate track history for efficient visualization
            if st.session_state.all_tracks_list:
                st.session_state.all_tracks_concatenated = np.concatenate(st.session_state.all_tracks_list, axis=0)
            else:
                st.session_state.all_tracks_concatenated = np.empty((0, 7))

            st.session_state.step2_completed = True
            st.success("Step 2 completed: Event processing and track generation finished.")
            st.info(f"Total track segments: {len(st.session_state.all_tracks_concatenated)}")
            st.info(f"Total unique event clusters: {len(st.session_state.events_for_parquet)}")

        except Exception as e:
            st.error(f"Error during Step 2 execution: {e}")
            st.text(traceback.format_exc())
            st.session_state.step2_completed = False
        finally:
            if hasattr(loop_iterator, 'close'):
                loop_iterator.close()

if st.button("Run Step 2: Process Events & Generate Tracks", key="run_step2", disabled=not st.session_state.get('step1_completed', False)):
    step2_process_events()

# --- Visualization Section ---
st.header("🔬 Time Slice Scrubber & Visualization")

rect_info = st.empty()
def plot_time_slice_data(slice_data, plot_width, plot_height, all_tracks_history):
    """
    Plots the data for a single time slice with two panels:
    1. Clusters and Detections from DBSCAN.
    2. Active Tracks from SORT, including their recent history (trajectory).
    """
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))

    # --- Plot 1: Clusters & Detections (Left Panel) ---
    ax1 = axs[0]
    raw_events_xy = slice_data["raw_events_xy"]
    cluster_points_xyt = slice_data["cluster_points_xyt"]
    dbscan_labels = slice_data["cluster_labels"]
    detections_for_sort = slice_data["detections_for_sort"]



    if raw_events_xy.shape[0] > 0:
        ax1.scatter(raw_events_xy['x'], raw_events_xy['y'], s=1, color='lightgrey', alpha=0.5, label='Filtered Events')

    if cluster_points_xyt.shape[0] > 0 and dbscan_labels.shape[0] == cluster_points_xyt.shape[0]:
        unique_dbscan_labels = np.unique(dbscan_labels[dbscan_labels != -1])
        dbscan_label_colors = {lbl: c for lbl, c in zip(unique_dbscan_labels, get_distinct_colors(len(unique_dbscan_labels)))}
        
        for label_val in unique_dbscan_labels:
            points = cluster_points_xyt[dbscan_labels == label_val]
            ax1.scatter(points[:, 0], points[:, 1], s=10, color=dbscan_label_colors.get(label_val), alpha=0.9, label=f'Cluster {label_val}')

    if detections_for_sort.shape[0] > 0:
        for det in detections_for_sort:
            x1, y1, x2, y2, _, orig_id = det
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor='crimson', facecolor='none', alpha=0.9)
            ax1.add_patch(rect)
            ax1.text(x1, y1 - 3, f"Det (OrigID: {int(orig_id)})", color='crimson', fontsize=7)
            with rect_info.container():
                st.info(f'Cluster Area: {(x2 - x1) * (y2 - y1)}')

    ax1.set_title(f"Clusters & Detections at T≈{slice_data['timestamp_end'] / 1e6:.3f}s")
    ax1.set_xlabel("X coordinate")
    ax1.set_ylabel("Y coordinate")
    ax1.set_xlim(0, plot_width)
    ax1.set_ylim(plot_height, 0)
    ax1.set_aspect('equal', adjustable='box')
    ax1.legend(fontsize='x-small', loc='upper right')

    # --- Plot 2: Tracks & History (Right Panel) ---
    ax2 = axs[1]
    tracked_bbs_slice = slice_data["tracked_objects_slice"]
    current_ts = slice_data['timestamp_end']
    
    if raw_events_xy.shape[0] > 0:
        ax2.scatter(raw_events_xy['x'], raw_events_xy['y'], s=1, color='lightgrey', alpha=0.5)

    if tracked_bbs_slice.shape[0] > 0:
        unique_track_ids = np.unique(tracked_bbs_slice[:, 4].astype(int))
        track_colors = {tid: color for tid, color in zip(unique_track_ids, get_distinct_colors(len(unique_track_ids)))}

        for bb in tracked_bbs_slice:
            x1, y1, x2, y2, track_id, _ = bb
            track_id = int(track_id)
            color = track_colors.get(track_id, '#000000')

            if all_tracks_history.shape[0] > 0:
                history_for_track = all_tracks_history[(all_tracks_history[:, 5] == track_id) & (all_tracks_history[:, 0] <= current_ts)]
                if history_for_track.shape[0] > 0:
                    centroids_x = (history_for_track[:, 1] + history_for_track[:, 3]) / 2
                    centroids_y = (history_for_track[:, 2] + history_for_track[:, 4]) / 2
                    ax2.plot(centroids_x, centroids_y, 'o-', color=color, markersize=2, linewidth=1, alpha=0.7)

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor=color, alpha=0.4)
            ax2.add_patch(rect)
            ax2.text(x1, y1 - 5, f"Track ID: {track_id}", color=color, fontsize=9, fontweight='bold')

    ax2.set_title(f"Active Tracks & Trajectories at T≈{current_ts / 1e6:.3f}s")
    ax2.set_xlabel("X coordinate")
    ax2.set_ylabel("Y coordinate")
    ax2.set_xlim(0, plot_width)
    ax2.set_ylim(plot_height, 0)
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout(pad=2.0)
    st.pyplot(fig)
    plt.close(fig)

if st.session_state.get('step2_completed', False) and st.session_state.time_slices_data:
    num_slices = len(st.session_state.time_slices_data)
    st.info(f"Data for {num_slices} time slices available. Use the slider to explore.")

    selected_slice_idx = st.slider(
        "Select Time Slice:",
        min_value=0,
        max_value=num_slices - 1,
        value=0,
        step=1,
        key="time_slice_slider"
    )

    if 0 <= selected_slice_idx < num_slices:
        current_slice_data = st.session_state.time_slices_data[selected_slice_idx]
        
        st.write(f"**Displaying Slice: {selected_slice_idx + 1} / {num_slices}** at timestamp ≈ {current_slice_data['timestamp_end'] / 1e6:.3f} seconds.")

        plot_w = st.session_state.get('sensor_width') or st.session_state.get('max_data_x') or 640
        plot_h = st.session_state.get('sensor_height') or st.session_state.get('max_data_y') or 480
        
        plot_time_slice_data(
            current_slice_data,
            plot_w,
            plot_h,
            st.session_state.get('all_tracks_concatenated', np.empty((0, 7)))
        )
        
        with st.expander("Show Raw Data for this Slice"):
            serializable_data = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in current_slice_data.items()
            }
            st.json(serializable_data)

elif st.session_state.get('step2_completed', False):
    st.warning("No time slice data was collected. Visualization is unavailable.")
else:
    st.info("Run Step 1 and Step 2 to generate data for visualization.")


# --- Step 3: Save Tracks and Event Clusters to Parquet ---
st.header("Step 3: Save Results")
st.markdown("This final step saves the generated tracks and the raw event data for each identified cluster into compressed Parquet files.")

with st.container():
    def step3_save_data():
        if not st.session_state.get('step2_completed', False):
            st.warning("Please run Step 2 first.")
            return
        
        if not st.session_state.all_tracks_list and not st.session_state.events_for_parquet:
            st.warning("No tracks or event clusters were generated. Nothing to save.")
            st.session_state.step3_completed = True
            return

        dt_us = st.session_state.config_dt
        st.info("Running Step 3: Saving data to Parquet files...")
        
        output_dir_path = Path(st.session_state.config_output_dir)
        input_file_stem = Path(st.session_state.config_input_file).stem
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # --- REVISED FILE NAMING LOGIC ---
        base_tracks_name = f"{input_file_stem}_{dt_us}_tracks"
        base_events_name = f"{input_file_stem}_{dt_us}_event_clusters"
        
        output_tracks_file = output_dir_path / f"{base_tracks_name}.parquet"
        output_event_clusters_file = output_dir_path / f"{base_events_name}.parquet"
        
        n = 1
        while output_tracks_file.exists() or output_event_clusters_file.exists():
            output_tracks_file = output_dir_path / f"{base_tracks_name}_{n}.parquet"
            output_event_clusters_file = output_dir_path / f"{base_events_name}_{n}.parquet"
            n += 1
        # --- END OF REVISION ---
        
        st.write(f"Target tracks file: `{output_tracks_file}`")
        st.write(f"Target event clusters file: `{output_event_clusters_file}`")

        try:
            # Save Tracks
            track_fields_types = [
                ("window_end_ts", pa.int64()), ("x1", pa.float64()), ("y1", pa.float64()),
                ("x2", pa.float64()), ("y2", pa.float64()), ("track_id", pa.int64()),
                ("original_event_cluster_id", pa.int64())
            ]
            tracks_schema = pa.schema(track_fields_types)

            final_tracks_array = st.session_state.all_tracks_concatenated
            if final_tracks_array.shape[0] > 0:
                tracks_table = pa.Table.from_arrays([pa.array(final_tracks_array[:, i]) for i in range(final_tracks_array.shape[1])], schema=tracks_schema)
                pq.write_table(tracks_table, output_tracks_file, compression='snappy')
                st.write(f"Saved {tracks_table.num_rows} track segments to '{output_tracks_file}'.")
            else:
                empty_table = pa.Table.from_arrays([pa.array([], type=field.type) for field in tracks_schema], schema=tracks_schema)
                pq.write_table(empty_table, output_tracks_file, compression='snappy')
                st.write(f"No tracks to save. Empty tracks file created.")

            # Save Event Clusters
            event_cluster_fields_types = [
                ('x', pa.float32()), ('y', pa.float32()), ('t', pa.float32()),
                ('cluster_id', pa.int64())
            ]
            event_schema = pa.schema(event_cluster_fields_types)
            
            with pq.ParquetWriter(output_event_clusters_file, event_schema, compression='snappy') as writer:
                if st.session_state.events_for_parquet:
                    for uniq_id_str, points_array in st.session_state.events_for_parquet.items():
                        if points_array.shape[0] == 0: continue
                        try:
                            cluster_id_val = np.int64(uniq_id_str)
                            
                            x_col = pa.array(points_array[:, 0])
                            y_col = pa.array(points_array[:, 1])
                            t_col = pa.array(points_array[:, 2])
                            cluster_id_col = pa.array(np.full(points_array.shape[0], cluster_id_val, dtype=np.int64))

                            batch_table = pa.Table.from_arrays([x_col, y_col, t_col, cluster_id_col], schema=event_schema)
                            writer.write_table(batch_table)
                        except ValueError:
                            st.warning(f"Skipping cluster with non-integer ID: {uniq_id_str}")
            st.write(f"Saved event clusters to '{output_event_clusters_file}'.")

            st.success(f"Step 3 completed: Parquet saving finished.")
            st.session_state.step3_completed = True

        except Exception as e:
            st.error(f"Error during Step 3 (Parquet Saving): {e}")
            st.text(traceback.format_exc())
            st.session_state.step3_completed = False

if st.button("Run Step 3: Save Data to Parquet Files", key="run_step3", disabled=not st.session_state.get('step2_completed', False)):
    step3_save_data()


# --- Optional: Display Code Details ---
with st.expander("Show Code Block Details (for reference)"):
    st.markdown("The key functions have been updated for better visualization and robustness.")
    st.markdown("#### Step 2: `step2_process_events()`")
    st.code("""
# ... (inside the loop)
# Collects detailed data for each time slice for visualization.
# ... (after the loop)
# NEW: Consolidates all track segments into a single NumPy array
st.session_state.all_tracks_concatenated = np.concatenate(...)
    """, language="python")

    st.markdown("#### Visualization Section: `plot_time_slice_data()`")
    st.code("""
# UPDATED to accept `all_tracks_history` as an argument.
def plot_time_slice_data(slice_data, plot_width, plot_height, all_tracks_history):
    # Plot 1 (Left): Clusters & Detections (as before)
    # ...
    # Plot 2 (Right): Tracks & History
    # For each active track in the current slice:
    # 1. Draws the current bounding box.
    # 2. Searches `all_tracks_history` for past points with the same track_id.
    # 3. Calculates the centroids of these historical points.
    # 4. Plots a line connecting the centroids to show the track's trajectory ("tail").
    pass
    """, language="python")

st.markdown("---")
st.markdown("Application ready.")
