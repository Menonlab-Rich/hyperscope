import traceback
from pathlib import Path

import joblib
import matplotlib.colors as mcolors  # Added for colors
import matplotlib.patches as patches  # Added for bounding boxes
import matplotlib.pyplot as plt  # Added for plotting
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import streamlit as st
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PolarityFilterAlgorithm
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
    - **Visualization**: After Step 2, a new section will appear allowing you to scrub through time slices and visualize clusters and tracks.
    """)

# --- Initialize session state variables ---
# These will persist across script reruns
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
    st.session_state.events_for_parquet = {} # Changed from HDF5 to Parquet
if 'all_tracks_list' not in st.session_state:
    st.session_state.all_tracks_list = []

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
    st.session_state.sensor_height = None # e.g., 480
if 'sensor_width' not in st.session_state:
    st.session_state.sensor_width = None  # e.g., 640
if 'max_data_x' not in st.session_state: # Fallback if sensor size not available
    st.session_state.max_data_x = 0
if 'max_data_y' not in st.session_state: # Fallback if sensor size not available
    st.session_state.max_data_y = 0


# --- Configuration Settings in Sidebar ---
st.sidebar.header("⚙️ Configuration Settings")

st.sidebar.markdown("### Input/Output")
# Provide sensible defaults if config object is not available or to make app standalone
default_raw_file = "data/raw/metavision/combo_nomod.raw" # Placeholder
default_output_dir = "data/interim/worms_output" # Placeholder

# Use session state to store config to ensure they persist if user navigates or widget reruns
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
    min_value=10, value=10000, step=1, # typical: 20ms=20000us, 50ms=50000us
    help="Length of each time window for event accumulation (in microseconds)."
)
st.session_state.config_start_ts = st.sidebar.number_input(
    "Start Timestamp (μs)",
    min_value=0, value=0, step=1, # Default to 0
    help="Timestamp in the recording to begin processing."
)
st.session_state.config_max_duration = st.sidebar.number_input(
    "Max Processing Duration (μs)",
    min_value=0, value=0, step=1,
    help="Maximum duration of the recording to process. Enter 0 for no limit (process until end of file after start_ts)."
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
    min_value=1, value=5, step=1, # Number of time windows
    help="Maximum number of consecutive frames a track can exist without a detection before being deleted."
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

# --- Helper function for colors ---
def get_distinct_colors(n):
    """Generates N distinct colors."""
    if n == 0:
        return []
    # Using a common colormap and picking spaced out colors
    # Using 'tab20' for good distinctiveness up to 20, then cycle if more needed
    if n <= 20:
        cmap = plt.cm.get_cmap('tab20', n)
        colors = [mcolors.to_hex(cmap(i)) for i in range(n)]
    else: # If more than 20, 'hsv' provides more, but distinctiveness can reduce
        colors = [mcolors.to_hex(plt.cm.hsv(i/n)) for i in range(n)]
    return colors

# --- Step 1: Prepare to process the events ---
st.header("Step 1: Initialize Processing Tools")
st.markdown("This step prepares all the necessary algorithms and data iterators based on the current configuration. It will count the number of event windows to process, which might take a moment for large files.")

with st.container(): # Changed from st.echo()
    def step1_prepare_processing():
        st.info("Running Step 1: Preparing processing tools...")
        try:
            # Use configured values
            input_file = Path(st.session_state.config_input_file)
            dt_us = st.session_state.config_dt
            print(dt_us)
            start_ts_us = st.session_state.config_start_ts
            max_duration_us = st.session_state.config_max_duration if st.session_state.config_max_duration > 0 else None
            eps = st.session_state.config_eps
            min_samples = st.session_state.config_min_samples
            sort_max_age = st.session_state.config_sort_max_age
            sort_min_hits = st.session_state.config_sort_min_hits
            sort_iou_thresh = st.session_state.config_sort_iou_thresh

            st.session_state.polarity_filter_alg = PolarityFilterAlgorithm(1) # Keep positive (polarity = 1) events
            st.session_state.events_buffer = PolarityFilterAlgorithm.get_empty_output_buffer()
            st.session_state.dbscan_object = clustering.DBScan(eps=eps, min_samples=min_samples, metric="euclidean") # Corrected metric spelling
            st.session_state.mot_tracker = Sort(
                max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_iou_thresh
            )

            if not input_file.exists():
                st.error(f"Input file not found: {input_file}")
                st.session_state.step1_completed = False
                return

            # Get sensor dimensions
            try:
                # For metavision_core.event_io.EventsIterator, .get_size() returns (height, width)
                temp_iter_for_size = EventsIterator(str(input_file), delta_t=1) # Open quickly just for size
                sensor_size = temp_iter_for_size.get_size() # This might be (height, width)
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


            # Count event windows for progress bar
            with st.spinner("Counting event windows for progress estimate..."):
                print(dt_us)
                print(max_duration_us)
                temp_evts_iter = EventsIterator(
                    str(input_file), start_ts=start_ts_us, delta_t=dt_us, max_duration=max_duration_us
                )
                n_windows = sum(1 for _ in temp_evts_iter)
                del temp_evts_iter # Explicitly delete to free resources if any
            
            st.session_state.n_windows_for_progress = n_windows
            st.write(f"Found {n_windows} event windows to process based on current settings.")

            # Re-create iterator for actual processing
            final_evts_iter = EventsIterator(
                str(input_file), start_ts=start_ts_us, delta_t=dt_us, max_duration=max_duration_us # dt_us is correct
            )
            st.session_state.loop_iterator_object = tqdm( # Use stqdm here
                final_evts_iter, desc="Processing Windows", total=n_windows, unit="Window"
            )

            # Reset data collectors when starting a new processing prep
            st.session_state.events_for_parquet = {}
            st.session_state.all_tracks_list = []
            st.session_state.time_slices_data = [] # Reset visualization data
            st.session_state.max_data_x = 0 # Reset data-driven limits
            st.session_state.max_data_y = 0


            st.session_state.step1_completed = True
            st.session_state.step2_completed = False # Reset step 2 status
            st.session_state.step3_completed = False # Reset step 3 status
            st.success("Step 1 completed: Processing tools and iterator are ready.")
            st.info(f"Iterator prepared with {n_windows} windows.")

        except FileNotFoundError:
            st.error(f"Error: Input file not found at '{st.session_state.config_input_file}'. Please check the path in settings.")
            st.session_state.step1_completed = False
        except Exception as e:
            st.error(f"Error during Step 1 initialization: {e}")
            st.text(traceback.format_exc())
            st.session_state.step1_completed = False

if st.button("Run Step 1: Initialize Tools & Iterator", key="run_step1"):
    step1_prepare_processing()

# --- Step 2: Process events to generate tracks ---
st.header("Step 2: Process Events and Generate Tracks")
st.markdown("This step iterates through the event data, applies filters, performs clustering to detect objects in each time window, and then tracks these objects across windows using the SORT algorithm. Progress will be shown below.")

with st.container(): # Changed from st.echo()
    def step2_process_events():
        if not st.session_state.get('step1_completed', False) or not st.session_state.loop_iterator_object:
            st.warning("Please run Step 1 first to prepare the tools and data iterator.")
            return

        st.info("Running Step 2: Processing events and generating tracks...")

        # Ensure all required objects are in session state from Step 1
        required_attrs = ['loop_iterator_object', 'polarity_filter_alg', 'events_buffer', 'mot_tracker', 'dbscan_object']
        if not all(hasattr(st.session_state, attr) and getattr(st.session_state, attr) is not None for attr in required_attrs):
            st.error("One or more necessary objects from Step 1 are missing or not initialized. Please re-run Step 1.")
            return

        # Local references for convenience
        loop_iterator = st.session_state.loop_iterator_object
        polarity_filter_alg = st.session_state.polarity_filter_alg
        events_buffer = st.session_state.events_buffer
        mot_tracker = st.session_state.mot_tracker
        dbscan = st.session_state.dbscan_object
        
        # Reset data collectors for this run
        st.session_state.events_for_parquet = {} 
        st.session_state.all_tracks_list = []
        st.session_state.time_slices_data = [] # Initialize for this run
        current_max_x_in_run = 0 # To determine plot limits if sensor size unknown
        current_max_y_in_run = 0

        try:
            for evts in loop_iterator: # loop_iterator is already a tqdm object
                current_window_start_ts = -1
                current_window_end_ts = -1 # This will be the timestamp for the slice

                if len(evts) > 0 : # evts is a structured numpy array from EventsIterator
                    current_window_start_ts = evts['t'][0]
                    current_window_end_ts = evts['t'][-1]
                
                polarity_filter_alg.process_events(evts, events_buffer)
                filtered_numpy = events_buffer.numpy() # Structured array: ('x', '<u2'), ('y', '<u2'), ('p', 'u1'), ('t', '<i8')

                # Initialize data structure for this slice (for visualization)
                slice_data_for_viz = {
                    "timestamp_start": current_window_start_ts,
                    "timestamp_end": current_window_end_ts,
                    "raw_events_xy": filtered_numpy[['x', 'y']].copy() if len(filtered_numpy) > 0 else np.empty((0,2), dtype=np.uint16),
                    "cluster_points_xyt": np.empty((0,3), dtype=np.float32), # x,y,t used for DBSCAN
                    "cluster_labels": np.empty(0, dtype=int), # DBSCAN output labels
                    "detections_for_sort": np.empty((0,6)), # x1,y1,x2,y2,conf,uniq_id (original_event_cluster_id)
                    "tracked_objects_slice": np.empty((0,6)) # x1,y1,x2,y2,track_id, original_event_cluster_id (linked)
                }

                if not len(filtered_numpy):
                    # Still update tracker with empty detections if no events in window
                    tracked_objects_output = mot_tracker.update(np.empty((0, 6))) # SORT needs to handle empty input
                    slice_data_for_viz["tracked_objects_slice"] = tracked_objects_output if tracked_objects_output.shape[0] > 0 else np.empty((0,6))
                    st.session_state.time_slices_data.append(slice_data_for_viz) # Append even if empty for scrubbing continuity
                    continue

                # If current_window_end_ts wasn't set from raw evts (e.g. evts was empty but filtered_numpy is not, unlikely but safe)
                if current_window_end_ts == -1 : current_window_end_ts = filtered_numpy[-1]["t"]
                slice_data_for_viz["timestamp_end"] = current_window_end_ts # Ensure it's updated

                # Update max x, y for plotting if sensor size not known
                if st.session_state.sensor_width is None and len(filtered_numpy['x']) > 0:
                    current_max_x_in_run = max(current_max_x_in_run, np.max(filtered_numpy['x']))
                if st.session_state.sensor_height is None and len(filtered_numpy['y']) > 0:
                    current_max_y_in_run = max(current_max_y_in_run, np.max(filtered_numpy['y']))

                # Prepare data for DBSCAN: x, y, t
                data_for_dbscan = np.array(
                    [(evt["x"], evt["y"], evt["t"]) for evt in filtered_numpy],
                    dtype=np.float32, order='C'
                )
                slice_data_for_viz["cluster_points_xyt"] = data_for_dbscan.copy()


                labels = dbscan.fit(data_for_dbscan) # Pass data_for_dbscan
                slice_data_for_viz["cluster_labels"] = labels.copy()
                unique_labels = np.unique(labels)

                processed_detections_for_sort = []
                for label_val in unique_labels:
                    if label_val == -1:  # Skip noise points
                        continue

                    points = data_for_dbscan[labels == label_val] # Original points (x,y,t)
                    
                    # Use a hash for cluster ID (original_event_cluster_id)
                    hash_str = joblib.hash(points)[:15] # Truncate hash
                    original_event_cluster_id = int(hash_str, 16)
                    st.session_state.events_for_parquet[str(original_event_cluster_id)] = points # Store (x,y,t) for saving

                    xy = points[:, :2] # x, y coordinates for bounding box
                    x1, y1 = np.min(xy, axis=0)
                    x2, y2 = np.max(xy, axis=0)
                    confidence = float(points.shape[0]) # Using number of points as confidence

                    if x1 < x2 and y1 < y2: # Ensure non-zero area
                        processed_detections_for_sort.append([x1, y1, x2, y2, confidence, original_event_cluster_id])
                    elif xy.shape[0] > 0: # Handle single point or collinear points by giving a 1x1 box
                        processed_detections_for_sort.append([x1, y1, x1 + 1, y1 + 1, confidence, original_event_cluster_id])
                
                detections_to_sort_np = np.empty((0,6))
                if processed_detections_for_sort:
                    detections_to_sort_np = np.array(processed_detections_for_sort, dtype=np.float64)
                
                slice_data_for_viz["detections_for_sort"] = detections_to_sort_np.copy() # Store for viz

                # Update SORT tracker. It expects [x1,y1,x2,y2,score, (optional other data)]
                # We pass [x1,y1,x2,y2,confidence, original_event_cluster_id]
                # The `Sort.update` output is typically [x1,y1,x2,y2,track_id] or [x1,y1,x2,y2,track_id, original_detection_index]
                tracked_objects_output = mot_tracker.update(detections_to_sort_np)

                # Process tracked_objects_output to ensure it has 6 columns: [x1,y1,x2,y2,track_id, original_event_cluster_id_linked]
                # This is crucial for visualization and consistent saving.
                final_tracked_objects_for_slice_and_saving = []
                if tracked_objects_output.shape[0] > 0:
                    if tracked_objects_output.shape[1] == 6: 
                        # Case 1: SORT returns the 6th column we passed (original_event_cluster_id)
                        # Or, if SORT returns original_detection_index, we use it to look up original_event_cluster_id from detections_to_sort_np
                        # Assuming for now the 6th column *is* the original_event_cluster_id if present.
                        # If it's an index, this logic needs adjustment:
                        # linked_original_ids = detections_to_sort_np[tracked_objects_output[:, 5].astype(int), 5]
                        # temp_tracked = np.hstack((tracked_objects_output[:, :5], linked_original_ids.reshape(-1,1)))
                        final_tracked_objects_for_slice_and_saving = tracked_objects_output.copy()

                    elif tracked_objects_output.shape[1] == 5: # x1,y1,x2,y2,track_id
                        # SORT only returned 5 columns. We add a placeholder for original_event_cluster_id.
                        placeholder_id_col = np.full((tracked_objects_output.shape[0], 1), -1, dtype=np.int64)
                        final_tracked_objects_for_slice_and_saving = np.hstack((tracked_objects_output, placeholder_id_col))
                    else: # Unexpected shape
                        st.error(f"Unexpected SORT output shape: {tracked_objects_output.shape}. Omitting tracks for this window.")
                        final_tracked_objects_for_slice_and_saving = np.empty((0,6))
                else: # No tracks returned
                     final_tracked_objects_for_slice_and_saving = np.empty((0,6))

                slice_data_for_viz["tracked_objects_slice"] = final_tracked_objects_for_slice_and_saving.copy()

                # Prepare for saving to all_tracks_list (Parquet)
                if final_tracked_objects_for_slice_and_saving.shape[0] > 0:
                    timestamp_col = np.full((final_tracked_objects_for_slice_and_saving.shape[0], 1), current_window_end_ts, dtype=np.int64)
                    # Columns: [ts, x1, y1, x2, y2, track_id, original_event_cluster_id]
                    tracks_with_ts_and_orig_id = np.hstack((timestamp_col, final_tracked_objects_for_slice_and_saving))
                    st.session_state.all_tracks_list.append(tracks_with_ts_and_orig_id)
                
                st.session_state.time_slices_data.append(slice_data_for_viz) # Add complete slice data
            
            # After loop, set max_data_x/y if sensor size wasn't available
            if st.session_state.sensor_width is None:
                st.session_state.max_data_x = current_max_x_in_run if current_max_x_in_run > 0 else 640 # Default fallback
            if st.session_state.sensor_height is None:
                st.session_state.max_data_y = current_max_y_in_run if current_max_y_in_run > 0 else 480 # Default fallback

            st.session_state.step2_completed = True
            st.success("Step 2 completed: Event processing and track generation finished.")
            total_segments = sum(len(t) for t in st.session_state.all_tracks_list) if st.session_state.all_tracks_list else 0
            total_clusters = len(st.session_state.events_for_parquet)
            st.info(f"Total accumulated track segments: {total_segments}")
            st.info(f"Total event clusters stored: {total_clusters}")
            st.info(f"Number of time slices captured for visualization: {len(st.session_state.time_slices_data)}")
            if st.session_state.sensor_width is None:
                 st.info(f"Max X observed (for plotting): {st.session_state.max_data_x}")
                 st.info(f"Max Y observed (for plotting): {st.session_state.max_data_y}")


        except Exception as e:
            st.error(f"Error during Step 2 execution: {e}")
            st.text(traceback.format_exc())
            st.session_state.step2_completed = False
        finally:
            if hasattr(loop_iterator, 'close'): # Close tqdm progress bar
                loop_iterator.close()

if st.button("Run Step 2: Process Events & Generate Tracks", key="run_step2", disabled=not st.session_state.get('step1_completed', False)):
    step2_process_events()

# --- Visualization Section ---
st.header("🔬 Time Slice Scrubber & Visualization")



def plot_time_slice_data(slice_data, plot_width, plot_height):
    fig, axs = plt.subplots(1, 2, figsize=(16, 7)) # Two side-by-side plots
    # plt.style.use('seaborn-v0_8-whitegrid') # Optional styling

    # --- Plot 1: Clusters (from DBSCAN points and Detections) ---
    ax1 = axs[0]
    cluster_points_xyt = slice_data["cluster_points_xyt"] # x,y,t used for DBSCAN
    raw_events_xy = slice_data["raw_events_xy"] # All filtered events in window x,y
    dbscan_labels = slice_data["cluster_labels"] # DBSCAN output labels
    detections_for_sort = slice_data["detections_for_sort"] # x1,y1,x2,y2,conf,original_event_cluster_id


    # Plot raw events as a faint background
    if raw_events_xy.shape[0] > 0:
        ax1.scatter(raw_events_xy[:]['x'], raw_events_xy[:]['y'], s=1, color='lightgrey', alpha=0.5, label='Filtered Events')

    if cluster_points_xyt.shape[0] > 0 and dbscan_labels.shape[0] == cluster_points_xyt.shape[0]:
        unique_dbscan_labels = np.unique(dbscan_labels[dbscan_labels != -1])
        
        dbscan_label_colors = {label: color for label, color in zip(unique_dbscan_labels, get_distinct_colors(len(unique_dbscan_labels)))}
        dbscan_label_colors[-1] = '#AAAAAA' # Noise color (gray)

        # Plot points that were part of identified DBSCAN clusters (not noise)
        for label_val in unique_dbscan_labels: # Plot actual clusters first
            points_in_label = cluster_points_xyt[dbscan_labels == label_val]
            ax1.scatter(points_in_label[:, 0], points_in_label[:, 1], 
                        s=10, label=f'DBSCAN Lbl {label_val}', # Small legend for DBSCAN labels
                        color=dbscan_label_colors.get(label_val, '#000000'), alpha=0.8)
        
        # Plot noise points if any
        noise_points = cluster_points_xyt[dbscan_labels == -1]
        if noise_points.shape[0] > 0:
             ax1.scatter(noise_points[:, 0], noise_points[:, 1], s=3, color=dbscan_label_colors[-1], alpha=0.6, label='DBSCAN Noise')


    # Draw bounding boxes for detections (which are derived from clusters)
    # Color these boxes by their original_event_cluster_id
    if detections_for_sort.shape[0] > 0:
        unique_orig_cluster_ids_in_dets = np.unique(detections_for_sort[:, 5].astype(int))
        orig_cluster_id_colors = {uid: color for uid, color in zip(unique_orig_cluster_ids_in_dets, get_distinct_colors(len(unique_orig_cluster_ids_in_dets)))}

        for det in detections_for_sort: # x1,y1,x2,y2,conf,original_event_cluster_id
            x1_d, y1_d, x2_d, y2_d, _, orig_id = det
            rect_color = orig_cluster_id_colors.get(int(orig_id), '#FF0000') # Red if ID not found
            rect = patches.Rectangle((x1_d, y1_d), x2_d - x1_d, y2_d - y1_d, linewidth=1.5, 
                                     edgecolor=rect_color, facecolor='none', alpha=0.9,
                                     label=f'Det (OrigID: {int(orig_id)})' if int(orig_id) in unique_orig_cluster_ids_in_dets[:5] else None) # Limit legend entries
            ax1.add_patch(rect)
            ax1.text(x1_d, y1_d - 3, f"OrigID:{int(orig_id)}", color=rect_color, fontsize=7, va='bottom', ha='left', fontweight='bold')

    ax1.set_title(f"Clusters & Detections at T~{slice_data['timestamp_end'] / 1e6:.3f}s")
    ax1.set_xlabel("X coordinate")
    ax1.set_ylabel("Y coordinate")
    ax1.set_xlim(0, plot_width)
    ax1.set_ylim(plot_height, 0) # Invert Y for typical image coordinates
    ax1.set_aspect('equal', adjustable='box')
    ax1.legend(fontsize='x-small', loc='upper right')


    # --- Plot 2: Tracks (from SORT) ---
    ax2 = axs[1]
    tracked_bbs_slice = slice_data["tracked_objects_slice"] # x1,y1,x2,y2,track_id, original_event_cluster_id_linked

    # Plot raw events as a faint background also on tracks plot for context
    if raw_events_xy.shape[0] > 0:
        ax2.scatter(raw_events_xy[:]['x'], raw_events_xy[:]['y'], s=1, color='lightgrey', alpha=0.5)


    if tracked_bbs_slice.shape[0] > 0:
        unique_track_ids = np.unique(tracked_bbs_slice[:, 4].astype(int)) # 5th column is track_id
        track_colors = {tid: color for tid, color in zip(unique_track_ids, get_distinct_colors(len(unique_track_ids)))}

        for bb in tracked_bbs_slice: # x1,y1,x2,y2,track_id, original_event_cluster_id_linked
            x1_t, y1_t, x2_t, y2_t, track_id, orig_cluster_id_linked = bb
            track_id = int(track_id)
            orig_cluster_id_linked = int(orig_cluster_id_linked)
            
            color = track_colors.get(track_id, '#000000') # Default to black
            rect = patches.Rectangle((x1_t, y1_t), x2_t - x1_t, y2_t - y1_t, linewidth=2, 
                                     edgecolor=color, facecolor=color, alpha=0.3, # Slight fill for visibility
                                     label=f'Track {track_id}' if track_id in unique_track_ids[:5] else None) # Limit legend
            ax2.add_patch(rect)
            
            # Annotate with track_id
            text_y_pos = y1_t - 3
            ax2.text(x1_t, text_y_pos, f"T_ID: {track_id}", color=color, fontsize=8, va='bottom', ha='left', fontweight='bold')
            if orig_cluster_id_linked != -1: # If we have the link to the original cluster ID
                 ax2.text(x1_t + (x2_t-x1_t)/2, y1_t + (y2_t-y1_t)/2, f"ClsID:{orig_cluster_id_linked}", 
                          color=mcolors.rgb_to_hsv(mcolors.to_rgb(color))[:2].tolist() + [0.3], # Darker version of track color for text
                          fontsize=6, ha='center', va='center', alpha=0.9)


    ax2.set_title(f"Active Tracks at T~{slice_data['timestamp_end'] / 1e6:.3f}s (SORT Track IDs)")
    ax2.set_xlabel("X coordinate")
    ax2.set_ylabel("Y coordinate")
    ax2.set_xlim(0, plot_width)
    ax2.set_ylim(plot_height, 0) # Invert Y
    ax2.set_aspect('equal', adjustable='box')
    
    if tracked_bbs_slice.shape[0] > 0 : ax2.legend(fontsize='x-small', loc='upper right')

    plt.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free memory




if st.session_state.get('step2_completed', False) and st.session_state.time_slices_data:
    num_slices = len(st.session_state.time_slices_data)
    st.info(f"Data for {num_slices} time slices available for scrubbing.")

    selected_slice_idx = st.slider(
        "Select Time Slice:",
        min_value=0,
        max_value=num_slices - 1,
        value=0, # Default to the first slice
        step=1,
        key="time_slice_slider"
    )

    if 0 <= selected_slice_idx < num_slices:
        current_slice_data = st.session_state.time_slices_data[selected_slice_idx]
        
        st.write(f"**Displaying Slice: {selected_slice_idx + 1} / {num_slices}**")
        ts_start_s = current_slice_data['timestamp_start'] / 1e6 if current_slice_data['timestamp_start'] != -1 else "N/A"
        ts_end_s = current_slice_data['timestamp_end'] / 1e6 if current_slice_data['timestamp_end'] != -1 else "N/A"

        st.write(f"Timestamp (approx. start of window): {ts_start_s} s")
        st.write(f"Timestamp (approx. end of window): {ts_end_s} s")


        plot_w = st.session_state.sensor_width if st.session_state.sensor_width is not None else st.session_state.max_data_x
        plot_h = st.session_state.sensor_height if st.session_state.sensor_height is not None else st.session_state.max_data_y
        
        if plot_w == 0: plot_w = 640 # Absolute fallback width
        if plot_h == 0: plot_h = 480 # Absolute fallback height

        plot_time_slice_data(current_slice_data, plot_w, plot_h)
        
        with st.expander("Raw data for this displayed slice (for debugging)"):
            # Prepare a version of the slice data that is JSON serializable (converts numpy arrays)
            serializable_slice_data = {}
            for k, v in current_slice_data.items():
                if isinstance(v, np.ndarray):
                    serializable_slice_data[k] = v.tolist()
                elif isinstance(v, (np.int64, np.int32, np.float32, np.float64)): # Handle numpy scalars
                    serializable_slice_data[k] = v.item()
                else:
                    serializable_slice_data[k] = v
            st.json(serializable_slice_data)


elif st.session_state.get('step2_completed', False):
    st.warning("No time slice data was collected during Step 2, or an error occurred (e.g., no events processed). Visualization is not available.")
else:
    st.markdown("Run Step 1 and Step 2 to generate data for visualization.")


# --- Step 3: Save Tracks and Event Clusters to Parquet ---
st.header("Step 3: Save Results")
st.markdown("This final step saves the generated tracks and the raw event data for each identified cluster into compressed Parquet files in your specified output directory. Parquet is an efficient columnar storage format.")

with st.container(): # Changed from st.echo()
    def step3_save_data():
        if not st.session_state.get('step2_completed', False):
            st.warning("Please run Step 2 first to generate tracks and event clusters.")
            return
        
        if not st.session_state.all_tracks_list and not st.session_state.events_for_parquet:
            st.warning("No tracks or event clusters were generated in Step 2. Nothing to save.")
            st.session_state.step3_completed = True # Mark as "complete" as there's nothing to do.
            return

        dt_us = st.session_state.config_dt
        st.info("Running Step 3: Saving data to Parquet files using PyArrow...")
        
        output_dir_path = Path(st.session_state.config_output_dir)
        input_file_stem = Path(st.session_state.config_input_file).stem

        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        output_tracks_file = output_dir_path / f"{input_file_stem}_{dt_us}_tracks.parquet"
        output_event_clusters_file = output_dir_path / f"{input_file_stem}_{dt_us}_event_clusters.parquet"
        
        st.write(f"Target tracks file: `{output_tracks_file}`")
        st.write(f"Target event clusters file: `{output_event_clusters_file}`")

        try:
            # Save Tracks
            # Schema expects: [ts, x1, y1, x2, y2, track_id, original_event_cluster_id]
            track_fields_types = [
                ("window_end_ts", pa.int64()), ("x1", pa.float64()), ("y1", pa.float64()),
                ("x2", pa.float64()), ("y2", pa.float64()), ("track_id", pa.int64()),
                ("original_event_cluster_id", pa.int64()) # This is the linked one
            ]
            tracks_schema = pa.schema(track_fields_types)

            if st.session_state.all_tracks_list:
                # all_tracks_list contains arrays, each of shape (N_tracks_in_window, 7)
                # The 7 columns are: timestamp_col, x1,y1,x2,y2,track_id, original_event_cluster_id_linked
                final_tracks_array = np.concatenate(st.session_state.all_tracks_list, axis=0)
                
                pa_arrays_for_tracks = []
                # Iterate using schema fields to ensure order and type
                for i, field in enumerate(tracks_schema): 
                    col_data = final_tracks_array[:, i]
                    # Ensure column data matches the schema type (e.g., cast if necessary)
                    if pa.types.is_integer(field.type) and not np.issubdtype(col_data.dtype, np.integer):
                        col_data = col_data.astype(np.int64)
                    elif pa.types.is_floating(field.type) and not np.issubdtype(col_data.dtype, np.floating):
                        col_data = col_data.astype(np.float64)
                    pa_arrays_for_tracks.append(pa.array(col_data, type=field.type))
                
                tracks_table = pa.Table.from_arrays(pa_arrays_for_tracks, schema=tracks_schema)
                pq.write_table(tracks_table, output_tracks_file, compression='snappy') # Or 'gzip'
                st.write(f"Saved {tracks_table.num_rows} track segments to '{output_tracks_file}'.")
            else:
                empty_arrays = [pa.array([], type=field.type) for field in tracks_schema]
                empty_table = pa.Table.from_arrays(empty_arrays, schema=tracks_schema)
                pq.write_table(empty_table, output_tracks_file, compression='snappy')
                st.write(f"No tracks to save. Empty tracks Parquet file created: '{output_tracks_file}'.")

            # Save Event Clusters using ParquetWriter for potentially lower memory use
            event_cluster_fields_types = [
                ('x', pa.float32()), ('y', pa.float32()), ('t', pa.float32()),
                ('cluster_id', pa.int64()) # This 'cluster_id' is the original_event_cluster_id (uniq_pts_id)
            ]
            event_schema = pa.schema(event_cluster_fields_types)
            
            writer = None
            data_written_for_events = False

            if st.session_state.events_for_parquet:
                st.write(f"Processing {len(st.session_state.events_for_parquet)} event clusters for Parquet saving...")
                # Use a progress bar for saving event clusters if many
                event_cluster_pbar_container = st.empty() # Container for progress bar
                
                processed_count = 0
                total_clusters_to_save = len(st.session_state.events_for_parquet)

                for uniq_id_str, points_array in st.session_state.events_for_parquet.items():
                    if points_array.shape[0] == 0: # Skip empty clusters
                        processed_count +=1
                        event_cluster_pbar_container.progress(processed_count / total_clusters_to_save if total_clusters_to_save > 0 else 0)
                        continue
                    try:
                        # This cluster_id_val is the original_event_cluster_id generated from hashing points
                        cluster_id_val = np.int64(uniq_id_str) 
                    except ValueError:
                        st.warning(f"Could not convert cluster_id '{uniq_id_str}' to int64. Skipping this cluster.")
                        processed_count +=1
                        event_cluster_pbar_container.progress(processed_count / total_clusters_to_save if total_clusters_to_save > 0 else 0)
                        continue 
                    
                    # points_array is (N,3) with x,y,t (already float32 from DBSCAN input prep)
                    x_col = pa.array(points_array[:, 0].astype(np.float32)) # Ensure correct type for schema
                    y_col = pa.array(points_array[:, 1].astype(np.float32))
                    t_col = pa.array(points_array[:, 2].astype(np.float32))
                    # Repeat the original_event_cluster_id for all points in that cluster
                    cluster_id_col_for_events = pa.array(np.full(points_array.shape[0], cluster_id_val, dtype=np.int64))

                    batch_table = pa.Table.from_arrays(
                        [x_col, y_col, t_col, cluster_id_col_for_events], schema=event_schema
                    )

                    if writer is None: # Initialize writer with the schema from the first batch
                        writer = pq.ParquetWriter(output_event_clusters_file, batch_table.schema, compression='snappy')
                    
                    writer.write_table(batch_table)
                    data_written_for_events = True
                    processed_count +=1
                    event_cluster_pbar_container.progress(processed_count / total_clusters_to_save if total_clusters_to_save > 0 else 0)
                
                event_cluster_pbar_container.empty() # Remove progress bar after completion

                if writer:
                    writer.close()
                    st.write(f"Saved event clusters to '{output_event_clusters_file}'.")
                elif not data_written_for_events and st.session_state.events_for_parquet: # All were empty/skipped
                    st.write("All event clusters were empty or skipped. Creating empty event clusters file.")
                    empty_event_table = pa.Table.from_arrays([pa.array([], type=field.type) for field in event_schema], schema=event_schema)
                    pq.write_table(empty_event_table, output_event_clusters_file, compression='snappy')

            # This block handles if st.session_state.events_for_parquet was initially empty
            # OR if all clusters were skipped and writer was never initialized.
            if not data_written_for_events: 
                if not st.session_state.events_for_parquet: # No clusters in session_state at all
                    st.write("No event clusters were generated to save.")
                # Ensure an empty file with schema is created if no writer was ever initialized
                if writer is None: 
                    empty_event_table = pa.Table.from_arrays([pa.array([], type=field.type) for field in event_schema], schema=event_schema)
                    pq.write_table(empty_event_table, output_event_clusters_file, compression='snappy')
                    st.write(f"Empty event clusters Parquet file created: '{output_event_clusters_file}'.")

            st.success(f"Step 3 completed: Parquet saving finished.")
            st.markdown(f"**Tracks saved to:** `{output_tracks_file}`")
            st.markdown(f"**Event clusters saved to:** `{output_event_clusters_file}`")
            st.session_state.step3_completed = True

        except Exception as e:
            st.error(f"Error during Step 3 (Parquet Saving): {e}")
            st.text(traceback.format_exc())
            st.session_state.step3_completed = False

if st.button("Run Step 3: Save Data to Parquet Files", key="run_step3", disabled=not st.session_state.get('step2_completed', False)):
    step3_save_data()


# --- Optional: Display Code Details ---
with st.expander("Show Code Block Details (for reference)"):
    st.markdown("""
    The core logic is divided into functions corresponding to each step.
    The parameters you set in the sidebar are used by these functions.
    Session state (`st.session_state`) is used to pass data and objects between steps.
    """)
    st.markdown("#### Step 1: `step1_prepare_processing()`")
    st.code("""
def step1_prepare_processing():
    # ... (Uses st.session_state.config_... values)
    # Initializes PolarityFilterAlgorithm, DBScan, Sort
    # Creates EventsIterator and wraps with tqdm
    # Tries to get sensor dimensions for plotting
    # Stores these objects in st.session_state
    pass
    """, language="python")

    st.markdown("#### Step 2: `step2_process_events()`")
    st.code("""
def step2_process_events():
    # Retrieves objects from st.session_state
    # Loops through event windows:
    #   - Filters events
    #   - Runs DBSCAN on filtered_numpy['x'], ['y'], ['t']
    #   - Creates bounding box detections from clusters (assigns original_event_cluster_id)
    #   - Updates SORT tracker with detections
    #   - Collects track data (linking to original_event_cluster_id if possible) and event cluster points for saving
    #   - **NEW**: Collects detailed data for each time slice:
    #       - `raw_events_xy`: (x,y) of all filtered events in window
    #       - `cluster_points_xyt`: (x,y,t) of points fed to DBSCAN
    #       - `cluster_labels`: output labels from DBSCAN
    #       - `detections_for_sort`: BBoxes [x1,y1,x2,y2,conf,original_event_cluster_id]
    #       - `tracked_objects_slice`: Tracked BBoxes [x1,y1,x2,y2,track_id, linked_original_event_cluster_id]
    #     into `st.session_state.time_slices_data` for visualization.
    # Stores collected data in st.session_state for saving and visualization
    pass
    """, language="python")

    st.markdown("#### Visualization Section: `plot_time_slice_data()`")
    st.code("""
def plot_time_slice_data(slice_data, plot_width, plot_height):
    # Uses matplotlib to create two side-by-side plots for a given time slice:
    # 1. Cluster & Detections Plot (Left):
    #    - Scatter plot of raw filtered events (faint background).
    #    - Scatter plot of points used in DBSCAN, colored by DBSCAN cluster label (noise is gray).
    #    - Overlays bounding boxes of detections (from clusters), colored by their 'original_event_cluster_id' and annotated.
    # 2. Track Plot (Right):
    #    - Scatter plot of raw filtered events (faint background).
    #    - Draws bounding boxes of active tracks from SORT.
    #    - Colors tracks by their 'track_id'.
    #    - Annotates with 'track_id' and, if available, the associated 'original_event_cluster_id'.
    # Plots are scaled using sensor dimensions or max observed data dimensions. Y-axis is inverted.
    pass
    """, language="python")

    st.markdown("#### Step 3: `step3_save_data()`")
    st.code("""
def step3_save_data():
    # Retrieves collected tracks and event clusters from st.session_state
    # Concatenates track data. Tracks Parquet includes:
    #   [window_end_ts, x1, y1, x2, y2, track_id, original_event_cluster_id (linked from SORT)]
    # Event Clusters Parquet includes points (x,y,t) associated with their 'original_event_cluster_id'.
    # Uses PyArrow to write tracks and event clusters to Parquet files.
    pass
    """, language="python")

st.markdown("---")
st.markdown("Application execution complete or awaiting next step.")
