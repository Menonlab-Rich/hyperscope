import streamlit as st
import polars as pl
import altair as alt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Import the custom rsklearn clustering library
from rsklearn import clustering
# Assuming a local configuration module for the data path
from hyperscope import config

def get_scale_factor(scale_str):
    if scale_str == 'μs':
        return 1
    elif scale_str == 'ms':
        return 1/1e3
    elif scale_str == 's':
        return 1/1e6
    else:
        raise ValueError('Scale string must be one of μs, ms, or s.')

st.set_page_config(page_title="Tracking Data Analysis", page_icon='🔬', layout="wide")

# --- Introduction ---
st.title("🔬 Analysis of Object Tracking Data")
st.markdown('''
**Date: 06/25/2025**

**Author: Rich Baird**
''')
st.markdown('---')
st.header("Analysis Pipeline")
st.markdown("""
**The analysis pipeline includes:**
1.  **Data Loading and Preprocessing:** Importing raw data and calculating fundamental properties like object centroids and displacements.
2.  **Calculating Speeds:** Determining instantaneous speeds for each movement step.
3.  **Temporal Dynamics:** A deep dive into the time intervals between consecutive detections within individual tracks.
4.  **Denoised 3D Trajectory Plot:** Applying a density-based filter to clean up the trajectory data for clear visualization using your `rsklearn` library.
5.  **Analysis Summary:** Key takeaways from the processed data.
""")

status = st.status("Initializing analysis...", expanded=True)
status.write('Loading required libraries...')

# --- Sidebar Controls ---
st.sidebar.title("Configuration")
st.sidebar.header("Data Source")
if 'data_path' not in st.session_state:
    st.session_state.data_path = config.INTERIM_DATA_DIR / 'worms' / 'combo_nomod_tracks.parquet'

data_path = st.sidebar.text_input('Data Path', st.session_state.data_path)
st.session_state.data_path = data_path

st.sidebar.header("Denoising Filter (rsklearn.DBSCAN)")
dbscan_eps = st.sidebar.slider("Epsilon (eps, µm)", 1.0, 50.0, 15.0, 0.5, key="r_eps")
dbscan_min_samples = st.sidebar.slider("Min Samples", 2, 50, 10, 1, key="r_min_samples")

st.sidebar.header("Sampling Duration and Scale")
sample_min_ts = st.sidebar.number_input("Min TS μs", 0.0, 60e6)
sample_max_ts = st.sidebar.number_input("Max TS μs", 0.0, 60e6)
ts_scale = st.sidebar.selectbox("Time Scale", ("μs", "ms", "s"))


if not st.sidebar.button('Run Analysis', type="primary"):
    st.info("Configure settings in the sidebar and click 'Run Analysis' to begin.")
    st.stop()

# --- Section 1: Data Loading and Initial Processing ---
with st.expander("Show Data Loading and Preprocessing", expanded=False):
    status.write('Loading and preprocessing data...')
    try:
        ds_initial = pl.read_parquet(data_path)
    except Exception as e:
        st.error(f"Fatal Error: Could not load Parquet file. Please check the path and file integrity: {e}")
        st.stop()

    if ds_initial.is_empty():
        st.warning("The Parquet file is empty. No data to analyze.")
        st.stop()

    ds = ds_initial.sort(['track_id', 'window_end_ts'])
    px_pitch_um = 15
    magnification = 10
    pixel_size_at_object_um = px_pitch_um / magnification

    ds = ds.with_columns([
        ((pl.col("x1") + pl.col("x2")) * 0.5).alias("cx_px"),
        ((pl.col("y1") + pl.col("y2")) * 0.5).alias("cy_px")
    ]).with_columns([
        (pl.col("cx_px") * pixel_size_at_object_um).alias('cx_um'),
        (pl.col("cy_px") * pixel_size_at_object_um).alias('cy_um'),
    ]).with_columns([
        (pl.col("cx_um") - pl.col("cx_um").shift(1).over("track_id")).alias("dx_um"),
        (pl.col("cy_um") - pl.col("cy_um").shift(1).over("track_id")).alias("dy_um"),
        (pl.col("window_end_ts") - pl.col("window_end_ts").shift(1).over("track_id")).alias("actual_dt_us")
    ])

    valid_displacements = ds.filter(pl.col("dx_um").is_not_null() & (pl.col("actual_dt_us") > 0))

    if valid_displacements.is_empty():
        st.warning("No valid displacements found. Cannot calculate velocity.")
        st.stop()

    ds_for_tracking = ds
    ds_for_metrics = valid_displacements.with_columns(
        (pl.col("actual_dt_us") / 1_000_000.0).alias("actual_dt_sec")
    ).with_columns(
        ((pl.col("dx_um").pow(2) + pl.col("dy_um").pow(2)).sqrt() / pl.col("actual_dt_sec")).alias("speed_um_per_sec")
    ).filter(pl.col("speed_um_per_sec").is_finite())

    st.success(f"Successfully processed data: `ds_for_metrics` contains {ds_for_metrics.height} valid steps.")
    avg_speed_umps_value = ds_for_metrics.select(pl.mean("speed_um_per_sec"))[0, 0]


# --- Section 2 & 3: Metrics ---
st.header("2. Overall Movement Characteristics")
st.metric(label="Overall Average Speed (µm/sec)", value=f"{avg_speed_umps_value:.2f}" if 'avg_speed_umps_value' in locals() else "N/A")
st.header("3. Temporal Dynamics")
with st.expander("Show Temporal Dynamics Plot", expanded=False):
    actual_dt_us_for_hist = ds_for_metrics.select("actual_dt_us")
    chart_linear = alt.Chart(actual_dt_us_for_hist).mark_bar().encode(
        alt.X("actual_dt_us:Q", bin=alt.Bin(maxbins=75), title="Intra-Track Step Duration (µs)"),
        alt.Y("count():Q", title="Frequency")
    ).properties(title="Distribution of Time Deltas").interactive()
    st.altair_chart(chart_linear, use_container_width=True)
st.markdown("---")

# --- Section 4: Denoised 3D Trajectory Plot (Using rsklearn) ---
st.header("4. Denoised 3D Trajectory Plot")
st.markdown("""
To visualize the primary motion and remove spurious noise, we apply a density-based filter using your high-performance **`rsklearn`** library. This algorithm groups closely packed points to identify the worm's main path while marking isolated points as noise.
""")

# --- Sidebar Controls for DBSCAN ---
status.write('Denoising trajectory with rsklearn...')
plot_df = ds_for_tracking.drop_nulls(subset=['cx_um', 'cy_um', 'window_end_ts'])

if not plot_df.is_empty():
    coords = plot_df.select(['cx_um', 'cy_um']).to_numpy().astype(np.float32)

    # --- Use the rsklearn library ---
    # The simple.py example shows the metric parameter, which we include here.
    db = clustering.DBScan(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="euclidean")
    labels = db.fit(coords)
    # --- End of rsklearn usage ---

    plot_df = plot_df.with_columns(pl.Series("label", labels))
    core_labels_df = plot_df.filter(pl.col('label') != -1)

    if not core_labels_df.is_empty():
        largest_cluster_label = core_labels_df['label'].mode()[0]
        filtered_plot_df = plot_df.filter(pl.col('label') == largest_cluster_label)
        
        original_points = len(plot_df)
        filtered_points = len(filtered_plot_df)
        removed_points = original_points - filtered_points
        
        st.metric(
            "Data Points on Main Path (After Filtering)",
            f"{filtered_points:,}",
            f"{removed_points:,} noise points removed ({removed_points/original_points:.1%})"
        )
    else:
        st.warning("rsklearn.DBSCAN did not find any core clusters. Showing unfiltered data.")
        filtered_plot_df = plot_df

else:
    filtered_plot_df = pl.DataFrame()

if not filtered_plot_df.is_empty():
    st.markdown("### Filtered Trajectory")
    if sample_max_ts == 0:
        sample_max_ts = filtered_plot_df['window_end_ts'].max()
    filtered_plot_df = filtered_plot_df.filter((pl.col('window_end_ts') < sample_max_ts) & (pl.col('window_end_ts') > sample_min_ts))

    x_data = filtered_plot_df['cx_um'].to_numpy()
    y_data = filtered_plot_df['cy_um'].to_numpy()
    ts_series = filtered_plot_df['window_end_ts'] - filtered_plot_df['window_end_ts'].min()
    z_data_seconds = ts_series.to_numpy() * get_scale_factor(ts_scale)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    scatter = ax.scatter(x_data, y_data, z_data_seconds, c=z_data_seconds, cmap='viridis', s=1.0)
    
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_zlabel(f"Time ({ts_scale})")
    ax.set_title("Denoised Worm Trajectory Over Time (via rsklearn)")
    
    cbar = fig.colorbar(scatter, shrink=0.5, aspect=10)
    cbar.set_label('Time (seconds)')
    
    fig.tight_layout()
    st.pyplot(fig)
else:
    st.info("No data to display in the 3D plot.")

st.markdown("---")

# --- Section 5: Analysis Summary ---
st.header("5. Analysis Summary")
status.write('Finalizing report...')
st.markdown(f"""
- The overall average speed was **{avg_speed_umps_value:.2f} µm/sec**.
- A density-based filter from the `rsklearn` library was used to isolate the primary path of the worm for clear visualization.
""")

status.update(state='complete', expanded=False, label="Analysis Complete!")
st.balloons()
