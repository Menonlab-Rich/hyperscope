import streamlit as st

st.set_page_config(page_title="Tracking Data Analysis", page_icon='🔬', layout="wide")

# --- Introduction ---
st.title("🔬 Analysis of Object Tracking Data")
st.markdown('''
**Date: 05/20/2025**

**Author: Rich Baird**
''')
st.markdown('---')
st.header("Analysis Pipeline")
st.markdown("""
**The analysis pipeline includes:**
1.  **Data Loading and Preprocessing:** Importing raw data and calculating fundamental properties like object centroids and displacements.
2.  **Calculating Speeds:** Determining instantaneous speeds for each movement step.
3.  **Overall Movement Characteristics:** Summarizing the general speed of tracked objects.
4.  **Temporal Dynamics:** A deep dive into the time intervals between consecutive detections within individual tracks.
5.  **Analysis Summary:** Key takeaways from the processed data.
""")

status = st.status("Initializing analysis...", expanded=True)
status.write('Loading required libraries...')

# These are imported after the streamlit setup
# So we can provide accurate status messages.
import polars as pl
import altair as alt
import pandas as pd # Retained for st.table convenience with list-of-tuples data
from hyperscope import config # Assuming this is a user-defined module for path config
import numpy as np

# --- Section 1: Data Loading and Initial Processing ---
st.header("1. Data Loading and Preprocessing")
status.write('Loading and preprocessing data...')
st.markdown("""
The first step is to load the raw tracking data. This data, typically from a `.parquet` file,
contains information like object coordinates (`x1`, `y1`, `x2`, `y2`) and timestamps (`window_end_ts`)
for each detected object, grouped by a `track_id`.

The raw data is then processed to:
- **Sort Data:** Ensure observations within each track are in chronological order.
- **Calculate Centroids:** Determine the center point (cx, cy) of each detected object in pixels.
- **Convert Units:** Convert pixel coordinates to physical units (micrometers, µm) using known pixel pitch and magnification.
- **Calculate Displacements:** For each track, find the change in position (`dx_um`, `dy_um`) and the actual time elapsed (`actual_dt_us`) between consecutive detections.
- **Filter Invalid Steps:** Remove any steps where displacement or time delta couldn't be reliably calculated (e.g., the first point in a track, or steps with non-positive time differences).

The code block below performs these operations. You can expand it to see the details.
""")

# This ds_for_metrics DataFrame will be populated by the calculation block
ds_for_metrics: pl.DataFrame = None
avg_speeds_df_final: pl.DataFrame = None
avg_speed_pps_value: float = None
avg_speed_umps_value: float = None

with st.expander("Show Data Loading and Preprocessing Code", expanded=False):
    with st.echo(code_location="below"):
        # Configuration for data path (ensure this path is correct for your environment)
        # Example: data_path = "path/to/your/combo_nomod_tracks.parquet"
        # Using hyperscope config as in the original script
        try:
            data_path = config.INTERIM_DATA_DIR / 'worms' / 'combo_nomod_tracks.parquet'
            st.caption(f"Attempting to load data from: `{data_path}`")
            ds_initial = pl.read_parquet(data_path)
        except Exception as e:
            st.error(f"Fatal Error: Could not load Parquet file. Please check the path and file integrity: {e}")
            st.error("Please ensure `hyperscope.config.INTERIM_DATA_DIR` is correctly configured or replace `data_path` with the direct path to your Parquet file.")
            status.update(state="error", expanded=True, label="Data Loading Failed!")
            st.stop()


        if ds_initial.is_empty():
            st.warning("The Parquet file is empty. No data to analyze.")
            status.update(state="error", expanded=False, label="Empty Data File")
            st.stop()

        st.markdown("#### Sample of Initial Loaded Data (First 5 rows)")
        st.dataframe(ds_initial.head(), use_container_width=True)

        ds = ds_initial.sort(['track_id', 'window_end_ts'])
        st.markdown("#### Sample of Sorted Data (First 5 rows)")
        st.dataframe(ds.head(), use_container_width=True)

        # Constants for unit conversion
        px_pitch_um = 15  # Pixel pitch of the camera sensor in micrometers
        magnification = 10      # Magnification of the optical system
        pixel_size_at_object_um = px_pitch_um / magnification
        st.write(f"**Microscope Setup:** Pixel Pitch = {px_pitch_um} µm, Magnification = {magnification}x")
        st.write(f"Calculated effective pixel size at object plane: **{pixel_size_at_object_um:.2f} µm/pixel**")


        # Calculate centroids in pixels
        ds = ds.with_columns([
            ((pl.col("x1").cast(pl.Float64) + pl.col("x2").cast(pl.Float64)) * 0.5).alias("cx_px"),
            ((pl.col("y1").cast(pl.Float64) + pl.col("y2").cast(pl.Float64)) * 0.5).alias("cy_px")
        ])

        # Convert centroids to micrometers
        ds = ds.with_columns([
            (pl.col("cx_px") * pixel_size_at_object_um).alias('cx_um'),
            (pl.col("cy_px") * pixel_size_at_object_um).alias('cy_um'),
        ])

        # Calculate displacements and time differences (delta t)
        # .over("track_id") ensures calculations are done per track
        ds = ds.with_columns([
            (pl.col("cx_px") - pl.col("cx_px").shift(1).over("track_id")).alias("dx_px"),
            (pl.col("cy_px") - pl.col("cy_px").shift(1).over("track_id")).alias("dy_px"),
            (pl.col("cx_um") - pl.col("cx_um").shift(1).over("track_id")).alias("dx_um"),
            (pl.col("cy_um") - pl.col("cy_um").shift(1).over("track_id")).alias("dy_um"),
            (pl.col("window_end_ts") - pl.col("window_end_ts").shift(1).over("track_id")).alias("actual_dt_us")
        ])

        # Filter out rows where displacements cannot be calculated (e.g., the first row of each track)
        # or where time delta is not positive.
        valid_displacements = ds.filter(
            pl.col("dx_px").is_not_null() &  # Ensure dx_px is calculated (not first point)
            pl.col("actual_dt_us").is_not_null() & # Ensure dt is calculated
            (pl.col("actual_dt_us") > 0) # Ensure time delta is positive
        )

        if valid_displacements.is_empty():
            st.warning("No valid displacements with positive time differences found after initial processing. Cannot calculate velocity or further metrics.")
            status.update(state="warning", expanded=False, label="No Valid Displacements")
            st.stop()

        st.markdown("#### Sample Data After Calculating Displacements (First 5 rows with valid displacements)")
        st.dataframe(valid_displacements.head().select(["track_id", "window_end_ts", "cx_um", "cy_um", "dx_um", "dy_um", "actual_dt_us"]), use_container_width=True)

        # --- Section 2: Calculating Speeds (within the data processing flow) ---
        status.write('Calculating instantaneous speeds...')
        # Convert delta time to seconds for speed calculation
        valid_displacements = valid_displacements.with_columns([
            (pl.col("actual_dt_us") / 1_000_000.0).alias("actual_dt_sec")
        ])

        # Calculate speed in pixels/sec and um/sec
        valid_displacements = valid_displacements.with_columns([
            ((pl.col("dx_px").pow(2) + pl.col("dy_px").pow(2)).sqrt() / pl.col("actual_dt_sec")).alias("speed_px_per_sec"),
            ((pl.col("dx_um").pow(2) + pl.col("dy_um").pow(2)).sqrt() / pl.col("actual_dt_sec")).alias("speed_um_per_sec"),
        ])

        # Filter out non-finite speeds (e.g., from division by zero if any dt_sec were zero, though filtered by actual_dt_us > 0)
        ds_for_metrics = valid_displacements.filter(
            pl.col("speed_px_per_sec").is_finite() &
            pl.col("speed_um_per_sec").is_finite()
        )

        if ds_for_metrics.is_empty():
            st.warning("No finite speeds were calculated. This might indicate issues with time delta calculations or all displacements being zero. Cannot proceed with further metrics.")
            status.update(state="warning", expanded=False, label="No Finite Speeds")
            st.stop()

        st.markdown("#### Sample Data with Calculated Speeds (`ds_for_metrics` - First 5 rows)")
        st.dataframe(ds_for_metrics.head().select(["track_id", "actual_dt_sec", "dx_um", "dy_um", "speed_px_per_sec", "speed_um_per_sec"]), use_container_width=True)
        st.success(f"Successfully processed data: `ds_for_metrics` now contains {ds_for_metrics.height} valid steps with calculated speeds.")

        # Calculate overall average speeds
        avg_speeds_df_final = ds_for_metrics.select([
            pl.mean("speed_px_per_sec").alias("average_speed_pps"),
            pl.mean("speed_um_per_sec").alias("average_speed_umps")
        ])
        avg_speed_pps_value = avg_speeds_df_final.get_column("average_speed_pps")[0]
        avg_speed_umps_value = avg_speeds_df_final.get_column("average_speed_umps")[0]

# Check if ds_for_metrics was successfully created
if ds_for_metrics is None or ds_for_metrics.is_empty():
    st.error("Core data processing failed or resulted in no data. Please check the data source and processing steps in the expander above.")
    status.update(state="error", expanded=False, label="Processing Error")
    st.stop()

# --- Section 2 (Display part): Calculated Speeds & Overall Movement ---
# The calculation is inside the expander, here we discuss and show results.
st.header("2. Calculated Speeds and Overall Movement Characteristics")
st.markdown("""
From the displacements and time intervals, we calculate the instantaneous speed for each step within a track.
This is done in both pixels per second (px/sec) and, more meaningfully, in micrometers per second (µm/sec).
The DataFrame `ds_for_metrics` now contains these speeds and is ready for further analysis.

To get a high-level understanding, we first look at the **overall average speed** across all valid movement steps from all tracks.
This gives a general sense of how fast the tracked objects are moving on average.
""")

col1_avg_overall, col2_avg_overall = st.columns(2)
with col1_avg_overall:
    st.metric(
        label="Overall Average Speed (pixels/sec)",
        value=f"{avg_speed_pps_value:.2f}" if avg_speed_pps_value is not None else "N/A"
    )
with col2_avg_overall:
    st.metric(
        label="Overall Average Speed (µm/sec)",
        value=f"{avg_speed_umps_value:.2f}" if avg_speed_umps_value is not None else "N/A"
    )
st.markdown("---")


# --- Section 3: Temporal Dynamics - Intra-Track Step Durations ---
st.header("3. Temporal Dynamics: Analyzing Intra-Track Step Durations")
status.write('Analyzing intra-track step durations...')
st.markdown("""
The time interval between consecutive detections for the *same* tracked object (referred to as `actual_dt_us` in microseconds)
is a crucial parameter. It can tell us about:
- **Tracking Consistency:** How regularly the system captures object positions.
- **Observation Frequency:** The typical rate at which data points are acquired for each track.
- **System Performance:** Whether the imaging or detection system introduces variable delays.

We will explore its distribution using histograms and then focus on its most common value (the mode).
""")

# --- Subsection 3.1: Distribution of Step Durations (`actual_dt_us`) ---
st.subheader("3.1. Distribution of Intra-Track Step Durations (`actual_dt_us`)")
st.markdown("""
A histogram visually represents the frequency of different step durations.
- The **Linear Scale Histogram** provides an overview of the distribution and helps identify the most populated duration ranges.
- The **Logarithmic Scale Histogram (X-axis)** is particularly useful when the data spans several orders of magnitude. It allows us to see details in ranges (especially shorter durations or the tail of longer durations) that might be compressed and less visible on a linear scale.
""")

if 'ds_for_metrics' in locals() and isinstance(ds_for_metrics, pl.DataFrame) and \
   not ds_for_metrics.is_empty() and 'actual_dt_us' in ds_for_metrics.columns:

    with st.expander("Technical Details: `actual_dt_us` Data for Histograms"):
        st.markdown("#### Debug Info for `actual_dt_us` (Data for Histogram):")
        actual_dt_us_series_for_debug = ds_for_metrics.get_column("actual_dt_us")
        min_val_debug = actual_dt_us_series_for_debug.min()
        max_val_debug = actual_dt_us_series_for_debug.max()
        mean_val_debug = actual_dt_us_series_for_debug.mean()
        median_val_debug = actual_dt_us_series_for_debug.median()
        non_positive_count_debug = ds_for_metrics.filter(pl.col("actual_dt_us") <= 0).height

        debug_stats_list = [
            ("Min `actual_dt_us` (µs)", f"{min_val_debug:.2f}" if min_val_debug is not None else "N/A"),
            ("Max `actual_dt_us` (µs)", f"{max_val_debug:.2f}" if max_val_debug is not None else "N/A"),
            ("Mean `actual_dt_us` (µs)", f"{mean_val_debug:.2f}" if mean_val_debug is not None else "N/A"),
            ("Median `actual_dt_us` (µs)", f"{median_val_debug:.2f}" if median_val_debug is not None else "N/A"),
            ("Count of non-positive (<=0) `actual_dt_us` values", non_positive_count_debug),
            ("Total steps for histogram", ds_for_metrics.height)
        ]
        debug_display_df_pd = pd.DataFrame(debug_stats_list, columns=["Statistic", "Value"]) # Pandas for st.table
        st.table(debug_display_df_pd)

        if non_positive_count_debug > 0:
            st.error(
                f"CRITICAL CHECK: Found {non_positive_count_debug} non-positive 'actual_dt_us' values in the "
                "data intended for the histogram. These should have been filtered out by `(actual_dt_us > 0)` during preprocessing."
            )

    status.write("Generating histogram plots for `actual_dt_us`...")
    actual_dt_us_for_hist_pl = ds_for_metrics.select("actual_dt_us")

    if not actual_dt_us_for_hist_pl.is_empty():
        hist_base_title = "Distribution of Time Deltas Between Consecutive Detections Within Tracks"

        st.markdown("##### Linear Scale Histogram of `actual_dt_us`")
        try:
            chart_linear = alt.Chart(actual_dt_us_for_hist_pl).mark_bar().encode(
                alt.X("actual_dt_us:Q", bin=alt.Bin(maxbins=75), title="Intra-Track Step Duration (µs)"),
                alt.Y("count():Q", title="Frequency (Number of Steps)")
            ).properties(
                title=f"{hist_base_title} (Linear Scale)"
            ).interactive()
            st.altair_chart(chart_linear, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating linear scale Altair chart: {e}")

        st.markdown("##### Logarithmic Scale Histogram of `actual_dt_us` (Manually Binned)")
        st.caption("This histogram uses logarithmically spaced bins to better visualize data spread across orders of magnitude. The X-axis itself is linear but represents these log-spaced bins.")

        positive_dt_us_series = ds_for_metrics.filter(pl.col("actual_dt_us") > 0).get_column("actual_dt_us")

        if not positive_dt_us_series.is_empty():
            min_val = positive_dt_us_series.min()
            max_val = positive_dt_us_series.max()

            if min_val is not None and max_val is not None and min_val > 0:
                data_for_binning_np = positive_dt_us_series.to_numpy()
                num_log_bins = 50
                if min_val == max_val:
                    log_bins = np.array([min_val * 0.9, max_val * 1.1])
                    if log_bins[0] <=0 : log_bins[0] = 1e-9
                else:
                    log_bins = np.logspace(np.log10(min_val), np.log10(max_val), num=num_log_bins + 1)
                    if log_bins[0] <= 0 : log_bins[0]= 1e-9

                counts, bin_edges = np.histogram(data_for_binning_np, bins=log_bins)
                binned_data_for_altair = [{"bin_start": bin_edges[i], "bin_end": bin_edges[i+1], "count": counts[i]} for i in range(len(counts))]
                
                binned_df_pd = pd.DataFrame(binned_data_for_altair)
                # Optional: Filter out bins with very low counts for clarity if needed
                # binned_df_pd = binned_df_pd[binned_df_pd['count'] > ... ]


                if not binned_df_pd.empty:
                    chart_manual_log_bins = alt.Chart(binned_df_pd).mark_bar().encode(
                        alt.X('bin_start:Q', title='Intra-Track Step Duration (µs) - Log Bins'),
                        alt.X2('bin_end:Q'),
                        alt.Y('count:Q', title='Frequency (Number of Steps)')
                    ).properties(
                        title=f"{hist_base_title} (Logarithmically Spaced Bins)"
                    ).interactive()
                    st.altair_chart(chart_manual_log_bins, use_container_width=True)
                else:
                    st.write("No data available for the manual log-binned histogram after processing.")
            else:
                st.info("Cannot create manual log-binned histogram: Min value of `actual_dt_us` is not strictly positive, or max value is missing.")
        else:
            st.info("No strictly positive `actual_dt_us` data available for manual log-binned histogram.")

        st.markdown("#### Summary Statistics for `actual_dt_us` (All Plotted Steps):")
        stats_series_pl = ds_for_metrics.get_column("actual_dt_us")
        stats_data_for_display = [
            ("Total Count of Steps", f"{stats_series_pl.len():,}"),
            ("Min Duration (µs)", f"{stats_series_pl.min():.0f}" if stats_series_pl.min() is not None else "N/A"),
            ("Mean Duration (µs)", f"{stats_series_pl.mean():.0f}" if stats_series_pl.mean() is not None else "N/A"),
            ("Median Duration (µs)", f"{stats_series_pl.median():.0f}" if stats_series_pl.median() is not None else "N/A"),
            ("Max Duration (µs)", f"{stats_series_pl.max():.0f}" if stats_series_pl.max() is not None else "N/A"),
            ("90th Percentile (µs)", f"{stats_series_pl.quantile(0.90, interpolation='linear'):.0f}" if stats_series_pl.quantile(0.90) is not None else "N/A"),
            ("95th Percentile (µs)", f"{stats_series_pl.quantile(0.95, interpolation='linear'):.0f}" if stats_series_pl.quantile(0.95) is not None else "N/A")
        ]
        summary_display_df_pd = pd.DataFrame(stats_data_for_display, columns=["Statistic", "Value"]) # Pandas for st.table
        st.table(summary_display_df_pd)
        st.caption("These statistics describe the overall distribution of time intervals between consecutive measurements within tracks.")

    else:
        st.warning("The `actual_dt_us` column data is empty, cannot generate histograms or summary statistics.")
else:
    st.warning("`ds_for_metrics` or the `actual_dt_us` column was not found or is empty. Ensure previous calculations were successful.")

st.markdown("---")

# --- Subsection 3.2: Focus on the Most Common Step Duration (Mode Analysis) ---
st.subheader("3.2. Analyzing the Most Common (Mode) Step Duration")
status.write('Analyzing mode intra-track step duration...')
st.markdown("""
The **mode** of `actual_dt_us` represents the single most frequently occurring time interval between detections within a track.
Analyzing characteristics of movements that happen at, or very near, this typical interval can provide insights into standard behavior
under common operating conditions of the tracking system.

We define a small tolerance (e.g., ±10 µs) around this mode to capture a representative set of these 'typical' steps.
For this subset of steps, we then calculate:
- The number of steps that fall into this range.
- The average actual time delta (which should be very close to the mode).
- The average displacement magnitude (how far objects typically move during this modal time interval).
""")

mode_exact_dt_us = None
count_steps_in_mode_range = 0
avg_actual_dt_in_mode_range = None
avg_displacement_magnitude_um_in_mode_range = None
tolerance_us = 10

if ds_for_metrics.is_empty() or "actual_dt_us" not in ds_for_metrics.columns:
    st.warning("Not enough data or 'actual_dt_us' column missing for mode analysis.")
else:
    mode_actual_dt_us_series = ds_for_metrics.get_column("actual_dt_us").mode()
    if not mode_actual_dt_us_series.is_empty():
        mode_exact_dt_us = mode_actual_dt_us_series[0]
        st.metric(label="Most Common (Mode) Intra-Track Step Duration (µs)", value=f"{mode_exact_dt_us:.0f}")

        dt_lower_bound = mode_exact_dt_us - tolerance_us
        dt_upper_bound = mode_exact_dt_us + tolerance_us
        
        st.write(f"Analyzing intra-track steps with `actual_dt_us` between **{dt_lower_bound:.0f} µs** and **{dt_upper_bound:.0f} µs** (i.e., mode ± {tolerance_us} µs).")

        if "dx_um" not in ds_for_metrics.columns or "dy_um" not in ds_for_metrics.columns:
            st.error("'dx_um' or 'dy_um' columns are missing. Cannot calculate average displacement for mode range.")
        else:
            steps_in_mode_range_df = ds_for_metrics.filter(
                (pl.col("actual_dt_us") >= dt_lower_bound) & (pl.col("actual_dt_us") <= dt_upper_bound)
            )
            count_steps_in_mode_range = steps_in_mode_range_df.height

            if count_steps_in_mode_range > 0:
                avg_actual_dt_in_mode_range = steps_in_mode_range_df.get_column("actual_dt_us").mean()
                steps_in_mode_range_df = steps_in_mode_range_df.with_columns(
                    (pl.col("dx_um").pow(2) + pl.col("dy_um").pow(2)).sqrt().alias("displacement_magnitude_um")
                )
                avg_displacement_magnitude_um_in_mode_range = steps_in_mode_range_df.get_column("displacement_magnitude_um").mean()
            else:
                st.write(f"No intra-track steps found with `actual_dt_us` in the defined mode range [{dt_lower_bound:.0f} µs - {dt_upper_bound:.0f} µs].")
    else:
        st.write("The mode of intra-track step durations could not be determined (e.g., no unique mode or no data).")

# Display metrics for the mode range analysis
cols_mode = st.columns(3)
with cols_mode[0]:
    st.metric(label="Number of Steps in Mode Range", value=f"{count_steps_in_mode_range:,}" if mode_exact_dt_us is not None else "N/A")
if mode_exact_dt_us is not None and count_steps_in_mode_range > 0:
    with cols_mode[1]:
        st.metric(label="Avg. `actual_dt_us` in Mode Range (µs)", value=f"{avg_actual_dt_in_mode_range:.2f}" if avg_actual_dt_in_mode_range is not None else "N/A")
    with cols_mode[2]:
        st.metric(label="Avg. Displacement in Mode Range (µm)", value=f"{avg_displacement_magnitude_um_in_mode_range:.2f}" if avg_displacement_magnitude_um_in_mode_range is not None else "N/A")
elif mode_exact_dt_us is not None:
     with cols_mode[1]:
        st.caption("No steps in mode range to calculate further averages.")


# --- Section 4: Analysis Summary ---
st.header("4. Analysis Summary")
status.write('Finalizing report...')
st.markdown(f"""
This analysis processed tracking data to reveal key movement characteristics.
- We started by loading raw data for **{ds_initial.height if 'ds_initial' in locals() and not ds_initial.is_empty() else 'N/A'}** initial detections,
  which resulted in **{ds_for_metrics.height if 'ds_for_metrics' in locals() and not ds_for_metrics.is_empty() else 'N/A'}** valid movement steps after preprocessing and speed calculation.
- The overall average speed of tracked objects was found to be approximately **{avg_speed_umps_value:.2f} µm/sec** if avg_speed_umps_value else 'N/A'.
""")

if mode_exact_dt_us is not None:
    st.markdown(f"""
- The most common time interval (`actual_dt_us`) between consecutive detections within a track (the mode) was **{mode_exact_dt_us:.0f} µs**.
  - Within a ±{tolerance_us} µs range around this mode, there were **{count_steps_in_mode_range:,}** steps.
  - For these steps, the average actual time delta was **{avg_actual_dt_in_mode_range:.2f} µs**  and the average displacement was **{avg_displacement_magnitude_um_in_mode_range:.2f} µm**.
""")
else:
    st.markdown("- The mode for `actual_dt_us` could not be determined from the current dataset.")

st.markdown("""
- The distribution of all `actual_dt_us` values (visualized in histograms and summarized by statistics) provides a detailed look at the tracker's temporal performance.

These insights can be valuable for understanding the behavior of the tracked objects and the performance of the tracking system.
""")

status.update(state='complete', expanded=False, label="Analysis Complete!")
st.balloons()
