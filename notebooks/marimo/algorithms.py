import marimo

__generated_with = "0.11.18"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import pandas as pd
    import sys
    from pathlib import Path
    from tqdm import tqdm
    from loguru import logger
    import altair as alt
    return Path, alt, logger, mo, np, pd, pl, sys, tqdm


@app.cell
def _(Path, mo):
    browser = mo.ui.file_browser(
        initial_path=Path("/mnt/d/rich/hyper-scope/data/interim/"),
        multiple=True,
    )

    browser
    return (browser,)


@app.cell
def _(browser, mo, np, pl):
    dfs = []
    if not mo.running_in_notebook():
        _pth = "/mnt/d/rich/hyper-scope/data/interim/optical_flow_worm_mod001hz_liquid_full.npy"
    elif browser.value:
        dfs = [pl.from_numpy(np.load(f.path)) for f in browser.value]
    return (dfs,)


@app.cell
def _(dfs, np, pl):
    means = []
    frequencies = [0, 10, 100, 500, 1000, 1500]
    for df in dfs:
        # Sort the dataframe by id and time
        df = df.sort(["id", "t"])

        # Define the time interval in microseconds
        interval = 25000

        # Group by id to process each object separately
        result = []

        for obj_id in df["id"].unique():
            # Get data for this object
            obj_df = df.filter(pl.col("id") == obj_id)

            # Calculate time bins (25000 microsecond intervals)
            obj_df = obj_df.with_columns(time_bin=pl.col("t") // interval)

            # Group by time bin and get the first position in each bin
            binned_positions = (
                obj_df.group_by("time_bin")
                .agg(
                    pl.col("center_x").first().alias("x"),
                    pl.col("center_y").first().alias("y"),
                )
                .sort("time_bin")
            )

            # Calculate Euclidean distances between consecutive positions
            if len(binned_positions) > 1:
                x_values = binned_positions["x"].to_numpy()
                y_values = binned_positions["y"].to_numpy()

                # Calculate consecutive differences
                x_diffs = np.diff(x_values)
                y_diffs = np.diff(y_values)

                # Calculate Euclidean distances
                distances = np.sqrt(x_diffs**2 + y_diffs**2)

                # Calculate average distance for this object
                avg_distance = np.mean(distances) if len(distances) > 0 else 0
                result.append(avg_distance)

        # Calculate the overall average across all objects
        overall_avg = np.mean(result) if result else 0
        means.append(overall_avg)
    return (
        avg_distance,
        binned_positions,
        df,
        distances,
        frequencies,
        interval,
        means,
        obj_df,
        obj_id,
        overall_avg,
        result,
        x_diffs,
        x_values,
        y_diffs,
        y_values,
    )


@app.cell
def _(alt, frequencies, means, mo, np, pl):
    results = pl.DataFrame({'frequency': frequencies, 'displacement': np.array(means) * 15/10})
    results.head()
    mo.ui.altair_chart(
    alt.Chart(results).mark_bar().encode(
        alt.X('frequency:O', title='Modulation Frequency (Hz)'),
        alt.Y('displacement', title='Displacement (μm)')
    ))
    return (results,)


@app.cell
def _(dfs, frequencies, np, pl):
    resampled = []

    def _():
        # Resample with anti-aliasing
        from scipy.interpolate import interp1d

        def handle_duplicates(t, x, y):
            """Remove duplicate timestamps that cause divide-by-zero errors"""
            # Get unique time indices (keeping the first occurrence)
            _, unique_indices = np.unique(t, return_index=True)
            unique_indices = np.sort(unique_indices)  # Sort to preserve original order

            # Return deduplicated arrays
            return t[unique_indices], x[unique_indices], y[unique_indices]

        def resample_with_gaps(x, y, t, target_dt, max_gap=0.001):
            """Robust resampling handling gaps and interpolation errors"""
            # Find large gaps
            gaps = np.where(np.diff(t) > max_gap)[0]
            print(f"Found {len(gaps)} gaps larger than {max_gap}s")

            # Split data at gaps
            split_indices = np.append(np.append(0, gaps + 1), len(t))

            x_resampled_list = []
            y_resampled_list = []
            t_resampled_list = []

            for i in range(len(split_indices) - 1):
                start_idx = split_indices[i]
                end_idx = split_indices[i + 1]

                # Get segment
                t_seg = t[start_idx:end_idx]
                x_seg = x[start_idx:end_idx]
                y_seg = y[start_idx:end_idx]

                # Skip if too few points
                if len(t_seg) < 2:
                    continue

                # Remove duplicate timestamps
                t_seg, x_seg, y_seg = handle_duplicates(t_seg, x_seg, y_seg)

                # Skip if insufficient unique points
                if len(t_seg) < 2:
                    continue

                # Create uniform time points
                t_uniform = np.arange(t_seg[0], t_seg[-1], target_dt)

                # Skip if resulting array is too small
                if len(t_uniform) < 2:
                    continue

                try:
                    # Linear interpolation with fallback values
                    x_interp = interp1d(
                        t_seg,
                        x_seg,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(x_seg[0], x_seg[-1]),
                    )
                    y_interp = interp1d(
                        t_seg,
                        y_seg,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(y_seg[0], y_seg[-1]),
                    )

                    # Resample using interpolation
                    x_resampled = x_interp(t_uniform)
                    y_resampled = y_interp(t_uniform)

                    # Check for invalid values (NaN or inf)
                    invalid_mask = (
                        np.isnan(x_resampled)
                        | np.isnan(y_resampled)
                        | np.isinf(x_resampled)
                        | np.isinf(y_resampled)
                    )

                    if np.any(invalid_mask):
                        # Filter out invalid values
                        valid_mask = ~invalid_mask
                        x_resampled_list.append(x_resampled[valid_mask])
                        y_resampled_list.append(y_resampled[valid_mask])
                        t_resampled_list.append(t_uniform[valid_mask])
                    else:
                        x_resampled_list.append(x_resampled)
                        y_resampled_list.append(y_resampled)
                        t_resampled_list.append(t_uniform)

                except Exception as e:
                    print(f"Interpolation error in segment {i}: {e}")

                    # Try fallback to nearest neighbor interpolation
                    try:
                        x_interp = interp1d(
                            t_seg, x_seg, kind="nearest", bounds_error=False
                        )
                        y_interp = interp1d(
                            t_seg, y_seg, kind="nearest", bounds_error=False
                        )

                        x_resampled = x_interp(t_uniform)
                        y_resampled = y_interp(t_uniform)

                        # Filter any remaining NaN/inf values
                        valid_mask = ~(
                            np.isnan(x_resampled)
                            | np.isnan(y_resampled)
                            | np.isinf(x_resampled)
                            | np.isinf(y_resampled)
                        )

                        x_resampled_list.append(x_resampled[valid_mask])
                        y_resampled_list.append(y_resampled[valid_mask])
                        t_resampled_list.append(t_uniform[valid_mask])
                    except:
                        # Skip segment if all interpolation attempts fail
                        print(f"Skipping segment {i} - all interpolation methods failed")
                        continue

            # Concatenate results
            if x_resampled_list:
                x_resampled = np.concatenate(x_resampled_list)
                y_resampled = np.concatenate(y_resampled_list)
                t_resampled = np.concatenate(t_resampled_list)
                return x_resampled, y_resampled, t_resampled
            else:
                print("No valid segments to resample")
                return np.array([]), np.array([]), np.array([])




        # Convert microseconds to seconds for easier interpretation
        for df, f in zip(dfs, frequencies):
            t_seconds = df["t"].to_numpy() / 1e6  # Convert to seconds
        
            # Choose appropriate new sampling rate
            # Using mean interval as base (1 microsecond is too fast)
            target_dt = 200e-6  # 200 microseconds (5kHz)
            # This is slightly larger than your mean interval
            # to ensure stable sampling
        
            # Create uniform time grid
            t_new = np.arange(t_seconds.min(), t_seconds.max(), target_dt)


            # Apply resampling
            x = df["center_x"].to_numpy()
            y = df["center_y"].to_numpy()
            x_resampled, y_resampled, t_resampled = resample_with_gaps(
                x, y, t_seconds, target_dt=200e-6, max_gap=0.001
            )
        
            # Create new dataframe
            df_resampled = pl.DataFrame(
                {
                    "t": t_resampled * 1e6,  # Convert back to microseconds
                    "center_x": x_resampled,
                    "center_y": y_resampled,
                }
            )
        
            # Verify new sampling
            dt_new = np.diff(df_resampled["t"])
            print(f"New sampling stats:")
            print(f"Mean interval: {dt_new.mean():.3f} microseconds")
            print(f"Std interval: {dt_new.std():.3f} microseconds")

            resampled.append(df_resampled)


    _()
    return (resampled,)


@app.cell
def _(alt, frequencies, np, pl, resampled):
    from scipy.fft import fft, fftfreq
    import matplotlib.pyplot as plt


    target_dt = 2500
    charts = []
    for df_resampled, _f in zip(resampled, frequencies):
        # Create complex representation of motion
        complex_motion = (
            df_resampled["center_x"].to_numpy() + 1j * df_resampled["center_y"].to_numpy()
        ).astype(np.complex128)
    
        # Perform FFT
        N = len(complex_motion)
        fourier_motion = fft(complex_motion)
    
        # Calculate frequencies (correctly using sampling period)
        freqs = fftfreq(N, d=target_dt)  # d is the sampling period
    
        # Filter positive frequencies only
        positive_mask = (freqs > 0) & (freqs < 25)
        pos_freqs = freqs[positive_mask]
        pos_fourier = fourier_motion[positive_mask]
    
        # Create results DataFrame
        fft_results = pl.DataFrame(
            {
                "frequency": pos_freqs,
                "amplitude_real": np.real(pos_fourier),
                "amplitude_imag": np.imag(pos_fourier),
                "magnitude": np.abs(pos_fourier),
                "phase": np.angle(pos_fourier),
            }
        )
    
        # # Create magnitude and phase plots
        magnitude_chart = (
            alt.Chart(fft_results)
            .mark_line()
            .encode(
                x=alt.X("frequency", title="Frequency of Motion (Hz)"),
                y=alt.Y("magnitude", title="Magnitude", scale=alt.Scale(type="log")),
            )
            .properties(title=f"FFT Magnitude Spectrum at {_f} Hz Modulation", width=600, height=300)
        )
    
        phase_chart = (
            alt.Chart(fft_results)
            .mark_line()
            .encode(
                x=alt.X("frequency", title="Frequency of (Hz)"),
                y=alt.Y("phase", title="Phase (radians)"),
            )
            .properties(title=f"FFT Phase Spectrum at {_f} Hz Modulation", width=600, height=300)
        )
    
        # # Display the plots
        charts.append(alt.vconcat(magnitude_chart, phase_chart))

    # Arrange in a grid (adjust columns as needed)
    num_cols = 3
    rows = []
    for i in range(0, len(charts), num_cols):
        row_charts = charts[i:i+num_cols]
        rows.append(alt.hconcat(*row_charts))

    alt.vconcat(*rows)
    return (
        N,
        charts,
        complex_motion,
        df_resampled,
        fft,
        fft_results,
        fftfreq,
        fourier_motion,
        freqs,
        i,
        magnitude_chart,
        num_cols,
        phase_chart,
        plt,
        pos_fourier,
        pos_freqs,
        positive_mask,
        row_charts,
        rows,
        target_dt,
    )


if __name__ == "__main__":
    app.run()
