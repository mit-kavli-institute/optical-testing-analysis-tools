"""
Photon Transfer Curve (PTC) Calculator for PIRT 1280SciCam
Main analysis script for processing lab testing data
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import CubicSpline

from analysis.utils.image_database import (
    get_dark_frames_for_exposure,
    get_light_frames_for_exposure,
)

# Import your utils (assuming they exist)
from analysis.utils.image_stack_processing import remove_outlier_layers

warnings.filterwarnings("ignore")


class PTCAnalyzer:
    """Main class for Photon Transfer Curve analysis"""

    def __init__(
        self, data_dir: str, output_dir: Optional[str] = None, outlier_std: float = 3.0
    ):
        """
        Initialize the PTC Analyzer

        Parameters:
        -----------
        data_dir : str
            Directory containing the raw data
        output_dir : str
            Directory to save the processed data: will be <output_dir>/processed
        outlier_std : float
            Number of standard deviations for outlier rejection
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir
        self.outlier_std = outlier_std
        self.processed_dir = self.output_dir / "processed"

        # Create processed subdirectories
        self.combined_dir = self.processed_dir / "combined"
        self.dark_sub_stack_dir = self.processed_dir / "dark_subtracted_stack"

        for dir_path in [self.combined_dir, self.dark_sub_stack_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def crawl_fits_files(self) -> pd.DataFrame:
        """
        Crawl through directory structure to find all FITS files

        Returns:
        --------
        pd.DataFrame with columns:
            - filename
            - filepath
            - frame_type
            - exposure_time
            - frame_time
            - filter_center
            - filter_bw
            - temperature
            - num_frames
        """
        data_records = []

        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".fits"):
                    filepath = Path(root) / file

                    # Determine frame type from directory structure
                    filepath_str = str(filepath).lower()  # Case-insensitive comparison
                    if "darks" in filepath_str:
                        frame_type = "dark"
                    elif "lights" in filepath_str:
                        frame_type = "light"
                    else:
                        continue  # Skip if not in darks or lights folder

                    # Read FITS header and data
                    with fits.open(filepath) as hdul:
                        header = hdul[0].header
                        data = hdul[0].data

                        # Extract header information
                        exposure_time = header.get("EXPTIME", np.nan)
                        frame_time = header.get("FRMTIME", np.nan)
                        temperature = header.get("TMP_CUR", np.nan)

                        # Parse filter information from FW1FILT, FW2FILT, etc.
                        filter_bw = np.nan
                        filter_center = np.nan
                        for fw_key in ["FW1FILT", "FW2FILT", "FW3FILT", "FW4FILT"]:
                            fw_value = header.get(fw_key, "")
                            if fw_value and fw_value != "EMPTY" and fw_value.strip():
                                # Parse filter string like 'FBH 1050-10'
                                parts = fw_value.split("-")
                                if len(parts) == 2:
                                    filter_bw = float(parts[1])
                                # Extract filter center from FILTER keyword
                                if "FILTER" in header:
                                    try:
                                        filter_center = float(header["FILTER"])
                                    except:
                                        # Try to parse from the FW value
                                        parts = fw_value.split()
                                        if len(parts) >= 2:
                                            filter_center = float(
                                                parts[1].split("-")[0]
                                            )
                                        else:
                                            filter_center = np.nan
                                break

                        # Count number of frames in stack
                        if data.ndim == 3:
                            num_frames = data.shape[0]
                        else:
                            num_frames = 1

                        record = {
                            "filename": file,
                            "filepath": str(filepath),
                            "frame_type": frame_type,
                            "exposure_time": exposure_time,
                            "frame_time": frame_time,
                            "filter_center": filter_center,
                            "filter_bw": filter_bw,
                            "temperature": temperature,
                            "num_frames": num_frames,
                        }

                        data_records.append(record)

        df = pd.DataFrame(data_records)
        return df.sort_values("exposure_time")

    def process_exposure_time(
        self, df: pd.DataFrame, exptime: float, combination_method: str = "mean"
    ) -> Dict:
        """
        Process all frames at a given exposure time

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with file information
        exptime : float
            Exposure time to process
        combination_method : str
            'mean' or 'median' for combining frames

        Returns:
        --------
        dict with processed data paths and arrays
        """

        # Get light and dark frames for this exposure time
        light_frame = get_light_frames_for_exposure(df, exptime)
        dark_frame = get_dark_frames_for_exposure(df, exptime)

        # Enforce that there is only one light and one dark frame
        if light_frame.shape[0] != 1:
            raise ValueError(
                f"Expected one light frame at exposure time {exptime}, but got {light_frame.shape[0]}"
            )
        if dark_frame.shape[0] != 1:
            raise ValueError(
                f"Expected one dark frame at exposure time {exptime}, but got {dark_frame.shape[0]}"
            )

        light_filepath = light_frame["filepath"].to_list()[0]
        dark_filepath = dark_frame["filepath"].to_list()[0]

        light_num_frames = light_frame["num_frames"].to_list()[0]
        dark_num_frames = dark_frame["num_frames"].to_list()[0]

        print(
            f"Light Stack: {light_frame['filename'].to_list()[0]}, n_frames = {light_frame['num_frames'].to_list()[0]}"
        )
        print(
            f"Dark Stack: {dark_frame['filename'].to_list()[0]}, n_frames = {dark_frame['num_frames'].to_list()[0]}"
        )

        # Load the data
        with fits.open(light_filepath) as hdul:
            light_data = hdul[0].data
            light_header = hdul[0].header
        with fits.open(dark_filepath) as hdul:
            dark_data = hdul[0].data
            dark_header = hdul[0].header

        print(f"Light data shape: {light_data.shape}")
        print(f"Dark data shape: {dark_data.shape}")

        # Combine the frames
        if combination_method.lower() in ["mean", "avg"]:
            dark_data_combined = np.nanmean(dark_data, axis=0)
            light_data_combined = np.nanmean(light_data, axis=0)
        elif combination_method.lower() in ["median", "med"]:
            dark_data_combined = np.nanmedian(dark_data, axis=0)
            light_data_combined = np.nanmedian(light_data, axis=0)
        else:
            raise ValueError(
                f"Unknown combination method: {combination_method}, only mean and median supported"
            )

        # Make the dark subtracted light stack
        dark_subtracted_stack = light_data - dark_data_combined
        dark_subtracted_header = light_header.copy()
        dark_subtracted_header["HISTORY"] = (
            f"Dark subtracted, {combination_method} combined"
        )

        ## Save the intermediate data products

        #### Dark-Subtracted Stack ####
        dark_subtracted_stack_filepath = (
            self.dark_sub_stack_dir
            / f"{exptime:.4f}_stack_{light_num_frames}_layers_darksub.fits"
        )
        dark_subtracted_stack_filepath.parent.mkdir(parents=True, exist_ok=True)
        fits.writeto(
            dark_subtracted_stack_filepath,
            dark_subtracted_stack,
            overwrite=True,
            header=dark_subtracted_header,
        )
        print(f"Saved dark subtracted stack to {dark_subtracted_stack_filepath}")

        #### Combined Dark-Subtracted Frame ####
        dark_subtracted_frame = light_data_combined - dark_data_combined
        dark_sub_dir = self.combined_dir / "dark_subtracted"
        dark_sub_dir.mkdir(parents=True, exist_ok=True)
        dark_subtracted_frame_filepath = dark_sub_dir / f"{exptime:.4f}_darksub.fits"
        fits.writeto(
            dark_subtracted_frame_filepath,
            dark_subtracted_frame,
            overwrite=True,
            header=dark_subtracted_header,
        )
        print(f"Saved dark subtracted frame to {dark_subtracted_frame_filepath}")

        ##### Light Frame ####
        light_dir = self.combined_dir / "light"
        light_dir.mkdir(parents=True, exist_ok=True)
        light_data_combined_filepath = light_dir / f"{exptime:.4f}_light.fits"
        light_data_combined_header = light_header.copy()
        light_data_combined_header["HISTORY"] = f"{combination_method} combined"
        fits.writeto(
            light_data_combined_filepath,
            light_data_combined,
            overwrite=True,
            header=light_data_combined_header,
        )
        print(f"Saved light data combined to {light_data_combined_filepath}")

        #### Dark Frame ####
        dark_dir = self.combined_dir / "dark"
        dark_dir.mkdir(parents=True, exist_ok=True)
        dark_data_combined_filepath = dark_dir / f"{exptime:.4f}_dark.fits"
        dark_data_combined_header = dark_header.copy()
        dark_data_combined_header["HISTORY"] = f"{combination_method} combined"
        fits.writeto(
            dark_data_combined_filepath,
            dark_data_combined,
            overwrite=True,
            header=dark_data_combined_header,
        )
        print(f"Saved dark data combined to {dark_data_combined_filepath}")

        # Return results dictionary
        results = {
            "light_stack": light_data,
            "dark_stack": dark_data,
            "light_combined": light_data_combined,
            "dark_combined": dark_data_combined,
            "dark_subtracted_stack": dark_subtracted_stack,
            "dark_subtracted_frame": dark_subtracted_frame,
            "light_combined_path": str(light_data_combined_filepath),
            "dark_combined_path": str(dark_data_combined_filepath),
            "dark_subtracted_stack_path": str(dark_subtracted_stack_filepath),
            "dark_subtracted_frame_path": str(dark_subtracted_frame_filepath),
        }

        return results

    def create_processed_dataframe(self) -> pd.DataFrame:
        """
        Create dataframe of processed dark-subtracted stacks
        """
        processed_records = []

        # Load all dark-subtracted stacks
        for filepath in self.dark_sub_stack_dir.glob("*.fits"):
            # Parse filename to get exposure time
            parts = filepath.name.split("_")
            exptime = float(parts[0])

            with fits.open(filepath) as hdul:
                data = hdul[0].data

                # Skip if data is not 3D
                if data.ndim != 3:
                    print(f"Warning: Skipping {filepath.name} - not a 3D stack")
                    continue

                # Apply outlier rejection
                cleaned_data, inlier_mask = remove_outlier_layers(
                    data, self.outlier_std
                )
                num_frames = data.shape[0]
                num_frames_cleaned = cleaned_data.shape[0]

                # Calculate statistics on cleaned data
                mean_frame = np.nanmean(cleaned_data, axis=0)
                var_frame = np.nanvar(cleaned_data, axis=0)

                record = {
                    "exposure_time": exptime,
                    "filename": filepath.name,
                    "filepath": str(filepath),
                    "num_frames": num_frames,
                    "data": data,
                    "num_frames_cleaned": num_frames_cleaned,
                    "outlier_mask": ~inlier_mask,  # Convert inlier to outlier mask
                    "cleaned_data": cleaned_data,
                    "mean_frame": mean_frame,
                    "var_frame": var_frame,
                }

                processed_records.append(record)

        return pd.DataFrame(processed_records).sort_values("exposure_time")

    def remove_outliers_iterative(
        self,
        df: pd.DataFrame,
        n_iters: int = 3,
        n_std: float = 3.0,
        mean_min: float = 100,
        mean_max: float = 50000,
    ) -> pd.DataFrame:
        """
        Remove outlier stacks based on variance vs mean plot
        """
        df = df.copy()
        df["outlier"] = False

        for iteration in range(n_iters):
            # Get all mean and variance values in range
            all_means = []
            all_vars = []
            indices = []

            for idx, row in df[~df["outlier"]].iterrows():
                mean_frame = row["mean_frame"]
                var_frame = row["var_frame"]

                # Select pixels in range
                mask = (mean_frame > mean_min) & (mean_frame < mean_max)
                all_means.extend(mean_frame[mask])
                all_vars.extend(var_frame[mask])
                indices.extend([idx] * np.sum(mask))

            all_means = np.array(all_means)
            all_vars = np.array(all_vars)
            indices = np.array(indices)

            if len(all_means) == 0:
                break

            # Fit slope
            slope, intercept = np.polyfit(all_means, all_vars, 1)

            # Calculate residuals
            predicted = slope * all_means + intercept
            residuals = all_vars - predicted
            std_residual = np.std(residuals)

            # Mark outliers
            outlier_mask = np.abs(residuals) > n_std * std_residual
            outlier_indices = np.unique(indices[outlier_mask])

            df.loc[outlier_indices, "outlier"] = True

            print(
                f"Iteration {iteration + 1}: Marked {len(outlier_indices)} outlier exposures"
            )

        return df

    def fit_photon_transfer_curve(
        self, df_filtered: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the pixel-wise photon transfer curve
        """
        # Stack all mean and variance frames
        arr_mean = np.stack(df_filtered["mean_frame"].values)
        arr_var = np.stack(df_filtered["var_frame"].values)

        # Calculate pixel-wise slope using centered differences
        x_mean = np.nanmean(arr_mean, axis=0)
        y_mean = np.nanmean(arr_var, axis=0)

        print(f"x_mean.shape: {x_mean.shape}")
        print(f"y_mean.shape: {y_mean.shape}")

        diff_x = arr_mean - x_mean
        diff_y = arr_var - y_mean

        num = np.nansum(diff_x * diff_y, axis=0)
        den = np.nansum(diff_x * diff_x, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            slope_image = num / den
            slope_image[den == 0] = np.nan

        intercept_image = y_mean - slope_image * x_mean
        gain_image = 1.0 / slope_image  # gain = 1/slope

        return slope_image, intercept_image, gain_image

    def plot_gain_histogram(
        self, slope_image: np.ndarray, gain_image: np.ndarray, nominal_gain: float = 4.2
    ):
        """
        Plot histogram of gains with analysis
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create histogram
        slope_flat = np.ravel(slope_image[~np.isnan(slope_image)])
        hist, bins = np.histogram(slope_flat, bins=1000, range=(0, 1))
        bin_centers = (bins[:-1] + bins[1:]) / 2

        ax.semilogy(bin_centers, hist, ".", alpha=0.5, label="Data")
        ax.set_title("Pixel-wise PTC: Slope Histogram (Inverse Gain)")

        # Calculate median gain
        median_gain = np.nanmedian(gain_image[~np.isnan(gain_image)])
        print(f"Median gain = {median_gain:.3f} e/ADU")
        print(f"1/median gain = {1/median_gain:.3f} ADU/e")

        # Fit spline to histogram peak region
        hist_clipcond = hist > 100
        if np.any(hist_clipcond):
            xhist_clip = bin_centers[hist_clipcond]
            yhist_clip = hist[hist_clipcond]

            # Spline fit
            cs = CubicSpline(xhist_clip, yhist_clip)
            xhist_fine = np.linspace(xhist_clip[0], xhist_clip[-1], 1000)
            yhist_fine = cs(xhist_fine)
            ax.plot(xhist_fine, yhist_fine, "r-", label="Spline Fit")

            # Find peak
            peak_idx = np.argmax(yhist_fine)
            peak_x = xhist_fine[peak_idx]
            peak_y = yhist_fine[peak_idx]

            print(f"Peak Slope = {peak_x:.3f} ADU/e")
            print(f"Peak Gain = {1/peak_x:.3f} e/ADU")

            ax.axvline(
                peak_x,
                color="k",
                linestyle="--",
                label=f"Peak: Slope={peak_x:.3f}, Gain={1/peak_x:.3f} e/ADU",
            )

            # Calculate FWHM
            half_max = peak_y / 2
            indices_above_half = np.where(yhist_fine > half_max)[0]
            if len(indices_above_half) > 0:
                left_idx = indices_above_half[0]
                right_idx = indices_above_half[-1]
                fwhm = xhist_fine[right_idx] - xhist_fine[left_idx]
                std_from_fwhm = fwhm / 2.355

                print(f"FWHM = {fwhm:.3f}")
                print(f"Std from FWHM = {std_from_fwhm:.3f}")

                ax.axvline(peak_x - fwhm / 2, color="b", linestyle="--", alpha=0.5)
                ax.axvline(
                    peak_x + fwhm / 2,
                    color="b",
                    linestyle="--",
                    alpha=0.5,
                    label=f"FWHM={fwhm:.3f}",
                )

                # Plot 3-sigma lines
                N = 3
                ax.axvline(
                    peak_x - N * std_from_fwhm, color="m", linestyle="--", alpha=0.5
                )
                ax.axvline(
                    peak_x + N * std_from_fwhm,
                    color="m",
                    linestyle="--",
                    alpha=0.5,
                    label=f"±{N}σ = ±{N*std_from_fwhm:.3f}",
                )

        # Plot nominal gain
        nominal_slope = 1 / nominal_gain
        ax.axvline(
            nominal_slope,
            color="g",
            linestyle="--",
            label=f"Nominal: Slope={nominal_slope:.3f}, Gain={nominal_gain:.1f} e/ADU",
        )

        ax.set_xlabel("Inverse Gain (ADU/e)")
        ax.set_ylabel("Number of pixels")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_variance_vs_mean(self, df: pd.DataFrame, slope: float = None):
        """
        Plot variance vs mean for all pixels
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot each exposure time
        colors = plt.cm.viridis(np.linspace(0, 1, len(df)))

        for idx, (_, row) in enumerate(df.iterrows()):
            if row.get("outlier", False):
                continue

            mean_frame = row["mean_frame"]
            var_frame = row["var_frame"]

            # Sample pixels for plotting
            mask = np.random.rand(*mean_frame.shape) < 0.01  # Plot 1% of pixels
            ax.scatter(
                mean_frame[mask],
                var_frame[mask],
                c=[colors[idx]],
                alpha=0.5,
                s=1,
                label=f"t={row['exposure_time']:.3f}s",
            )

        if slope is not None:
            # Plot fitted line
            x_range = np.array(
                [0, np.nanmax([row["mean_frame"].max() for _, row in df.iterrows()])]
            )
            ax.plot(
                x_range,
                slope * x_range,
                "r--",
                linewidth=2,
                label=f"Fit: slope={slope:.3f}, gain={1/slope:.2f} e/ADU",
            )

        ax.set_xlabel("Mean Signal (ADU)")
        ax.set_ylabel("Variance (ADU²)")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        # Only show legend for first few exposure times
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 10:
            ax.legend(
                handles[:5] + handles[-1:],
                labels[:5] + labels[-1:],
                loc="upper left",
                fontsize=8,
            )
        else:
            ax.legend(loc="upper left", fontsize=8)

        ax.set_title("Photon Transfer Curve: Variance vs Mean")
        plt.tight_layout()
        return fig


def main(
    data_dir: str,
    output_dir: Optional[str] = None,
    outlier_std: float = 3.0,
    n_iters: int = 3,
    mean_min: float = 100,
    mean_max: float = 50000,
):
    """
    Main function to run the complete PTC analysis

    Parameters:
    -----------
    data_dir : str
        Path to the top-level data directory
    output_dir : str, optional
        Directory to save processed data. If None, uses data_dir
    outlier_std : float
        Number of standard deviations for outlier rejection
    n_iters : int
        Number of iterations for outlier rejection
    mean_min : float
        Minimum mean value for fitting
    mean_max : float
        Maximum mean value for fitting
    """

    print("=" * 60)
    print("Photon Transfer Curve Analysis")
    print("=" * 60)

    # Initialize analyzer
    analyzer = PTCAnalyzer(data_dir, output_dir, outlier_std)

    # Step 1-2: Crawl files and build database
    print("\n1. Crawling FITS files...")
    df_raw = analyzer.crawl_fits_files()
    print(f"   Found {len(df_raw)} FITS files")
    print(f"   Exposure times: {df_raw['exposure_time'].unique()}")

    # Step 3-4: Process each exposure time
    print("\n2. Processing exposure times...")
    unique_exptimes = df_raw["exposure_time"].unique()

    for exptime in unique_exptimes:
        if np.isnan(exptime):
            continue
        print(f"   Processing {exptime:.4f}s...")
        analyzer.process_exposure_time(df_raw, exptime)

    # Step 5: Create processed dataframe
    print("\n3. Creating processed dataframe...")
    df_processed = analyzer.create_processed_dataframe()
    print(f"   Processed {len(df_processed)} exposure times")

    # Step 6: Remove outliers
    print("\n4. Removing outliers iteratively...")
    df_filtered = analyzer.remove_outliers_iterative(
        df_processed, n_iters, outlier_std, mean_min, mean_max
    )
    print(f"   {(~df_filtered['outlier']).sum()} good exposures remaining")

    # Step 7: Fit photon transfer curve
    print("\n5. Fitting photon transfer curve...")
    df_good = df_filtered[~df_filtered["outlier"]]

    if len(df_good) < 2:
        print("   ERROR: Not enough good data points for fitting!")
        return None

    slope_image, intercept_image, gain_image = analyzer.fit_photon_transfer_curve(
        df_good
    )

    # Calculate statistics
    median_gain = np.nanmedian(gain_image)
    mean_gain = np.nanmean(gain_image)
    std_gain = np.nanstd(gain_image)

    print(f"\n   RESULTS:")
    print(f"   Median Gain: {median_gain:.3f} e/ADU")
    print(f"   Mean Gain:   {mean_gain:.3f} e/ADU")
    print(f"   Std Gain:    {std_gain:.3f} e/ADU")

    # Step 8: Create plots
    print("\n6. Creating plots...")

    # Variance vs Mean plot
    fig_var = analyzer.plot_variance_vs_mean(df_filtered, np.nanmedian(slope_image))
    plt.savefig(
        analyzer.processed_dir / "variance_vs_mean.png", dpi=150, bbox_inches="tight"
    )

    # Gain histogram
    fig_hist = analyzer.plot_gain_histogram(slope_image, gain_image)
    plt.savefig(
        analyzer.processed_dir / "gain_histogram.png", dpi=150, bbox_inches="tight"
    )

    # Save gain image as FITS
    output_gain = analyzer.processed_dir / "gain_image.fits"
    fits.writeto(output_gain, gain_image, overwrite=True)
    print(f"   Saved gain image to {output_gain}")

    plt.show()

    return {
        "df_raw": df_raw,
        "df_processed": df_processed,
        "df_filtered": df_filtered,
        "slope_image": slope_image,
        "intercept_image": intercept_image,
        "gain_image": gain_image,
        "median_gain": median_gain,
    }


if __name__ == "__main__":
    # Example usage
    data_directory = "/path/to/WINTER - Public Data Workspace/pirt_camera/lab_testing/ramps/20250729_1050"
    results = main(data_directory)
