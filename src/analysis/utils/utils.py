import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, SqrtStretch, ZScaleInterval


def plot_image(
    data,
    thresh=3.0,
    ax=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    norm=None,
    return_norm=False,
    title=None,
):
    """
    Plot an image with plt.imshow, auto-thresholded via sigma_clipped_stats,
    and optionally restrict to a window defined by (x_min:x_max, y_min:y_max).

    Parameters
    ----------
    data : 2D np.ndarray
        The input image to plot.
    thresh : float, optional
        The sigma threshold to use in sigma_clipped_stats. Default is 3.0.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If None, a new figure and axes will be created.
    x_min : int, optional
        Minimum x index (column) for the window. If None, defaults to 0.
    x_max : int, optional
        Maximum x index (column) for the window (non-inclusive). If None, defaults to data.shape[1].
    y_min : int, optional
        Minimum y index (row) for the window. If None, defaults to 0.
    y_max : int, optional
        Maximum y index (row) for the window (non-inclusive). If None, defaults to data.shape[0].
    norm : matplotlib.colors.Normalize, optional
        A normalization object to pass to imshow. If None, defaults to a linear normalization.
    return_norm : bool, optional
        If True, return the normalization object. Default is False
    title : str, optional
        Title for the plot. Default is None.
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the image was plotted.
    norm : astropy.visualization.ImageNormalize, optional
        The normalization object used for the plot. Only returned if return_norm is True
    """
    # Create ax if not provided
    if ax is None:
        fig, ax = plt.subplots()

    # Handle default window boundaries
    if x_min is None:
        x_min = 0
    if x_max is None:
        x_max = data.shape[1]
    if y_min is None:
        y_min = 0
    if y_max is None:
        y_max = data.shape[0]

    # Slice the data to the desired window
    windowed_data = data[y_min:y_max, x_min:x_max]

    # Compute stats on the windowed region
    _, med, std = sigma_clipped_stats(windowed_data, sigma=thresh)

    # Create a normalization object if not provided
    if norm is None:
        # Plot
        im = ax.imshow(
            windowed_data,
            vmin=med - 3 * std,
            vmax=med + 3 * std,
            cmap="gray",
            origin="lower",
        )
    else:
        if norm == "zscale":
            norm = ImageNormalize(
                windowed_data,
                interval=ZScaleInterval(),
                stretch=SqrtStretch(),
            )
        elif norm == "minmax":
            norm = ImageNormalize(
                windowed_data,
                vmin=np.nanmin(windowed_data),
                vmax=np.nanmax(windowed_data),
            )
        # Plot
        im = ax.imshow(
            windowed_data,
            cmap="gray",
            origin="lower",
            norm=norm,
        )

    cbar = plt.colorbar(im, ax=ax)

    # Set the title if provided
    if title is not None:
        ax.set_title(title)

    # Return the normalization object if requested
    if return_norm:
        return ax, norm
    else:
        return ax


def write_image(image, filename, header=None, overwrite=True):
    """
    Write an image to a fits file
    :param image:
    :param filename:
    :param header:
    :param overwrite:
    :return:
    """
    # make parent directories if they don't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    hdu = fits.PrimaryHDU(image, header=header)
    hdu.writeto(filename, overwrite=overwrite)


# Plotting method for histogram
def histogram_peak_in_band(
    data: np.ndarray,
    mask: np.ndarray | None = None,
    bins: int = 100,
    freq_thresh: int = 0,
    value_min: float = -np.inf,
    value_max: float = np.inf,
):
    """
    Compute the peak bin center within a value and frequency band.

    Returns
    -------
    peak_value : float
        Bin center with the highest count among selected bins, or np.nan if none.
    hist : np.ndarray
        Histogram counts.
    edges : np.ndarray
        Histogram bin edges.
    bin_centers : np.ndarray
        Centers of the bins.
    selected_bins : np.ndarray (bool)
        True for bins that were considered (within band and above freq_thresh).
    """
    if mask is not None:
        values = data[mask].ravel()
    else:
        values = data.ravel()

    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, None, None, None, None

    hist, edges = np.histogram(values, bins=bins)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    selected_bins = (
        (bin_centers >= value_min) & (bin_centers <= value_max) & (hist >= freq_thresh)
    )

    if not np.any(selected_bins):
        peak_value = np.nan
    else:
        # argmax only over selected bins
        idx_local = np.argmax(hist[selected_bins])
        idx_global = np.where(selected_bins)[0][idx_local]
        peak_value = float(bin_centers[idx_global])

    return peak_value, hist, edges, bin_centers, selected_bins


# Plot the histogram of an image
def plot_hist(ax, hist, edges, centers, selected, peak_val, band_min, band_max, title):
    if hist is None:
        ax.set_title(title + "\n(no data)")
        return

    width = edges[1] - edges[0]
    # Full histogram
    ax.bar(centers, hist, width=width, align="center", alpha=0.5, label="all bins")

    # Highlight the bins used for the peak search
    ax.bar(
        centers[selected],
        hist[selected],
        width=width,
        align="center",
        alpha=0.8,
        color="C1",
        label="used for peak",
    )

    # Shade the value band we requested
    ax.axvspan(band_min, band_max, color="C2", alpha=0.1, label="value band")

    if np.isfinite(peak_val):
        ax.axvline(peak_val, color="r", linestyle="--", linewidth=1, label="peak")

    ax.set_title(title)
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.legend()


def plot_image_with_histogram(
    data,
    title=None,
    bad_mask=None,
    band_min=15000,
    band_max=50000,
    bins=100,
    freq_thresh=10,
):
    """
    Plot an image and its histogram with peak detection.

    Parameters
    ----------
    data : numpy.ndarray
        Image data to plot
    title : str, optional
        Title for the plot (default: None)
    bad_mask : numpy.ndarray, optional
        Boolean mask of bad pixels (default: None, creates default mask)
    band_min : float, optional
        Minimum value for histogram peak search (default: 15000)
    band_max : float, optional
        Maximum value for histogram peak search (default: 50000)
    bins : int, optional
        Number of bins for histogram (default: 100)
    freq_thresh : int, optional
        Minimum frequency threshold for bins to be considered (default: 10)
    Returns
    -------
    dict
        Dictionary containing:
        - 'peak_value': histogram peak value in DN
        - 'fig': matplotlib figure object
        - 'axes': matplotlib axes objects
    """
    # Apply the bad pixel mask
    if bad_mask is None:
        # bad_mask = make_bad_mask(data.shape, active='even', col_phase=2, invert=False)
        bad_mask = np.zeros_like(data, dtype=bool)

    # Calculate histogram peak
    peak_value, hist, edges, centers, selected = histogram_peak_in_band(
        data,
        mask=~bad_mask,
        bins=bins,
        freq_thresh=freq_thresh,
        value_min=band_min,
        value_max=band_max,
    )

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the image with bad pixels masked
    ax = axes[1]
    """data_plot = data.copy()
    data_plot[bad_mask] = np.nan
    im = ax.imshow(
        data_plot,
        cmap="gray",
        vmin=np.percentile(data, 1),
        vmax=np.percentile(data, 99),
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("DN")"""
    plot_image(
        data,
        ax=ax,
        norm="zscale",
        # title="Image (bad pixels masked)",
    )

    # Plot histogram
    hist_title = (
        f"Histogram of image: Peak in [{band_min}, {band_max}] is {peak_value:.1f} DN"
    )
    plot_hist(
        axes[0],
        hist,
        edges,
        centers,
        selected,
        peak_value,
        band_min,
        band_max,
        hist_title,
    )

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()

    return {"peak_value": peak_value, "fig": fig, "axes": axes}
