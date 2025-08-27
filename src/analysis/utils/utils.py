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
