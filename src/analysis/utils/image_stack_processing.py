# image_stack_processing.py

from typing import Tuple

import numpy as np


def remove_outlier_layers(
    stack: np.ndarray, N: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outlier layers from a 3D image stack based on median intensity.

    Parameters
    ----------
    stack : np.ndarray
        3D numpy array corresponding to image data, shape (n_layers, height, width).
    N : float, optional
        Number of standard deviations from the global median to use as the
        outlier threshold (default is 3.0).

    Returns
    -------
    cleaned_stack : np.ndarray
        Image stack with outlier layers removed.
    inlier_mask : np.ndarray
        Boolean array of shape (n_layers,) indicating which layers were kept.

    Notes
    -----
    Outliers are identified by comparing the median of each layer with the global
    median of all layer medians. A layer is kept if its median lies within
    ``N * std`` of the global median.

    Examples
    --------
    >>> cleaned_stack, inlier_mask = remove_outlier_layers(image_stack)
    >>> cleaned_stack.shape
    (n_inliers, height, width)
    >>> inlier_mask.sum()
    n_inliers
    """
    if stack.ndim != 3:
        raise ValueError("Input stack must be a 3D numpy array.")

    # Calculate per-layer medians
    layer_medians = np.median(stack, axis=(1, 2))

    # Global median and standard deviation
    global_median = np.median(layer_medians)
    global_std = np.std(layer_medians)

    if global_std == 0:
        # If all layers are identical, keep everything
        inlier_mask = np.ones_like(layer_medians, dtype=bool)
    else:
        # Identify inliers
        inlier_mask = np.abs(layer_medians - global_median) <= N * global_std

    # Filter the stack
    cleaned_stack = stack[inlier_mask]

    return cleaned_stack, inlier_mask
