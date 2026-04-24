# base.py

"""Base classes for analysis pipelines."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.io import fits


class DataManager:
    """Base class for data management in analysis pipelines."""

    def __init__(self, data_dir: str | Path, output_dir: Optional[str | Path] = None):

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir
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
            - temperature
            - num_frames
        """
        data_records = []

        filepaths = self._get_files_in_directory(self.data_dir)
        # Walk through all subdirectories
        for filepath in filepaths:
            if filepath.endswith(".fits"):

                # ignore any file that has mask in the name
                if "mask" in filepath:
                    continue

                # Extract any information from filename if needed
                # (This can be overridden in subclasses)
                filename_info = self._parse_filename(filepath)

                # Read FITS header and data
                with fits.open(filepath) as hdul:
                    header = hdul[0].header
                    data = hdul[0].data

                    # Extract header information
                    # Camera-specific methods to get exposure time and temperature
                    exposure_time = self._get_exposure_time(filename_info, header)
                    temperature = self._get_temperature(filename_info, header)
                    frame_type = self._get_frame_type(filename_info, header)

                    # Count number of frames in stack
                    if data.ndim == 3:
                        num_frames = data.shape[0]
                    else:
                        num_frames = 1

                    record = {
                        "filename": os.path.basename(filepath),
                        "filepath": str(filepath),
                        "frame_type": frame_type,
                        "exposure_time": exposure_time,
                        "temperature": temperature,
                        "num_frames": num_frames,
                        "data": data,
                        "header": header,
                    }

                    data_records.append(record)

        df = pd.DataFrame(data_records)
        return df.sort_values("exposure_time")

    def _get_files_in_directory(self, directory: Path) -> list:
        from glob import glob

        pattern = os.path.join(directory, "*.fits")
        return glob(pattern)

    def _parse_filename(self, filepath: Path) -> dict:
        """Camera-specific method to parse filename. Override in subclasses if needed."""
        return {}

    def _get_exposure_time(
        self, filename_info: dict, header: fits.Header | dict
    ) -> float | None:
        """Camera-specific method to extract exposure time from FITS header."""
        # Default implementation; override in subclasses if needed
        return header.get("EXPTIME", None)  # Example key

    def _get_temperature(
        self, filename_info: dict, header: fits.Header | dict
    ) -> float | None:
        """Camera-specific method to extract temperature from FITS header."""
        # Default implementation; override in subclasses if needed
        return header.get("TEMP", None)  # Example key

    def _get_frame_type(
        self, filename_info: dict, header: fits.Header | dict
    ) -> str | None:
        """Camera-specific method to determine frame type from FITS header."""
        # Default implementation; override in subclasses if needed
        return header.get("FRAMETYP", "unknown")  # Example key
