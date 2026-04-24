"""
PTC analysis for WINTER sensor data.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from astropy.io import fits

from analysis.pipelines.base import DataManager


class WinterDataManager(DataManager):
    """Data manager for WINTER sensor data analysis."""

    def _parse_filename(self, filepath: str | Path) -> dict:
        """
        Parse FITS filename with format:
        <addr>_<YYYYMMDD>_texp_<exposure time>_<frame type>_lockin_<amplitude>V_freq_<frequency>Hz_gain_<gain>_<temperature>C_<filter>nm_<timestamp>[_ch_<channel>].fits

        Examples:
            "bench_20251209_texp_0.4_light_lockin_0.062950V_freq_200Hz_gain_100000000.0_-8.60C_1050nm_20251209-175653-845.fits"
            "bench_20251209_texp_0.4_dark_lockin_0.000000V_freq_200Hz_gain_100000000.0_-8.38C_1050nm_20251209-175750-807_ch_3.fits"
        """
        # Remove .fits extension
        filename = os.path.basename(filepath)
        name = filename.replace(".fits", "")

        # Split by underscore
        parts = name.split("_")

        # Extract components
        addr = parts[0]
        date_str = parts[1]

        # Find texp index and get exposure time
        texp_idx = parts.index("texp")
        exposure_time = float(parts[texp_idx + 1])

        # Frame type (light or dark)
        frame_type = parts[texp_idx + 2]

        # Find lockin index and get amplitude (remove 'V')
        lockin_idx = parts.index("lockin")
        lockin_amplitude = float(parts[lockin_idx + 1].replace("V", ""))

        # Find freq index and get frequency (remove 'Hz')
        freq_idx = parts.index("freq")
        chopper_freq = float(parts[freq_idx + 1].replace("Hz", ""))

        # Find gain index and get gain value
        gain_idx = parts.index("gain")
        gain = float(parts[gain_idx + 1])

        # Temperature (find part ending with 'C', remove 'C')
        temp_parts = [p for p in parts if p.endswith("C")]
        temperature = float(temp_parts[0].replace("C", ""))

        # Filter wavelength (find part ending with 'nm', remove 'nm')
        filter_parts = [p for p in parts if p.endswith("nm")]
        filter_wavelength = int(filter_parts[0].replace("nm", ""))

        # Check for channel value
        channel = None
        if "ch" in parts:
            ch_idx = parts.index("ch")
            channel = int(parts[ch_idx + 1])
            # Timestamp is before the channel
            timestamp_str = parts[ch_idx - 1]
        else:
            # Timestamp is the last part
            timestamp_str = parts[-1]

        # Parse timestamp: YYYYMMDD-HHMMSS-mmm
        dt = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S-%f")

        return {
            "addr": addr,
            "exposure_time": exposure_time,
            "frame_type": frame_type,
            "lockin_amplitude": lockin_amplitude,
            "chopper_freq": chopper_freq,
            "gain": gain,
            "temperature": temperature,
            "filter": filter_wavelength,
            "exposure_start_time": dt,
            "channel": channel,
            "filename": filename,
            "filepath": str(filepath),
        }

    def _get_exposure_time(
        self, filename_info: dict, header: fits.Header | dict
    ) -> float:
        """Extract exposure time from FITS header for WINTER camera."""
        return header.get("EXP_ACT", None)

    def _get_temperature(
        self, filename_info: dict, header: fits.Header | dict
    ) -> float:
        """Extract temperature from FITS header for WINTER camera."""
        return filename_info.get("temperature", None)

    def _get_frame_type(self, filename_info: dict, header: fits.Header | dict) -> str:
        """Extract frame type from FITS header for WINTER camera."""

        return filename_info.get("frame_type", "unknown")

    def split_data_into_channels(
        self,
        data: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        """
        Splits a 2D array into 8 channels by sampling rows and columns
        at regular offsets.

        Each of the 8 channels is extracted by:
        1. Choosing a row offset (j::2), where j = 1 for the first four channels
            (channels 0..3) and j = 0 for the latter four (channels 4..7).
        2. Choosing a column offset ((3 - i) % 4::4), where i is the channel index.

        Parameters
        ----------
        data : numpy.typing.NDArray[Any]
            2D array of shape (height, width). Must be evenly divisible so that
            height % 2 == 0 and width % 4 == 0.

        Returns
        -------
        channels_3d : numpy.typing.NDArray[Any]
            3D array of shape (8, height//2, width//4). The first dimension indexes
            the 8 channels, and the remaining two dimensions are the downsampled rows
            and columns for each channel.
        """
        height, width = data.shape

        # Number of channels to produce
        channels = 8

        # Prepare the output array
        data_8ch = np.zeros((channels, height // 2, width // 4), dtype=data.dtype)

        # Fill each of the 8 channels
        for i in range(channels):
            # Row offset (use j=1 for channels 0..3, j=0 for channels 4..7)
            j = 1 if (3 - i) >= 0 else 0
            # Column offset is (3 - i) % 4
            data_8ch[i] = data[j::2, (3 - i) % 4 :: 4]

        return data_8ch

    def make_datasec_mask(
        self, header: fits.Header, shape: tuple[int, int], invert: bool = False
    ) -> np.ndarray:
        """
        Function to create a boolean mask for the data section of an image
        """
        datasec = header["DATASEC"].replace("[", "").replace("]", "").split(",")
        datasec_xmin = int(datasec[0].split(":")[0]) - 1  # Convert to 0-based index
        datasec_xmax = int(datasec[0].split(":")[1])  # Non-inclusive
        datasec_ymin = int(datasec[1].split(":")[0]) - 1  # Convert to 0-based index
        datasec_ymax = int(datasec[1].split(":")[1])  # Non-inclusive

        mask = np.zeros(shape, dtype=bool)
        mask[datasec_ymin:datasec_ymax, datasec_xmin:datasec_xmax] = True
        if invert:
            mask = ~mask
        return mask

    def process_and_channelize_data(self, data_dir: str | Path) -> dict:
        """
        Process and channelize the data for WINTER camera.

        Takes in the directory with the full frame images,
        chunks it up by channel and saves to sub directories, 1-8.

        eg <data_dir>/ch_1/, <data_dir>/ch_2/, ..., <data_dir>/ch_8/

        Use the split_data_into_channels function to split each image into 8 channels.

        Make a master mask the first light image with make_datasec_mask,
        then split that mask up into 8 channels as well, and save each channel mask
        into the respective channel directory as 'master_mask_ch_X.fits'

        Then make a dictionary that maps channel number to a dataframe, and run crawl_fits_files
        on each channel directory to populate the dataframes.

        Parameters
        ----------
        data_dir : str | Path
            Directory containing full frame FITS images

        Returns
        -------
        dict
            Dictionary with channelized data, mapping channel to directory.
        """
        data_dir = Path(data_dir)

        # Create channel subdirectories
        num_channels = 8
        channel_dirs = {}
        for ch in range(1, num_channels + 1):
            ch_dir = data_dir / f"ch_{ch}"
            ch_dir.mkdir(exist_ok=True)
            channel_dirs[ch] = ch_dir

        # Get all FITS files in the directory
        fits_files = sorted(data_dir.glob("*.fits"))

        if len(fits_files) == 0:
            raise ValueError(f"No FITS files found in {data_dir}")

        print(f"Found {len(fits_files)} FITS files to process")

        # Create master mask from first light image
        master_mask = None
        for filepath in fits_files:
            filename_info = self._parse_filename(filepath)
            if filename_info["frame_type"] == "light":
                print(f"Creating master mask from {filepath.name}")
                with fits.open(filepath) as hdul:
                    header = hdul[0].header
                    data = hdul[0].data
                    master_mask = self.make_datasec_mask(
                        header, data.shape, invert=False
                    )
                break

        if master_mask is None:
            raise ValueError("No light frames found to create master mask")

        # Split master mask into channels and save
        master_mask_channels = self.split_data_into_channels(master_mask)
        for ch in range(1, num_channels + 1):
            mask_filepath = channel_dirs[ch] / f"master_mask_ch_{ch}.fits"
            fits.writeto(
                mask_filepath,
                master_mask_channels[ch - 1].astype(np.uint8),
                overwrite=True,
            )
            print(f"Saved mask for channel {ch} to {mask_filepath}")

        # Process each FITS file
        for i, filepath in enumerate(fits_files):
            print(f"Processing {i+1}/{len(fits_files)}: {filepath.name}")

            with fits.open(filepath) as hdul:
                header = hdul[0].header
                data = hdul[0].data

            # Split data into 8 channels
            data_channels = self.split_data_into_channels(data)

            # Parse filename to reconstruct channel-specific filenames
            filename_info = self._parse_filename(filepath)

            # Save each channel
            for ch in range(1, num_channels + 1):
                # Create new filename with channel prefix
                original_name = filepath.stem  # filename without .fits
                new_filename = f"{original_name}_ch_{ch}.fits"
                ch_filepath = channel_dirs[ch] / new_filename

                # Create new header with channel info
                new_header = header.copy()
                new_header["CHANNEL"] = ch
                new_header["ORIGFILE"] = filepath.name

                # Save channel data
                fits.writeto(
                    ch_filepath,
                    data_channels[ch - 1],
                    header=new_header,
                    overwrite=True,
                )

        print(f"\nChannelization complete. Processing channel directories...")

        # Create dictionary mapping channel to the new directory
        channel_directories = {
            ch: channel_dirs[ch] for ch in range(1, num_channels + 1)
        }

        return channel_directories
