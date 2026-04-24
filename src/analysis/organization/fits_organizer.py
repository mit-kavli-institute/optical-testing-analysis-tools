#!/usr/bin/env python3
"""
FITS File Organizer
Sorts FITS files by OBSTYPE into organized directory structure
"""

import os
import shutil
from pathlib import Path

from astropy.io import fits


def organize_fits_files(source_dir, output_dir=None):
    """
    Organize FITS files from source directory into sorted subdirectories.

    Args:
        source_dir: Path to directory containing FITS files
        output_dir: Output directory path. Can be:
                   - None (default): creates 'sorted' in source_dir
                   - Relative name: creates that folder in source_dir
                   - Absolute path: uses that full path
    """
    source_path = Path(source_dir)

    # Verify source directory exists
    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' not found")
        return

    # Determine output directory
    if output_dir is None:
        sorted_dir = source_path / "sorted"
    else:
        output_path = Path(output_dir)
        # Check if it's an absolute path
        if output_path.is_absolute():
            sorted_dir = output_path
        else:
            # Treat as relative name in source directory
            sorted_dir = source_path / output_dir
    darks_dir = sorted_dir / "darks"
    science_dir = sorted_dir / "science"
    focus_dir = sorted_dir / "focus"

    sorted_dir.mkdir(exist_ok=True)
    darks_dir.mkdir(exist_ok=True)
    science_dir.mkdir(exist_ok=True)
    focus_dir.mkdir(exist_ok=True)

    # Find all FITS files
    fits_files = list(source_path.glob("*.fits")) + list(source_path.glob("*.fit"))

    if not fits_files:
        print(f"No FITS files found in '{source_dir}'")
        return

    print(f"Found {len(fits_files)} FITS files")
    print("Organizing files...\n")

    # Statistics
    stats = {"DARK": 0, "SCIENCE": 0, "FOCUS": 0, "OTHER": 0, "ERROR": 0}

    # Process each file
    for fits_file in fits_files:
        try:
            # Read header
            with fits.open(fits_file) as hdul:
                header = hdul[0].header
                obstype = header.get("OBSTYPE", "").strip().upper()

                if obstype == "DARK":
                    # Get exposure time and create subfolder
                    exptime = header.get("EXPTIME", "unknown")
                    exptime_str = (
                        f"{exptime}s"
                        if isinstance(exptime, (int, float))
                        else str(exptime)
                    )
                    dest_dir = darks_dir / exptime_str
                    dest_dir.mkdir(exist_ok=True)
                    stats["DARK"] += 1

                elif obstype == "SCIENCE":
                    # Get target name and create subfolder
                    targname = header.get("TARGNAME", "unknown").strip()
                    # Sanitize target name for folder
                    targname = "".join(
                        c for c in targname if c.isalnum() or c in (" ", "-", "_")
                    ).strip()
                    if not targname:
                        targname = "unknown"
                    dest_dir = science_dir / targname
                    dest_dir.mkdir(exist_ok=True)
                    stats["SCIENCE"] += 1

                elif obstype == "FOCUS":
                    dest_dir = focus_dir
                    stats["FOCUS"] += 1

                else:
                    print(f"Unknown OBSTYPE '{obstype}' in {fits_file.name} - skipping")
                    stats["OTHER"] += 1
                    continue

                # Copy file to destination
                dest_file = dest_dir / fits_file.name
                shutil.copy2(fits_file, dest_file)
                print(f"✓ {fits_file.name} -> {dest_dir.relative_to(sorted_dir)}")

        except Exception as e:
            print(f"✗ Error processing {fits_file.name}: {e}")
            stats["ERROR"] += 1

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Dark frames:    {stats['DARK']}")
    print(f"Science frames: {stats['SCIENCE']}")
    print(f"Focus frames:   {stats['FOCUS']}")
    print(f"Other/Unknown:  {stats['OTHER']}")
    print(f"Errors:         {stats['ERROR']}")
    print(f"\nOrganized files saved to: {sorted_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fits_organizer.py <source_directory> [output_directory]")
        print("\nExamples:")
        print("  python fits_organizer.py /path/to/fits/files")
        print("  python fits_organizer.py /path/to/fits/files organized")
        print("  python fits_organizer.py /path/to/fits/files /absolute/path/to/output")
        sys.exit(1)

    source_directory = sys.argv[1]
    output_directory = sys.argv[2] if len(sys.argv) > 2 else None
    organize_fits_files(source_directory, output_directory)
