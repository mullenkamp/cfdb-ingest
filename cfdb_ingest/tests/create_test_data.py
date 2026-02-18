"""
Generate vertically-subset WRF test files for CI.

Uses ncks to extract selected variables and the bottom N_Z+1 vertical levels
from full wrfout files, writing compressed NetCDF4 output to
cfdb_ingest/tests/data/.

Usage:
    uv run python -m cfdb_ingest.tests.create_test_data
"""

import pathlib
import shlex
import subprocess

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_DIR = pathlib.Path("/home/mike/data/wrf/tests/cfdb_ingest")
OUTPUT_DIR = pathlib.Path(__file__).parent / "data"

SOURCE_FILES = [
    "wrfout_d01_2023-02-12_00:00:00.nc",
    "wrfout_d01_2023-02-13_00:00:00.nc",
]

VARIABLES = [
    "Times", "XLAT", "XLONG", "XTIME", "XLONG_U", "XLONG_V", "XLAT_U", "XLAT_V",
    "T2", "T", "U", "U10", "V", "V10", "COSALPHA", "SINALPHA", "TSK", "HGT",
    "PH", "PSFC", "SWDOWN", "GLW", "SNOWH", "RAINNC", "RAINC", "P", "PB",
    "Q2", "QVAPOR", "PHB",
]

# Vertical levels to keep (bottom 16 of 32 unstaggered, covers ~5000 m)
N_Z = 16  # unstaggered levels; staggered-z variables get N_Z+1


def create_subset_file(src_path, dst_path):
    """Read a full wrfout and write a vertically subsetted copy via ncks."""
    vars_str = ",".join(VARIABLES)
    cmd_str = (
        f"ncks -O -4 -L 5"
        f" --cnk_dmn Time,1 --cnk_dmn bottom_top,{N_Z}"
        f" --cnk_dmn south_north,111 --cnk_dmn west_east,99"
        f" -d bottom_top,0,{N_Z - 1} -d bottom_top_stag,0,{N_Z}"
        f" -v {vars_str}"
        f" {src_path} {dst_path}"
    )
    subprocess.run(shlex.split(cmd_str), capture_output=True, text=True, check=True)

    size_mb = dst_path.stat().st_size / (1024 * 1024)
    print(f"  {dst_path.name}: {size_mb:.1f} MB")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Source directory: {SOURCE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"  Vertical: {N_Z} unstaggered levels")
    print()

    for filename in SOURCE_FILES:
        src_path = SOURCE_DIR / filename
        dst_path = OUTPUT_DIR / filename
        if not src_path.exists():
            print(f"  SKIP (not found): {src_path}")
            continue
        print(f"Processing {filename}...")
        create_subset_file(src_path, dst_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
