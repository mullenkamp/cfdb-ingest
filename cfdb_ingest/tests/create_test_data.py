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
    "T2", "T", "U", "U10", "V", "V10", "W", "COSALPHA", "SINALPHA", "TSK", "HGT",
    "PH", "PSFC", "SWDOWN", "GLW", "SNOWH", "RAINNC", "RAINC", "P", "PB",
    "Q2", "QVAPOR", "PHB",
    "XLAND", "SST", "SEAICE", "SNOW",     # surface variables for WPS
    "SMOIS", "TSLB", "DZS",               # soil variables
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
        f" --cnk_dmn soil_layers_stag,4"
        f" -d bottom_top,0,{N_Z - 1} -d bottom_top_stag,0,{N_Z}"
        f" -v {vars_str}"
        f" {src_path} {dst_path}"
    )
    subprocess.run(shlex.split(cmd_str), capture_output=True, text=True, check=True)

    size_mb = dst_path.stat().st_size / (1024 * 1024)
    print(f"  {dst_path.name}: {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Real WRF pressure-level output (0.7.0): a 50 x 50 box over the Southern Alps from the Hetzner
# `plev_test_2023-02` run f_on (C1's 5 levels x 5 fields, extrap_below_grnd = 1, as the pipeline
# uploaded it), plus the same box of that day's wrfout for the rotation cross-check. The box holds
# cells missing at 900 AND 850 hPa in every frame (terrain up to ~1840 m). WRF's own layout is kept:
# one frame per chunk, all levels per chunk, the plane in 2 x 2 tiles; deflate 1 as the pipeline writes.
# ---------------------------------------------------------------------------

PLEV_SOURCE_DIR = pathlib.Path("/mnt/hdd_raid0/wrf/wvt/plev_2023-02-11/f_on_s3")
PLEV_OUTPUT_DIR = OUTPUT_DIR / "wrf_plevels"
PLEV_DAY = "2023-02-12_00_00_00"
PLEV_BOX = dict(south_north=(60, 109), west_east=(130, 179))
PLEV_STATIC_VARIABLES = ["Times", "XLAT", "XLONG", "COSALPHA", "SINALPHA", "HGT", "PSFC"]


def create_plevel_subset():
    """Crop the real wrfplevels day and its wrfout to PLEV_BOX (ncks), keeping WRF's 2 x 2 tiling."""
    PLEV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (j0, j1), (i0, i1) = PLEV_BOX["south_north"], PLEV_BOX["west_east"]
    crop = f"-d south_north,{j0},{j1} -d west_east,{i0},{i1}"
    tiles = f"--cnk_dmn Time,1 --cnk_dmn south_north,{(j1 - j0 + 2) // 2} --cnk_dmn west_east,{(i1 - i0 + 2) // 2}"
    for name, extra in ((f"wrfplevels_d01_{PLEV_DAY}.nc", "--cnk_dmn num_press_levels_stag,5"),
                        (f"wrfout_d01_{PLEV_DAY}.nc", "-v " + ",".join(PLEV_STATIC_VARIABLES))):
        src, dst = PLEV_SOURCE_DIR / name, PLEV_OUTPUT_DIR / name
        if not src.exists():
            print(f"  SKIP (not found): {src}")
            continue
        cmd = f"ncks -O -4 -L 1 {tiles} {extra} {crop} {src} {dst}"
        subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=True)
        print(f"  {dst.relative_to(OUTPUT_DIR)}: {dst.stat().st_size / 1024:.0f} KiB")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_plevel_subset()

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
