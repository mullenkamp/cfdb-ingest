"""
Generate small spatial-subset WRF test files for CI.

Reads full wrfout files from a local path and writes subset NetCDF4/HDF5
files to cfdb_ingest/tests/data/. The subset preserves all 24 timesteps but
keeps only the bottom 16 vertical levels (~5000 m) and crops the spatial
domain to 60x55 grid points centered on the bbox (172.0, -44.0, 173.0, -43.0)
test region.

Output files include proper NetCDF4 dimensions and coordinate variables
(XLAT, XLONG, ZNU, etc.) so they can be opened in Panoply like normal
wrfout files.

Usage:
    uv run python -m cfdb_ingest.tests.create_test_data
"""

import pathlib

import h5py
import numpy as np
import pyproj

from cfdb_ingest.wrf import _wrf_attr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_DIR = pathlib.Path("/home/mike/data/wrf/tests/nudge_tests/test_d03_no_nudge")
OUTPUT_DIR = pathlib.Path(__file__).parent / "data"

SOURCE_FILES = [
    "wrfout_d03_2015-10-26_00:00:00.nc",
    "wrfout_d03_2015-10-27_00:00:00.nc",
]

# Spatial subset indices (in the original 534y x 315x grid)
Y_START, Y_END = 183, 243  # 60 unstaggered y points
X_START, X_END = 213, 268  # 55 unstaggered x points

# Vertical levels to keep (bottom 16 of 32 unstaggered, covers ~0-5000 m)
N_Z = 16  # unstaggered levels; staggered-z variables get N_Z+1

# Global attributes to copy verbatim
ATTRS_COPY = [
    "MAP_PROJ", "DX", "DY", "TRUELAT1", "TRUELAT2", "STAND_LON",
]

# ---------------------------------------------------------------------------
# Variable-to-dimension mapping
# ---------------------------------------------------------------------------
# Each entry: variable name -> tuple of dimension names for that variable.
# Spatial slicing is inferred from the dimension names.

DIM_2D = ("Time", "south_north", "west_east")
DIM_3D = ("Time", "bottom_top", "south_north", "west_east")
DIM_3D_STAG_Z = ("Time", "bottom_top_stag", "south_north", "west_east")
DIM_3D_STAG_X = ("Time", "bottom_top", "south_north", "west_east_stag")
DIM_3D_STAG_Y = ("Time", "bottom_top", "south_north_stag", "west_east")
DIM_2D_STAG_X = ("Time", "south_north", "west_east_stag")
DIM_2D_STAG_Y = ("Time", "south_north_stag", "west_east")
DIM_TIME_Z = ("Time", "bottom_top")
DIM_TIME_ZSTAG = ("Time", "bottom_top_stag")
DIM_TIME = ("Time",)
DIM_TIMES = ("Time", "DateStrLen")

VARIABLE_DIMS = {
    # Character time array
    "Times": DIM_TIMES,
    # 2D spatial (unstaggered)
    "COSALPHA": DIM_2D, "SINALPHA": DIM_2D,
    "T2": DIM_2D, "U10": DIM_2D, "V10": DIM_2D, "PSFC": DIM_2D,
    "TSK": DIM_2D, "SWDOWN": DIM_2D, "GLW": DIM_2D, "SNOWH": DIM_2D,
    "HGT": DIM_2D, "RAINNC": DIM_2D, "RAINC": DIM_2D, "Q2": DIM_2D,
    "XLAT": DIM_2D, "XLONG": DIM_2D,
    # 3D unstaggered
    "T": DIM_3D, "P": DIM_3D, "PB": DIM_3D, "QVAPOR": DIM_3D,
    # 3D staggered in z
    "PH": DIM_3D_STAG_Z, "PHB": DIM_3D_STAG_Z,
    # 3D staggered in x
    "U": DIM_3D_STAG_X,
    # 3D staggered in y
    "V": DIM_3D_STAG_Y,
    # 2D coordinate arrays on staggered grids
    "XLAT_U": DIM_2D_STAG_X, "XLONG_U": DIM_2D_STAG_X,
    "XLAT_V": DIM_2D_STAG_Y, "XLONG_V": DIM_2D_STAG_Y,
    # Vertical coordinate arrays
    "ZNU": DIM_TIME_Z, "ZNW": DIM_TIME_ZSTAG,
    # Time coordinate
    "XTIME": DIM_TIME,
}


def _compute_new_center(src_h5):
    """Compute CEN_LAT/CEN_LON for the center of the spatial subset."""
    attrs = src_h5.attrs
    dx = _wrf_attr(attrs, "DX")
    dy = _wrf_attr(attrs, "DY")
    cen_lat = _wrf_attr(attrs, "CEN_LAT")
    cen_lon = _wrf_attr(attrs, "CEN_LON")
    truelat1 = _wrf_attr(attrs, "TRUELAT1")
    truelat2 = _wrf_attr(attrs, "TRUELAT2")
    stand_lon = _wrf_attr(attrs, "STAND_LON")

    nx_orig = _wrf_attr(attrs, "WEST-EAST_PATCH_END_UNSTAG") - _wrf_attr(attrs, "WEST-EAST_PATCH_START_UNSTAG") + 1
    ny_orig = (
        _wrf_attr(attrs, "SOUTH-NORTH_PATCH_END_UNSTAG") - _wrf_attr(attrs, "SOUTH-NORTH_PATCH_START_UNSTAG") + 1
    )

    crs = pyproj.CRS.from_cf({
        "grid_mapping_name": "lambert_conformal_conic",
        "standard_parallel": [truelat1, truelat2],
        "longitude_of_central_meridian": stand_lon,
        "latitude_of_projection_origin": cen_lat,
        "false_easting": 0.0,
        "false_northing": 0.0,
    })

    transformer_to_proj = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    transformer_to_geo = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    center_x, center_y = transformer_to_proj.transform(cen_lon, cen_lat)

    ny_sub = Y_END - Y_START
    nx_sub = X_END - X_START
    sub_center_j = Y_START + (ny_sub - 1) / 2.0
    sub_center_i = X_START + (nx_sub - 1) / 2.0

    new_center_x = center_x + (sub_center_i - (nx_orig - 1) / 2.0) * dx
    new_center_y = center_y + (sub_center_j - (ny_orig - 1) / 2.0) * dy

    new_cen_lon, new_cen_lat = transformer_to_geo.transform(new_center_x, new_center_y)
    return new_cen_lat, new_cen_lon


def _slice_for_dim(dim_name):
    """Return the index slice to apply along a given dimension."""
    return {
        "Time": slice(None),
        "DateStrLen": slice(None),
        "bottom_top": slice(None, N_Z),
        "bottom_top_stag": slice(None, N_Z + 1),
        "south_north": slice(Y_START, Y_END),
        "south_north_stag": slice(Y_START, Y_END + 1),
        "west_east": slice(X_START, X_END),
        "west_east_stag": slice(X_START, X_END + 1),
    }[dim_name]


def _dim_size(dim_name):
    """Return the output size for a given dimension (before data is known)."""
    return {
        "bottom_top": N_Z,
        "bottom_top_stag": N_Z + 1,
        "south_north": Y_END - Y_START,
        "south_north_stag": Y_END - Y_START + 1,
        "west_east": X_END - X_START,
        "west_east_stag": X_END - X_START + 1,
        "DateStrLen": 19,
    }.get(dim_name)  # Time returns None → handled as unlimited


def create_subset_file(src_path, dst_path):
    """Read a full wrfout and write a spatially subsetted copy with NetCDF4 dimensions."""
    ny_sub = Y_END - Y_START
    nx_sub = X_END - X_START

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        new_cen_lat, new_cen_lon = _compute_new_center(src)

        # --- Global attributes ---
        dst.attrs["_NCProperties"] = "version=2,h5py=cfdb_ingest"
        for attr in ATTRS_COPY:
            dst.attrs[attr] = src.attrs[attr]
        dst.attrs["CEN_LAT"] = np.array([new_cen_lat], dtype="float32")
        dst.attrs["CEN_LON"] = np.array([new_cen_lon], dtype="float32")
        dst.attrs["SOUTH-NORTH_PATCH_START_UNSTAG"] = np.array([1], dtype="int32")
        dst.attrs["SOUTH-NORTH_PATCH_END_UNSTAG"] = np.array([ny_sub], dtype="int32")
        dst.attrs["WEST-EAST_PATCH_START_UNSTAG"] = np.array([1], dtype="int32")
        dst.attrs["WEST-EAST_PATCH_END_UNSTAG"] = np.array([nx_sub], dtype="int32")

        # --- Create NetCDF4 dimension scales ---
        dim_scales = {}
        dim_order = [
            "Time", "bottom_top", "bottom_top_stag",
            "south_north", "south_north_stag",
            "west_east", "west_east_stag",
            "DateStrLen",
        ]
        for i, dim_name in enumerate(dim_order):
            size = _dim_size(dim_name)
            if size is None:
                # Unlimited dimension (Time)
                ds = dst.create_dataset(dim_name, shape=(0,), maxshape=(None,), dtype="f4")
            else:
                ds = dst.create_dataset(dim_name, shape=(size,), dtype="f4")
            ds.make_scale(dim_name)
            ds.attrs["_Netcdf4Dimid"] = np.int32(i)
            dim_scales[dim_name] = ds

        # --- Write variables and attach dimension scales ---
        for var_name, dim_names in VARIABLE_DIMS.items():
            if var_name not in src:
                continue

            # Build the slicing tuple
            sel = tuple(_slice_for_dim(d) for d in dim_names)
            data = src[var_name][sel]

            # Times is a byte/char array — no compression
            if var_name == "Times":
                ds = dst.create_dataset(var_name, data=data)
            else:
                ds = dst.create_dataset(var_name, data=data, compression="gzip", compression_opts=6)

            # Attach dimension scales
            for ax, dim_name in enumerate(dim_names):
                ds.dims[ax].attach_scale(dim_scales[dim_name])

    size_mb = dst_path.stat().st_size / (1024 * 1024)
    print(f"  {dst_path.name}: {size_mb:.1f} MB")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Source directory: {SOURCE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Spatial subset: y=[{Y_START}:{Y_END}], x=[{X_START}:{X_END}]")
    print(f"  Unstaggered: {Y_END - Y_START}y x {X_END - X_START}x")
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
