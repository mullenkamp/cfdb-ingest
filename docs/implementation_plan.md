# Implementation Plan: cfdb-ingest Package

## Context

cfdb-ingest is a new package to convert external file formats into cfdb, standardizing variable names and metadata to CF conventions. The cfdb project (at `../cfdb`) already has a `netcdf4_to_cfdb()` function in `cfdb/tools.py` that does structural conversion but no source-specific variable mapping, unit conversion, or CRS extraction. cfdb-ingest fills that gap with source-aware converters, starting with WRF output files.

## Module Layout

```
cfdb_ingest/
  __init__.py           # Existing, add public API exports
  base.py               # Base class for HDF5/netCDF4 ingestion via h5py
  wrf.py                # WRF output file converter
  cli.py                # Typer CLI application
  tests/
    conftest.py         # Test fixtures
    test_wrf.py         # WRF converter tests
```

## 1. Dependencies (`pyproject.toml`)

Add runtime dependencies:
```
cfdb
h5py
numpy
pyproj
geointerp
rechunkit
typer
```

---

## 2. Base Class (`cfdb_ingest/base.py`)

An abstract-ish base class `H5Ingest` that provides the common workflow for converting HDF5/netCDF4 files to cfdb via h5py. Subclasses override methods to provide source-specific behavior.

### 2a. Initialization

```python
class H5Ingest:
    def __init__(self, input_paths)
```

**Parameters:**
- `input_paths`: str, Path, or list thereof — source HDF5/netCDF4 file(s)

**What `__init__` does:**
1. Normalize `input_paths` to a sorted list of `pathlib.Path` objects
2. Validate all files exist and are readable via h5py
3. Call subclass methods to derive metadata from the files:
   - `_init_crs()` → sets `self.crs` (pyproj.CRS)
   - `_init_time()` → sets `self.times` (np.ndarray of datetime64), `self._file_time_map` (which file has which time indices)
   - `_init_spatial_coords()` → sets `self.x`, `self.y` (1D np.ndarrays) and optionally `self.lat`, `self.lon` if geographic
   - `_init_variables()` → sets `self.variables` (dict of available variable info)
4. Compute `self.bbox_geographic` — the lat/lon extent of the domain (derived from CRS + x/y or directly from lat/lon)

**Exposed attributes** (available to user before `convert()`):
- `crs` — pyproj.CRS object
- `times` — np.ndarray of all timestamps across all files (sorted, deduplicated)
- `x`, `y` — 1D spatial coordinate arrays (projected)
- `variables` — dict describing available variables (name, cfdb name, source vars, units, dimensions)
- `bbox_geographic` — tuple of (min_lon, min_lat, max_lon, max_lat) in WGS84
- `input_paths` — list of source file paths

### 2b. Convert Method

```python
def convert(self, cfdb_path, variables=None, start_date=None, end_date=None,
            bbox=None, target_levels=None, max_mem=2**27, dataset_type='grid', **cfdb_kwargs)
```

**Parameters:**
- `cfdb_path`: str or Path — output cfdb file path
- `variables`: list of str or None — variable names to convert (accepts both source and cfdb names). None = all mapped variables found in source.
- `start_date`, `end_date`: str or np.datetime64 — optional time range filter
- `bbox`: tuple of 4 floats or None — `(min_lon, min_lat, max_lon, max_lat)` in WGS84 degrees. Clips the spatial domain to the bounding box.
- `target_levels`: list/array of float or None — target height levels (meters) for 3D variable interpolation. Required if any 3D variables are requested.
- `max_mem`: int — memory budget in bytes for rechunkit read buffers (default `2**27` = 128 MiB). Larger values reduce the number of HDF5 reads when source chunks are small relative to the target chunk shape.
- `dataset_type`: passed to `cfdb.open_dataset` (default `'grid'`)
- `**cfdb_kwargs`: extra kwargs for `cfdb.open_dataset` (compression, etc.)

**What `convert()` does:**
1. **Resolve variables**: Map user-provided names to the internal variable mapping (handles both source and cfdb names)
2. **Filter time**: Subset `self.times` by start_date/end_date, determine which source files to read
3. **Filter space (bbox)**: Transform bbox from WGS84 to the source CRS, find the x/y index ranges that fall within, subset x/y arrays
4. **Create cfdb dataset**: `cfdb.open_dataset(cfdb_path, 'n', dataset_type=dataset_type, **cfdb_kwargs)`
5. **Create coordinates**:
   - Time: `ds.create.coord.time(data=filtered_times)`
   - Spatial: `ds.create.coord.x(data=filtered_x)`, `ds.create.coord.y(data=filtered_y)`
   - Height (if 3D vars): `ds.create.coord.height(data=target_levels)`
6. **Set CRS**: `ds.create.crs.from_user_input(self.crs, x_coord='x', y_coord='y')`
7. **Create and populate data variables**: For each variable:
   - Create cfdb data var using template method (e.g., `ds.create.data_var.air_temp(...)`) or generic, with `chunk_shape=(1, ny, nx)` for 2D or `(1, 1, ny, nx)` for 3D
   - Use `rechunkit.rechunker()` to iterate over source HDF5 data in optimal read order, yielding full x/y spatial slabs per timestep (see section 2e)
   - Apply transforms/unstaggering per slab, write to cfdb
   - For 3D vars: read source levels the same way via rechunkit, apply `geointerp.GridInterpolator.regrid_levels()` to interpolate to target_levels, then write. Only necessary for WRF output.

### 2c. Abstract Methods (subclasses must implement)

- `_parse_crs(h5) -> pyproj.CRS` — extract CRS from a source file
- `_parse_time(h5) -> np.ndarray[datetime64]` — extract time array from a source file
- `_parse_spatial_coords(h5) -> dict` — return spatial coordinate arrays
- `_get_variable_mapping() -> dict` — return the full variable mapping dict
- `_read_variable(h5, var_key, time_idx, spatial_slice) -> np.ndarray` — read and transform a variable for one timestep

### 2d. Concrete Helper Methods

- `resolve_variables(variables)` — resolve user-provided names (either source or cfdb) against the mapping dict, return list of mapping keys
- `_bbox_to_indices(bbox)` — transform WGS84 bbox to source CRS, compute x/y slice indices
- `_open_file(path)` — open an h5py.File (context manager)

### 2e. Chunking Strategy and rechunkit

**Target cfdb chunk shape:**
- 2D variables: `(1, ny, nx)` — one full x/y slab per timestep
- 3D variables: `(1, 1, ny, nx)` — one full x/y slab per timestep per level

Each chunk is a complete spatial field, which is the natural access pattern for gridded meteorological data.

**Why rechunkit is needed:**
WRF files may have arbitrary HDF5 chunk layouts (e.g., `(1, 100, 100)`) or contiguous storage. When the target chunk is `(1, ny, nx)` (full spatial slab), naively reading one target chunk at a time would re-read overlapping source chunks multiple times. rechunkit's LCM-based read buffer minimizes this:
1. Computes the optimal read buffer shape via `calc_source_read_chunk_shape(source_chunks, target_chunks, itemsize, max_mem)`
2. Fills the buffer with source chunks (each read exactly once if memory allows)
3. Yields multiple target chunks from a single buffer fill

**Usage in convert():**
```python
import rechunkit

source_chunk_shape = h5_var.chunks or rechunkit.guess_chunk_shape(h5_var.shape, h5_var.dtype.itemsize)
target_chunk_shape = (1, ny, nx)  # full x/y per timestep

for target_slices, data in rechunkit.rechunker(
    h5_var.__getitem__,       # source callable
    h5_var.shape,             # full array shape
    h5_var.dtype,             # numpy dtype
    source_chunk_shape,       # HDF5 native chunks
    target_chunk_shape,       # desired output layout
    max_mem,                  # memory budget (default 2**27 = 128 MiB)
    sel=time_spatial_sel,     # optional sub-selection (time + bbox filtering)
):
    # target_slices = (slice(t, t+1), slice(0, ny), slice(0, nx))
    # data has shape (1, ny, nx) — one complete spatial slab
    # Apply transforms (unstagger, wind rotation, etc.) then write to cfdb
```

For variables that need multiple source WRF variables (e.g., RAINNC+RAINC for precip, U10+V10 for wind), rechunkit is used per source variable, and results are combined after reading.

For unchunked (contiguous) WRF variables, `rechunkit.guess_chunk_shape()` estimates a source chunk shape using highly composite numbers for optimal rechunking.

**cfdb chunk_shape parameter:**
When creating cfdb data variables, pass the target chunk shape explicitly:
```python
ds.create.data_var.air_temp(('time', 'y', 'x'), chunk_shape=(1, ny, nx))
```

---

## 3. WRF Converter (`cfdb_ingest/wrf.py`)

`class WrfIngest(H5Ingest)` — implements WRF-specific logic.

### 3a. Variable Mapping (`wrf_variable_mapping` dict)

A module-level dict mapping variable keys to cfdb standardization info. Each entry describes one output variable.

```python
wrf_variable_mapping = {
    # --- 2D surface variables ---
    'T2': {
        'cfdb_name': 'air_temp',           # cfdb short name (uses cfdb template)
        'source_vars': ['T2'],              # WRF variables needed
        'transform': None,                  # No transform (already in K)
        'dims': '2d',                       # 2D surface variable
    },
    'RAINNC': {
        'cfdb_name': 'precip',
        'source_vars': ['RAINNC', 'RAINC'],
        'transform': 'precip_increment',    # Compute (RAINNC+RAINC) diff between timesteps
        'dims': '2d',
    },
    'WIND': {
        'cfdb_name': 'wind_speed',
        'source_vars': ['U10', 'V10'],
        'transform': 'wind_speed',          # Rotate grid->earth, compute sqrt(u^2+v^2)
        'dims': '2d',
    },
    'WIND_DIR': {
        'cfdb_name': 'wind_direction',
        'source_vars': ['U10', 'V10'],
        'transform': 'wind_direction',      # Rotate grid->earth, compute direction
        'dims': '2d',
    },
    'RH2': {
        'cfdb_name': 'relative_humidity',
        'source_vars': ['RH2'],
        'transform': None,
        'dims': '2d',
    },
    'TSK': {
        'cfdb_name': 'soil_temp',
        'source_vars': ['TSK'],
        'transform': None,
        'dims': '2d',
    },

    # --- 3D variables (require level interpolation) ---
    'T': {
        'cfdb_name': 'air_temp_3d',
        'source_vars': ['T', 'P', 'PB'],
        'transform': 'potential_to_actual_temp',  # Convert theta to T using pressure
        'dims': '3d',
        'level_source': 'geopotential_height',     # Source levels derived from PH+PHB
    },
    # Additional 3D mappings as needed
}
```

Users can pass either WRF keys (`'T2'`, `'RAINNC'`, `'WIND'`), source var names (`'T2'`, `'U10'`), or cfdb names (`'air_temp'`, `'precip'`, `'wind_speed'`) in the `variables` parameter. The `resolve_variables` method handles all three.

### 3b. CRS Extraction (`_parse_crs`)

Read WRF global attributes and build a pyproj CRS:
- `MAP_PROJ=1` (Lambert Conformal Conic): Use `TRUELAT1`, `TRUELAT2`, `STAND_LON`, `CEN_LAT`
- `MAP_PROJ=2` (Polar Stereographic): Use `TRUELAT1`, `STAND_LON`
- `MAP_PROJ=3` (Mercator): Use `TRUELAT1`, `STAND_LON`
- `MAP_PROJ=6` (Lat-Lon): Use EPSG:4326

For Lambert Conformal (most common), construct via:
```python
pyproj.CRS.from_cf({
    'grid_mapping_name': 'lambert_conformal_conic',
    'standard_parallel': [truelat1, truelat2],
    'longitude_of_central_meridian': stand_lon,
    'latitude_of_projection_origin': cen_lat,
    'false_easting': 0.0,
    'false_northing': 0.0,
})
```

Also store cone factor for wind rotation: `self._cone_factor` (computed from TRUELAT1, TRUELAT2).

### 3c. Spatial Coordinates (`_parse_spatial_coords`)

Compute projected x/y coordinate arrays from WRF grid info:
1. Read `DX`, `DY` (grid spacing in meters), `CEN_LAT`, `CEN_LON` from global attrs
2. Get grid dimensions: `south_north` and `west_east` dimension sizes (from any non-staggered 2D variable)
3. Transform center point (CEN_LAT, CEN_LON) to projected coordinates using the CRS
4. Compute 1D arrays:
   - `x[i] = center_x + (i - center_i) * DX` where `center_i = (nx - 1) / 2`
   - `y[j] = center_y + (j - center_j) * DY` where `center_j = (ny - 1) / 2`

For `MAP_PROJ=6` (lat-lon): use first row of `XLONG` and first column of `XLAT` directly as lon/lat coordinates.

### 3d. Time Parsing (`_parse_time`)

WRF stores time as a character array `Times` with shape `(n_times, 19)`, format `"YYYY-MM-DD_HH:MM:SS"`.
- Read the byte array, decode to strings
- Replace `_` with `T` for ISO format
- Parse to `numpy.datetime64[m]` (minute resolution, matching cfdb default time dtype)

### 3e. WRF Unstaggering

WRF uses Arakawa C-grid staggering:
- **`west_east_stag`**: has `nx + 1` points (U wind, etc.)
- **`south_north_stag`**: has `ny + 1` points (V wind, etc.)
- **`bottom_top_stag`**: has `nz + 1` points (W wind, PH geopotential, etc.)

Unstaggering averages adjacent points along the staggered dimension:
```python
def unstagger(data, axis):
    slices_lo = [slice(None)] * data.ndim
    slices_hi = [slice(None)] * data.ndim
    slices_lo[axis] = slice(None, -1)
    slices_hi[axis] = slice(1, None)
    return (data[tuple(slices_lo)] + data[tuple(slices_hi)]) / 2.0
```

Applied automatically in `_read_variable` when a source variable has a staggered dimension:
- Check if any dimension name ends in `_stag`
- Identify the staggered axis index
- Unstagger along that axis before returning

### 3f. 3D Variable Level Interpolation

For WRF 3D variables on model (eta) levels:

1. **Compute source levels** (geopotential height at each grid point):
   - Read `PH` (perturbation geopotential) and `PHB` (base geopotential) — both on `bottom_top_stag`
   - Unstagger vertically: `geopotential_height = unstagger((PH + PHB) / 9.81, axis=z_axis)`
   - Result has shape `(nz, ny, nx)` — the actual height in meters at each 3D grid point

2. **Set up regrid function** (once per conversion):
   ```python
   from geointerp import GridInterpolator
   gi = GridInterpolator(from_crs=self.crs)
   regrid_func = gi.regrid_levels(target_levels, axis=0, method='linear')
   ```

3. **Per timestep**: call `regrid_func(data_3d, source_levels_3d)` to interpolate from eta levels to the target height levels

4. **Output coordinates**: `(time, height, y, x)` where height is the target_levels array

### 3g. Data Transforms

Implemented as methods on `WrfIngest`:

- **`_transform_precip_increment`**: Sum `RAINNC + RAINC` for current and previous timestep, compute difference. First timestep of each file uses the previous file's last timestep if available; very first timestep overall gets NaN.
- **`_transform_wind_speed`**: Read U10/V10, rotate from grid-relative to earth-relative using the cone factor and grid-point longitude, compute `sqrt(u^2 + v^2)`.
- **`_transform_wind_direction`**: Same rotation, compute meteorological direction (direction FROM which wind blows): `(270 - atan2(v, u) * 180/pi) % 360`.
- **`_transform_potential_to_actual_temp`**: Convert WRF perturbation potential temperature `T` (which is theta - 300) to actual temperature using pressure: `T_actual = (T + 300) * (P + PB) / 100000) ^ (R/Cp)`.

### 3h. Wind Rotation (Grid-Relative to Earth-Relative)

WRF stores U10/V10 relative to the model grid, not geographic north/east. For Lambert Conformal:

```python
# Cone factor
cone = (np.log(np.cos(np.radians(truelat1))) - np.log(np.cos(np.radians(truelat2)))) / \
       (np.log(np.tan(np.radians(90 - abs(truelat1)) / 2)) - np.log(np.tan(np.radians(90 - abs(truelat2)) / 2)))

# Rotation angle per grid point (uses 2D longitude field or computed from x coords)
diff = longitude - stand_lon
alpha = np.radians(diff * cone)

# Rotate
u_earth = u_grid * np.cos(alpha) - v_grid * np.sin(alpha)
v_earth = u_grid * np.sin(alpha) + v_grid * np.cos(alpha)
```

For Mercator and lat-lon: no rotation needed.
For Polar Stereographic: different rotation formula based on longitude.

### 3i. Multi-file Handling

When multiple wrfout files are passed:
1. During `__init__`: open each file, extract the time array, close
2. Sort files chronologically by first timestep
3. Build `self._file_time_map`: list of `(path, time_array, global_start_idx)` tuples
4. Concatenate all time arrays into `self.times` (deduplicate overlapping timestamps)
5. During `convert()`: open files as needed based on which timesteps pass the date filter

### 3j. Bbox Filtering

When `bbox=(min_lon, min_lat, max_lon, max_lat)` is passed to `convert()`:
1. Transform the 4 corners (and intermediate edge points for curved projections) from WGS84 to the source CRS using pyproj
2. Compute the bounding range in projected coordinates: `x_min, x_max, y_min, y_max`
3. Find the index ranges in `self.x` and `self.y` that fall within this projected bbox
4. Subset x/y coordinate arrays
5. Apply the spatial slice when reading every variable

---

## 4. Public API (`cfdb_ingest/__init__.py`)

Export the main classes:
```python
from cfdb_ingest.wrf import WrfIngest
from cfdb_ingest.base import H5Ingest
```

### Usage Examples

**Basic usage — inspect then convert:**
```python
from cfdb_ingest import WrfIngest

wrf = WrfIngest('/path/to/wrfout_d04_2020-01-04_00_00_00')

# Inspect metadata before converting
print(wrf.crs)
print(wrf.times)
print(wrf.variables)
print(wrf.bbox_geographic)

# Convert selected variables with filtering
wrf.convert(
    cfdb_path='output.cfdb',
    variables=['air_temp', 'precip', 'wind_speed'],
    start_date='2020-01-04T06:00',
    end_date='2020-01-04T18:00',
)
```

**Multi-file with bbox and 3D levels:**
```python
wrf = WrfIngest([
    '/path/to/wrfout_d04_2020-01-04_00_00_00',
    '/path/to/wrfout_d04_2020-01-05_00_00_00',
])

wrf.convert(
    cfdb_path='output_3d.cfdb',
    variables=['air_temp', 'T'],  # Mix of 2D and 3D
    bbox=(170.0, -47.0, 178.0, -34.0),
    target_levels=[100, 500, 1000, 2000, 5000],
)
```

---

## 5. CLI (`cfdb_ingest/cli.py`)

A Typer CLI application providing command-line access to all conversion processes. Registered as a console script entry point in `pyproject.toml`.

### Entry Point (`pyproject.toml`)

```toml
[project.scripts]
cfdb-ingest = "cfdb_ingest.cli:app"
```

### App Structure

The top-level `app` is a `typer.Typer()` instance. Each source format gets its own command (or command group if it grows). For now, there is a single `wrf` command.

### `wrf` Command

```
cfdb-ingest wrf INPUT_PATHS... CFDB_PATH [OPTIONS]
```

**Arguments:**
- `INPUT_PATHS` — one or more wrfout file paths (positional, variadic)
- `CFDB_PATH` — output cfdb file path (positional)

**Options:**
- `--variables`, `-v` — comma-separated list of variable names (WRF or cfdb names). Omit for all mapped variables.
- `--start-date`, `-s` — start date filter (ISO format, e.g., `2020-01-04` or `2020-01-04T06:00`)
- `--end-date`, `-e` — end date filter (ISO format)
- `--bbox`, `-b` — bounding box as `min_lon,min_lat,max_lon,max_lat` (comma-separated floats)
- `--target-levels`, `-l` — comma-separated height levels in meters for 3D variable interpolation
- `--max-mem` — rechunkit read buffer size in bytes (default 134217728)
- `--compression` — cfdb compression algorithm (`zstd` or `lz4`, default `zstd`)

**Implementation:** Parses CLI arguments, constructs a `WrfIngest` instance, calls `convert()`. Minimal logic — delegates everything to the library classes.

```python
import typer
from pathlib import Path
from typing import Optional, List
from typing_extensions import Annotated

app = typer.Typer(help="Convert file formats to cfdb.")

@app.command()
def wrf(
    input_paths: Annotated[List[Path], typer.Argument(help="One or more wrfout file paths.")],
    cfdb_path: Annotated[Path, typer.Argument(help="Output cfdb file path.")],
    variables: Annotated[Optional[str], typer.Option("--variables", "-v", help="Comma-separated variable names.")] = None,
    start_date: Annotated[Optional[str], typer.Option("--start-date", "-s", help="Start date (ISO format).")] = None,
    end_date: Annotated[Optional[str], typer.Option("--end-date", "-e", help="End date (ISO format).")] = None,
    bbox: Annotated[Optional[str], typer.Option("--bbox", "-b", help="Bounding box: min_lon,min_lat,max_lon,max_lat")] = None,
    target_levels: Annotated[Optional[str], typer.Option("--target-levels", "-l", help="Comma-separated height levels (m).")] = None,
    max_mem: Annotated[int, typer.Option(help="Read buffer size in bytes.")] = 2**27,
    compression: Annotated[str, typer.Option(help="Compression: zstd or lz4.")] = "zstd",
):
    from cfdb_ingest.wrf import WrfIngest

    wrf = WrfIngest(input_paths)

    var_list = [v.strip() for v in variables.split(",")] if variables else None
    bbox_tuple = tuple(float(x) for x in bbox.split(",")) if bbox else None
    levels = [float(x) for x in target_levels.split(",")] if target_levels else None

    wrf.convert(
        cfdb_path=cfdb_path,
        variables=var_list,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox_tuple,
        target_levels=levels,
        max_mem=max_mem,
        compression=compression,
    )
```

### CLI Usage Examples

```bash
# Basic conversion
cfdb-ingest wrf wrfout_d04_2020-01-04_00_00_00 output.cfdb

# With variable and date filtering
cfdb-ingest wrf wrfout_d04_*.nc output.cfdb \
    -v air_temp,precip,wind_speed \
    -s 2020-01-04 -e 2020-01-05

# With bbox clipping and 3D levels
cfdb-ingest wrf wrfout_d04_*.nc output.cfdb \
    -v air_temp,T \
    -b 170.0,-47.0,178.0,-34.0 \
    -l 100,500,1000,2000,5000
```

---

## 6. Key cfdb API Calls Used

From `../cfdb` (see `cfdb/creation.py`, `cfdb/utils.py`):

```python
import cfdb

with cfdb.open_dataset(cfdb_path, 'n', dataset_type='grid') as ds:
    # Create coordinates using templates (auto-sets dtype + CF attrs)
    ds.create.coord.time(data=time_array)
    ds.create.coord.x(data=x_array)
    ds.create.coord.y(data=y_array)
    ds.create.coord.height(data=target_levels)  # For 3D vars

    # Set CRS
    ds.create.crs.from_user_input(pyproj_crs, x_coord='x', y_coord='y')

    # Create data vars using templates
    air_temp = ds.create.data_var.air_temp(('time', 'y', 'x'))
    air_temp_3d = ds.create.data_var.generic('air_temperature_3d',
        ('time', 'height', 'y', 'x'), dtype=cfdb.dtypes.dtype(...))

    # Write data per timestep
    air_temp[time_slice] = data_2d
    air_temp_3d[time_slice] = data_3d  # After regrid_levels
```

### Key cfdb patterns:
- **Template coord methods**: `ds.create.coord.time()`, `.x()`, `.y()`, `.height()` — auto-set dtype/attrs from `cfdb/utils.py:default_dtype_params` and `default_attrs`
- **Template data var methods**: `ds.create.data_var.air_temp()`, `.precip()`, `.wind_speed()`, etc. — same auto-setup
- **Generic methods**: `ds.create.coord.generic(name, ...)`, `ds.create.data_var.generic(name, ...)` — for non-template variables
- **CRS**: `ds.create.crs.from_user_input(crs, x_coord, y_coord)` — sets CRS string in metadata and axis flags on coordinates

---

## 6. Files to Create/Modify

| File | Action |
|------|--------|
| `pyproject.toml` | Add runtime dependencies (cfdb, h5py, numpy, pyproj, geointerp, rechunkit, typer) + console script entry point |
| `cfdb_ingest/__init__.py` | Add public API exports (WrfIngest, H5Ingest) |
| `cfdb_ingest/base.py` | **New** — H5Ingest base class |
| `cfdb_ingest/wrf.py` | **New** — WrfIngest class + wrf_variable_mapping |
| `cfdb_ingest/cli.py` | **New** — Typer CLI application with `wrf` command |
| `cfdb_ingest/tests/test_wrf.py` | **New** — Tests for WRF conversion |

---

## 7. Verification

- Unit tests with a real or synthetic wrfout file confirming:
  - `__init__` correctly derives CRS, times, x/y coords, and available variables
  - CRS is correctly extracted for Lambert Conformal (and other MAP_PROJ types)
  - Time coordinate is correctly parsed from WRF `Times` character array
  - x/y projected coordinates are computed correctly from DX/DY
  - Staggered variables are correctly unstaggered
  - Variable data is written with correct cfdb standard names and CF attributes
  - Date range filtering selects correct timesteps
  - Bbox filtering correctly clips the spatial domain
  - Multi-file input produces correct merged time dimension
  - Precipitation increments are computed correctly (including across file boundaries)
  - Wind speed/direction rotation and computation are correct
  - 3D variables are correctly interpolated to target levels via geointerp.regrid_levels

---

## 8. Reference: cfdb Internals

### cfdb standard variable templates (from `cfdb/utils.py`)

**Coordinate defaults (`default_dtype_params`):**
- `time`: `datetime64[m]`, encoded as `int32`, offset `-36816481`
- `x`: `float32`, precision 1
- `y`: `float32`, precision 1
- `height`: `float64` -> `uint32`, precision 3, offset `-1`
- `lat`: `float64` -> `uint32`, precision 6, offset `-90.000001`
- `lon`: `float64` -> `uint32`, precision 6, offset `-180.000001`

**Data variable defaults (`default_dtype_params`):**
- `air_temp`: `float32` -> `uint16`, precision 2, offset `-61`
- `precip`: `float32` -> `uint16`, precision 2, offset `-1`
- `wind_speed`: `float32` -> `uint16`, precision 2, offset `-1`
- `wind_direction`: `float32` -> `uint16`, precision 1, offset `-1`
- `relative_humidity`: `float32` -> `uint16`, precision 1, offset `-1`
- `soil_temp`: `float32` -> `uint16`, precision 2, offset `-61`

**CF attributes (`default_attrs`):**
- `air_temp`: standard_name=`air_temperature`, units=`K`
- `precip`: standard_name=`precipitation_amount`, units=`mm`
- `wind_speed`: standard_name=`wind_speed`, units=`m/s`
- `wind_direction`: standard_name=`wind_to_direction`, units=`deg`
- `relative_humidity`: standard_name=`relative_humidity`, units=`m^3/m^3`
- `soil_temp`: standard_name=`soil_temperature`, units=`K`

### cfdb creation API (from `cfdb/creation.py`)

- `ds.create.coord.time(data, step, **kwargs)` — auto-sets dtype, attrs from templates
- `ds.create.coord.x(data, step, **kwargs)` — same for x projected coordinate
- `ds.create.coord.y(data, step, **kwargs)` — same for y projected coordinate
- `ds.create.coord.height(data, step, **kwargs)` — same for height coordinate
- `ds.create.coord.generic(name, data, dtype, chunk_shape, step, axis)` — for non-template coords
- `ds.create.data_var.air_temp(coords, **kwargs)` — auto-sets dtype, attrs from templates
- `ds.create.data_var.generic(name, coords, dtype, chunk_shape)` — for non-template vars
- `ds.create.crs.from_user_input(crs, x_coord, y_coord)` — sets CRS and axis metadata

### geointerp API (from `geointerp/grid.py`)

- `gi = GridInterpolator(from_crs=crs)` — create interpolator
- `regrid_func = gi.regrid_levels(target_levels, axis=0, method='linear')` — returns a callable
- `result = regrid_func(data_3d, source_levels_3d)` — interpolate data from variable source levels to fixed target levels

### rechunkit API (from `rechunkit/main.py`)

- `rechunkit.guess_chunk_shape(shape, itemsize, target_chunk_size=2**21)` — estimate chunk shape for unchunked data, using highly composite numbers for optimal LCM alignment
- `rechunkit.rechunker(source, shape, dtype, source_chunk_shape, target_chunk_shape, max_mem, sel=None)` — main rechunking generator. `source` is any callable that accepts a tuple of slices and returns an ndarray. Yields `(target_slices, data)` pairs, reading source data in LCM-aligned buffer groups that minimize re-reads.
- `rechunkit.calc_source_read_chunk_shape(source_chunk_shape, target_chunk_shape, itemsize, max_mem)` — compute optimal read buffer shape (LCM of source/target, constrained to memory budget)
- `rechunkit.calc_n_reads_rechunker(shape, itemsize, source_chunk_shape, target_chunk_shape, max_mem, sel=None)` — estimate `(n_reads, n_writes)` without moving data, useful for benchmarking
