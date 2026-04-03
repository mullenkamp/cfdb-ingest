## cfdb-ingest Library Reference

`cfdb-ingest` is a Python package for converting model outputs (WRF) and reanalysis data (ERA5) into the `cfdb` database format. It provides both a Python API and a CLI.

### Installation

```bash
pip install cfdb-ingest
# or
uv add cfdb-ingest
```

### Python API: WRF Ingestion

Convert WRF `wrfout` netCDF files to `cfdb`. It supports surface variables, 3D variables interpolated to height or pressure levels, and soil variables.

```python
from cfdb_ingest import WrfIngest

# Initialize with a single file, a list of files, or a directory path
wrf = WrfIngest('/path/to/wrfout/')

# Basic conversion of surface variables
wrf.convert(
    cfdb_path='output.cfdb',
    variables=['T2', 'WIND10', 'precip'], # Map keys, source names, or cfdb names
    start_date='2023-02-12T06:00',        # Optional time filter
    end_date='2023-02-12T18:00',
    bbox=(165.0, -47.0, 175.0, -40.0),    # Optional spatial subset (min_lon, min_lat, max_lon, max_lat)
)

# 3D Variables: Interpolate to height levels (meters)
wrf.convert(
    cfdb_path='output_heights.cfdb',
    variables=['T', 'WIND'],
    target_levels=[100.0, 500.0, 1000.0, 2000.0],
)

# 3D Variables: Interpolate to pressure levels (Pascals)
wrf.convert(
    cfdb_path='output_pressure.cfdb',
    variables=['T', 'U', 'V', 'GHT', 'RH'],
    target_levels=[100000, 92500, 85000, 70000, 50000],
    vertical_coord='pressure',
)

# Accessing metadata before conversion:
crs = wrf.crs                # pyproj.CRS
times = wrf.times            # numpy datetime64 array
x_coords = wrf.x
y_coords = wrf.y
var_dict = wrf.variables     # dict of available variables
bbox = wrf.bbox_geographic   # (min_lon, min_lat, max_lon, max_lat)
```

### Python API: ERA5 Ingestion

Convert ERA5 NetCDF files (e.g., from the NCAR RDA archive) to `cfdb`. Automatically handles combining single-variable ERA5 files into a single database.

```python
from cfdb_ingest import Era5Ingest

# Initialize with directory or list of files (mix of surface and pressure-level files)
era5 = Era5Ingest('/path/to/era5/')

# Basic conversion
era5.convert(
    cfdb_path='era5_output.cfdb',
    variables=['SP', 'VAR_2T', 'T', 'U', 'V'],
    start_date='2020-01-01',
    end_date='2020-01-31',
    bbox=(170.0, -40.0, 175.0, -35.0), # (min_lon, min_lat, max_lon, max_lat)
)

# Auto-detects pressure levels from source files, or specify explicitly:
era5.convert(
    cfdb_path='era5_output.cfdb',
    variables=['T', 'U', 'V'],
    target_levels=[100000, 85000, 70000, 50000], # Pa
)

# Compute Vertically Integrated Moisture Flux (VIMF) natively during ingestion
# (Requires Q, U, V source files to be present)
era5.convert(
    cfdb_path='era5_vimf.cfdb',
    variables=['VIMF_U', 'VIMF_V'],
)

# Split mode: Create a separate cfdb file for each variable in a directory
era5.convert(
    cfdb_path='/output/dir/',
    variables=['SP', 'VAR_2T', 'T'],
    split=True, 
)
```

### CLI Usage

`cfdb-ingest` provides a command-line interface for both WRF and ERA5.

**WRF:**
```bash
cfdb-ingest wrf wrfout_d01_*.nc output.cfdb -v T2,WIND10 -s 2023-02-12 -e 2023-02-13

# WPS Export Preset (auto-selects variables and pressure levels needed for metgrid)
cfdb-ingest wrf /path/to/wrfout/ output.cfdb --preset wps -s 2023-02-10 -e 2023-02-10_06
```

**ERA5:**
```bash
# Combined surface + pressure level
cfdb-ingest era5 /path/to/era5/*.nc output.cfdb -v SP,VAR_2T,T,U,V -s 2020-01-01 -e 2020-01-31

# Split mode (one cfdb per variable)
cfdb-ingest era5 /path/to/era5/*.nc /output/dir/ --split -v SP,VAR_2T,T
```

**WPS Intermediate Export:**
Convert a `cfdb` file (previously ingested with `--preset wps`) to the WPS intermediate format:
```bash
cfdb-to-int output.cfdb -s 2023-02-10 -e 2023-02-10_06
```

### Advanced Options (Python `convert` method & CLI)
- `chunk_shape`: Tuple `(time, z, y, x)` to override default chunking (default is `(1, 1, ny, nx)`). CLI: `-c 1,1,50,50`
- `compression`: `'zstd'` or `'lz4'` (default is `'zstd'`).
- `max_mem`: Memory budget for read buffers (bytes).

### Key Rules
- Both classes support passing a directory path directly instead of explicitly globbing `*.nc` files.
- Spatial subsets (`bbox`) take tuples of `(min_lon, min_lat, max_lon, max_lat)`.
- When variables exist as both surface and 3D variants (e.g. Temperature), the surface variant automatically gets suffixed with its height in the resulting `cfdb` (e.g. `air_temperature_2m`).
