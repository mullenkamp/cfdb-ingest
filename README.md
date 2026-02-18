# cfdb-ingest

<p align="center">
    <em>Convert meteorological model output to cfdb with standardized CF conventions</em>
</p>

[![build](https://github.com/mullenkamp/cfdb-ingest/workflows/Build/badge.svg)](https://github.com/mullenkamp/cfdb-ingest/actions)
[![codecov](https://codecov.io/gh/mullenkamp/cfdb-ingest/branch/master/graph/badge.svg)](https://codecov.io/gh/mullenkamp/cfdb-ingest)
[![PyPI version](https://badge.fury.io/py/cfdb-ingest.svg)](https://badge.fury.io/py/cfdb-ingest)

---

**Documentation**: <a href="https://mullenkamp.github.io/cfdb-ingest/" target="_blank">https://mullenkamp.github.io/cfdb-ingest/</a>

**Source Code**: <a href="https://github.com/mullenkamp/cfdb-ingest" target="_blank">https://github.com/mullenkamp/cfdb-ingest</a>

---

## Table of Contents

- [Overview](#overview)
- [Supported Formats](#supported-formats)
- [WRF Variables](#wrf-variables)
- [Installation](#installation)
- [Python API](#python-api)
- [CLI](#cli)
- [Development](#development)
- [License](#license)

## Overview

cfdb-ingest converts meteorological file formats (netCDF4/HDF5, GRIB2, etc.) from various model outputs into [cfdb](https://github.com/mullenkamp/cfdb). It standardizes variable names and attributes to be consistent with [CF conventions](https://cfconventions.org/), making it straightforward to work with datasets from different sources through a single interface.

Key features:

- **Automatic variable mapping** -- source variable names are translated to CF-standard names with proper metadata (standard_name, units, encoding)
- **Wind rotation** -- grid-relative wind components are rotated to earth-relative using COSALPHA/SINALPHA
- **3D level interpolation** -- eta-level variables are interpolated to user-specified height levels above ground
- **Variable merging** -- surface and level-interpolated variants of the same quantity (e.g. T2 at 2 m and T at arbitrary heights) are merged into a single output variable
- **Spatial and temporal filtering** -- subset by bounding box (WGS84) and/or date range before writing
- **Multi-file support** -- seamlessly spans multiple input files, including cross-file precipitation accumulation
- **Configurable chunking** -- tune output chunk shapes for different access patterns

## Supported Formats

| Source | Class | CRS Projections |
|--------|-------|-----------------|
| WRF (wrfout) | `WrfIngest` | Lambert Conformal Conic, Polar Stereographic, Mercator, Lat-Lon |

## WRF Variables

Surface variables (fixed height above ground):

| Key | cfdb Name | Height | Source Vars | Transform |
|-----|-----------|--------|-------------|-----------|
| `T2` | `air_temp` | 2 m | T2 | direct |
| `PSFC` | `surface_pressure` | 0 m | PSFC | direct |
| `Q2` | `specific_humidity` | 2 m | Q2 | direct |
| `RAIN` | `precip` | 0 m | RAINNC, RAINC | accumulation increment |
| `WIND10` | `wind_speed` | 10 m | U10, V10 | wind rotation |
| `WIND_DIR10` | `wind_direction` | 10 m | U10, V10 | wind rotation |
| `U10` | `u_wind` | 10 m | U10, V10 | wind rotation |
| `V10` | `v_wind` | 10 m | U10, V10 | wind rotation |
| `TSK` | `soil_temp` | 0 m | TSK | direct |
| `SWDOWN` | `shortwave_radiation` | 0 m | SWDOWN | direct |
| `GLW` | `longwave_radiation` | 0 m | GLW | direct |
| `SNOWH` | `snow_depth` | 0 m | SNOWH | direct |
| `SLP` | `mslp` | 0 m | PSFC, T2, HGT | hypsometric reduction |

3D level-interpolated variables (interpolated to user-specified `target_levels`):

| Key | cfdb Name | Source Vars | Transform |
|-----|-----------|-------------|-----------|
| `T` | `air_temp` | T, P, PB, PH, PHB | potential to actual temperature |
| `WIND` | `wind_speed` | U, V, PH, PHB | unstagger + rotation |
| `WIND_DIR` | `wind_direction` | U, V, PH, PHB | unstagger + rotation |
| `U` | `u_wind` | U, V, PH, PHB | unstagger + rotation |
| `V` | `v_wind` | U, V, PH, PHB | unstagger + rotation |
| `Q` | `specific_humidity` | QVAPOR, PH, PHB | mixing ratio to specific humidity |

## Installation

Requires Python >= 3.11.

```bash
pip install cfdb-ingest
```

## Python API

### Basic conversion

```python
from cfdb_ingest import WrfIngest

wrf = WrfIngest('wrfout_d01_2023-02-12_00:00:00.nc')

# Convert selected variables for a time window
wrf.convert(
    cfdb_path='output.cfdb',
    variables=['T2', 'WIND10', 'precip'],
    start_date='2023-02-12T06:00',
    end_date='2023-02-12T18:00',
)
```

### Multi-file input

```python
wrf = WrfIngest([
    'wrfout_d01_2023-02-12_00:00:00.nc',
    'wrfout_d01_2023-02-13_00:00:00.nc',
])

# All timesteps across both files are merged automatically
wrf.convert(cfdb_path='output.cfdb', variables=['T2'])
```

### Spatial subsetting with a bounding box

```python
wrf.convert(
    cfdb_path='output.cfdb',
    variables=['T2'],
    bbox=(165.0, -47.0, 175.0, -40.0),  # (min_lon, min_lat, max_lon, max_lat)
)
```

### 3D level interpolation

```python
# Interpolate 3D temperature and wind to specific heights above ground
wrf.convert(
    cfdb_path='output.cfdb',
    variables=['T', 'WIND'],
    target_levels=[100.0, 500.0, 1000.0, 2000.0],
    bbox=(165.0, -47.0, 175.0, -40.0),
)
```

### Merging surface and 3D variables

Variables sharing a cfdb name are automatically merged. For example, `T2` (2 m) and `T` (levels) both map to `air_temp` and produce a single output variable spanning all heights:

```python
wrf.convert(
    cfdb_path='output.cfdb',
    variables=['T2', 'T'],
    target_levels=[100.0, 500.0],
)
# Output height coordinate: [2.0, 100.0, 500.0]
```

### Custom chunk shape

The output chunk shape defaults to `(1, 1, ny, nx)` (one full spatial slab per timestep per height level). Override it for different access patterns:

```python
wrf.convert(
    cfdb_path='output.cfdb',
    variables=['T2'],
    chunk_shape=(1, 1, 50, 50),  # (time, z, y, x)
)
```

### Inspecting metadata before conversion

```python
wrf = WrfIngest('wrfout_d01_2023-02-12_00:00:00.nc')

wrf.crs                # pyproj.CRS
wrf.times              # numpy datetime64 array
wrf.x, wrf.y           # 1D projected coordinate arrays
wrf.variables          # dict of available variable mappings
wrf.bbox_geographic    # (min_lon, min_lat, max_lon, max_lat)
```

### Variable name resolution

`variables` accepts mapping keys (`T2`), source variable names (`RAINNC`), or cfdb names (`air_temp`). When a cfdb name maps to multiple keys, all are included:

```python
wrf.resolve_variables(['air_temp'])  # ['T2', 'T']
wrf.resolve_variables(['RAINNC'])    # ['RAIN']
wrf.resolve_variables(None)          # all available keys
```

## CLI

cfdb-ingest provides a `cfdb-ingest` command with a `wrf` subcommand.

### Basic usage

```bash
cfdb-ingest wrf wrfout_d01_2023-02-12_00:00:00.nc output.cfdb \
    -v T2,WIND10 \
    -s 2023-02-12T06:00 \
    -e 2023-02-12T18:00
```

### All options

```
cfdb-ingest wrf [OPTIONS] INPUT_PATHS... CFDB_PATH
```

| Option | Short | Description |
|--------|-------|-------------|
| `--variables` | `-v` | Comma-separated variable names |
| `--start-date` | `-s` | Start date (ISO format) |
| `--end-date` | `-e` | End date (ISO format) |
| `--bbox` | `-b` | Bounding box: `min_lon,min_lat,max_lon,max_lat` |
| `--target-levels` | `-l` | Comma-separated height levels in meters |
| `--chunk-shape` | `-c` | Output chunk shape: `time,z,y,x` (e.g. `1,1,50,50`) |
| `--max-mem` | | Read buffer size in bytes (default: 128 MiB) |
| `--compression` | | Compression algorithm: `zstd` or `lz4` |

### Examples

```bash
# Convert with spatial subset
cfdb-ingest wrf wrfout_d01_*.nc output.cfdb \
    -v T2 -b 165.0,-47.0,175.0,-40.0

# 3D temperature at specific height levels
cfdb-ingest wrf wrfout_d01_*.nc output.cfdb \
    -v T -l 100,500,1000,2000 -b 165.0,-47.0,175.0,-40.0

# Custom chunk shape for time-series access patterns
cfdb-ingest wrf wrfout_d01_*.nc output.cfdb \
    -v T2,WIND10 -c 24,1,50,50
```

## Development

### Setup environment

We use [UV](https://docs.astral.sh/uv/) to manage the development environment and production build.

```bash
uv sync
```

### Run tests

```bash
uv run pytest
```

### Lint and format

```bash
uv run lint
```

## License

This project is licensed under the terms of the Apache Software License 2.0.
