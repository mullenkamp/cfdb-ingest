# ERA5 Ingestion

## Overview

`Era5Ingest` converts ERA5 NetCDF files from the [NCAR ERA5 archive](https://rda.ucar.edu/datasets/d633000/) to cfdb. ERA5 files from NCAR have one variable per file, with surface products as monthly files and pressure level products as daily files.

## Python API

### Basic conversion

```python
from cfdb_ingest import Era5Ingest

era5 = Era5Ingest('/path/to/era5/*.nc')

era5.convert(
    cfdb_path='output.cfdb',
    variables=['SP', 'VAR_2T', 'T'],
    start_date='2020-01-01',
    end_date='2020-01-31',
)
```

### Input paths

ERA5 files can be provided as individual paths, a list, or a directory:

```python
# Single file
era5 = Era5Ingest('e5.oper.an.sfc.128_134_sp.ll025sc.2020010100_2020013123.nc')

# Multiple files
era5 = Era5Ingest([
    'e5.oper.an.sfc.128_134_sp.ll025sc.2020010100_2020013123.nc',
    'e5.oper.an.pl.128_130_t.ll025sc.2020010100_2020010123.nc',
])

# Directory (finds all *.nc files)
era5 = Era5Ingest('/path/to/era5/')

# Multiple directories
era5 = Era5Ingest(['/path/to/sfc/', '/path/to/pl/'])
```

### Spatial subsetting

```python
era5.convert(
    cfdb_path='output.cfdb',
    variables=['SP'],
    bbox=(170.0, -40.0, 175.0, -35.0),  # (min_lon, min_lat, max_lon, max_lat)
)
```

### Pressure level variables

Pressure levels are auto-detected from the source files. You can also specify them explicitly:

```python
era5.convert(
    cfdb_path='output.cfdb',
    variables=['T', 'U', 'V'],
    target_levels=[100000, 85000, 70000, 50000],  # Pa
)
```

### Combined vs split output

**Combined (default):** All variables in one cfdb file.

```python
era5.convert(
    cfdb_path='output.cfdb',
    variables=['SP', 'VAR_2T', 'T'],
)
```

**Split:** One cfdb file per variable, written to a directory.

```python
era5.convert(
    cfdb_path='/output/dir/',
    variables=['SP', 'VAR_2T', 'T'],
    split=True,
)
# Creates: surface_pressure.cfdb, air_temperature.cfdb (for T)
```

### Variable name resolution

`variables` accepts mapping keys (e.g. `SP`, `VAR_2T`, `T`), source variable names, or cfdb names:

```python
era5.resolve_variables(['surface_pressure'])  # ['SP']
era5.resolve_variables(['air_temperature'])   # ['VAR_2T', 'T'] (both surface and pl)
era5.resolve_variables(None)                  # all available
```

### Inspecting metadata

```python
era5 = Era5Ingest('/path/to/era5/')

era5.crs                # pyproj.CRS (always EPSG:4326)
era5.times              # numpy datetime64 array (union across all files)
era5.x                  # longitude array
era5.y                  # latitude array (ascending)
era5.variables          # dict of available variable mappings
era5.bbox_geographic    # (min_lon, min_lat, max_lon, max_lat)
```

## Height Coordinates

Surface variables are stored with named height coordinates indicating their measurement height:

| Height | Coordinate | Example variables |
|--------|-----------|-------------------|
| 0 m | `height_0m` | SP, MSL, SSTK, CI, snow fields, albedo, all invariant vars |
| 2 m | `height_2m` | VAR_2T, VAR_2D |
| 10 m | `height_10m` | VAR_10U, VAR_10V, U10N, V10N |
| 100 m | `height_100m` | VAR_100U, VAR_100V |

Pressure level variables use a `pressure` coordinate with `axis='Z'`. Named height coordinates do not have `axis='Z'` when a pressure coordinate is also present.

When a variable name conflicts between surface and pressure levels (e.g. `air_temperature` from both `VAR_2T` and `T`), the surface variant is suffixed: `air_temperature_2m`.

## Geopotential Transform

The `Z` variable in ERA5 is geopotential (m2 s-2), not geopotential height (m). When ingested:

- Pressure level Z (`Z_PL`) is converted to `geopotential_height` by dividing by g (9.80665)
- Invariant Z (`Z_INV`) is converted to `terrain_height` by dividing by g

## CLI

### Basic usage

```bash
cfdb-ingest era5 /path/to/era5/*.nc output.cfdb \
    -v SP,VAR_2T,T,U,V \
    -s 2020-01-01 -e 2020-01-31
```

### Options

```
cfdb-ingest era5 [OPTIONS] INPUT_PATHS... OUTPUT_PATH
```

| Option | Short | Description |
|--------|-------|-------------|
| `--variables` | `-v` | Comma-separated variable names |
| `--split` | | Create one cfdb file per variable |
| `--start-date` | `-s` | Start date (ISO format) |
| `--end-date` | `-e` | End date (ISO format) |
| `--bbox` | `-b` | Bounding box: `min_lon,min_lat,max_lon,max_lat` |
| `--target-levels` | `-l` | Comma-separated pressure levels in Pa (auto-detected if omitted) |
| `--chunk-shape` | `-c` | Output chunk shape: `time,z,y,x` (e.g. `1,1,50,50`) |
| `--compression` | | Compression algorithm: `zstd` or `lz4` |

### Examples

```bash
# Surface variables only
cfdb-ingest era5 /path/to/sfc/*.nc output.cfdb \
    -v SP,VAR_2T,VAR_10U,VAR_10V

# Pressure level variables with spatial subset
cfdb-ingest era5 /path/to/pl/*.nc output.cfdb \
    -v T,U,V,Q -b 170.0,-40.0,175.0,-35.0

# Combined surface + pressure level
cfdb-ingest era5 /path/to/era5/*.nc output.cfdb \
    -v SP,VAR_2T,T,U,V

# Split mode: one cfdb per variable
cfdb-ingest era5 /path/to/era5/*.nc /output/dir/ --split \
    -v SP,VAR_2T,T
```
