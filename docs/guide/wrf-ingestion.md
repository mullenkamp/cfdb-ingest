# WRF Ingestion

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

You can also pass a directory path and all `wrfout*` files will be found automatically:

```python
wrf = WrfIngest('/path/to/wrfout/')
```

### Spatial subsetting with a bounding box

```python
wrf.convert(
    cfdb_path='output.cfdb',
    variables=['T2'],
    bbox=(165.0, -47.0, 175.0, -40.0),  # (min_lon, min_lat, max_lon, max_lat)
)
```

### 3D level interpolation (height)

```python
# Interpolate 3D temperature and wind to specific heights above ground
wrf.convert(
    cfdb_path='output.cfdb',
    variables=['T', 'WIND'],
    target_levels=[100.0, 500.0, 1000.0, 2000.0],
    bbox=(165.0, -47.0, 175.0, -40.0),
)
```

### 3D level interpolation (pressure)

Use `vertical_coord='pressure'` to interpolate to pressure levels instead of height levels. Target levels are in Pa:

```python
wrf.convert(
    cfdb_path='output.cfdb',
    variables=['T', 'U', 'V', 'GHT', 'RH', 'Q_SH'],
    target_levels=[100000, 92500, 85000, 70000, 50000, 30000, 20000, 10000],
    vertical_coord='pressure',
)
```

### Surface and 3D variables

Surface variables are stored as `(time, y, x)` without a vertical dimension. 3D level-interpolated variables are stored as `(time, height, y, x)` or `(time, pressure, y, x)`.

When converting both surface and 3D variants of the same variable (e.g., `T2` and `T` both map to `air_temp`), they are stored as separate cfdb variables. The surface variant gets a `_sfc` suffix:

```python
wrf.convert(
    cfdb_path='output.cfdb',
    variables=['T2', 'T'],
    target_levels=[100.0, 500.0],
)
# Creates: air_temperature (time, height, y, x) and air_temp_sfc (time, y, x)
```

### Soil variables

Soil moisture and temperature are stored on a `depth` coordinate derived from WRF's DZS (soil layer thicknesses):

```python
wrf.convert(
    cfdb_path='output.cfdb',
    variables=['SMOIS', 'TSLB'],
)
```

### Custom chunk shape

The output chunk shape defaults to `(1, 1, ny, nx)` for 4D variables and `(1, ny, nx)` for surface variables. Override for 4D variables:

```python
wrf.convert(
    cfdb_path='output.cfdb',
    variables=['T'],
    target_levels=[100.0, 500.0],
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

### Basic usage

```bash
cfdb-ingest wrf wrfout_d01_2023-02-12_00:00:00.nc output.cfdb \
    -v T2,WIND10 \
    -s 2023-02-12T06:00 \
    -e 2023-02-12T18:00
```

### Options

```
cfdb-ingest wrf [OPTIONS] INPUT_PATHS... CFDB_PATH
```

| Option | Short | Description |
|--------|-------|-------------|
| `--variables` | `-v` | Comma-separated variable names |
| `--preset` | | Variable preset: `wps` selects all variables needed for cfdb-to-int export |
| `--start-date` | `-s` | Start date (ISO format) |
| `--end-date` | `-e` | End date (ISO format) |
| `--bbox` | `-b` | Bounding box: `min_lon,min_lat,max_lon,max_lat` |
| `--target-levels` | `-l` | Comma-separated target levels (meters for height, Pa for pressure) |
| `--vertical-coord` | | Vertical coordinate: `height` (default) or `pressure` |
| `--chunk-shape` | `-c` | Output chunk shape: `time,z,y,x` (e.g. `1,1,50,50`) |
| `--max-mem` | | Read buffer size in bytes (default: 536 MiB) |
| `--compression` | | Compression algorithm: `zstd` or `lz4` (default: `zstd`) |

### Examples

```bash
# Convert with spatial subset
cfdb-ingest wrf wrfout_d01_*.nc output.cfdb \
    -v T2 -b 165.0,-47.0,175.0,-40.0

# 3D temperature at specific height levels
cfdb-ingest wrf wrfout_d01_*.nc output.cfdb \
    -v T -l 100,500,1000,2000 -b 165.0,-47.0,175.0,-40.0

# WPS preset -- all variables, pressure levels, and settings in one flag
cfdb-ingest wrf /path/to/wrfout/ output.cfdb \
    --preset wps -s 2023-02-10 -e 2023-02-10_06

# WPS preset with custom pressure levels
cfdb-ingest wrf /path/to/wrfout/ output.cfdb \
    --preset wps -l 100000,85000,70000,50000,30000,20000,10000

# Custom chunk shape for time-series access patterns
cfdb-ingest wrf wrfout_d01_*.nc output.cfdb \
    -v T2,WIND10 -c 24,1,50,50
```
