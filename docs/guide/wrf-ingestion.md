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

### Projection and coordinates (0.6.0)

Lambert (MAP_PROJ 1), polar-stereographic (2) and Mercator (3) grids are stored with 1-D projected `x`/`y`
(metres) and a CRS built from the file's own metadata on **WRF's earth model**: the WPS projection
parameters (`TRUELAT1/2`, `STAND_LON`) on a sphere of 6 370 000 m (WPS `constants_module.F`,
`EARTH_RADIUS_M`), which is what `XLAT`/`XLONG` were computed on. The Lambert latitude of origin is a
convention -- WPS anchors its grid at a known corner and has no such parameter; any value gives the same
geometry with shifted `y` -- and cfdb-ingest uses `MOAD_CEN_LAT` (the outermost domain's centre), so every
nest of one run shares one CRS. The polar hemisphere follows the sign of `TRUELAT1`, as WPS does.

`x`/`y` are the file's own cell centres (`XLAT`/`XLONG`) projected through that CRS and fitted to a regular
`DX`/`DY` lattice, then checked both ways within `WrfIngest.xy_tolerance_m` (25 m; healthy files measure
1.6-4.7 m, the float32 precision of `XLAT`/`XLONG`). A file whose projection attributes do not describe its
grid is refused rather than written with misplaced coordinates. Without `XLAT`/`XLONG` the lattice is laid
around `CEN_LAT`/`CEN_LON` with a warning. Every input file's grid header (projection, `DX`/`DY`,
dimensions, `BUCKET_MM`, `PREC_ACC_DT`) must match the first file's.

!!! warning "Outputs of cfdb-ingest < 0.6.0"
    Earlier versions built these CRSs on the WGS84 ellipsoid, placing cells up to several km (2.5 km on a
    3 km NZ d03, 5-8 km on 12-27 km domains) from where WRF computed them, and gave each nest its own
    origin. Appending new output to such a dataset (grid `extend` or a forecast archive) is refused with a
    "rebuild" message; rebuild those datasets from the wrfout.

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

Surface variables are stored as `(time, height_Xm, y, x)` with a named height coordinate indicating their measurement height. 3D level-interpolated variables are stored as `(time, height, y, x)` or `(time, pressure, y, x)`.

When converting both surface and 3D variants of the same variable (e.g., `T2` and `T` both map to `air_temp`), they are stored as separate cfdb variables. The surface variant is suffixed with its height:

```python
wrf.convert(
    cfdb_path='output.cfdb',
    variables=['T2', 'T'],
    target_levels=[100.0, 500.0],
)
# Creates: air_temperature (time, height, y, x) and air_temperature_2m (time, height_2m, y, x)
# (the surface variant takes a height suffix on the full cfdb-vars name because the name also
# exists on levels; a name present at two heights, e.g. 10 m and 100 m winds, is suffixed at both)
```

`squeeze_height=True` (0.6.0, grid mode, surface variables only) stores surface fields as `(time, y, x)`
without the length-1 height axis; the height moves to a `height` attribute (e.g. `'0 m'`), and
`chunk_shape` may be given as 3-D `(time, y, x)`.

### Soil variables

Soil moisture and temperature are stored on a `depth` coordinate derived from WRF's DZS (soil layer thicknesses):

```python
wrf.convert(
    cfdb_path='output.cfdb',
    variables=['SMOIS', 'TSLB'],
)
```

### Potential temperature, vorticity, and surface fluxes

Beyond temperature, humidity, and wind, the mapping also covers several other field groups. They are
selected by key like any other variable:

- **Potential temperature** -- `THETA2`/`THETA_E2` (2 m) and `THETA`/`THETA_E` (3D level-interpolated)
- **Relative vorticity** -- `VORT10` (10 m) and `VORT` (3D level-interpolated)
- **Surface energy and land fields** -- `HFX` (sensible heat flux), `QFX` (moisture flux), `ALBEDO`,
  `EMISS` (emissivity), and `LU_INDEX` (MODIS land-use category)

See the [WRF Variables reference](../reference/wrf-variables.md) for the full list.

### Precipitation and accumulators

`RAIN` and `RAIN_TR` are stored as per-timestep accumulation increments: each value is the difference
between successive accumulated totals, so the cfdb variable holds the precipitation that fell during
each interval rather than a running total. Several details are handled automatically:

- **Cross-file boundaries** -- the increment is computed across input files, so multi-file
  conversions produce a continuous series with no gap or spike where files join.
- **Bucket counters** -- when a `wrfout` uses WRF's bucket accumulators (a non-zero `BUCKET_MM`
  attribute with companion `I_RAINNC`/`I_RAINC` overflow counts), the true total is reconstructed as
  `RAINNC + BUCKET_MM * I_RAINNC` before differencing.
- **Negative increments** -- small negative differences (which can occur with nudging or two-way
  nesting feedback) are clipped to zero.
- **One run only** (0.6.0) -- the running totals restart at every cold start, so differencing across
  runs (different `SIMULATION_START_DATE`) would store the first interval of each later run as 0. A window
  whose frames come from more than one run is refused (files outside the window don't matter); a window
  starting at a run's first frame gets NaN there, not a difference against the previous run. Use
  `PREC_ACC` for a record stitched from independent runs.

`PREC_ACC` (0.5.0) stores the same quantity -- precipitation over the output interval, cfdb `precip`
-- from WRF's own windowed accumulators `PREC_ACC_NC + PREC_ACC_C` (namelist `prec_acc_dt` set to the
history interval). Each frame already holds the increment since the previous frame, so it needs no
previous-frame state: it is the source to use when an init is ingested one file at a time, and it
gives 0 rather than NaN at lead 0. Not valid on a two-way-nested *parent* domain, whose `PREC_ACC_*`
the child's feedback overwrites -- use it on the innermost domain. `RAIN` and `PREC_ACC` cannot be
requested together (both write `precip`). Since 0.6.0 negative `PREC_ACC` values (float noise, e.g. from
offline backfills of bucket differences) are clipped to 0, as `RAIN`'s increments are.

Both are labelled, by default, with WRF's frame time -- the END of the interval they accumulate. For an
interval-start convention, see [Interval-start labels](#interval-start-labels-060).

### Interval-start labels (0.6.0)

`convert(..., time_label='start')` labels each accumulation by the start of its interval: a `PREC_ACC`
frame at `t` (precipitation over `(t - PREC_ACC_DT, t]`) is stored at `t - PREC_ACC_DT`. It applies to
accumulations only (`PREC_ACC`, `RAIN`, `RAIN_TR`) -- all fields of one conversion share one time axis, so
mixing in an instantaneous field is refused -- and to grid mode only. Details:

- `PREC_ACC_DT` must equal the frame spacing, so each frame is one output interval.
- `start_date` / `end_date` select the **labels**: the window `[b0, b1)` is `start_date=b0,
  end_date=b1 - 1h`, and needs the input frame at `b1` (usually the next file's first frame).
- A frame whose interval starts before **its own file's** `SIMULATION_START_DATE` (a run's lead-0 frame,
  WRF's zero-initialised accumulator) is dropped, and `PREC_ACC` frames must lie on their run's
  `SIMULATION_START_DATE + k * PREC_ACC_DT` grid. Both are evaluated per file, because one conversion of a
  stitched hindcast spans several runs.
- The dataset records `time.attrs['time_label'] = 'interval_start'` and `interval_minutes`, and the
  variable `cell_methods = 'time: sum (interval: 60 minutes)'`.

### Column-integrated and moisture-transport variables

Precipitable water and vertically integrated moisture flux are available as surface (`height_0m`)
variables:

```python
wrf.convert(
    cfdb_path='output.cfdb',
    variables=['PWAT', 'VIMF_U', 'VIMF_V', 'IVT'],
)
```

- `PWAT` / `PWAT_TR` -- total / tracer precipitable water
- `VIMF_U` / `VIMF_V` -- eastward / northward vertically integrated moisture flux
- `VIMF_TR_U` / `VIMF_TR_V` -- tracer moisture flux components
- `IVT` -- integrated vapour transport magnitude

### Native passthrough vs. computed fallback

Some variables can be obtained either directly from the source file or reconstructed from 3D fields.
Recent WRF builds (`wrf-auto-runs-intel-wvt:1.12`+) emit these fields directly, so they are read as a
native passthrough. On older `wrfout` files that lack them, the same cfdb output is computed on the
fly from the variable's fallback 3D fields.

Resolution is automatic and per-variable: if the native source variable is present in the file it is
used as-is; otherwise the fallback source variables and transform are applied. No flag or
configuration is required -- the output cfdb variable is identical either way.

- **Native or computed fallback:** `SLP` (native, else hypsometric from `PSFC`/`T2`/`HGT`),
  `PWAT`, `PWAT_TR`, `VIMF_U`, `VIMF_V` (native, else integrated from `QVAPOR`/wind/pressure)
- **Native only** (require a WRF build that emits them): `VIMF_TR_U`, `VIMF_TR_V`, `IVT`

### Forecast mode

A WRF run can be stored as one init of a `grid_forecast` dataset -- `(forecast_reference_time, forecast_period, level, y, x)` -- and successive runs appended:

```python
wrf.convert('forecasts.cfdb', variables=['T2', 'RAIN', 'U10', 'V10'], dataset_type='grid_forecast')
# init from SIMULATION_START_DATE (else START_DATE, else the first timestep); leads are hours since it
wrf_next.convert('forecasts.cfdb', variables=['T2', 'RAIN', 'U10', 'V10'], dataset_type='grid_forecast',
                 forecast_reference_time='2026-09-13T12')
```

The target may also be an open cfdb `Dataset` / `EDataset` handle. Every (init, variable, level) lead-span is buffered and written once, so keep forecast-mode ingests to 2-D variables (each buffered span is `n_lead_in_call × ny × nx`). Relative humidity is a 0-1 fraction. See [Forecast Datasets](forecast-datasets.md).

An init can also be ingested **one file at a time** as a running forecast produces them:

```python
axis = range(0, 145)
WrfIngest(day1).convert(ds, variables=['T2', 'PREC_ACC', 'WIND10'], dataset_type='grid_forecast',
                        leads=axis, chunk_shape=(1, 1, 1, ny, nx), mark_complete=False)
WrfIngest(day2).convert(ds, variables=['T2', 'PREC_ACC', 'WIND10'], dataset_type='grid_forecast',
                        leads=axis, chunk_shape=(1, 1, 1, ny, nx), mark_complete=False)
# ... until forecast.missing_chunks(ds, init) == [], then mark (forecast_archive.push_and_mark)
```

`leads=` fixes the full axis at creation, `mark_complete=False` keeps the init a back-fill target, one lead per chunk keeps every call to its own chunks, and `PREC_ACC` (not `RAIN`) keeps precipitation stateless across files. Details in [Forecast Datasets](forecast-datasets.md#incremental-inits-050).

### Custom chunk shape

All variables are stored as 4D. The output chunk shape defaults to `(1, 1, ny, nx)`. Override:

```python
wrf.convert(
    cfdb_path='output.cfdb',
    variables=['T'],
    target_levels=[100.0, 500.0],
    chunk_shape=(1, 1, 50, 50),  # (time, z, y, x)
)
```

With `squeeze_height=True` a 3-D `(time, y, x)` chunk shape is accepted (a 3-D shape without it is
refused). Every output chunk is written exactly once per call, also when `start_date` drops leading
frames (0.6.0: write blocks are aligned to the output chunk grid; before, a dropped first frame rewrote
the first chunk once per timestep).

Values a variable's packed dtype cannot store (e.g. more than 654.35 mm in an hour for `precipitation`,
uint16 at 0.01 mm) are refused before writing (0.6.0); cfdb would otherwise store them as missing.

### Building a long dataset in bands (0.6.0)

`convert(..., extend=True)` (grid mode, WRF) writes into an existing dataset -- or creates it -- instead of
replacing it, so a decades-long record can be built in time bands, and extended forward or backward later.
`cfdb_path` may be a path or an open cfdb `Dataset` / `EDataset` handle.

```python
import cfdb
import numpy as np
from cfdb_ingest import WrfIngest, grid

KW = dict(variables=['PREC_ACC'], extend=True, time_label='start', squeeze_height=True,
          chunk_shape=(840, 24, 24))
with cfdb.open_dataset('precip.cfdb') as ds:          # after the first band exists
    bands = grid.time_bands(grid.time_anchor(ds), '1980-01-01', '1990-07-01', 840, 60)
for b0, b1 in bands:
    files = ...  # the wrfout files holding frames (b0, b1]
    r = WrfIngest(files).convert('precip.cfdb', start_date=str(b0), end_date=str(b1 - np.timedelta64(1, 'h')), **KW)
    if r['missing_frames']:
        print(b0, 'missing', len(r['missing_frames']))   # re-run this band once the files exist
```

- **Time axis.** Created with an explicit numeric step (never `step=True`, which stores no step for a
  one-frame axis). A window may extend either end of the stored axis -- across a gap too: cfdb auto-fills
  the skipped slots as placeholders, never written and read as missing -- or overwrite stored times (a
  placeholder, an interior band, a re-run), in any order. A window extending both ends at once, or off the
  stored step grid, is refused, as is a target whose time axis has no step. The step is the accumulation
  interval with `time_label='start'`, else the smallest spacing of all input frames.
- **Missing frames are written as missing and reported.** The window covers every step from `start_date` to
  `end_date` (default: the first and last frame present); frames absent from the input are left unwritten
  (read as missing) and listed in the result's `missing_frames`. Re-run the window once the files exist --
  writes are idempotent overwrites -- to fill them. Chunks may therefore be partly written; nothing tracks
  completeness, so check the values where it matters (e.g. no missing values in a published range).
  `RAIN` (differenced totals) is refused across a hole, which would span several intervals.
- **Chunks.** Bands are aligned to the dataset's absolute time index 0 (its first time at creation), not the
  calendar: use `grid.time_bands(grid.time_anchor(ds), ...)`. Every output chunk is written once per call.
  An existing variable keeps its chunk shape and encoding (a mismatch is refused before anything is
  written; `chunk_shape=None` uses the stored one). The CRS, `x`/`y`, time step and time labelling must
  match the target. `grid.missing_chunks(ds, start, end)` lists chunks never written (key presence, no fetch).
- **Re-runs** overwrite and are idempotent, but the store is log-structured: run `ds.prune()` afterwards to
  reclaim the old chunk versions.
- `convert` returns `{'status', 'time_index', 'n_times', 'n_new', 'gap_filled', 'missing_frames',
  'variables'}` in grid mode.

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
| `--compression` | | cfdb compression, for new datasets: `zstd_shuffle`, `zstd`, `lz4_shuffle` or `lz4` (default: cfdb's own, `zstd_shuffle` from cfdb 0.10) |
| `--forecast` | | Treat the files as ONE forecast run and write a `grid_forecast` dataset (appends to an existing one) |
| `--init` | | Forecast mode: the run's init (ISO); default `SIMULATION_START_DATE` / `START_DATE` |
| `--forecast-step-minutes` | | Forecast mode: init step baked into a new dataset (default 360) |
| `--overwrite` | | Forecast mode: replace an init the dataset already holds complete |
| `--leads` | | Forecast mode: the FULL lead axis (hours) for a NEW dataset as `start:stop:step` (stop inclusive, e.g. `0:144:1`) when this call holds part of the run |
| `--no-mark-complete` | | Forecast mode: a partial call -- leave the init unmarked |

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
