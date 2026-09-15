# WPS Export

cfdb-ingest can export cfdb datasets to WPS intermediate format files for use with metgrid.exe. Two
workflows use it: WRF output ingested into cfdb and exported to drive a new subdomain in a different
coordinate system, and an [IFS forecast cycle](ifs-ingestion.md) ingested into a `grid_forecast`
dataset and exported to force a WRF forecast.

Variables are matched by their stored cfdb-vars names after stripping any height suffix
(`air_temperature`, `air_temperature_2m` and the short `air_temp` all resolve alike), 4-D surface
variables `(time, height_Xm, y, x)` are handled, relative humidity is written in percent from the
stored 0-1 fraction, and `SKINTEMP` comes from `skin_temperature` (IFS) or `soil_temperature` (WRF `TSK`)
-- a dataset holding both is refused rather than first-matched.
When one quantity is stored at several heights (IFS `u_wind_10m` and `u_wind_100m`), the WPS
surface field takes the screen / anemometer one -- 2 m for `TT`/`DEWPT`/`RH`, 10 m for `UU`/`VV` --
and the others are simply not exported; two heights with neither the WPS one is refused.

## The WPS Preset

The `--preset wps` flag for the `cfdb-ingest wrf` command auto-selects everything needed for WPS export:

- **All required variables** (see list below)
- **Pressure-level interpolation** (`--vertical-coord pressure`)
- **26 standard pressure levels** (1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10 hPa)

The user can override the pressure levels with `--target-levels` and add extra variables with `--variables` (they are merged with the preset list).

### Variables included in the WPS preset

**3D pressure-level fields:**

| Key | Description |
|-----|-------------|
| `T` | Temperature (from potential temperature) |
| `U` | U-wind component (unstaggered, earth-relative) |
| `V` | V-wind component (unstaggered, earth-relative) |
| `GHT` | Geopotential height |
| `RH` | Relative humidity |
| `Q_SH` | Specific humidity |

**Surface fields:**

| Key | Description |
|-----|-------------|
| `PSFC` | Surface pressure |
| `SLP` | Mean sea level pressure |
| `TSK` | Skin temperature |
| `T2` | 2m temperature |
| `U10` | 10m U-wind (earth-relative) |
| `V10` | 10m V-wind (earth-relative) |
| `TD2` | 2m dewpoint temperature |
| `RH2` | 2m relative humidity |
| `XLAND` | Land-sea mask |
| `HGT` | Terrain height |
| `SNOWH` | Physical snow depth |
| `SST_VAR` | Sea surface temperature |
| `SEAICE_VAR` | Sea ice fraction |
| `SNOW_VAR` | Snow water equivalent |

**Soil fields:**

| Key | Description |
|-----|-------------|
| `SMOIS` | Soil moisture (per layer) |
| `TSLB` | Soil temperature (per layer) |

## Full Pipeline

### Step 1: Ingest WRF output

```bash
cfdb-ingest wrf /path/to/wrfout/ output.cfdb \
    --preset wps \
    -s 2023-02-10T00:00 -e 2023-02-10T06:00
```

### Step 2: Export to WPS intermediate format

```bash
cfdb-to-int output.cfdb -s 2023-02-10 -e 2023-02-10_06 -h 6
```

This produces files named `WRF:2023-02-10_00`, `WRF:2023-02-10_06`, etc. The prefix may include a
directory (`-p /run/WRF`).

### Forecast datasets: one init

For a `grid_forecast` dataset select the init; one file is written per lead, named by the valid time:

```bash
cfdb-to-int nz_ifs.cfdb --init 2026-09-13T00 -h 3 -p IFS     # IFS:2026-09-13_00, IFS:2026-09-13_03, ...
```

`--start-date` / `--end-date` then filter by valid time. An init the dataset does not mark complete is
refused before any file is written.

### Step 3: Run metgrid.exe

Set `fg_name` in your `namelist.wps` to point to the output files:

```
&metgrid
 fg_name = '/path/to/output/WRF'
 ...
/
```

Then run `metgrid.exe` as usual.

## cfdb-to-int CLI

Available as both a subcommand and a standalone command:

```bash
# As subcommand:
cfdb-ingest cfdb-to-int output.cfdb -s 2023-02-10 -e 2023-02-10_06 -h 6

# As standalone command:
cfdb-to-int output.cfdb -s 2023-02-10 -e 2023-02-10_06 -h 6
```

### Options

```
cfdb-to-int [OPTIONS] CFDB_PATH
```

| Option | Short | Description |
|--------|-------|-------------|
| `--start-date` | `-s` | First valid time to convert (default: the first available) |
| `--end-date` | `-e` | Last valid time to convert (default: the last available) |
| `--hour-interval` | `-h` | Interval in hours between records (default: 6) |
| `--prefix` | `-p` | Output file prefix, may include a directory (default: `WRF`) |
| `--init` | `-i` | `grid_forecast` datasets: the forecast init to export (required there) |

## Python API

```python
from cfdb_ingest.cfdb_to_int import convert_cfdb_to_int
from datetime import datetime

convert_cfdb_to_int(
    cfdb_path='output.cfdb',
    output_prefix='WRF',
    start_date=datetime(2023, 2, 10),
    end_date=datetime(2023, 2, 10, 6),
    hour_interval=6,
)

# a forecast dataset: one init, every lead
convert_cfdb_to_int('nz_ifs.cfdb', output_prefix='IFS', init='2026-09-13T00', hour_interval=3)
```

Reads are chunk-aligned: a `grid` dataset is read one (time, level) slab at a time, a `grid_forecast`
one (init, level) chunk-row at a time.

## Implementation Notes

### WPS intermediate file writing

The WPS intermediate file format is handled by the [wrf_to_int](https://github.com/wrf_to_int) package, which provides the shared `IntermediateFile`, `Projections`, `MapProjection`, and `write_slab` tools used by both `cfdb-to-int` and `era5_to_int`. See the [wrf_to_int library API](https://github.com/wrf_to_int#library-api) for details on building custom converters.

Key conventions for the WPS intermediate file format:

- **Projection codes**: Lat-Lon=0, Mercator=1, Lambert Conformal=3, Gaussian=4, Polar Stereographic=5, Cassini=6
- **dx/dy**: must be in **km** (metgrid multiplies by 1000)
- **earth_radius**: must be in **km** (6371.229, not 6371229.0)
