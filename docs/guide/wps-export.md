# WPS Export

cfdb-ingest can export cfdb datasets to WPS intermediate format files for use with metgrid.exe. This enables a workflow where WRF output is first ingested into cfdb, then exported to WPS format for driving a new subdomain in a different coordinate system.

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

This produces files named `WRF:2023-02-10_00`, `WRF:2023-02-10_06`, etc.

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
| `--start-date` | `-s` | Starting date-time to convert (required) |
| `--end-date` | `-e` | Ending date-time to convert (required) |
| `--hour-interval` | `-h` | Interval in hours between records (default: 6) |
| `--prefix` | `-p` | Output file prefix (default: `WRF`) |

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
```

## Implementation Notes

### WPS intermediate file format

The WPS intermediate file writer was originally based on code from [era5_to_int](https://github.com/era5_to_int). During testing with metgrid.exe, several issues were discovered that only manifested with non-LATLON projections:

**Projection codes**: The WPS intermediate file format uses different integer codes than WPS internal codes:

| Projection | File code |
|---|---|
| Lat-Lon | 0 |
| Mercator | 1 |
| Lambert Conformal | 3 |
| Gaussian | 4 |
| Polar Stereographic | 5 |
| Cassini | 6 |

**dx/dy units**: metgrid reads dx/dy from the file and multiplies by 1000 -- values must be in **km**, not meters.

**earth_radius units**: Same convention -- metgrid multiplies by 1000, so the value must be in **km** (6371.229), not meters (6371229.0).
