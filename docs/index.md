# cfdb-ingest

Convert meteorological model output to [cfdb](https://github.com/mullenkamp/cfdb) with standardized CF conventions.

## Overview

cfdb-ingest converts meteorological file formats (netCDF4/HDF5) from various model outputs into cfdb. It standardizes variable names and attributes to be consistent with [CF conventions](https://cfconventions.org/), making it straightforward to work with datasets from different sources through a single interface.

## Supported Sources

| Source | Class | Description |
|--------|-------|-------------|
| WRF (wrfout) | `WrfIngest` | All variables in one file per time range. Lambert Conformal, Polar Stereographic, Mercator, and Lat-Lon projections. |
| ERA5 (NCAR) | `Era5Ingest` | One variable per file. Surface, pressure level, and invariant products on a regular lat-lon grid (EPSG:4326). |
| IFS (ECMWF open data) | `IfsIngest` | One forecast cycle of GRIB2 files, into a `grid_forecast` dataset (init × lead). Requires the `ifs` extra. |

## Key Features

- **Automatic variable mapping** -- source variable names are translated to CF-standard names with proper metadata via [cfdb-vars](https://github.com/mullenkamp/cfdb-vars)
- **Named height coordinates** -- surface variables at specific heights (0m, 2m, 10m, 100m) get their own named coordinates (e.g. `height_2m`), coexisting cleanly with pressure-level variables; a name that also exists on levels, or at more than one height, is stored with a height suffix (`air_temperature_2m`, `u_wind_10m` / `u_wind_100m`)
- **Forecast datasets** -- a WRF run or an IFS cycle stored as `grid_forecast` `(forecast_reference_time, forecast_period, level, y, x)`, appended one init at a time with every chunk written exactly once (see [Forecast Datasets](guide/forecast-datasets.md))
- **Wind rotation** (WRF) -- grid-relative wind components are rotated to earth-relative
- **VIMF computation** (ERA5) -- native calculation of vertically integrated moisture flux from Q, U, and V
- **3D level interpolation** (WRF) -- eta-level variables are interpolated to user-specified height or pressure levels
- **Auto pressure level detection** (ERA5) -- pressure levels are read directly from source files
- **Split or combined output** (ERA5) -- create one cfdb per variable or combine into a single dataset
- **Soil variables** -- soil moisture and temperature stored on a depth coordinate
- **WPS intermediate file export** -- convert cfdb datasets (`grid`, or one init of a `grid_forecast`) to WPS intermediate format for metgrid.exe
- **One relative-humidity convention** -- a 0-1 fraction in every source (cfdb-vars `relative_humidity`); the WPS exporter writes percent
- **Spatial and temporal filtering** -- subset by bounding box (WGS84) and/or date range
- **Multi-file support** -- seamlessly spans multiple input files
- **Configurable chunking** -- tune output chunk shapes for different access patterns

## High Performance

`cfdb-ingest` is optimized for processing high-resolution meteorological datasets:

- **Vectorized processing** -- transformations and vertical integrations (like VIMF) are fully vectorized using NumPy.
- **Efficient I/O** -- utilizes the `rechunkit` engine to minimize HDF5 read operations when extracting spatial and temporal subsets.
- **Intelligent Caching** -- manages HDF5 C-level chunk caches and per-timestep Python caches to eliminate redundant data access.
- **Parallel Startup** -- multi-threaded file scanning ensures fast initialization even with thousands of input files.
