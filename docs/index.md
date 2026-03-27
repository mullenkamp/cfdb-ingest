# cfdb-ingest

Convert meteorological model output to [cfdb](https://github.com/mullenkamp/cfdb) with standardized CF conventions.

## Overview

cfdb-ingest converts meteorological file formats (netCDF4/HDF5) from various model outputs into cfdb. It standardizes variable names and attributes to be consistent with [CF conventions](https://cfconventions.org/), making it straightforward to work with datasets from different sources through a single interface.

## Supported Sources

| Source | Class | Description |
|--------|-------|-------------|
| WRF (wrfout) | `WrfIngest` | All variables in one file per time range. Lambert Conformal, Polar Stereographic, Mercator, and Lat-Lon projections. |
| ERA5 (NCAR) | `Era5Ingest` | One variable per file. Surface, pressure level, and invariant products on a regular lat-lon grid (EPSG:4326). |

## Key Features

- **Automatic variable mapping** -- source variable names are translated to CF-standard names with proper metadata via [cfdb-vars](https://github.com/mullenkamp/cfdb-vars)
- **Named height coordinates** -- surface variables at specific heights (0m, 2m, 10m, 100m) get their own named coordinates (e.g. `height_2m`), coexisting cleanly with pressure-level variables
- **Wind rotation** (WRF) -- grid-relative wind components are rotated to earth-relative
- **3D level interpolation** (WRF) -- eta-level variables are interpolated to user-specified height or pressure levels
- **Auto pressure level detection** (ERA5) -- pressure levels are read directly from source files
- **Split or combined output** (ERA5) -- create one cfdb per variable or combine into a single dataset
- **Soil variables** -- soil moisture and temperature stored on a depth coordinate
- **WPS intermediate file export** -- convert cfdb datasets to WPS intermediate format for metgrid.exe
- **Spatial and temporal filtering** -- subset by bounding box (WGS84) and/or date range
- **Multi-file support** -- seamlessly spans multiple input files
- **Configurable chunking** -- tune output chunk shapes for different access patterns
