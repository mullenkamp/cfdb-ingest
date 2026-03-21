# cfdb-ingest

Convert meteorological model output to [cfdb](https://github.com/mullenkamp/cfdb) with standardized CF conventions.

## Overview

cfdb-ingest converts meteorological file formats (netCDF4/HDF5) from various model outputs into cfdb. It standardizes variable names and attributes to be consistent with [CF conventions](https://cfconventions.org/), making it straightforward to work with datasets from different sources through a single interface.

## Key Features

- **Automatic variable mapping** -- source variable names are translated to CF-standard names with proper metadata (standard_name, units, encoding)
- **Wind rotation** -- grid-relative wind components are rotated to earth-relative using COSALPHA/SINALPHA
- **3D level interpolation** -- eta-level variables are interpolated to user-specified height or pressure levels
- **Pressure-level mode** -- optionally interpolate 3D variables to pressure levels (Pa) for use with WPS/metgrid
- **Soil variables** -- soil moisture and temperature stored on a depth coordinate
- **WPS intermediate file export** -- convert cfdb datasets to WPS intermediate format for metgrid.exe
- **Spatial and temporal filtering** -- subset by bounding box (WGS84) and/or date range before writing
- **Multi-file support** -- seamlessly spans multiple input files, including cross-file precipitation accumulation
- **Configurable chunking** -- tune output chunk shapes for different access patterns

## Supported Formats

| Source | Class | CRS Projections |
|--------|-------|-----------------|
| WRF (wrfout) | `WrfIngest` | Lambert Conformal Conic, Polar Stereographic, Mercator, Lat-Lon |
