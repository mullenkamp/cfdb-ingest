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

## Overview

cfdb-ingest converts meteorological file formats (netCDF4/HDF5) from various model outputs into [cfdb](https://github.com/mullenkamp/cfdb). It standardizes variable names and attributes to be consistent with [CF conventions](https://cfconventions.org/), making it straightforward to work with datasets from different sources through a single interface.

Supported sources:

- **WRF** -- wrfout NetCDF files (all variables in one file per time range)
- **ERA5** -- NCAR ERA5 NetCDF files (one variable per file, surface + pressure level + invariant products)

Key features:

- **Automatic variable mapping** -- source variable names are translated to CF-standard names with proper metadata via [cfdb-vars](https://github.com/mullenkamp/cfdb-vars)
- **Named height coordinates** -- surface variables at specific heights (0m, 2m, 10m, 100m) get their own named coordinates (e.g. `height_2m`), allowing them to coexist with pressure-level variables without ambiguity
- **Wind rotation** (WRF) -- grid-relative wind components are rotated to earth-relative
- **VIMF computation** (ERA5) -- native calculation of vertically integrated moisture flux from Q, U, and V
- **3D level interpolation** (WRF) -- eta-level variables are interpolated to user-specified height or pressure levels
- **Auto pressure level detection** (ERA5) -- pressure levels are read directly from source files
- **Split or combined output** (ERA5) -- create one cfdb per variable or combine into a single dataset
- **WPS intermediate file export** -- convert cfdb datasets to WPS intermediate format for metgrid.exe
- **Spatial and temporal filtering** -- subset by bounding box and/or date range
- **Multi-file support** -- seamlessly spans multiple input files

## Performance

cfdb-ingest is designed for high-performance processing of large meteorological datasets:

- **Vectorized rechunking** -- utilizes [rechunkit](https://github.com/mullenkamp/rechunkit) for optimized HDF5 reads, even when extracting small spatial subsets across many timesteps.
- **Parallel initialization** -- multi-threaded file scanning and metadata extraction for fast startup.
- **HDF5 Chunk Caching** -- intelligent management of the HDF5 chunk cache to prevent redundant I/O during per-timestep transformations.
- **Synchronized multi-variable rechunking** -- synchronized iteration for derived variables (like VIMF) to eliminate redundant reads of shared source variables.

## Installation

Requires Python >= 3.10.

```bash
pip install cfdb-ingest
```

## Quick Start

### WRF

```python
from cfdb_ingest import WrfIngest

wrf = WrfIngest('wrfout_d01_2023-02-12_00:00:00.nc')
wrf.convert(
    cfdb_path='output.cfdb',
    variables=['T2', 'WIND10'],
    start_date='2023-02-12T06:00',
    end_date='2023-02-12T18:00',
)
```

```bash
cfdb-ingest wrf wrfout_d01_*.nc output.cfdb -v T2,WIND10 -s 2023-02-12T06:00 -e 2023-02-12T18:00
```

For WPS export, use the `--preset wps` flag:

```bash
cfdb-ingest wrf /path/to/wrfout/ output.cfdb --preset wps -s 2023-02-10 -e 2023-02-10_06
cfdb-to-int output.cfdb -s 2023-02-10 -e 2023-02-10_06
```

### ERA5

```python
from cfdb_ingest import Era5Ingest

era5 = Era5Ingest('/path/to/era5/*.nc')
era5.convert(
    cfdb_path='era5.cfdb',
    variables=['SP', 'VAR_2T', 'T', 'U', 'V'],
    start_date='2020-01-01',
    end_date='2020-01-31',
)
```

```bash
# Combined: multiple variables in one cfdb
cfdb-ingest era5 /path/to/era5/*.nc output.cfdb -v SP,VAR_2T,T,U,V -s 2020-01-01 -e 2020-01-31

# Split: one cfdb file per variable
cfdb-ingest era5 /path/to/era5/*.nc /output/dir/ --split -v SP,T
```

See the [full documentation](https://mullenkamp.github.io/cfdb-ingest/) for details.

## License

This project is licensed under the terms of the Apache Software License 2.0.
