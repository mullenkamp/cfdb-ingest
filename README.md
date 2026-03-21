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

Key features:

- **Automatic variable mapping** -- source variable names are translated to CF-standard names with proper metadata
- **Wind rotation** -- grid-relative wind components are rotated to earth-relative
- **3D level interpolation** -- eta-level variables are interpolated to user-specified height or pressure levels
- **WPS intermediate file export** -- convert cfdb datasets to WPS intermediate format for metgrid.exe
- **Spatial and temporal filtering** -- subset by bounding box and/or date range
- **Multi-file support** -- seamlessly spans multiple input files

## Installation

Requires Python >= 3.10.

```bash
pip install cfdb-ingest
```

## Quick Start

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

See the [full documentation](https://mullenkamp.github.io/cfdb-ingest/) for details.

## License

This project is licensed under the terms of the Apache Software License 2.0.
