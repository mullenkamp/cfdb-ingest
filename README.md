# cfdb-ingest

<p align="center">
    <em>File format conversions to cfdb</em>
</p>

[![build](https://github.com/mullenkamp/cfdb-ingest/workflows/Build/badge.svg)](https://github.com/mullenkamp/cfdb-ingest/actions)
[![codecov](https://codecov.io/gh/mullenkamp/cfdb-ingest/branch/master/graph/badge.svg)](https://codecov.io/gh/mullenkamp/cfdb-ingest)
[![PyPI version](https://badge.fury.io/py/cfdb-ingest.svg)](https://badge.fury.io/py/cfdb-ingest)

---

**Source Code**: <a href="https://github.com/mullenkamp/cfdb-ingest" target="_blank">https://github.com/mullenkamp/cfdb-ingest</a>

---
## Overview
This repo contains a python package with tools to convert file types (e.g. netcdf4/hdf5, grib2, etc) formatted according to different organizations or   
model outputs (e.g. WRF output, ERA5 data from the ECMWF) and convert them to cfdb. It standardize the variable names and the attributes/metadata to be consistent with the CF conventions and cfdb. The python package will make it easier to utilize the power of cfdb from external datasets with varied formatting.

## Development

### Setup environment

We use [UV](https://docs.astral.sh/uv/) to manage the development environment and production build. 

```bash
uv sync
```

### Run unit tests

You can run all the tests with:

```bash
uv run pytest
```

### Format the code

Execute the following command to apply linting and check typing:

```bash
uv run lint
```

## License

This project is licensed under the terms of the Apache Software License 2.0.
