# Quick Start

## Python API

```python
from cfdb_ingest import WrfIngest

wrf = WrfIngest('wrfout_d01_2023-02-12_00:00:00.nc')

wrf.convert(
    cfdb_path='output.cfdb',
    variables=['T2', 'WIND10', 'precip'],
    start_date='2023-02-12T06:00',
    end_date='2023-02-12T18:00',
)
```

## CLI

```bash
cfdb-ingest wrf wrfout_d01_2023-02-12_00:00:00.nc output.cfdb \
    -v T2,WIND10 \
    -s 2023-02-12T06:00 \
    -e 2023-02-12T18:00
```

## WPS Export

To prepare data for WPS/metgrid, use the `--preset wps` flag which auto-selects all required variables and pressure levels:

```bash
# Step 1: Ingest
cfdb-ingest wrf /path/to/wrfout/ output.cfdb --preset wps \
    -s 2023-02-10 -e 2023-02-10_06

# Step 2: Export to WPS intermediate format
cfdb-to-int output.cfdb -s 2023-02-10 -e 2023-02-10_06
```

See the [WRF Ingestion](../guide/wrf-ingestion.md) and [WPS Export](../guide/wps-export.md) guides for full details.
