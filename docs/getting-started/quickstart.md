# Quick Start

## WRF

### Python API

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

### CLI

```bash
cfdb-ingest wrf wrfout_d01_2023-02-12_00:00:00.nc output.cfdb \
    -v T2,WIND10 \
    -s 2023-02-12T06:00 \
    -e 2023-02-12T18:00
```

### WPS Export

To prepare data for WPS/metgrid, use the `--preset wps` flag which auto-selects all required variables and pressure levels:

```bash
# Step 1: Ingest
cfdb-ingest wrf /path/to/wrfout/ output.cfdb --preset wps \
    -s 2023-02-10 -e 2023-02-10_06

# Step 2: Export to WPS intermediate format
cfdb-to-int output.cfdb -s 2023-02-10 -e 2023-02-10_06
```

## ERA5

### Python API

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

### CLI

```bash
# Combined: multiple variables in one cfdb
cfdb-ingest era5 /path/to/era5/*.nc output.cfdb \
    -v SP,VAR_2T,T,U,V \
    -s 2020-01-01 -e 2020-01-31

# Split: one cfdb file per variable
cfdb-ingest era5 /path/to/era5/*.nc /output/dir/ --split \
    -v SP,T
```

See the [WRF Ingestion](../guide/wrf-ingestion.md), [ERA5 Ingestion](../guide/era5-ingestion.md), and [WPS Export](../guide/wps-export.md) guides for full details.
