# IFS Ingestion

`IfsIngest` reads one **ECMWF IFS open-data forecast cycle** (GRIB2) and appends it to a cfdb
`grid_forecast` dataset. It needs the `ifs` extra:

```bash
pip install 'cfdb-ingest[ifs]'      # eccodes + eccodeslib (bundled libeccodes with CCSDS support)
```

## The source

ECMWF publishes its 0.25° IFS forecasts as open data (CC-BY-4.0) on AWS (`s3://ecmwf-forecasts`) and
`https://data.ecmwf.int/forecasts`, one GRIB2 file per forecast step:

```
YYYYMMDD/HHz/ifs/0p25/oper/YYYYMMDDHH0000-<step>h-oper-fc.grib2   (+ .index)
```

An `IfsIngest` takes the files of ONE cycle (any split of that cycle's messages -- whole step files,
or byte-range subsets assembled from the `.index`) and refuses files that mix inits.

Encoding facts the reader handles (verified against the production files):

- the grid's first longitude is 180°E, so the array seam sits on the dateline -- longitudes are
  rolled to run 0 → 360 and a bounding box may cross the dateline;
- latitudes scan north → south and are flipped to ascending;
- `grid_ccsds` packing (why `eccodeslib` is required);
- soil layers are `typeOfLevel=soilLayer` with layer **indices** 1..4;
- `sithick` is bitmapped over land (missing → no ice);
- orography (`z` on `surface`) exists only in the 0 h message set (broadcast to every lead);
- the IFS `r` field ranges outside 0-100 % and is not used: RH is diagnosed from `q` and `t`;
- 00/12z cycles are 3-hourly to 144 h then 6-hourly -- pass `max_lead_hours=144`.

## Usage

```bash
cfdb-ingest ifs /data/ifs/2026091300/ nz_ifs.cfdb --preset wps --bbox 142,-54,192,-14 --max-lead-hours 144
```

```python
from cfdb_ingest.ifs import IfsIngest, IFS_WPS_PRESET_KEYS

ingest = IfsIngest('/data/ifs/2026091300/')        # pass 1: index every message's headers
print(ingest.init, ingest.leads, ingest.pressure_levels)
result = ingest.convert('nz_ifs.cfdb', variables=IFS_WPS_PRESET_KEYS,
                        bbox=(142, -54, 192, -14), max_lead_hours=144)
# {'init': '2026-09-13T00:00', 'init_index': 0, 'status': 'new', 'autofilled': 0,
#  'n_leads': 49, 'variables': [...], 'chunk_writes': 106}
```

`variables` accepts mapping keys (`T2`), GRIB shortNames (`2t`) or cfdb names (`air_temp`); see
[IFS Variables](../reference/ifs-variables.md). `target_levels` (Pa) subsets the native levels
without interpolation.

## Appending cycles

The target may be an existing dataset (same bbox / levels) or an open cfdb handle:

```python
from cfdb import open_edataset

with open_edataset(remote_conn, 'nz_ifs.cfdb', flag='w') as ds:
    IfsIngest('/data/ifs/2026091312/').convert(ds, variables=IFS_WPS_PRESET_KEYS,
                                               bbox=(142, -54, 192, -14), max_lead_hours=144)
    ds.push()
    ds.prune()
```

Rules (see [Forecast Datasets](forecast-datasets.md) for the full set):

- a **path** to a remote-backed (S3) file is refused -- pass the `open_edataset` handle, whose
  `push()` / `prune()` the caller owns;
- a missed cycle leaves an auto-filled slot on the init axis (six-hourly step), which a later
  ingest of that cycle fills (`status='backfill'`);
- an init the dataset already holds **complete** is immutable unless `overwrite=True`;
- every (init, variable, level) chunk-row is written exactly once (`chunk_writes` in the result).

Peak memory is one global field plus one chunk-row `(n_lead, ny, nx)`. Size scales with the box:
the 49-lead cycle for `bbox=(142, -54, 192, -14)` -- 201 x 161 points at 0.25 deg, all 14 levels,
the WPS preset plus the extras bundle, 29 variables -- is ~220 MB (packed templates; see
[IFS Variables](../reference/ifs-variables.md) for the encodings and the range guard).

Two things the ingest does **not** check, because it cannot: that every step of the cycle is on
disk, and that every step is complete. `leads` come from the files it is given, so a cycle with one
step file missing ingests as 48 leads and is marked complete. A downloader must therefore pass the
explicit step files (not the directory), compare `IfsIngest(files).leads` with the step set it
expects **before** `convert`, and check `result['n_leads']` after. The message set it has to fetch
comes from the same mapping the ingest uses:

```python
from cfdb_ingest.ifs import required_messages
required_messages(IFS_WPS_PRESET_KEYS)
# {('pl', 't', False), ('pl', 'u', False), ..., ('sfc', 'z', True), ('soil', 'sot', False), ...}
```

`category` is `pl` (every pressure level), `soil` (every layer) or `sfc`; `invariant=True` marks a
source that exists in the 0 h file only.

The ingest refuses, rather than guesses at, three upstream changes (0.4.2): a source variable in
other GRIB `units` than `cfdb_ingest.ifs.IFS_SOURCE_UNITS` lists, the same field/level/step twice,
and a source on a level type it cannot place (`snowLayer` in IFS Cycle 50r2). Each raises a
`ValueError` naming the file and message; the fix is a mapping change, never a silent conversion.

Synthetic cycles for tests come from `cfdb_ingest.ifs_synthetic.write_cycle(out_dir, init, steps,
levels_hpa)` -- real GRIB2 with the production quirks (dateline seam, CCSDS packing, `soilLayer`
indices, `sithick` bitmap, 0 h-only orography) and closed-form values for assertions.

## Downstream: WRF forcing

```bash
cfdb-to-int nz_ifs.cfdb --init 2026-09-13T00 -h 3 -p /run/IFS
```

writes `IFS:2026-09-13_00`, `IFS:2026-09-13_03`, ... for `metgrid.exe` -- see [WPS Export](wps-export.md).
