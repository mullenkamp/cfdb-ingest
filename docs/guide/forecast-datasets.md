# Forecast Datasets

cfdb's `grid_forecast` type replaces the single `time` axis with a pair:
`forecast_reference_time` (the run's init) × `forecast_period` (the lead). cfdb-ingest writes every
forecast dataset -- from IFS (`IfsIngest`) or from WRF (`WrfIngest(...).convert(..., dataset_type='grid_forecast')`)
-- with one layout:

```
(forecast_reference_time, forecast_period, <pressure | height_Xm | depth>, y, x)
```

i.e. the grid layout `(time, z, y, x)` with `time` split into (init, lead). The rules live in
`cfdb_ingest/forecast.py` and are shared by both sources.

## Axis conventions (baked in at creation)

| coordinate | dtype | step | notes |
|---|---|---|---|
| `forecast_reference_time` | `datetime64[m]` | **explicit, in minutes** (IFS: 360 = six-hourly; WRF: `forecast_step_minutes`) | `step=True` would infer nothing from the single init a new dataset starts with; the step is what lets a missed cycle be back-filled |
| `forecast_period` | int32 | from the data (IFS 3 h, WRF 1 h) | `attrs['units'] = 'h'` -- cfdb has no timedelta dtype, so `init + lead` needs the unit; `forecast.valid_times(ds, init_idx)` does it right |

## Chunking and writes

Storage chunk = `(1, n_lead, 1, ty, tx)`: one init, every lead, one level, a spatial tile
(`forecast.forecast_chunk_shape` -- full extent when a row is ≤ 8 MB, else ~6 MB tiles). One
(variable, level, init) is therefore one chunk-row, which is both the natural write unit and the
natural read unit for WRF.

The `ForecastWriter` buffers exactly one chunk-row per live (variable, level) and writes it once,
the moment its last lead arrives. Appending an init never rewrites another init's chunks, and
`prune()` after a fresh ingest reclaims nothing but cfdb's own metadata rewrite. For per-file
sources (WRF), every batch variable's rows stay live until the last file -- keep forecast-mode WRF
ingests to 2-D variables.

## Appending, back-filling, overwriting

`convert(target, ...)` takes a path (created if missing, else appended to) or an open cfdb
`Dataset` / `EDataset` handle (never closed by the ingest; the caller pushes and prunes).

- A path whose file is **remote-backed** is refused: through a plain `open_dataset` a
  read-modify-write on a chunk that was never fetched would replace the remote chunk with blank + new.
  Through `open_edataset` ebooklet pulls the chunk first, so pass that handle.
- `status='new'`: appended; cfdb auto-fills skipped inits (`autofilled` counts them).
- `status='backfill'`: the slot exists but the init is not marked complete -- an auto-filled gap, or an
  ingest that did not finish. Writing into it is the recovery path.
- `status='overwrite'`: the init is complete and `overwrite=True` was passed; otherwise a complete init
  is immutable.
- Inits before the axis origin are refused (no prepending history); off-grid inits are refused, never snapped.

Completion is recorded in `ds.attrs['complete_inits']` when every variable of an init has been
written; `cfdb-to-int --init` refuses inits without it, so a crashed ingest cannot feed WRF a
partial boundary-condition set. Recovery story: the local file is scratch until `push()`; discard an
unpushed crash and re-pull, or re-run the ingest (it back-fills).

## Retention

```python
with open_edataset(remote_conn, 'nz_ifs.cfdb', flag='w') as ds:
    ds['forecast_reference_time'].truncate(start='2026-08-15T00:00')   # keep the step, drop old inits
    ds.push()
    ds.prune()
```

`truncate` keeps the step and deletes whole out-of-range chunks; deletes reach the remote only
through an `open_edataset` handle. Recommendation for EDatasets that receive one init per push:
create them with `num_groups=None` (per-key objects) so a push moves exactly the new init's chunks.

## Reading

`.interp()` raises `NotImplementedError` on forecast types (two non-spatial dims). Read one init
chunk-aligned -- `ds[var][init_idx, :, k, :, :]` -- rather than `.data` on a whole variable.
