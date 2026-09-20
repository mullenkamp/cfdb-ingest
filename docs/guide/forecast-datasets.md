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

The `ForecastWriter` buffers exactly one lead-span per live (variable, level) -- the leads THIS
call supplies -- and writes it once, the moment its last lead arrives. Appending an init never
rewrites another init's chunks, and `prune()` after a fresh ingest reclaims nothing but cfdb's own
metadata rewrite. For per-file sources (WRF), every batch variable's span stays live until the last
file -- keep forecast-mode WRF ingests to 2-D variables.

For an init that is written **incrementally** (below) use a one-lead chunk, `chunk_shape=(1, 1, 1,
ny, nx)`: file boundaries are hours and chunks are indices, so only one lead per chunk guarantees
that a later call never rewrites an earlier call's chunk, whatever the output cadence — the writer
flushes contiguous runs of the leads it FILLED, never the gaps between them, so a source coarser than
the axis (3-hourly frames on an hourly axis) leaves the gaps absent (`missing_chunks` sees them) rather
than storing NaN chunks. A 3 km NZ
domain is ~0.7 MB per chunk; ~145 chunks per (init, variable).

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

## Incremental inits (0.5.0)

An init can be built up over several `convert()` calls -- one per daily wrfout file as a running
forecast writes them -- with three rules:

```python
axis = range(0, 145)                      # the FULL run, hourly
for path in daily_files:                  # as each one completes
    WrfIngest(path).convert(ds, variables=[...], dataset_type='grid_forecast',
                            leads=axis, chunk_shape=(1, 1, 1, ny, nx), mark_complete=False)
    ds.push()                             # the init is unmarked: a partial push is harmless
missing = forecast.missing_chunks(ds, init)   # [] when every chunk of the init is present
```

- **`leads=`** creates the dataset with the whole lead axis, because the axis is fixed at creation
  and every later call maps its subset onto it by value (`lead_index_map`). On an existing dataset it
  must equal the stored axis. The creating call's own leads must lie on it.
- **`mark_complete=False`** leaves the init unmarked, so `place_init` treats the next call as a
  back-fill (no `overwrite=` needed) and no `complete_inits`-gated reader can take the partial init
  for a whole one. Only the caller that has checked `missing_chunks` marks it -- through
  `forecast_archive.push_and_mark` for an EDataset (chunks first, marker second).
- **`forecast.missing_chunks(ds, init)`** enumerates every chunk key of the init and probes presence
  without fetching (`key in blt`); it is the completeness check, and the sweep a reader should run
  after `load()`ing an init from a remote (ebooklet skips keys absent from the remote index silently).

Precipitation for an incremental WRF ingest comes from `PREC_ACC` (WRF's own per-interval
accumulators), not `RAIN`: differencing running totals needs the previous frame, which the previous
file held. See [WRF Ingestion](wrf-ingestion.md#precipitation-and-accumulators).

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
`truncate` is positional: an unmarked init (a run that never finished) older than the cutoff is
dropped with the marked ones. The rolling-window policy itself -- `keep_days`, `min_keep`, protected
inits, the two-commit push -- lives in [`forecast_archive`](forecast-archive.md).

## Reading

`.interp()` raises `NotImplementedError` on forecast types (two non-spatial dims). Read one init
chunk-aligned -- `ds[var][init_idx, :, k, :, :]` -- rather than `.data` on a whole variable.
