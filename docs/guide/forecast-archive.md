# Forecast Archive

`cfdb_ingest.forecast_archive` is the shared protocol for a pipeline that keeps a `grid_forecast`
dataset on S3 (an ebooklet-backed EDataset): opening the target, refusing to create one by accident,
a rolling-window retention policy, a push that never advertises an init whose chunks did not land,
and the process guards that make an interrupted run recoverable. It needs the `archive` extra
(`pip install cfdb-ingest[archive]`).

It was lifted from `ifs-download` (reviewed dual-blind there) so that every forecast pipeline -- the
IFS downloader, the WRF forecast ingest -- shares one implementation.

## Opening

```python
from cfdb_ingest import forecast_archive as fa

remote = {'access_key_id': ..., 'access_key': ..., 'bucket': ..., 'endpoint_url': ..., 'db_key': ...}
with fa.open_target(remote, '/scratch/wrf_fc.cfdb', 'c', lock_timeout=120) as ds:
    fa.refuse_create(ds, allow_create=False)      # a scheduled run never creates the dataset
    ...
with fa.open_target('https://.../public.cfdb', '/scratch/status.cfdb', 'r') as ds:   # public URL: read-only, no lock
    done = fa.done_inits(ds)
```

`open_target` takes an `S3Connection` kwargs dict, a public `db_url` string (`flag='r'` only), or
`None` for a plain local file; `flag='r'` on a target that does not exist yields `None`. The handle
is always closed on exit -- for an EDataset that is what releases the remote write lock.

## Retention

`inits_to_drop(complete, keep_days, protect, min_keep)` is a pure function returning
`(cutoff, dropped)` for a prefix cut of `forecast_reference_time`; `apply_retention(ds, ...)` applies
it, removes the dropped inits from `complete_inits` and appends a history line. Auto-filled empty
slots are not inits; `min_keep` complete inits always survive; `protect` lowers the cutoff below an
init written this run. `truncate` is positional, so an unmarked init older than the cutoff is dropped
with the marked ones.

## Publishing: two commits

```python
fa.push_and_mark(ds, [init])     # unmark -> push chunks (one forced retry) -> mark -> push metadata
```

ebooklet commits metadata on any partially successful push, so the completion marker must never
ride the chunk push. On failure (`PushFailed`) the init stays unmarked on the remote and the local
file keeps the journal; `recover(ds, inits)` finishes the job next run. An **incremental** writer
pushes its partial calls with `push_twice(ds)` (the init is unmarked throughout) and calls
`push_and_mark` once, after `forecast.missing_chunks(ds, init)` is empty.

## Process guards

- `run_lock(work_dir)` -- an exclusive flock for one pipeline's writes to one archive.
- `install_sigterm_handler()` -- SIGTERM becomes `SystemExit`, so `with` blocks close their dataset and
  release the remote lock (a SIGKILLed process cannot; see below).
- `write_sidecar` / `read_sidecar` / `clear_sidecar` -- record the stage a run is in (`'ingest'`,
  `'push'`) and the inits involved, for the next run to finish or discard.
- `break_locks(remote)` -- remove every other lock ticket on the db object. `force_lock=True` on open
  breaks only tickets older than two hours; a ticket left by a process of your own that died minutes
  ago (a hook killed on timeout) blocks every later open until then. Safe only for a writer that is
  the single writer by construction (job flock + one job per archive at a time).

What is deliberately *not* here: the IFS-specific completeness gates around `IfsIngest.convert`
(they assume a whole init per call) and the "discard the working file of an interrupted ingest"
recovery step (it assumes nothing reached the remote, false once an init is pushed incrementally) --
an incremental writer reconciles with `forecast.missing_chunks` instead.
