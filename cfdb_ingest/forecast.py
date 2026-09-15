"""
Shared rules for cfdb ``grid_forecast`` targets: the (forecast_reference_time, forecast_period)
axis pair, appending one init at a time, and writing every (init, level) chunk exactly once.

Layout produced by every forecast-mode ingest in this package::

    (forecast_reference_time, forecast_period, <level coord>, y, x)

i.e. the grid layout ``(time, z, y, x)`` with ``time`` split into (init, lead). One init per
storage chunk, all leads in that chunk, one level, a spatial tile -- so the natural write unit
(one variable, one level, one init) and the natural read unit (one init for WRF) are both one
row of chunks, and appending an init never rewrites another init's chunks.

Two conventions decided once and baked into a dataset at creation:

- ``forecast_reference_time`` is ``datetime64[m]`` with an EXPLICIT step in minutes (six-hourly
  cycles: 360). ``step=True`` infers nothing from a single-value axis, and the step is what lets a
  missed cycle be back-filled later (cfdb auto-fills the skipped slot on append).
- ``forecast_period`` is an int32 lead with ``attrs['units']`` set (``'h'``); cfdb has no timedelta
  dtype, so ``init + lead`` is only meaningful once the unit is read back.

Chunk policy (cfdb): the write loop mirrors the chunk shape. ``ForecastWriter`` buffers exactly one
chunk-row per (variable, level) -- ``(n_lead, ny, nx)`` -- and flushes it the moment its last lead
arrives, so no chunk is written twice within one ingest call and nothing larger than one chunk-row
per live (variable, level) is ever held.
"""

import contextlib
import math
import pathlib
from typing import Optional, Union

import cfdb
import numpy as np

FRT = 'forecast_reference_time'
LEAD = 'forecast_period'
COMPLETE_INITS_ATTR = 'complete_inits'

_UNIT_TO_NP = {
    'h': 'h',
    'hour': 'h',
    'hours': 'h',
    'min': 'm',
    'minute': 'm',
    'minutes': 'm',
    's': 's',
    'second': 's',
    'seconds': 's',
    'D': 'D',
    'd': 'D',
    'day': 'D',
    'days': 'D',
}


######################################################
# Target handling


def is_dataset(obj) -> bool:
    """True for an open cfdb Dataset / EDataset handle (duck-typed)."""
    return hasattr(obj, 'create') and hasattr(obj, 'coord_names') and hasattr(obj, 'data_var_names')


@contextlib.contextmanager
def open_target(target, *, dataset_type: str = 'grid_forecast', **cfdb_kwargs):
    """
    Yield ``(ds, created)`` for a forecast target.

    ``target`` is either a path (opened with ``flag='c'``: created if missing, else appended to;
    closed here) or an already-open cfdb Dataset / EDataset handle (yielded as-is, never closed --
    the caller owns ``push()`` / ``prune()``).

    A path whose file was created as a remote (S3-backed) dataset is refused: through plain
    ``open_dataset`` a read-modify-write on a chunk that was never fetched would replace the remote
    chunk with blank + new. Pass the ``open_edataset`` handle instead.
    """
    if is_dataset(target):
        yield target, len(target.coord_names) == 0
        return

    path = pathlib.Path(target)
    created = not path.exists()
    ds = cfdb.open_dataset(str(path), 'c', dataset_type=dataset_type, allow_partial=True, **cfdb_kwargs)
    try:
        if not created and ds._sys_meta.remote:
            raise ValueError(
                f'{path} is a remote-backed (S3) cfdb file. Appending through a plain path would '
                f'blank chunks that were never fetched; open it with cfdb.open_edataset and pass the handle.'
            )
        yield ds, created
    finally:
        ds.close()


def validate_target(
    ds,
    *,
    x_name: str,
    y_name: str,
    x: np.ndarray,
    y: np.ndarray,
    levels: Optional[np.ndarray] = None,
    depths: Optional[np.ndarray] = None,
) -> None:
    """
    An existing target must be a ``grid_forecast`` with the forecast axis pair and the same spatial
    (and level / depth) coordinates as the incoming data. Raises ValueError naming the first mismatch.
    """
    if ds.dataset_type != 'grid_forecast':
        raise ValueError(f"target dataset_type is {ds.dataset_type!r}, expected 'grid_forecast'")
    for name in (FRT, LEAD, x_name, y_name):
        if name not in ds.coord_names:
            raise ValueError(f'target lacks the {name!r} coordinate')
    for name, values in ((x_name, x), (y_name, y), ('pressure', levels), ('depth', depths)):
        if values is None:
            continue
        if name not in ds.coord_names:
            raise ValueError(f'target lacks the {name!r} coordinate')
        stored = ds[name].data
        if len(stored) != len(values) or not np.allclose(stored, values):
            raise ValueError(
                f'target {name!r} coordinate does not match the incoming data '
                f'({len(stored)} vs {len(values)} values; first {stored[:3]} vs {np.asarray(values)[:3]})'
            )


######################################################
# Axis rules


def lead_step(leads) -> Optional[int]:
    """
    The single step of a lead axis (None for one lead). ValueError if the cadence is irregular --
    e.g. IFS 00/12z runs are 3-hourly to 144 h then 6-hourly, which one axis cannot hold.
    """
    leads = np.asarray(leads)
    if leads.size < 2:
        return None
    diffs = np.unique(np.diff(leads))
    if diffs.size != 1:
        raise ValueError(
            f'forecast leads are not regularly spaced (steps {diffs.tolist()}); '
            f'restrict the leads (e.g. max_lead_hours=144 for IFS open data) so one step covers them all'
        )
    return int(diffs[0])


def check_init_on_grid(init, origin, step_minutes: int) -> None:
    """
    Refuse (never snap) an init that is not a whole number of steps from the axis origin.
    cfdb refuses off-grid appends itself since 0.9.7; this gives the domain-specific message first.
    """
    delta = int((np.datetime64(init, 'm') - np.datetime64(origin, 'm')).astype('int64'))
    if delta % step_minutes:
        raise ValueError(
            f'{FRT} {init} is not on the declared grid: it is {delta} min from the axis origin {origin}, '
            f'not a multiple of the {step_minutes} min step. Refusing rather than snapping.'
        )


def create_forecast_coords(ds, init, leads, *, step_minutes: int, lead_units: str = 'h') -> None:
    """Create the (init, lead) axis pair on a new dataset."""
    ds.create.coord.forecast_reference_time(
        data=np.array([np.datetime64(init, 'm')], dtype='datetime64[m]'), step=int(step_minutes)
    )
    leads = np.asarray(leads, dtype='int32')
    step = lead_step(leads)
    lead = ds.create.coord.forecast_period(data=leads, step=step if step is not None else False)
    lead.attrs['units'] = lead_units


def lead_index_map(ds, leads) -> np.ndarray:
    """
    Position of each incoming lead on the stored ``forecast_period`` axis. Every incoming lead
    must exist on the axis (the chunk shape was sized to it); leads are matched by VALUE, never by
    a scalar offset, so a source whose interval differs from the stored step cannot be misplaced.
    """
    stored = np.asarray(ds[LEAD].data)
    leads = np.asarray(leads)
    pos = np.searchsorted(stored, leads)
    ok = (pos < len(stored)) & (stored[np.minimum(pos, len(stored) - 1)] == leads)
    if not ok.all():
        missing = leads[~ok].tolist()
        raise ValueError(
            f'leads {missing} are not on the stored {LEAD} axis {stored.tolist()} '
            f'(units {ds[LEAD].attrs.get("units")!r})'
        )
    return pos.astype('int64')


def valid_times(ds, init_idx: int) -> np.ndarray:
    """``init + lead`` for one init, honouring the lead axis's ``units`` attr (required)."""
    units = ds[LEAD].attrs.get('units')
    if units not in _UNIT_TO_NP:
        raise ValueError(
            f'{LEAD} has units {units!r}; expected one of {sorted(_UNIT_TO_NP)} -- a bare integer lead '
            f'cannot be added to a datetime without its unit'
        )
    init = np.datetime64(ds[FRT].data[init_idx], 'm')
    leads = np.asarray(ds[LEAD].data).astype('int64')
    return init + leads.astype(f'timedelta64[{_UNIT_TO_NP[units]}]')


def complete_inits(ds) -> list:
    attrs = ds.attrs.data
    return list(attrs.get(COMPLETE_INITS_ATTR) or [])


def is_init_complete(ds, init) -> bool:
    return str(np.datetime64(init, 'm')) in complete_inits(ds)


def mark_init_complete(ds, init) -> None:
    """Record that every variable of ``init`` has been written (the exporter refuses inits without it)."""
    key = str(np.datetime64(init, 'm'))
    inits = complete_inits(ds)
    if key not in inits:
        inits.append(key)
        ds.attrs[COMPLETE_INITS_ATTR] = sorted(inits)


def unmark_init_complete(ds, init) -> None:
    key = str(np.datetime64(init, 'm'))
    inits = complete_inits(ds)
    if key in inits:  # only touch the attrs record when there is something to remove
        ds.attrs[COMPLETE_INITS_ATTR] = [i for i in inits if i != key]


def append_history(ds, line: str) -> None:
    prev = ds.attrs.data.get('history')
    ds.attrs['history'] = f'{prev}\n{line}' if prev else line


def _init_has_chunks(ds, init_idx: int) -> bool:
    """Fallback probe for datasets predating the completion marker: does any data var hold a chunk for this init?"""
    for name in ds.data_var_names:
        dv = ds[name]
        if FRT not in dv.coord_names:
            continue
        sel = tuple(init_idx if c == FRT else 0 for c in dv.coord_names)
        if dv.get_chunk(sel, missing_none=True) is not None:
            return True
    return False


def place_init(ds, init, *, step_minutes: int, overwrite: bool = False) -> dict:
    """
    Locate or append the slot for ``init`` on an existing target.

    Returns ``{'index', 'status', 'autofilled'}`` with status one of:

    - ``'new'``: appended (cfdb auto-fills any skipped inits; ``autofilled`` counts them);
    - ``'backfill'``: the slot exists but the init is not marked complete (an auto-filled gap, or a
      previous ingest that did not finish) -- writing into it is the recovery path;
    - ``'overwrite'``: the init is complete and ``overwrite=True`` was passed.

    A complete init is immutable without ``overwrite=True``. Inits before the axis origin are refused.
    """
    init64 = np.datetime64(init, 'm')
    stored = np.asarray(ds[FRT].data).astype('datetime64[m]')
    if stored.size == 0:
        raise ValueError(f'target has an empty {FRT} axis')

    hits = np.where(stored == init64)[0]
    if hits.size:
        idx = int(hits[0])
        has_marker = COMPLETE_INITS_ATTR in ds.attrs.data
        complete = is_init_complete(ds, init64) if has_marker else _init_has_chunks(ds, idx)
        if complete and not overwrite:
            raise ValueError(
                f'{FRT} {init64} already holds a complete run -- forecast runs are immutable. '
                f'Pass overwrite=True to replace it deliberately.'
            )
        return {'index': idx, 'status': 'overwrite' if complete else 'backfill', 'autofilled': 0}

    if init64 < stored[0]:
        raise ValueError(f'{FRT} {init64} predates the axis start {stored[0]}; refusing to prepend history')

    check_init_on_grid(init64, stored[0], step_minutes)
    old_len = stored.size
    ds[FRT].append(np.array([init64], dtype='datetime64[m]'))
    new_len = len(ds[FRT].data)
    return {'index': new_len - 1, 'status': 'new', 'autofilled': int(new_len - old_len - 1)}


######################################################
# Chunking


def forecast_chunk_shape(
    n_lead: int, ny: int, nx: int, itemsize: int = 4, target_bytes: int = 6 * 2**20, max_bytes: int = 8 * 2**20
) -> tuple:
    """
    ``(1, n_lead, 1, ty, tx)``: one init, every lead, one level, one spatial tile.

    Full spatial extent when one (init, level) row fits in ``max_bytes``; otherwise square-ish
    tiles sized to ``target_bytes`` (pre-compression), the last tile never a sliver because tile
    sizes come from ceil-division of the extent by the tile count.
    """
    row_bytes = n_lead * ny * nx * itemsize
    if row_bytes <= max_bytes:
        return (1, int(n_lead), 1, int(ny), int(nx))
    n_tiles = math.ceil(row_bytes / target_bytes)
    n_y = max(1, round(math.sqrt(n_tiles * ny / nx)))
    n_x = max(1, math.ceil(n_tiles / n_y))
    ty, tx = math.ceil(ny / n_y), math.ceil(nx / n_x)
    while n_lead * ty * tx * itemsize > max_bytes:
        if ty >= tx:
            n_y += 1
        else:
            n_x += 1
        ty, tx = math.ceil(ny / n_y), math.ceil(nx / n_x)
    return (1, int(n_lead), 1, int(ty), int(tx))


class ForecastWriter:
    """
    The only thing that writes forecast data variables.

    Blocks arrive per source file / per rechunkit block, indexed by OUTPUT time index ``t``
    (position in the ingest's filtered time axis). ``lead_index[t]`` is that time's position on the
    stored ``forecast_period`` axis, so placement is by lead value. Each (variable, level) gets one
    ``(n_lead_stored, ny, nx)`` buffer -- one chunk-row, the write unit -- which is written with a
    single ``set`` and discarded as soon as every expected lead has arrived. ``close()`` writes any
    buffer that never completed (a source that stops short), still once.

    Memory: one chunk-row per (variable, level) that is live at the same time. Per-file sources that
    interleave variables (WRF's batch path) keep every batch variable's rows live until the last file;
    keep forecast-mode WRF ingests to 2-D variables for that reason.
    """

    def __init__(self, ds, init_idx: int, lead_index: np.ndarray, ny: int, nx: int):
        self.ds = ds
        self.init_idx = int(init_idx)
        self.lead_index = np.asarray(lead_index, dtype='int64')
        self.n_lead_stored = len(ds[LEAD].data)
        self.ny = int(ny)
        self.nx = int(nx)
        self._buffers = {}  # (var name, level_idx) -> [ndarray, filled mask]
        self.writes = 0  # number of chunk-row writes issued (tests assert one per (var, level))

    def _buffer(self, data_var, level_idx: int):
        key = (data_var.name, int(level_idx))
        buf = self._buffers.get(key)
        if buf is None:
            arr = np.full((self.n_lead_stored, self.ny, self.nx), np.nan, dtype='float32')
            buf = [arr, np.zeros(self.n_lead_stored, dtype=bool), data_var]
            self._buffers[key] = buf
        return key, buf

    def put(
        self,
        data_var,
        level_idx: int,
        t_index: Union[int, slice, np.ndarray],
        block: np.ndarray,
        ys: slice = slice(None),
        xs: slice = slice(None),
    ) -> None:
        """
        Accumulate ``block`` for one level. ``block`` is ``(n, ny_sub, nx_sub)`` for a slice/array of
        output time indices, or ``(ny_sub, nx_sub)`` for a single int index.
        """
        block = np.asarray(block)
        if isinstance(t_index, (int, np.integer)):
            t_idx = np.array([int(t_index)])
            block = block[np.newaxis, ...]
        elif isinstance(t_index, slice):
            t_idx = np.arange(*t_index.indices(len(self.lead_index)))
        else:
            t_idx = np.asarray(t_index, dtype='int64')
        if len(t_idx) != block.shape[0]:
            raise ValueError(f'block has {block.shape[0]} timesteps for {len(t_idx)} indices')
        positions = self.lead_index[t_idx]
        key, (arr, filled, _) = self._buffer(data_var, level_idx)
        arr[positions, ys, xs] = block
        filled[positions] = True
        if filled.sum() >= len(self.lead_index):
            self._flush_key(key)

    def _flush_key(self, key) -> None:
        # Write the span of leads this ingest supplies. For a run that covers the whole stored axis
        # (the normal case) that is the full chunk-row; a partial re-ingest (e.g. a start_date
        # filter with overwrite=True) leaves the leads it did not supply untouched.
        arr, _, data_var = self._buffers.pop(key)
        level_idx = key[1]
        lo, hi = int(self.lead_index.min()), int(self.lead_index.max()) + 1
        data_var[(self.init_idx, slice(lo, hi), level_idx, slice(None), slice(None))] = arr[lo:hi]
        self.writes += 1

    def flush(self, data_var=None) -> None:
        """Write every pending buffer (for one variable, or all)."""
        for key in list(self._buffers):
            if data_var is None or key[0] == data_var.name:
                self._flush_key(key)

    def close(self) -> None:
        self.flush()
