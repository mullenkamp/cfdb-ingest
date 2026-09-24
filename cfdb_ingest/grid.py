"""
Rules for growing one cfdb ``grid`` dataset across many ``convert(..., extend=True)`` calls: a long
record (e.g. a decades-long hindcast) built in time bands, extended forward or backward later.

Conventions baked into a dataset at creation:

- ``time`` carries an EXPLICIT numeric step (minutes). cfdb then auto-fills the slots skipped when a
  later call appends or prepends across a gap: that is deliberate, a gap is a placeholder for data to
  come. Its slots are never written and read as missing. (``step=True`` infers nothing from a one-frame axis --
  cfdb then stores ``step=None`` and would accept a gap as an irregular jump with no placeholder slots,
  so a stored step of None is refused.)
- A call writes one window at the step. Frames missing from the input are unwritten slots (read as
  missing) and are listed in ``convert``'s result (``missing_frames``); re-running the window once the files
  exist fills them (writes are idempotent overwrites). Partially written chunks are fine: nothing here
  tracks completeness -- rebuild a range by re-running it, and check a dataset's values (e.g. no missing
  values in a published range) where completeness matters.
- Time chunks are anchored at the dataset's absolute index 0 (its first time at creation), not at the
  calendar; ``time_bands`` enumerates bands on that grid, and ``convert`` aligns every write to it, so
  each output chunk is written exactly once per call.
"""
import numpy as np
from cfdb import indexers as _indexers

from cfdb_ingest import forecast as _fc

TIME = 'time'
TIME_LABEL_ATTR = 'time_label'


def _minutes(step_minutes) -> np.timedelta64:
    return np.timedelta64(int(step_minutes), 'm')


def validate_target(ds, *, crs, x_name, y_name, x, y, levels=None, depths=None, step_minutes: int,
                    time_label: str) -> None:
    """
    An existing target for ``extend`` must be a ``grid`` dataset with a numeric time step equal to the
    incoming one, the same time labelling, the incoming CRS and the same spatial (level / depth)
    coordinates. Raises ValueError naming the first mismatch.
    """
    if ds.dataset_type != 'grid':
        raise ValueError(f"target dataset_type is {ds.dataset_type!r}, expected 'grid'")
    for name in (TIME, x_name, y_name):
        if name not in ds.coord_names:
            raise ValueError(f'target lacks the {name!r} coordinate')
    stored_step = ds[TIME].step
    if stored_step is None:
        raise ValueError(
            "target 'time' has no step (created with step=True from a single frame?); without one cfdb "
            'cannot place a gap as placeholder slots. Rebuild the target with cfdb-ingest >= 0.6.0.'
        )
    if int(stored_step) != int(step_minutes):
        raise ValueError(f"target 'time' step is {stored_step} min, incoming frames are {step_minutes} min apart")
    stored_label = ds[TIME].attrs.data.get(TIME_LABEL_ATTR, 'interval_end')
    if stored_label != f'interval_{time_label}':
        raise ValueError(f'target is labelled {stored_label}; refusing to add interval_{time_label} labels')
    _fc.check_crs(ds, crs)
    _fc.validate_spatial(ds, x_name=x_name, y_name=y_name, x=x, y=y, levels=levels, depths=depths)


def place_times(ds, labels, *, step_minutes: int) -> dict:
    """
    Place a window of consecutive ``labels`` on an existing target's time axis.

    Labels may overlap the stored axis (the overwrite path: placeholder gaps, interior bands, re-runs)
    and/or extend ONE end of it; an extension across a gap auto-fills the gap's slots (placeholders).
    Everything is validated before the coordinate is touched. Refused: labels that are not consecutive
    at the step, labels off the stored step grid, and a window that extends both ends at once.

    Returns ``{'status', 'index', 'n_new', 'gap_filled', 'abs_start'}``: ``index`` is the position of
    ``labels[0]`` on the (possibly extended) axis and ``abs_start`` its absolute chunk-grid index
    (``origin + index``).
    """
    step = _minutes(step_minutes)
    labels = np.asarray(labels).astype('datetime64[m]')
    if labels.size == 0:
        raise ValueError('no times to place')
    if labels.size > 1:
        d = np.diff(labels)
        if np.any(d != step):
            holes = labels[:-1][d != step]
            raise ValueError(f'incoming times are not consecutive at {step_minutes} min (breaks after '
                             f'{[str(h) for h in holes[:5]]}); a window must hold every frame')
    coord = ds[TIME]
    stored = np.asarray(coord.data).astype('datetime64[m]')
    if stored.size == 0:
        raise ValueError("target has an empty 'time' axis")
    t0, tn = stored[0], stored[-1]
    off = int((labels[0] - t0) / np.timedelta64(1, 'm')) % int(step_minutes)
    if off:
        raise ValueError(f'{labels[0]} is {off} min off the target time grid (origin {t0}, step '
                         f'{step_minutes} min); refusing rather than snapping')
    before = labels[labels < t0]
    after = labels[labels > tn]
    if before.size and after.size:
        raise ValueError(f'incoming {labels[0]}..{labels[-1]} extends both before {t0} and after {tn}; '
                         f'extend one end per call')
    gap = 0
    if after.size:
        gap = int((after[0] - tn) / step) - 1
        coord.append(after)
    elif before.size:
        gap = int((t0 - before[-1]) / step) - 1
        coord.prepend(before)
    placed = np.asarray(coord.data).astype('datetime64[m]')
    lo = int(np.searchsorted(placed, labels[0]))
    if not np.array_equal(placed[lo:lo + labels.size], labels):
        raise RuntimeError(f"placed 'time' values do not match the incoming labels at index {lo}")
    n_new = int(before.size + after.size)
    if n_new == labels.size:
        status = 'append' if after.size else 'prepend'
    elif n_new:
        status = ('append' if after.size else 'prepend') + '+overwrite'
    else:
        status = 'overwrite'
    return {'status': status, 'index': lo, 'n_new': n_new, 'gap_filled': gap,
            'abs_start': int(coord.origin) + lo}


def time_anchor(ds, step_minutes: int = None) -> np.datetime64:
    """The time at absolute index 0 of ``ds``'s time-chunk grid (its first time at creation)."""
    coord = ds[TIME]
    step = _minutes(step_minutes if step_minutes is not None else coord.step)
    return np.datetime64(coord.data[0], 'm') - int(coord.origin) * step


def time_bands(anchor, start, stop, chunk_t: int, step_minutes: int) -> list:
    """
    The half-open bands ``[b0, b1)`` of ``chunk_t`` steps, on the chunk grid anchored at ``anchor``,
    that cover ``[start, stop)`` -- ascending. The first and last may reach outside ``[start, stop)``;
    clip the window passed to ``convert`` to the frames that exist.
    """
    step = _minutes(step_minutes)
    width = chunk_t * step
    anchor = np.datetime64(anchor, 'm')
    start, stop = np.datetime64(start, 'm'), np.datetime64(stop, 'm')
    k0 = int(np.floor((start - anchor) / width))
    k1 = int(np.ceil((stop - anchor) / width))
    return [(anchor + k * width, anchor + (k + 1) * width) for k in range(k0, k1)]


def missing_chunks(ds, start=None, end=None) -> list:
    """
    ``(var_name, chunk_start)`` for every chunk of every time-dimensioned data variable, within the
    stored times ``[start, end]`` (inclusive; default the whole axis), that is absent from the store.
    Probes key presence only (no data fetched, also on an EDataset).
    """
    times = np.asarray(ds[TIME].data).astype('datetime64[m]')
    lo = 0 if start is None else int(np.searchsorted(times, np.datetime64(start, 'm')))
    hi = len(times) if end is None else int(np.searchsorted(times, np.datetime64(end, 'm'), side='right'))
    if hi <= lo:
        return []
    missing = []
    for name in ds.data_var_names:
        dv = ds[name]
        if TIME not in dv.coord_names:
            continue
        sel = tuple(slice(lo, hi) if c == TIME else slice(None) for c in dv.coord_names)
        slices = _indexers.index_combo_all(sel, dv.get_coord_origins(), dv.shape)
        for key in _indexers.slices_to_keys(slices, name, dv.chunk_shape):
            if key not in dv._blt:
                start_idx = tuple(int(v) for v in key.rsplit('!', 1)[-1].split('.'))
                missing.append((name, start_idx))
    return missing
