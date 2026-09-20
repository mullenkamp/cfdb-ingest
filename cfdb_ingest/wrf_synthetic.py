"""
Deterministic synthetic wrfout files for tests -- a running forecast written one daily file at a
time, with the attributes and variables ``WrfIngest`` needs and closed-form values a test can
predict without reading the file back.

Nothing here is a real WRF field; the point is a generator that any downstream package can import
(``from cfdb_ingest.wrf_synthetic import write_run``) instead of committing binary fixtures.

Values: every 2-D field at output time index ``t`` (lead ``L`` hours since ``simulation_start``) is
``base(name, L) + Y_SLOPE * j + X_SLOPE * i`` -- a lead-dependent constant plus a small spatial
gradient that catches a transposed axis; the precipitation family uses the gradient
multiplicatively, ``base(name, L) * (1 + Y_SLOPE * j + X_SLOPE * i)``, so a zero increment stays
exactly zero. The closed forms are :func:`expected`. Precipitation is consistent between the windowed
accumulators and the running totals: ``PREC_ACC_NC[t]`` is the increment over the window ending at
``t`` (0 at lead 0, like WRF's zero-initialised accumulator) and ``RAINNC[t]`` is the cumulative sum
of those increments, so ``RAIN`` (differenced) and ``PREC_ACC`` (summed) agree everywhere except
lead 0 (NaN vs 0).
"""

import pathlib
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np

Y_SLOPE = 0.01
X_SLOPE = 0.001

# Fields the generator knows how to write. 'PREC_ACC_*' and 'RAIN*' are derived from the same
# per-lead increment; everything else is a smooth function of the lead.
FIELDS = (
    'T2', 'PSFC', 'Q2', 'U10', 'V10', 'TSK', 'SWDOWN', 'PBLH',
    'PREC_ACC_NC', 'PREC_ACC_C', 'RAINNC', 'RAINC', 'SNOWNC', 'SNOW_ACC_NC',
)
DEFAULT_FIELDS = ('T2', 'PSFC', 'U10', 'V10', 'PREC_ACC_NC', 'PREC_ACC_C', 'RAINNC', 'RAINC')


def increment_nc(lead: int) -> float:
    """Non-convective precipitation (mm) over the window ending at ``lead`` hours; 0 at lead 0."""
    return 0.0 if lead <= 0 else 0.5 + 0.1 * (lead % 7)


def increment_c(lead: int) -> float:
    """Convective precipitation (mm) over the window ending at ``lead`` hours; 0 at lead 0."""
    return 0.0 if lead <= 0 else 0.25 * (lead % 3)


def base(name: str, lead: int) -> float:
    """The spatially-uniform part of field ``name`` at lead ``lead`` (hours)."""
    if name == 'T2':
        return 280.0 + 0.1 * lead
    if name == 'TSK':
        return 281.0 + 0.1 * lead
    if name == 'PSFC':
        return 101300.0 - 2.0 * lead
    if name == 'Q2':
        return 0.008 + 1e-5 * lead
    if name == 'U10':
        return 3.0 + 0.05 * lead
    if name == 'V10':
        return -1.0 + 0.02 * lead
    if name == 'SWDOWN':
        return 400.0 * max(0.0, np.sin(np.pi * (lead % 24) / 24.0))
    if name == 'PBLH':
        return 500.0 + 10.0 * (lead % 24)
    if name == 'PREC_ACC_NC':
        return increment_nc(lead)
    if name == 'PREC_ACC_C':
        return increment_c(lead)
    if name == 'SNOW_ACC_NC':
        return 0.1 * increment_nc(lead)
    if name == 'RAINNC':
        return float(sum(increment_nc(k) for k in range(lead + 1)))
    if name == 'RAINC':
        return float(sum(increment_c(k) for k in range(lead + 1)))
    if name == 'SNOWNC':
        return float(sum(0.1 * increment_nc(k) for k in range(lead + 1)))
    raise KeyError(f'no closed form for {name!r}; known fields: {FIELDS}')


PRECIP_FIELDS = ('PREC_ACC_NC', 'PREC_ACC_C', 'SNOW_ACC_NC', 'RAINNC', 'RAINC', 'SNOWNC')


def expected(name: str, lead: int, ny: int, nx: int) -> np.ndarray:
    """The (ny, nx) float32 field ``name`` at ``lead`` hours, exactly as :func:`write_wrfout` wrote it."""
    j = np.arange(ny, dtype='float64')[:, None]
    i = np.arange(nx, dtype='float64')[None, :]
    gradient = Y_SLOPE * j + X_SLOPE * i
    if name in PRECIP_FIELDS:
        return (base(name, lead) * (1.0 + gradient)).astype('float32')
    return (base(name, lead) + gradient).astype('float32')


def _times_array(times: Sequence[np.datetime64]) -> np.ndarray:
    rows = []
    for t in times:
        s = str(np.datetime64(t, 's')).replace('T', '_')
        if len(s) != 19:
            raise ValueError(f'unexpected timestamp width: {s!r}')
        rows.append([c.encode() for c in s])
    return np.array(rows, dtype='S1')


def write_wrfout(
    path,
    start_time,
    n_times: int,
    ny: int,
    nx: int,
    *,
    hour_step: int = 1,
    simulation_start=None,
    variables: Iterable[str] = DEFAULT_FIELDS,
    bucket_mm: Optional[float] = None,
    lat0: float = -45.0,
    lon0: float = 170.0,
    dlat: float = 0.05,
    dlon: float = 0.05,
) -> pathlib.Path:
    """
    Write one lat-lon (``MAP_PROJ=6``) wrfout with ``n_times`` frames from ``start_time`` every
    ``hour_step`` hours, holding ``variables`` (each ``(time, ny, nx)`` float32 with the closed forms
    of :func:`expected`), ``Times``, ``XLAT``/``XLONG``, ``COSALPHA``/``SINALPHA`` (no rotation), and
    the attributes ``WrfIngest`` reads. ``simulation_start`` (default ``start_time``) becomes
    ``SIMULATION_START_DATE`` and ``START_DATE``, so a later segment of a run still reports the run's
    init and its leads count from it. ``bucket_mm`` adds ``BUCKET_MM`` and ``I_RAINNC``/``I_RAINC``
    counters (the totals then wrap at the bucket, as WRF's do).
    """
    path = pathlib.Path(path)
    start = np.datetime64(start_time, 'm')
    sim_start = np.datetime64(simulation_start if simulation_start is not None else start_time, 'm')
    times = [start + np.timedelta64(k * hour_step, 'h') for k in range(n_times)]
    leads = [int((t - sim_start).astype('timedelta64[m]').astype('int64') // 60) for t in times]
    if min(leads) < 0:
        raise ValueError('start_time precedes simulation_start')

    lats = (lat0 + dlat * np.arange(ny)).astype('float32')
    lons = (lon0 + dlon * np.arange(nx)).astype('float32')
    xlat = np.broadcast_to(lats[:, None], (ny, nx)).astype('float32')
    xlong = np.broadcast_to(lons[None, :], (ny, nx)).astype('float32')

    def stamp(dt64):
        return str(np.datetime64(dt64, 's')).replace('T', '_')

    with h5py.File(path, 'w') as h5:
        h5.attrs['MAP_PROJ'] = np.int32(6)
        h5.attrs['TITLE'] = ' OUTPUT FROM WRF V4 SYNTHETIC (cfdb_ingest.wrf_synthetic)'.ljust(74)
        h5.attrs['CEN_LAT'] = np.float32(lats[ny // 2])
        h5.attrs['CEN_LON'] = np.float32(lons[nx // 2])
        h5.attrs['DX'] = np.float32(dlon * 111000.0)
        h5.attrs['DY'] = np.float32(dlat * 111000.0)
        h5.attrs['SIMULATION_START_DATE'] = stamp(sim_start)
        h5.attrs['START_DATE'] = stamp(sim_start)
        if bucket_mm is not None:
            h5.attrs['BUCKET_MM'] = np.float32(bucket_mm)

        h5.create_dataset('Times', data=_times_array(times))
        h5.create_dataset('XLAT', data=np.broadcast_to(xlat, (n_times, ny, nx)))
        h5.create_dataset('XLONG', data=np.broadcast_to(xlong, (n_times, ny, nx)))
        h5.create_dataset('COSALPHA', data=np.ones((n_times, ny, nx), dtype='float32'))
        h5.create_dataset('SINALPHA', data=np.zeros((n_times, ny, nx), dtype='float32'))

        for name in variables:
            field = np.stack([expected(name, lead, ny, nx) for lead in leads])
            if bucket_mm is not None and name in ('RAINNC', 'RAINC'):
                counts = np.floor(field / bucket_mm).astype('int32')
                h5.create_dataset(f'I_{name}', data=counts, dtype='int32')
                field = (field - counts * bucket_mm).astype('float32')
            h5.create_dataset(name, data=field, chunks=(1, ny, nx), compression='gzip', compression_opts=1)
    return path


def write_run(
    directory,
    init,
    lead_hours: int,
    ny: int,
    nx: int,
    *,
    frames_per_file: int = 24,
    hour_step: int = 1,
    domain: str = 'd02',
    end_frame_file: bool = True,
    **kwargs,
) -> list:
    """
    Write one forecast run as the daily files WRF produces: ``frames_per_file`` frames per file from
    ``init`` to ``init + lead_hours``, plus (``end_frame_file``) the single-frame file WRF writes at
    the final hour. Files are named with the pipeline's colon-free archived names
    (``wrfout_d02_2026-09-19_00_00_00.nc``). Returns the paths in time order.
    """
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    init64 = np.datetime64(init, 'm')
    n_total = lead_hours // hour_step + 1
    paths = []
    t = 0
    while t < n_total:
        n = min(frames_per_file, n_total - t)
        if n == 1 and t == n_total - 1 and not end_frame_file:
            break
        start = init64 + np.timedelta64(t * hour_step, 'h')
        name = f'wrfout_{domain}_{str(np.datetime64(start, "s")).replace("T", "_").replace(":", "_")}.nc'
        paths.append(write_wrfout(directory / name, start, n, ny, nx, hour_step=hour_step,
                                  simulation_start=init64, **kwargs))
        t += n
    return paths
