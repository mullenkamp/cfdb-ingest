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


WPS_EARTH_RADIUS_M = 6370000.0


def wps_ijll(map_proj: int, i, j, *, lat1: float, lon1: float, dx: float, truelat1: float,
             truelat2: Optional[float] = None, stdlon: float = 0.0, knowni: float = 1.0, knownj: float = 1.0,
             re_m: float = WPS_EARTH_RADIUS_M):
    """
    Latitude/longitude of grid point(s) ``(i, j)`` (1-based, as WPS counts) for WRF map projections
    1 (Lambert conformal), 2 (polar stereographic) and 3 (Mercator): a numpy port of WPS
    ``geogrid/src/module_map_utils.F`` (``set_lc``/``lc_cone``/``ijll_lc``, ``set_ps``/``ijll_ps``,
    ``set_merc``/``ijll_merc``), with ``(lat1, lon1)`` the known point at ``(knowni, knownj)``.

    It exists so tests can build XLAT/XLONG the way WPS does -- independently of pyproj, so a CRS built by
    cfdb-ingest is checked against WRF's own geometry rather than against itself.
    """
    rad = np.pi / 180.0
    deg = 180.0 / np.pi
    i = np.asarray(i, dtype='float64')
    j = np.asarray(j, dtype='float64')
    hemi = -1.0 if truelat1 < 0 else 1.0
    rebydx = re_m / dx

    if map_proj == 1:
        t2 = truelat1 if truelat2 is None else truelat2
        if abs(truelat1 - t2) > 0.1:
            cone = ((np.log10(np.cos(truelat1 * rad)) - np.log10(np.cos(t2 * rad)))
                    / (np.log10(np.tan((45.0 - abs(truelat1) / 2.0) * rad))
                       - np.log10(np.tan((45.0 - abs(t2) / 2.0) * rad))))
        else:
            cone = np.sin(abs(truelat1) * rad)
        deltalon1 = lon1 - stdlon
        deltalon1 = deltalon1 - 360.0 if deltalon1 > 180.0 else (deltalon1 + 360.0 if deltalon1 < -180.0 else deltalon1)
        rsw = (rebydx * np.cos(truelat1 * rad) / cone
               * (np.tan((90.0 * hemi - lat1) * rad / 2.0) / np.tan((90.0 * hemi - truelat1) * rad / 2.0)) ** cone)
        arg = cone * deltalon1 * rad
        polei = hemi * knowni - hemi * rsw * np.sin(arg)
        polej = hemi * knownj + rsw * np.cos(arg)
        chi1 = (90.0 - hemi * truelat1) * rad
        chi2 = (90.0 - hemi * t2) * rad
        xx = hemi * i - polei
        yy = polej - hemi * j
        r = np.sqrt(xx * xx + yy * yy) / rebydx
        lon = np.mod(stdlon + deg * np.arctan2(hemi * xx, yy) / cone + 360.0, 360.0)
        if chi1 == chi2:
            chi = 2.0 * np.arctan((r / np.tan(chi1)) ** (1.0 / cone) * np.tan(chi1 * 0.5))
        else:
            chi = 2.0 * np.arctan((r * cone / np.sin(chi1)) ** (1.0 / cone) * np.tan(chi1 * 0.5))
        lat = (90.0 - chi * deg) * hemi
    elif map_proj == 2:
        reflon = stdlon + 90.0
        scale_top = 1.0 + hemi * np.sin(truelat1 * rad)
        ala1 = lat1 * rad
        rsw = rebydx * np.cos(ala1) * scale_top / (1.0 + hemi * np.sin(ala1))
        alo1 = (lon1 - reflon) * rad
        polei = knowni - rsw * np.cos(alo1)
        polej = knownj - hemi * rsw * np.sin(alo1)
        xx = i - polei
        yy = (j - polej) * hemi
        r2 = xx ** 2 + yy ** 2
        gi2 = (rebydx * scale_top) ** 2
        lat = deg * hemi * np.arcsin((gi2 - r2) / (gi2 + r2))
        arccos = np.arccos(np.clip(xx / np.sqrt(r2), -1.0, 1.0))
        lon = np.where(yy > 0, reflon + deg * arccos, reflon - deg * arccos)
    elif map_proj == 3:
        dlon = dx / (re_m * np.cos(rad * truelat1))
        rsw = 0.0 if lat1 == 0 else np.log(np.tan(0.5 * ((lat1 + 90.0) * rad))) / dlon
        lat = 2.0 * np.arctan(np.exp(dlon * (rsw + j - knownj))) * deg - 90.0
        lon = (i - knowni) * dlon * deg + lon1
    else:
        raise ValueError(f'wps_ijll supports MAP_PROJ 1, 2, 3; got {map_proj}')
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    lon = np.where(lon < -180.0, lon + 360.0, lon)
    return lat, lon


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
    map_proj: int = 6,
    projection: Optional[dict] = None,
) -> pathlib.Path:
    """
    Write one lat-lon (``MAP_PROJ=6``) wrfout with ``n_times`` frames from ``start_time`` every
    ``hour_step`` hours, holding ``variables`` (each ``(time, ny, nx)`` float32 with the closed forms
    of :func:`expected`), ``Times``, ``XLAT``/``XLONG``, ``COSALPHA``/``SINALPHA`` (no rotation), and
    the attributes ``WrfIngest`` reads. ``simulation_start`` (default ``start_time``) becomes
    ``SIMULATION_START_DATE`` and ``START_DATE``, so a later segment of a run still reports the run's
    init and its leads count from it. ``bucket_mm`` adds ``BUCKET_MM`` and ``I_RAINNC``/``I_RAINC``
    counters (the totals then wrap at the bucket, as WRF's do). Files holding a ``PREC_ACC_*`` field
    carry ``PREC_ACC_DT`` = the history interval in minutes, as WRF writes it.

    ``map_proj`` 1, 2 or 3 writes a projected grid instead: ``projection`` gives ``dx`` (m) and ``truelat1``
    (and ``truelat2``, ``stdlon``), the south-west cell centre is ``(lat0, lon0)``, and XLAT/XLONG come
    from :func:`wps_ijll` (WPS's own formulas on its 6 370 km sphere).
    """
    variables = tuple(variables)
    path = pathlib.Path(path)
    start = np.datetime64(start_time, 'm')
    sim_start = np.datetime64(simulation_start if simulation_start is not None else start_time, 'm')
    times = [start + np.timedelta64(k * hour_step, 'h') for k in range(n_times)]
    leads = [int((t - sim_start).astype('timedelta64[m]').astype('int64') // 60) for t in times]
    if min(leads) < 0:
        raise ValueError('start_time precedes simulation_start')

    if map_proj == 6:
        lats = (lat0 + dlat * np.arange(ny)).astype('float32')
        lons = (lon0 + dlon * np.arange(nx)).astype('float32')
        xlat = np.broadcast_to(lats[:, None], (ny, nx)).astype('float32')
        xlong = np.broadcast_to(lons[None, :], (ny, nx)).astype('float32')
    else:
        proj = dict(projection or {})
        jj, ii = np.meshgrid(np.arange(1, ny + 1), np.arange(1, nx + 1), indexing='ij')
        la, lo = wps_ijll(map_proj, ii, jj, lat1=lat0, lon1=lon0, dx=proj['dx'], truelat1=proj['truelat1'],
                          truelat2=proj.get('truelat2'), stdlon=proj.get('stdlon', 0.0))
        xlat = la.astype('float32')
        xlong = lo.astype('float32')
        lats, lons = xlat[:, nx // 2], xlong[ny // 2, :]

    def stamp(dt64):
        return str(np.datetime64(dt64, 's')).replace('T', '_')

    with h5py.File(path, 'w') as h5:
        h5.attrs['MAP_PROJ'] = np.int32(map_proj)
        h5.attrs['TITLE'] = ' OUTPUT FROM WRF V4 SYNTHETIC (cfdb_ingest.wrf_synthetic)'.ljust(74)
        h5.attrs['CEN_LAT'] = np.float32(xlat[ny // 2, nx // 2])
        h5.attrs['CEN_LON'] = np.float32(xlong[ny // 2, nx // 2])
        if map_proj == 6:
            h5.attrs['DX'] = np.float32(dlon * 111000.0)
            h5.attrs['DY'] = np.float32(dlat * 111000.0)
        else:
            h5.attrs['DX'] = np.float32(proj['dx'])
            h5.attrs['DY'] = np.float32(proj['dx'])
            h5.attrs['TRUELAT1'] = np.float32(proj['truelat1'])
            h5.attrs['TRUELAT2'] = np.float32(proj.get('truelat2', proj['truelat1']))
            h5.attrs['STAND_LON'] = np.float32(proj.get('stdlon', 0.0))
            h5.attrs['MOAD_CEN_LAT'] = np.float32(xlat[ny // 2, nx // 2])
        h5.attrs['SIMULATION_START_DATE'] = stamp(sim_start)
        h5.attrs['START_DATE'] = stamp(sim_start)
        if bucket_mm is not None:
            h5.attrs['BUCKET_MM'] = np.float32(bucket_mm)
        if any(name.startswith('PREC_ACC') or name == 'SNOW_ACC_NC' for name in variables):
            # WRF's accumulation window (namelist prec_acc_dt, minutes) = the history interval here.
            h5.attrs['PREC_ACC_DT'] = np.float32(hour_step * 60)

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
