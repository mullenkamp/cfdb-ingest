"""
Generate a small synthetic ECMWF IFS open-data forecast cycle as real GRIB2, with every encoding
quirk of the production files that IfsIngest has to handle:

- regular_ll grid whose first longitude is 180 (so the array seam sits at the dateline),
  latitudes scanning north -> south;
- ``grid_ccsds`` packing;
- soil layers on ``typeOfFirstFixedSurface=151`` (``soilLayer``) with layer INDICES 1..4;
- ``sithick`` carrying a bitmap (missing over land);
- orography (``z`` on ``surface``) only in the 0 h message set, plus ``sdor``/``slor`` decoys;
- accumulated ``tp`` / ``ssrd`` / ``strd``.

Every field is a closed-form function of (lat, lon, level, step) exported here so tests can assert
values without re-reading the GRIB. It is a public module (since 0.4.1) so that downstream packages
-- ``ifs-download`` -- can build the same cycles in their own tests; cfdb-ingest's fixtures generate
cycles into a session temp dir (the generator is deterministic and sub-second, so nothing binary is
committed). Run as a module to write a cycle somewhere for manual inspection:

    uv run python -m cfdb_ingest.ifs_synthetic /tmp/ifs_cycle
"""

import pathlib
import sys

import numpy as np

from cfdb_ingest.ifs import _require_eccodes

NI, NJ = 72, 37  # 5-degree grid, 180E first, 90N first
DLON = DLAT = 5.0
LON0, LAT0 = 180.0, 90.0
STEPS = (0, 3, 6)
LEVELS_HPA = (1000, 850, 500)
SOIL_LAYERS = (1, 2, 3, 4)
MISSING = 9999.0
G = 9.80665

def grid_lons_lats():
    """Raw (unrolled) grid as written: lons 180, 185, ... 535 (mod 360), lats 90 -> -90."""
    lons = (LON0 + np.arange(NI) * DLON) % 360.0
    lats = LAT0 - np.arange(NJ) * DLAT
    return lons, lats


def is_land(lat, lon):
    """Synthetic land mask: a block in the south-west Pacific (NZ-ish, west of the dateline) plus Antarctica."""
    lon = np.asarray(lon) % 360.0
    lat = np.asarray(lat)
    return ((lon >= 165) & (lon <= 180) & (lat >= -50) & (lat <= -30)) | (lat <= -80)


# --- closed forms -------------------------------------------------------------------------------


def t_pl(lat, lon, level, step):
    return 220.0 + 0.3 * (lat + 90) / 180 * 60 + 0.05 * level + 0.5 * step + 0.01 * (lon % 360)


def u_pl(lat, lon, level, step):
    return 10.0 + 0.02 * level / 100 + 0.1 * lat + 0.2 * step


def v_pl(lat, lon, level, step):
    return -5.0 + 0.05 * (lon % 360) / 10 + 0.1 * step


def q_pl(lat, lon, level, step):
    return 0.002 + 0.000005 * level + 0.00001 * step


def gh_pl(lat, lon, level, step):
    return {1000: 100.0, 850: 1450.0, 500: 5600.0}[int(level)] + 0.5 * lat + step


def r_pl(lat, lon, level, step):  # decoy: deliberately outside 0-100
    return -10.0 + 1.5 * (lat + 90)


def t2(lat, lon, step):
    return 280.0 + 0.2 * lat + 0.01 * (lon % 360) + step


def td2(lat, lon, step):
    return t2(lat, lon, step) - 5.0


def u10(lat, lon, step):
    return 3.0 + 0.05 * lat + 0.1 * step


def v10(lat, lon, step):
    return -2.0 + 0.02 * (lon % 360) / 10


def u100(lat, lon, step):
    return u10(lat, lon, step) * 1.5


def v100(lat, lon, step):
    return v10(lat, lon, step) * 1.5


def msl(lat, lon, step):
    return 101325.0 - 20.0 * lat + 100.0 * step


def sp(lat, lon, step):
    return msl(lat, lon, step) - 50.0 * np.where(is_land(lat, lon), 1.0, 0.0) * 5


def skt(lat, lon, step):
    return 285.0 + 0.1 * lat + 0.5 * step


def lsm(lat, lon, step):
    return np.where(is_land(lat, lon), 1.0, 0.0)


def sd(lat, lon, step):  # m of water equivalent
    return np.where(lat > 60, 0.05, 0.0) + np.where(lat <= -80, 10.0, 0.0)


def rsn(lat, lon, step):
    return 250.0 + step


def sithick(lat, lon, step):  # NaN where the bitmap masks (land); thickness where ice
    return np.where(is_land(lat, lon), np.nan, np.where(lat >= 80, 1.5, 0.0))


def tp(lat, lon, step):  # accumulated from init, metres
    return step * 0.001 * (1.0 + (lon % 360) / 360.0)


def fg10(lat, lon, step):
    return 8.0 + 0.1 * abs(lat) + step


def tcwv(lat, lon, step):
    return 20.0 - 0.1 * abs(lat) + step


def mucape(lat, lon, step):
    return 100.0 + 5.0 * step + 0.5 * (lon % 360) / 10


def ssrd(lat, lon, step):  # accumulated J m-2
    return step * 3600.0 * 300.0 * (1.0 + 0.001 * lat)


def strd(lat, lon, step):
    return step * 3600.0 * 350.0


def z_sfc(lat, lon, step):  # geopotential m2 s-2, negative in one sea region
    return G * np.where(is_land(lat, lon), 200.0 + lat, -20.0)


def sdor(lat, lon, step):
    return 10.0


def slor(lat, lon, step):
    return 0.01


def sot(lat, lon, layer, step):
    return 283.0 + 0.1 * lat - 0.5 * layer + 0.1 * step


def vsw(lat, lon, layer, step):
    return 0.2 + 0.05 * layer + 0.001 * step


PL_FIELDS = {'t': t_pl, 'u': u_pl, 'v': v_pl, 'q': q_pl, 'gh': gh_pl, 'r': r_pl}
SFC_FIELDS = {
    '2t': t2,
    '2d': td2,
    '10u': u10,
    '10v': v10,
    '100u': u100,
    '100v': v100,
    'msl': msl,
    'sp': sp,
    'skt': skt,
    'lsm': lsm,
    'sd': sd,
    'rsn': rsn,
    'sithick': sithick,
    'tp': tp,
    '10fg': fg10,
    'tcwv': tcwv,
    'mucape': mucape,
    'ssrd': ssrd,
    'strd': strd,
}
STEP0_ONLY_FIELDS = {'z': z_sfc, 'sdor': sdor, 'slor': slor}
SOIL_FIELDS = {'sot': sot, 'vsw': vsw}
ACCUMULATED = {'tp', 'ssrd', 'strd'}
BITMAPPED = {'sithick'}


def field_values(fn, level_or_layer=None, step=0):
    """Evaluate a closed form on the raw grid -> (NJ, NI) float64, NaN where a bitmap should mask."""
    lons, lats = grid_lons_lats()
    lon2, lat2 = np.meshgrid(lons, lats)
    if level_or_layer is None:
        return np.asarray(fn(lat2, lon2, step), dtype='float64') + np.zeros_like(lon2)
    return np.asarray(fn(lat2, lon2, level_or_layer, step), dtype='float64') + np.zeros_like(lon2)


def _new_message(ec, sample, short_name, date, time, step):
    h = ec.codes_grib_new_from_samples(sample)
    ec.codes_set(h, 'centre', 'ecmf')
    ec.codes_set(h, 'Ni', NI)
    ec.codes_set(h, 'Nj', NJ)
    ec.codes_set(h, 'latitudeOfFirstGridPointInDegrees', LAT0)
    ec.codes_set(h, 'longitudeOfFirstGridPointInDegrees', LON0)
    ec.codes_set(h, 'latitudeOfLastGridPointInDegrees', LAT0 - (NJ - 1) * DLAT)
    ec.codes_set(h, 'longitudeOfLastGridPointInDegrees', (LON0 + (NI - 1) * DLON) % 360.0)
    ec.codes_set(h, 'iDirectionIncrementInDegrees', DLON)
    ec.codes_set(h, 'jDirectionIncrementInDegrees', DLAT)
    ec.codes_set(h, 'jScansPositively', 0)
    ec.codes_set(h, 'dataDate', date)
    ec.codes_set(h, 'dataTime', time)
    if short_name in ACCUMULATED:
        ec.codes_set(h, 'stepType', 'accum')
        ec.codes_set(h, 'startStep', 0)
        ec.codes_set(h, 'endStep', step)
    else:
        ec.codes_set(h, 'stepType', 'instant')
        ec.codes_set(h, 'step', step)
    return h


def _finish(ec, h, values, f):
    ec.codes_set(h, 'packingType', 'grid_ccsds')
    ec.codes_set(h, 'bitsPerValue', 16)
    vals = np.asarray(values, dtype='float64').ravel()
    if np.isnan(vals).any():
        ec.codes_set(h, 'bitmapPresent', 1)
        ec.codes_set(h, 'missingValue', MISSING)
        vals = np.where(np.isnan(vals), MISSING, vals)
    ec.codes_set_values(h, vals)
    ec.codes_write(h, f)
    ec.codes_release(h)


def write_cycle(out_dir, init='2026-09-13T00', steps=STEPS, levels_hpa=LEVELS_HPA, extras=True):
    """
    Write one cycle as ``<out_dir>/YYYYMMDDHH0000-<step>h-oper-fc.grib2`` files. Returns the paths.

    ``extras=False`` omits the non-WRF surface bundle (10fg, tcwv, mucape, ssrd, strd, 100u/v).
    """
    ec = _require_eccodes()

    init64 = np.datetime64(init, 'm')
    date = int(np.datetime_as_string(init64, unit='D').replace('-', ''))
    hour = int(str(init64)[11:13])
    time = hour * 100
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    extras_set = {'10fg', 'tcwv', 'mucape', 'ssrd', 'strd', '100u', '100v'}
    for step in steps:
        path = out_dir / f'{date}{hour:02d}0000-{step}h-oper-fc.grib2'
        with open(path, 'wb') as f:
            for short, fn in SFC_FIELDS.items():
                if not extras and short in extras_set:
                    continue
                h = _new_message(ec, 'regular_ll_sfc_grib2', short, date, time, step)
                ec.codes_set(h, 'shortName', short)
                _finish(ec, h, field_values(fn, None, step), f)
            if step == steps[0]:
                for short, fn in STEP0_ONLY_FIELDS.items():
                    h = _new_message(ec, 'regular_ll_sfc_grib2', short, date, time, step)
                    ec.codes_set(h, 'shortName', short)
                    _finish(ec, h, field_values(fn, None, step), f)
            for short, fn in PL_FIELDS.items():
                for level in levels_hpa:
                    h = _new_message(ec, 'regular_ll_pl_grib2', short, date, time, step)
                    ec.codes_set(h, 'shortName', short)
                    ec.codes_set(h, 'typeOfLevel', 'isobaricInhPa')
                    ec.codes_set(h, 'level', int(level))
                    _finish(ec, h, field_values(fn, level, step), f)
            for short, fn in SOIL_FIELDS.items():
                for layer in SOIL_LAYERS:
                    h = _new_message(ec, 'regular_ll_sfc_grib2', short, date, time, step)
                    ec.codes_set(h, 'shortName', short)
                    ec.codes_set(h, 'typeOfFirstFixedSurface', 151)
                    ec.codes_set(h, 'scaleFactorOfFirstFixedSurface', 0)
                    ec.codes_set(h, 'scaledValueOfFirstFixedSurface', layer - 1)
                    ec.codes_set(h, 'typeOfSecondFixedSurface', 151)
                    ec.codes_set(h, 'scaleFactorOfSecondFixedSurface', 0)
                    ec.codes_set(h, 'scaledValueOfSecondFixedSurface', layer)
                    _finish(ec, h, field_values(fn, layer, step), f)
        paths.append(path)
    return paths


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print('usage: python -m cfdb_ingest.ifs_synthetic <out_dir>', file=sys.stderr)
        return 2
    out = pathlib.Path(argv[0])
    paths = write_cycle(out)
    total = sum(p.stat().st_size for p in paths)
    print(f'wrote {len(paths)} files, {total / 1024:.0f} KiB, to {out}')


if __name__ == '__main__':
    sys.exit(main())
