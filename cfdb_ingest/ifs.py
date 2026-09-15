"""
ECMWF IFS open-data forecast (GRIB2) -> cfdb ``grid_forecast``.

One ``IfsIngest`` handles ONE forecast cycle: the per-step GRIB2 files ``<init>-<step>h-oper-fc.grib2``
(or any split of that cycle's messages). The result is appended to a ``grid_forecast`` dataset laid
out as ``(forecast_reference_time, forecast_period, <level>, latitude, longitude)`` -- see
``cfdb_ingest.forecast`` for the axis rules, and the WPS preset for what WRF needs.

Reading is two-pass and memory-bounded:

1. index every message's headers (``shortName``, level type, level, step, byte offset) without
   decoding values;
2. per (variable, level), decode each lead's global field, clip it to the bbox, apply the transform,
   and hand the ``(n_lead, ny, nx)`` block -- one chunk-row -- to the ``ForecastWriter``.

Peak memory is one global field plus one chunk-row.

Encoding facts this module relies on (verified on the production files, 2026-09-15): regular_ll
grid whose first longitude is 180 (the array seam sits on the dateline -> rolled to 0..360 here),
latitudes north -> south (flipped to ascending), ``grid_ccsds`` packing (needs the ``eccodeslib``
wheel), soil layers as ``typeOfLevel=soilLayer`` with layer indices 1..4, ``sithick`` bitmapped over
land, orography (``z`` on ``surface``) only in the 0 h message set, IFS ``r`` outside 0-100 % (unused:
RH is diagnosed from ``q``/``t``), ``tp``/``ssrd``/``strd`` accumulated from the init.

Requires the ``ifs`` extra: ``pip install 'cfdb-ingest[ifs]'``.
"""

import datetime
import pathlib
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pyproj

from cfdb_ingest import forecast as fc
from cfdb_ingest import thermo
from cfdb_ingest.base import create_cfdb_data_var, group_variables, resolve_variable_keys

G = 9.80665

# Cumulative layer bottoms (m) of the IFS/ERA5 soil scheme: 0-7, 7-28, 28-100, 100-289 cm.
# cfdb_to_int renders them as SM000007 / SM007028 / SM028100 / SM100289.
IFS_SOIL_DEPTHS = np.array([0.07, 0.28, 1.00, 2.89])

# Entry format follows the other sources: cfdb_name (cfdb-vars SHORT name), source_vars (GRIB
# shortNames), transform, height ('levels' | 'soil' | float metres). Extra keys: ``invariant``
# (source exists at one step only -> broadcast to every lead), ``attrs`` (extra CF attrs, applied on
# top of the cfdb-vars template), ``dtype`` (override the template's packed encoding; default = the
# template, whose precision is finer than the GRIB's own 12-16-bit quantisation for every field here).
IFS_VARIABLE_MAPPING = {
    # --- pressure levels ---------------------------------------------------------------------
    'T': {'cfdb_name': 'air_temp', 'source_vars': ['t'], 'transform': None, 'height': 'levels'},
    'U': {'cfdb_name': 'u_wind', 'source_vars': ['u'], 'transform': None, 'height': 'levels'},
    'V': {'cfdb_name': 'v_wind', 'source_vars': ['v'], 'transform': None, 'height': 'levels'},
    'Q': {'cfdb_name': 'specific_humidity', 'source_vars': ['q'], 'transform': None, 'height': 'levels'},
    'GH': {'cfdb_name': 'geopotential_height', 'source_vars': ['gh'], 'transform': None, 'height': 'levels'},
    'RH': {
        'cfdb_name': 'relative_humidity',
        'source_vars': ['q', 't'],
        'transform': 'rh_from_q_t_p',
        'height': 'levels',
    },
    # --- fixed-height surface fields ---------------------------------------------------------
    'T2': {'cfdb_name': 'air_temp', 'source_vars': ['2t'], 'transform': None, 'height': 2.0},
    'TD2': {'cfdb_name': 'dew_temp', 'source_vars': ['2d'], 'transform': None, 'height': 2.0},
    'RH2': {'cfdb_name': 'relative_humidity', 'source_vars': ['2t', '2d'], 'transform': 'rh_from_t_td', 'height': 2.0},
    'U10': {'cfdb_name': 'u_wind', 'source_vars': ['10u'], 'transform': None, 'height': 10.0},
    'V10': {'cfdb_name': 'v_wind', 'source_vars': ['10v'], 'transform': None, 'height': 10.0},
    'U100': {'cfdb_name': 'u_wind', 'source_vars': ['100u'], 'transform': None, 'height': 100.0},
    'V100': {'cfdb_name': 'v_wind', 'source_vars': ['100v'], 'transform': None, 'height': 100.0},
    'FG10': {'cfdb_name': 'wind_gust', 'source_vars': ['10fg'], 'transform': None, 'height': 10.0},
    'MSL': {'cfdb_name': 'mslp', 'source_vars': ['msl'], 'transform': None, 'height': 0.0},
    'SP': {'cfdb_name': 'surface_pressure', 'source_vars': ['sp'], 'transform': None, 'height': 0.0},
    'SKT': {'cfdb_name': 'skin_temp', 'source_vars': ['skt'], 'transform': None, 'height': 0.0},
    'SST': {
        'cfdb_name': 'sea_surface_temp',
        'source_vars': ['skt', 'lsm'],
        'transform': 'skt_over_water',
        'height': 0.0,
    },
    # float32: the template is a 0/1 flag and would drop the IFS fraction
    'LSM': {'cfdb_name': 'land_sea_mask', 'source_vars': ['lsm'], 'transform': None, 'height': 0.0, 'dtype': 'float32'},
    'SEAICE': {'cfdb_name': 'sea_ice', 'source_vars': ['sithick'], 'transform': 'thickness_to_flag', 'height': 0.0},
    'SD': {'cfdb_name': 'snow_water_equiv', 'source_vars': ['sd'], 'transform': 'm_to_kg_m2', 'height': 0.0},
    'SNOWH': {
        'cfdb_name': 'snow_depth',
        'source_vars': ['sd', 'rsn'],
        'transform': 'snow_physical_depth',
        'height': 0.0,
    },
    'RSN': {'cfdb_name': 'snow_density', 'source_vars': ['rsn'], 'transform': None, 'height': 0.0},
    'TP': {'cfdb_name': 'precip', 'source_vars': ['tp'], 'transform': 'accumulation_increment_m_to_mm', 'height': 0.0},
    'SSRD': {
        'cfdb_name': 'shortwave_radiation',
        'source_vars': ['ssrd'],
        'transform': 'accumulation_to_mean_flux',
        'height': 0.0,
    },
    'STRD': {
        'cfdb_name': 'longwave_radiation',
        'source_vars': ['strd'],
        'transform': 'accumulation_to_mean_flux',
        'height': 0.0,
    },
    'TCWV': {'cfdb_name': 'pwat', 'source_vars': ['tcwv'], 'transform': None, 'height': 0.0},
    # float32: the packed template caps at 6552 J kg-1 and tropical CAPE exceeds it (a value past the cap packs as missing)
    'CAPE': {'cfdb_name': 'cape', 'source_vars': ['mucape'], 'transform': None, 'height': 0.0, 'dtype': 'float32'},
    'Z_SFC': {
        'cfdb_name': 'terrain_height',
        'source_vars': ['z'],
        'transform': 'geopotential_to_height',
        'height': 0.0,
        'invariant': True,
    },
    # --- soil ---------------------------------------------------------------------------------
    'SOT': {'cfdb_name': 'soil_layer_temp', 'source_vars': ['sot'], 'transform': None, 'height': 'soil'},
    'VSW': {'cfdb_name': 'soil_moisture', 'source_vars': ['vsw'], 'transform': 'clip_nonneg', 'height': 'soil'},
}

# The rows cfdb_to_int consumes for WRF forcing.
IFS_WPS_PRESET_KEYS = [
    'T',
    'U',
    'V',
    'Q',
    'GH',
    'RH',
    'T2',
    'TD2',
    'RH2',
    'U10',
    'V10',
    'MSL',
    'SP',
    'SKT',
    'SST',
    'LSM',
    'SEAICE',
    'SD',
    'SNOWH',
    'Z_SFC',
    'SOT',
    'VSW',
]

# Transforms on ONE lead: fn(sources: {shortName: 2-D}, level_pa) -> 2-D. Stack transforms act on
# the whole (n_lead, ny, nx) stack of a single source: fn(stack, lead_hours) -> stack.
_STACK_TRANSFORMS = {'accumulation_increment_m_to_mm', 'accumulation_to_mean_flux'}


def _t_rh_from_q_t_p(src, level_pa):
    return thermo.rh_from_q_t_p(src['q'], src['t'], level_pa)


def _t_rh_from_t_td(src, level_pa):
    return thermo.rh_from_t_td(src['2t'], src['2d'])


def _t_skt_over_water(src, level_pa):
    return np.where(src['lsm'] < 0.5, src['skt'], np.nan).astype('float32')


def _t_thickness_to_flag(src, level_pa):
    thick = np.nan_to_num(src['sithick'], nan=0.0)  # bitmap-masked land arrives as NaN (_read_field) -> no ice
    return (thick > 0.0).astype('float32')


def _t_m_to_kg_m2(src, level_pa):
    return (src['sd'] * 1000.0).astype('float32')


def _t_snow_physical_depth(src, level_pa):
    rsn = np.where(src['rsn'] > 0.0, src['rsn'], np.nan)
    return (src['sd'] * 1000.0 / rsn).astype('float32')


def _t_clip_nonneg(src, level_pa):
    return np.clip(next(iter(src.values())), 0.0, None).astype('float32')  # GRIB rounding gives -1e-12 soil moisture


def _t_geopotential_to_height(src, level_pa):
    return (src['z'] / G).astype('float32')


def _t_accumulation_increment_m_to_mm(stack, lead_hours):
    out = np.full_like(stack, np.nan)
    out[1:] = np.clip(np.diff(stack, axis=0), 0.0, None) * 1000.0
    return out


def _t_accumulation_to_mean_flux(stack, lead_hours):
    out = np.full_like(stack, np.nan)
    seconds = (np.diff(lead_hours) * 3600.0).astype('float64')[:, np.newaxis, np.newaxis]
    out[1:] = np.diff(stack, axis=0) / seconds
    return out


_TRANSFORMS = {
    'rh_from_q_t_p': _t_rh_from_q_t_p,
    'rh_from_t_td': _t_rh_from_t_td,
    'skt_over_water': _t_skt_over_water,
    'thickness_to_flag': _t_thickness_to_flag,
    'm_to_kg_m2': _t_m_to_kg_m2,
    'snow_physical_depth': _t_snow_physical_depth,
    'geopotential_to_height': _t_geopotential_to_height,
    'clip_nonneg': _t_clip_nonneg,
    'accumulation_increment_m_to_mm': _t_accumulation_increment_m_to_mm,
    'accumulation_to_mean_flux': _t_accumulation_to_mean_flux,
}

_CATEGORY = {'isobaricInhPa': 'pl', 'soilLayer': 'soil'}


def _require_eccodes():
    try:
        import eccodes  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "IfsIngest needs the 'ifs' extra (eccodes + eccodeslib): pip install 'cfdb-ingest[ifs]'"
        ) from e
    return eccodes


def packed_range(dt) -> Optional[Tuple[float, float]]:
    """
    The representable ``(lo, hi)`` of a cfdb packed dtype, or None for an unpacked one. Code 0 is
    the missing marker, so ``lo`` is one step above the offset; ``hi`` is the largest code.
    """
    if getattr(dt, 'precision', None) is None or getattr(dt, 'offset', None) is None:
        return None
    step = 10.0**-dt.precision
    return dt.offset + step, dt.offset + float(np.iinfo(dt.dtype_encoded).max) * step


def check_packed_range(block: np.ndarray, rng: Tuple[float, float], what: str) -> None:
    """
    Refuse a block a packed template cannot hold: cfdb encodes a value above ``hi`` as MISSING
    (silently, per cell) and a value below ``lo`` wraps in the code space, so either would poke holes
    into the archive rather than fail. NaN cells (accumulation lead 0, SST over land) are expected.
    """
    finite = block[np.isfinite(block)]
    if finite.size == 0:
        return
    lo, hi = rng
    bmin, bmax = float(finite.min()), float(finite.max())
    if bmin < lo or bmax > hi:
        raise ValueError(
            f'{what}: values {bmin:.6g}..{bmax:.6g} exceed the packed template range {lo:.6g}..{hi:.6g} '
            f'(a value past the range would be stored as missing); give the mapping entry a wider dtype'
        )


def _category(info) -> str:
    h = info['height']
    return 'pl' if h == 'levels' else 'soil' if h == 'soil' else 'sfc'


def required_messages(variables: Optional[List[str]] = None) -> Set[Tuple[str, str, bool]]:
    """
    The GRIB messages a set of mapping keys needs, as ``{(category, shortName, invariant)}``.

    ``category`` is ``'pl'`` (all pressure levels), ``'soil'`` (all four layers) or ``'sfc'``;
    ``invariant`` is True for sources that exist in the 0 h file only (orography ``z``) and so must
    be fetched for the first step alone. ``variables`` are resolved exactly as ``IfsIngest.convert``
    resolves them (mapping keys, GRIB shortNames or cfdb short names; None = every mapping row), so a
    downloader that fetches this set is guaranteed to feed the ingest everything it will ask for.
    """
    keys = resolve_variable_keys(IFS_VARIABLE_MAPPING, variables)
    out = set()
    for key in keys:
        info = IFS_VARIABLE_MAPPING[key]
        cat = _category(info)
        invariant = bool(info.get('invariant'))
        for sv in info['source_vars']:
            out.add((cat, sv, invariant))
    # a source needed by both an invariant row and a per-step row must be fetched every step
    per_step = {(c, s) for c, s, inv in out if not inv}
    return {(c, s, inv and (c, s) not in per_step) for c, s, inv in out}


class IfsIngest:
    """
    Ingest one ECMWF IFS open-data forecast cycle (GRIB2) into a cfdb ``grid_forecast`` dataset.

    Parameters
    ----------
    input_paths : str, Path, or list
        The cycle's GRIB2 files, or directories of them (``*.grib2``). All messages must share one
        ``dataDate``/``dataTime`` (one init).
    """

    file_glob_pattern = '*.grib2'
    x_coord_name = 'longitude'
    y_coord_name = 'latitude'

    def __init__(self, input_paths: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]]):
        _require_eccodes()
        if isinstance(input_paths, (str, pathlib.Path)):
            input_paths = [input_paths]
        paths = []
        for p in input_paths:
            p = pathlib.Path(p)
            if p.is_dir():
                paths.extend(sorted(p.glob(self.file_glob_pattern)))
            elif p.exists():
                paths.append(p)
            else:
                raise FileNotFoundError(f'{p} does not exist')
        if not paths:
            raise FileNotFoundError(f'no GRIB2 files found in {input_paths}')
        self.input_paths = sorted(set(paths))
        self.crs = pyproj.CRS.from_epsg(4326)
        self.soil_depths = IFS_SOIL_DEPTHS
        self._index_messages()

    # ------------------------------------------------------------------ pass 1: headers

    def _index_messages(self):
        ec = _require_eccodes()
        catalog: Dict[Tuple[str, str], Dict[float, Dict[int, Tuple[pathlib.Path, int]]]] = {}
        inits = set()
        grid = None
        for path in self.input_paths:
            with open(path, 'rb') as f:
                while True:
                    h = ec.codes_grib_new_from_file(f, headers_only=True)
                    if h is None:
                        break
                    try:
                        short = ec.codes_get(h, 'shortName')
                        cat = _CATEGORY.get(ec.codes_get(h, 'typeOfLevel'), 'sfc')
                        level = float(ec.codes_get(h, 'level')) if cat != 'sfc' else 0.0
                        step = int(ec.codes_get(h, 'endStep'))
                        offset = int(ec.codes_get(h, 'offset'))
                        date = int(ec.codes_get(h, 'dataDate'))
                        time = int(ec.codes_get(h, 'dataTime'))
                        inits.add((date, time))
                        if grid is None:
                            grid = {
                                k: ec.codes_get(h, k)
                                for k in (
                                    'Ni',
                                    'Nj',
                                    'latitudeOfFirstGridPointInDegrees',
                                    'longitudeOfFirstGridPointInDegrees',
                                    'iDirectionIncrementInDegrees',
                                    'jDirectionIncrementInDegrees',
                                    'jScansPositively',
                                )
                            }
                    finally:
                        ec.codes_release(h)
                    catalog.setdefault((short, cat), {}).setdefault(level, {})[step] = (path, offset)
        if grid is None:
            raise ValueError('no GRIB messages found')
        if len(inits) != 1:
            raise ValueError(f'input files mix forecast inits: {sorted(inits)}')
        if grid['jScansPositively'] != 0:
            raise ValueError('unexpected grid: latitudes scanning south -> north')
        ((date, time),) = inits
        self.init = np.datetime64(
            f'{date // 10000:04d}-{(date // 100) % 100:02d}-{date % 100:02d}T' f'{time // 100:02d}:{time % 100:02d}',
            'm',
        )
        self._grid = grid
        self._catalog = catalog

        # native axes: roll longitudes so the stored axis runs 0 -> 360 (the seam is at 180 in the file)
        ni, nj = int(grid['Ni']), int(grid['Nj'])
        lons_raw = (
            grid['longitudeOfFirstGridPointInDegrees'] + np.arange(ni) * grid['iDirectionIncrementInDegrees']
        ) % 360.0
        self._roll = int(np.argmin(lons_raw))
        self._lon_axis = np.roll(lons_raw, -self._roll)
        self._lat_axis = (
            grid['latitudeOfFirstGridPointInDegrees'] - np.arange(nj) * grid['jDirectionIncrementInDegrees']
        )

        steps = set()
        for levels in catalog.values():
            for by_step in levels.values():
                steps.update(by_step)
        self.leads = np.array(sorted(steps), dtype='int64')
        t_levels = catalog.get(('t', 'pl'), {})
        self.pressure_levels = np.array(sorted(lev * 100.0 for lev in t_levels), dtype='float64')  # Pa ascending

        self.variables = {}
        for key, info in IFS_VARIABLE_MAPPING.items():
            cat = _category(info)
            if all((sv, cat) in catalog for sv in info['source_vars']):
                self.variables[key] = info

    # ------------------------------------------------------------------ helpers

    def resolve_variables(self, variables: Optional[List[str]]) -> List[str]:
        return resolve_variable_keys(self.variables, variables)

    def _bbox_selection(self, bbox):
        """(x indices into the rolled lon axis, may wrap; y indices north->south; lon values; ascending lat values)."""
        ni = len(self._lon_axis)
        if bbox is None:
            ii = np.arange(ni)
            lon_out = self._lon_axis
            jj = np.arange(len(self._lat_axis))
        else:
            min_lon, min_lat, max_lon, max_lat = bbox
            lo, hi = min_lon % 360.0, max_lon % 360.0
            if hi <= lo:
                hi += 360.0
            lon_ext = np.concatenate([self._lon_axis, self._lon_axis + 360.0])
            sel = np.where((lon_ext >= lo - 1e-9) & (lon_ext <= hi + 1e-9))[0]
            if sel.size == 0:
                raise ValueError(f'bbox {bbox} selects no longitudes')
            ii = sel % ni
            lon_out = lon_ext[sel]
            jj = np.where((self._lat_axis >= min_lat - 1e-9) & (self._lat_axis <= max_lat + 1e-9))[0]
            if jj.size == 0:
                raise ValueError(f'bbox {bbox} selects no latitudes')
        lat_out = self._lat_axis[jj][::-1]  # ascending
        return ii, jj, lon_out, lat_out

    def _read_field(self, short: str, cat: str, level: float, step: int) -> np.ndarray:
        """One global field decoded, missing -> NaN, as (Nj, Ni) float64 in the file's own layout."""
        ec = _require_eccodes()
        path, offset = self._catalog[(short, cat)][level][step]
        with open(path, 'rb') as f:
            f.seek(offset)
            h = ec.codes_grib_new_from_file(f)
            try:
                ec.codes_set(h, 'missingValue', np.nan)
                values = ec.codes_get_values(h)
            finally:
                ec.codes_release(h)
        return values.reshape(int(self._grid['Nj']), int(self._grid['Ni']))

    def _clip(self, field: np.ndarray, ii, jj) -> np.ndarray:
        rolled = np.roll(field, -self._roll, axis=1)
        return rolled[jj][::-1][:, ii]

    def _source_step(self, short: str, cat: str, level: float, step: int, invariant: bool) -> int:
        by_step = self._catalog[(short, cat)][level]
        if step in by_step:
            return step
        if invariant and by_step:
            return sorted(by_step)[0]
        raise KeyError(f'{short} ({cat}, level {level}) has no message for step {step}')

    def _read_block(self, info: dict, level: float, leads: np.ndarray, ii, jj) -> np.ndarray:
        """(n_lead, ny, nx) float32 for one mapping entry at one level, transform applied."""
        cat = _category(info)
        transform = info.get('transform')
        invariant = bool(info.get('invariant'))
        level_pa = level * 100.0 if cat == 'pl' else None
        out = None
        for i, step in enumerate(leads):
            src = {}
            for sv in info['source_vars']:
                s = self._source_step(sv, cat, level, int(step), invariant)
                src[sv] = self._clip(self._read_field(sv, cat, level, s), ii, jj)
            if transform is None or transform in _STACK_TRANSFORMS:
                field = next(iter(src.values()))
            else:
                field = _TRANSFORMS[transform](src, level_pa)
            if out is None:
                out = np.empty((len(leads),) + field.shape, dtype='float32')
            out[i] = field
        if transform in _STACK_TRANSFORMS:
            out = _TRANSFORMS[transform](out, np.asarray(leads, dtype='float64')).astype('float32')
        return out

    # ------------------------------------------------------------------ pass 2: convert

    def convert(
        self,
        target,
        variables: Optional[List[str]] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        target_levels: Optional[List[float]] = None,
        max_lead_hours: Optional[int] = None,
        chunk_shape: Optional[Tuple[int, ...]] = None,
        overwrite: bool = False,
        **cfdb_kwargs,
    ) -> dict:
        """
        Append this cycle to a ``grid_forecast`` dataset.

        Parameters
        ----------
        target : path, or an open cfdb Dataset / EDataset
            A path is created if missing, else appended to (a remote-backed file is refused -- pass
            the ``open_edataset`` handle, whose ``push()`` / ``prune()`` the caller owns).
        variables : list of str or None
            Mapping keys, GRIB shortNames, or cfdb short names; None = every available field.
        bbox : (min_lon, min_lat, max_lon, max_lat) or None
            Degrees; longitudes in either -180..180 or 0..360, and the box may cross the dateline.
            Stored longitudes run 0..360 (a box such as 160..190 is stored as 160..190).
        target_levels : list of float (Pa) or None
            Subset of the native pressure levels (no interpolation); None = all.
        max_lead_hours : int or None
            Drop leads beyond this. IFS 00/12z cycles are 3-hourly to 144 h then 6-hourly; one
            regular lead axis cannot hold both, so pass 144 for those.
        chunk_shape : 5-tuple or None
            ``(1, n_lead, 1, ty, tx)``; default from ``forecast.forecast_chunk_shape``.
        overwrite : bool
            Replace an init the target already holds complete.
        **cfdb_kwargs
            For ``cfdb.open_dataset`` when creating a new file (e.g. compression).

        Returns
        -------
        dict with init, init_index, status ('new' | 'backfill' | 'overwrite'), autofilled, n_leads,
        variables, chunk_writes.
        """
        var_keys = self.resolve_variables(variables)
        leads = self.leads if max_lead_hours is None else self.leads[self.leads <= max_lead_hours]
        if leads.size == 0:
            raise ValueError('no leads selected')
        fc.lead_step(leads)

        if target_levels is not None:
            wanted = np.array(sorted(float(v) for v in target_levels))
            missing = [v for v in wanted if not np.isclose(self.pressure_levels, v).any()]
            if missing:
                raise ValueError(
                    f'target_levels {missing} Pa are not native IFS levels {self.pressure_levels.tolist()}'
                )
            levels_pa = wanted
        else:
            levels_pa = self.pressure_levels

        ii, jj, lon_out, lat_out = self._bbox_selection(bbox)
        ny, nx = len(lat_out), len(lon_out)

        level_vars, surface_vars, soil_vars, _ = group_variables(self.variables, var_keys)
        has_levels = bool(level_vars) and levels_pa.size > 0
        has_soil = bool(soil_vars)
        fixed_heights = {h for h, _ in surface_vars.values()}
        storage_chunk = tuple(chunk_shape) if chunk_shape is not None else fc.forecast_chunk_shape(len(leads), ny, nx)
        if len(storage_chunk) != 5:
            raise ValueError(f'chunk_shape must be 5-D (init, lead, z, y, x), got {storage_chunk}')

        with fc.open_target(target, **cfdb_kwargs) as (ds, created):
            if created:
                fc.create_forecast_coords(ds, self.init, leads, step_minutes=360)
                ds.create.coord.lat(data=lat_out, step=True)
                ds.create.coord.lon(data=lon_out, step=True)
                if has_levels:
                    ds.create.coord.pressure(data=levels_pa)
                if has_soil:
                    ds.create.coord.depth(data=self.soil_depths, axis=None)
                for h in sorted(fixed_heights):
                    axis = None if (has_levels or len(fixed_heights) > 1) else 'z'
                    ds.create.coord.generic(f'height_{int(h)}m', data=np.array([h], dtype='float64'), axis=axis)
                ds.create.crs.from_user_input(self.crs, x_coord=self.x_coord_name, y_coord=self.y_coord_name)
                for k, v in self._dataset_attrs().items():
                    ds.attrs[k] = v
                placed = {'index': 0, 'status': 'new', 'autofilled': 0}
            else:
                fc.validate_target(
                    ds,
                    x_name=self.x_coord_name,
                    y_name=self.y_coord_name,
                    x=lon_out,
                    y=lat_out,
                    levels=levels_pa if has_levels else None,
                    depths=self.soil_depths if has_soil else None,
                )
                placed = fc.place_init(ds, self.init, step_minutes=int(ds[fc.FRT].step or 360), overwrite=overwrite)
                fc.append_history(
                    ds,
                    f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} '
                    f'IfsIngest appended init {self.init}',
                )
            fc.unmark_init_complete(ds, self.init)

            writer = fc.ForecastWriter(ds, placed['index'], fc.lead_index_map(ds, leads), ny, nx)
            n_lead = len(leads)

            def _write(stored_name, coord_names, keys, level_values):
                data_var = create_cfdb_data_var(
                    ds,
                    stored_name,
                    coord_names,
                    storage_chunk,
                    dtype=next((self.variables[k].get('dtype') for k in keys if self.variables[k].get('dtype')), None),
                    attrs=next((self.variables[k].get('attrs') for k in keys if self.variables[k].get('attrs')), None),
                )
                packed = packed_range(data_var.dtype)
                for key in keys:
                    for k_idx, level in enumerate(level_values):
                        block = self._read_block(self.variables[key], level, leads, ii, jj)
                        if packed is not None:
                            check_packed_range(block, packed, f'{data_var.name} level {level}')
                        writer.put(data_var, k_idx, slice(0, n_lead), block)

            # every coordinate exists before the first data variable (cfdb creation order rule)
            for stored_name, keys in level_vars.items():
                _write(stored_name, (fc.FRT, fc.LEAD, 'pressure', 'latitude', 'longitude'), keys, levels_pa / 100.0)
            for stored_name, (h, keys) in surface_vars.items():
                _write(stored_name, (fc.FRT, fc.LEAD, f'height_{int(h)}m', 'latitude', 'longitude'), keys, [0.0])
            for stored_name, keys in soil_vars.items():
                _write(
                    stored_name,
                    (fc.FRT, fc.LEAD, 'depth', 'latitude', 'longitude'),
                    keys,
                    [float(i + 1) for i in range(len(self.soil_depths))],
                )

            writer.close()
            fc.mark_init_complete(ds, self.init)
            if placed['status'] == 'overwrite' and not fc.is_dataset(target):
                ds.prune()

        return {
            'init': str(self.init),
            'init_index': placed['index'],
            'status': placed['status'],
            'autofilled': placed['autofilled'],
            'n_leads': int(n_lead),
            'variables': list(var_keys),
            'chunk_writes': int(writer.writes),
        }

    def _dataset_attrs(self) -> dict:
        return {
            'Conventions': 'CF-1.11',
            'source': 'ECMWF IFS open data (oper, 0.25 deg), CC-BY-4.0 -- https://www.ecmwf.int/en/forecasts/datasets/open-data',
            'history': f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} created by cfdb-ingest IfsIngest',
            'source_files': [p.name for p in self.input_paths],
        }
