"""
Convert a cfdb dataset to WPS intermediate files (one file per valid time) for ``metgrid.exe``.

Handles both dataset layouts cfdb-ingest produces:

- ``grid``: ``(time, <level|height_Xm|depth>, y, x)`` -- one file per selected ``time``;
- ``grid_forecast``: ``(forecast_reference_time, forecast_period, ..., y, x)`` -- one INIT is
  selected (``init=``) and one file is written per lead, named by the valid time ``init + lead``.

Variable matching is on the stored (cfdb-vars full) names after stripping a height suffix, so
``air_temperature``, ``air_temperature_2m`` and the short spellings in the tables all line up.
Relative humidity is stored as a fraction and written to WPS in percent.

Reads are chunk-aligned: a ``grid`` dataset is read one (time, level) slab at a time (its storage
chunk), a ``grid_forecast`` one (init, level) row at a time -- never a whole variable.
"""

import pathlib
import re
from datetime import datetime
from typing import Optional, Union

import numpy as np
import pyproj
import typer
import cfdb
from cfdb_vars import short_name_map
from typing_extensions import Annotated

from wrf_to_int import IntermediateFile, Projections, MapProjection, write_slab

from cfdb_ingest import forecast as fc
from cfdb_ingest import thermo

_HEIGHT_SUFFIX = re.compile(r'_(\d+)m$')
_DATE_FORMATS = ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M', '%Y-%m-%dT%H', '%Y-%m-%d_%H']


def _canon(name: str) -> str:
    """Stored-name spelling of ``name`` with any height suffix removed: ``air_temp_2m`` -> ``air_temperature``."""
    m = _HEIGHT_SUFFIX.search(name)
    base = name[: m.start()] if m else name
    return short_name_map.get(base, base)


######################################################
# Projection extraction from cfdb CRS


def _xy_coord_names(ds):
    """The x and y coordinate names, from the coordinates' axis metadata (falling back to conventional names)."""
    x_name = y_name = None
    for name in ds.coord_names:
        axis = ds[name].axis
        axis = getattr(axis, 'value', axis)
        if axis == 'x':
            x_name = name
        elif axis == 'y':
            y_name = name
    if x_name is None:
        x_name = next((n for n in ('x', 'longitude', 'lon') if n in ds.coord_names), None)
    if y_name is None:
        y_name = next((n for n in ('y', 'latitude', 'lat') if n in ds.coord_names), None)
    if x_name is None or y_name is None:
        raise ValueError(f'could not identify the x/y coordinates among {ds.coord_names}')
    return x_name, y_name


def _extract_projection(ds, x_name, y_name):
    """
    Extract a WPS-compatible projection from a cfdb dataset's CRS and coordinates (dx/dy in km).
    """
    crs = ds.crs
    if crs is None:
        raise ValueError('dataset has no CRS; cannot build the WPS projection record')
    x = ds[x_name].data
    y = ds[y_name].data

    cf_params = crs.to_cf()
    grid_mapping = cf_params.get('grid_mapping_name', '')

    if grid_mapping == 'latitude_longitude' or crs.to_epsg() == 4326:
        return MapProjection(
            projType=Projections.LATLON,
            startLat=float(y[0]),
            startLon=float(x[0]),
            startI=1.0,
            startJ=1.0,
            deltaLat=float(y[1] - y[0]),
            deltaLon=float(x[1] - x[0]),
        )

    # projected grids: startLat/startLon from the SW corner
    transformer = pyproj.Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    sw_lon, sw_lat = transformer.transform(float(x[0]), float(y[0]))
    dx_km = float(x[1] - x[0]) / 1000.0
    dy_km = float(y[1] - y[0]) / 1000.0

    if grid_mapping == 'lambert_conformal_conic':
        std_parallel = cf_params.get('standard_parallel', [0.0, 0.0])
        if isinstance(std_parallel, (int, float)):
            truelat1 = truelat2 = float(std_parallel)
        else:
            truelat1, truelat2 = float(std_parallel[0]), float(std_parallel[1])
        return MapProjection(
            projType=Projections.LC,
            startLat=sw_lat,
            startLon=sw_lon,
            startI=1.0,
            startJ=1.0,
            deltaLat=0.0,
            deltaLon=0.0,
            dx=dx_km,
            dy=dy_km,
            truelat1=truelat1,
            truelat2=truelat2,
            xlonc=float(cf_params.get('longitude_of_central_meridian', 0.0)),
        )
    if grid_mapping == 'polar_stereographic':
        return MapProjection(
            projType=Projections.PS,
            startLat=sw_lat,
            startLon=sw_lon,
            startI=1.0,
            startJ=1.0,
            deltaLat=0.0,
            deltaLon=0.0,
            dx=dx_km,
            dy=dy_km,
            truelat1=float(cf_params.get('standard_parallel', 60.0)),
            xlonc=float(cf_params.get('straight_vertical_longitude_from_pole', 0.0)),
        )
    if grid_mapping == 'mercator':
        return MapProjection(
            projType=Projections.MERC,
            startLat=sw_lat,
            startLon=sw_lon,
            startI=1.0,
            startJ=1.0,
            deltaLat=0.0,
            deltaLon=0.0,
            dx=dx_km,
            dy=dy_km,
            truelat1=float(cf_params.get('standard_parallel', 0.0)),
            xlonc=float(cf_params.get('longitude_of_projection_origin', 0.0)),
        )
    raise ValueError(f'Unsupported CRS grid mapping: {grid_mapping}')


######################################################
# Variable tables: cfdb name (short or full) -> WPS field

# 3D pressure-level variables: (cfdb_name, WPS_field, units, description, scale)
PRESSURE_LEVEL_VARS = [
    ('air_temp', 'TT', 'K', 'Temperature', 1.0),
    ('u_wind', 'UU', 'm s-1', 'U-component of wind', 1.0),
    ('v_wind', 'VV', 'm s-1', 'V-component of wind', 1.0),
    ('geopotential_height', 'GHT', 'm', 'Geopotential height', 1.0),
    ('relative_humidity', 'RH', '%', 'Relative humidity', 100.0),
    ('specific_humidity', 'SPECHUMD', 'kg kg-1', 'Specific humidity', 1.0),
]

# 2D surface variables: (cfdb_name, WPS_field, xlvl, units, description, scale)
SURFACE_VARS = [
    ('surface_pressure', 'PSFC', 200100.0, 'Pa', 'Surface pressure', 1.0),
    ('mslp', 'PMSL', 201300.0, 'Pa', 'Mean sea level pressure', 1.0),
    ('skin_temp', 'SKINTEMP', 200100.0, 'K', 'Skin temperature', 1.0),
    ('soil_temp', 'SKINTEMP', 200100.0, 'K', 'Skin temperature', 1.0),  # WRF TSK is stored as soil_temperature
    ('air_temp', 'TT', 200100.0, 'K', 'Temperature', 1.0),
    ('u_wind', 'UU', 200100.0, 'm s-1', 'U-component of wind', 1.0),
    ('v_wind', 'VV', 200100.0, 'm s-1', 'V-component of wind', 1.0),
    ('dew_temp', 'DEWPT', 200100.0, 'K', 'Dewpoint temperature', 1.0),
    ('relative_humidity', 'RH', 200100.0, '%', 'Relative humidity', 100.0),
    ('land_sea_mask', 'LANDSEA', 200100.0, '0/1 Flag', 'Land-sea mask', 1.0),
    ('sea_ice', 'SEAICE', 200100.0, 'fraction', 'Sea ice fraction', 1.0),
    ('sea_surface_temp', 'SST', 200100.0, 'K', 'Sea surface temperature', 1.0),
    (
        'terrain_height',
        'SOILHGT',
        200100.0,
        'm',
        'Terrain height',
        1.0,
    ),  # a surface field: METGRID.TBL fills GHT from SOILHGT(200100)
    ('snow_depth', 'SNOWH', 200100.0, 'm', 'Physical snow depth', 1.0),
    ('snow_water_equiv', 'SNOW', 200100.0, 'kg m-2', 'Water equivalent snow depth', 1.0),
]

# Soil layer variables: (cfdb_name, WPS prefix, units, description)
SOIL_VARS = [
    ('soil_moisture', 'SM', 'm3 m-3', 'Soil moisture'),
    ('soil_layer_temp', 'ST', 'K', 'Soil temperature'),
]

# Fields whose WPS meaning is a 0/1 flag; IFS stores a fraction.
_FLAG_FIELDS = {'LANDSEA'}


######################################################
# Dataset introspection


def _classify_data_vars(ds):
    """
    {'levels': {canon: dv}, 'surface': {canon: dv}, 'soil': {canon: dv}} keyed by canonical name.

    A variable is a level field when it has the ``pressure`` coordinate, a soil field when it has
    ``depth``, and a surface field otherwise (a ``height_Xm`` coordinate, or the legacy 3-D layout).
    """
    groups = {'levels': {}, 'surface': {}, 'soil': {}}
    for name in ds.data_var_names:
        dv = ds[name]
        coords = dv.coord_names
        canon = _canon(name)
        group = 'levels' if 'pressure' in coords else 'soil' if 'depth' in coords else 'surface'
        if canon in groups[group]:
            raise ValueError(
                f'two variables resolve to {canon!r} in the {group} group: '
                f'{groups[group][canon].name!r} and {name!r}'
            )
        groups[group][canon] = dv
    return groups


def _parse_dt(value) -> Optional[np.datetime64]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return np.datetime64(value.replace(second=0, microsecond=0), 'm')
    return np.datetime64(value, 'm')


def _records(ds, init, start_date, end_date, hour_interval):
    """
    The (index_prefix, valid_time) records to export. ``index_prefix`` is ``(t_idx,)`` for a grid
    dataset or ``(init_idx, lead_idx)`` for a grid_forecast one.
    """
    if ds.dataset_type == 'grid_forecast':
        inits = np.asarray(ds[fc.FRT].data).astype('datetime64[m]')
        if init is None:
            raise ValueError(
                f'init= is required for a grid_forecast dataset; available inits: ' f'{[str(i) for i in inits]}'
            )
        init64 = _parse_dt(init)
        hits = np.where(inits == init64)[0]
        if hits.size == 0:
            raise ValueError(f'init {init64} is not in the dataset; available inits: {[str(i) for i in inits]}')
        init_idx = int(hits[0])
        if fc.COMPLETE_INITS_ATTR in ds.attrs.data:
            if not fc.is_init_complete(ds, init64):
                raise ValueError(
                    f'init {init64} is not marked complete in the dataset (ingest unfinished?); refusing '
                    f'to write a partial boundary-condition set'
                )
        elif not fc._init_has_chunks(ds, init_idx):
            raise ValueError(f'init {init64} holds no data')
        valids = fc.valid_times(ds, init_idx)
        prefixes = [(init_idx, k) for k in range(len(valids))]
    else:
        if init is not None:
            raise ValueError('init= only applies to grid_forecast datasets')
        valids = np.asarray(ds['time'].data).astype('datetime64[m]')
        prefixes = [(t,) for t in range(len(valids))]

    start = _parse_dt(start_date) if start_date is not None else valids[0]
    end = _parse_dt(end_date) if end_date is not None else valids[-1]
    wanted = set()
    curr = start
    step = np.timedelta64(int(hour_interval), 'h')
    while curr <= end:
        wanted.add(curr)
        curr = curr + step
    return [(p, v) for p, v in zip(prefixes, valids) if v in wanted]


def _slab(dv, prefix, level_idx=None) -> np.ndarray:
    """One 2-D slab of a grid variable, read as a single chunk-aligned selection."""
    sel = tuple(prefix)
    if level_idx is not None:
        sel = sel + (level_idx,)
    elif dv.ndims > len(prefix) + 2:
        sel = sel + (0,)
    sel = sel + (slice(None), slice(None))
    arr = np.asarray(dv[sel].data, dtype='float64')
    return arr.reshape(arr.shape[-2:])


def _row(dv, init_idx, level_idx=None) -> np.ndarray:
    """One (init, level) chunk-row of a grid_forecast variable: (n_lead, ny, nx)."""
    sel = (init_idx, slice(None))
    if level_idx is not None:
        sel = sel + (level_idx,)
    elif dv.ndims > 4:
        sel = sel + (0,)
    sel = sel + (slice(None), slice(None))
    arr = np.asarray(dv[sel].data, dtype='float64')
    return arr.reshape((arr.shape[1],) + arr.shape[-2:])


def _finish(slab, scale, wps):
    out = np.asarray(slab, dtype='float64')
    if scale != 1.0:
        out = out * scale
    if wps in _FLAG_FIELDS:
        out = np.where(np.isnan(out), np.nan, (out >= 0.5).astype('float64'))
    return out


######################################################
# Main conversion


def convert_cfdb_to_int(
    cfdb_path: Union[str, pathlib.Path],
    output_prefix: str = 'WRF',
    start_date=None,
    end_date=None,
    hour_interval: int = 6,
    init=None,
):
    """
    Convert a cfdb dataset to WPS intermediate files.

    Parameters
    ----------
    cfdb_path : str or Path
        Path to the cfdb dataset (``grid`` or ``grid_forecast``).
    output_prefix : str
        Prefix for the output files: ``'WRF'`` produces ``WRF:YYYY-MM-DD_HH`` in the current
        directory; a prefix with a directory part writes there (the directory must exist).
    start_date, end_date : str, datetime, np.datetime64 or None
        Valid-time range to convert (defaults: the dataset's / the init's full range).
    hour_interval : int
        Interval in hours between output records, counted from ``start_date``.
    init : str, datetime, np.datetime64 or None
        ``grid_forecast`` only (required there): the forecast init to export. Refused unless the
        dataset marks it complete.

    Returns
    -------
    list of pathlib.Path
        The files written.
    """
    with cfdb.open_dataset(str(cfdb_path)) as ds:
        x_name, y_name = _xy_coord_names(ds)
        proj = _extract_projection(ds, x_name, y_name)
        map_source = f'cfdb ({pathlib.Path(cfdb_path).name})'
        groups = _classify_data_vars(ds)
        records = _records(ds, init, start_date, end_date, hour_interval)
        if not records:
            raise ValueError('no records selected')
        is_forecast = ds.dataset_type == 'grid_forecast'

        pressure_levels = np.asarray(ds['pressure'].data) if 'pressure' in ds.coord_names else None
        soil_depths = np.asarray(ds['depth'].data) if 'depth' in ds.coord_names else None

        # the field list: (group, canonical cfdb name, WPS field, xlvl or None, units, desc, scale)
        fields = []
        used_wps = {}
        for cfdb_name, wps, units, desc, scale in PRESSURE_LEVEL_VARS:
            canon = _canon(cfdb_name)
            if canon in groups['levels']:
                fields.append(('levels', canon, wps, None, units, desc, scale))
        for cfdb_name, wps, xlvl, units, desc, scale in SURFACE_VARS:
            canon = _canon(cfdb_name)
            if canon in groups['surface']:
                if wps in used_wps:
                    raise ValueError(
                        f'both {used_wps[wps]!r} and {canon!r} would be written as WPS {wps}; '
                        f'the dataset must hold only one of them'
                    )
                used_wps[wps] = canon
                fields.append(('surface', canon, wps, xlvl, units, desc, scale))
        # 2 m RH diagnosed from T/Td when the dataset carries no surface relative humidity
        rh2_from_td = (
            'RH' not in used_wps
            and 'air_temperature' in groups['surface']
            and 'dew_point_temperature' in groups['surface']
        )
        soil_fields = [(_canon(n), p, u, d) for n, p, u, d in SOIL_VARS if _canon(n) in groups['soil']]

        print(f'Records:        {len(records)} ({records[0][1]} .. {records[-1][1]}), interval {hour_interval} h')
        if pressure_levels is not None:
            print(f'Pressure levels ({len(pressure_levels)}): {[int(p / 100) for p in pressure_levels]} hPa')
        print(
            f'Fields:         {sorted({f[2] for f in fields} | ({"RH"} if rh2_from_td else set()))}'
            f'{" + soil " + str([p for _, p, _, _ in soil_fields]) if soil_fields else ""}'
        )

        files = {}
        try:
            for prefix, valid in records:
                dt = valid.astype(datetime)
                files[prefix] = (
                    IntermediateFile(output_prefix, dt.strftime('%Y-%m-%d_%H')),
                    dt.strftime('%Y-%m-%d_%H') + ':00:00',
                )

            def slabs(dv, level_idx=None):
                """Yield (prefix, slab) per record, reading one chunk-aligned unit at a time."""
                if is_forecast:
                    row = _row(dv, records[0][0][0], level_idx)
                    for prefix, _ in records:
                        yield prefix, row[prefix[1]]
                else:
                    for prefix, _ in records:
                        yield prefix, _slab(dv, prefix, level_idx)

            def emit(dv, wps, xlvl, units, desc, scale, level_idx=None):
                for prefix, slab in slabs(dv, level_idx):
                    intfile, hdate = files[prefix]
                    write_slab(intfile, _finish(slab, scale, wps), xlvl, proj, wps, hdate, units, map_source, desc)

            for group, canon, wps, xlvl, units, desc, scale in fields:
                dv = groups[group][canon]
                if group == 'levels':
                    for k in range(len(pressure_levels)):
                        emit(dv, wps, float(pressure_levels[k]), units, desc, scale, level_idx=k)
                else:
                    emit(dv, wps, xlvl, units, desc, scale)

            if rh2_from_td:
                t_slabs = dict(slabs(groups['surface']['air_temperature']))
                for prefix, td in slabs(groups['surface']['dew_point_temperature']):
                    intfile, hdate = files[prefix]
                    rh = thermo.rh_from_t_td(t_slabs[prefix], td) * 100.0
                    write_slab(intfile, rh, 200100.0, proj, 'RH', hdate, '%', map_source, 'Relative humidity')

            for canon, wps_prefix, units, desc in soil_fields:
                dv = groups['soil'][canon]
                # Depth coordinate holds cumulative layer bottoms (m); layer k spans bottom[k-1]..bottom[k].
                for d_idx in range(len(soil_depths)):
                    top_cm = 0 if d_idx == 0 else int(round(float(soil_depths[d_idx - 1]) * 100.0))
                    bot_cm = int(round(float(soil_depths[d_idx]) * 100.0))
                    emit(dv, f'{wps_prefix}{top_cm:03d}{bot_cm:03d}', 200100.0, units, desc, 1.0, level_idx=d_idx)
        finally:
            for intfile, _ in files.values():
                intfile.close()

    written = [pathlib.Path(intfile.filename_) for intfile, _ in files.values()]
    print(f'\nWrote {len(written)} files.')
    return written


######################################################
# Standalone CLI

app = typer.Typer()


@app.command()
def main(
    cfdb_path: Annotated[pathlib.Path, typer.Argument(help='Input cfdb dataset path.', exists=True, resolve_path=True)],
    start_date: Annotated[
        Optional[datetime],
        typer.Option(
            '--start-date',
            '-s',
            help='First valid time to convert (default: the first available).',
            formats=_DATE_FORMATS,
        ),
    ] = None,
    end_date: Annotated[
        Optional[datetime],
        typer.Option(
            '--end-date',
            '-e',
            help='Last valid time to convert (default: the last available).',
            formats=_DATE_FORMATS,
        ),
    ] = None,
    hour_interval: Annotated[int, typer.Option('--hour-interval', '-h', help='Interval in hours between records.')] = 6,
    output_prefix: Annotated[
        str, typer.Option('--prefix', '-p', help='Output file prefix (may include a directory).')
    ] = 'WRF',
    init: Annotated[
        Optional[datetime],
        typer.Option(
            '--init',
            '-i',
            help='grid_forecast datasets: the forecast init to export (required).',
            formats=_DATE_FORMATS,
        ),
    ] = None,
):
    """Convert a cfdb dataset to WPS intermediate files for metgrid.exe."""
    convert_cfdb_to_int(
        cfdb_path=cfdb_path,
        output_prefix=output_prefix,
        start_date=start_date,
        end_date=end_date,
        hour_interval=hour_interval,
        init=init,
    )
