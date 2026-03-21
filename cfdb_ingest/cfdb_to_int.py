"""
Convert a cfdb dataset (with pressure-level 3D variables) to WPS intermediate files.
"""
import pathlib
from datetime import datetime, timedelta
from typing import Union

import numpy as np
import pyproj
import typer
import cfdb
from typing_extensions import Annotated

from cfdb_ingest import WPSUtils


######################################################
# Projection extraction from cfdb CRS


class MapProjection:
    """WPS intermediate file projection parameters."""

    def __init__(self, projType, startLat, startLon, startI, startJ, deltaLat, deltaLon,
                 dx=0.0, dy=0.0, truelat1=0.0, truelat2=0.0, xlonc=0.0):
        self.projType = projType
        self.startLat = startLat
        self.startLon = startLon
        self.startI = startI
        self.startJ = startJ
        self.deltaLat = deltaLat
        self.deltaLon = deltaLon
        self.dx = dx
        self.dy = dy
        self.truelat1 = truelat1
        self.truelat2 = truelat2
        self.xlonc = xlonc


def _extract_projection(ds):
    """
    Extract WPS-compatible projection from a cfdb dataset's CRS and coordinates.

    WPS intermediate file expects dx/dy in km.
    """
    crs = ds.crs
    x = ds['x'].data
    y = ds['y'].data

    cf_params = crs.to_cf()
    grid_mapping = cf_params.get('grid_mapping_name', '')

    # Compute startLat/startLon from SW corner
    transformer = pyproj.Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    sw_lon, sw_lat = transformer.transform(float(x[0]), float(y[0]))

    if grid_mapping == 'lambert_conformal_conic':
        std_parallel = cf_params.get('standard_parallel', [0.0, 0.0])
        if isinstance(std_parallel, (int, float)):
            truelat1 = truelat2 = float(std_parallel)
        else:
            truelat1, truelat2 = float(std_parallel[0]), float(std_parallel[1])

        dx_m = float(x[1] - x[0])
        dy_m = float(y[1] - y[0])

        return MapProjection(
            projType=WPSUtils.Projections.LC,
            startLat=sw_lat, startLon=sw_lon,
            startI=1.0, startJ=1.0,
            deltaLat=0.0, deltaLon=0.0,
            dx=dx_m / 1000.0, dy=dy_m / 1000.0,
            truelat1=truelat1, truelat2=truelat2,
            xlonc=float(cf_params.get('longitude_of_central_meridian', 0.0)),
        )

    elif grid_mapping == 'polar_stereographic':
        dx_m = float(x[1] - x[0])
        dy_m = float(y[1] - y[0])
        truelat1 = float(cf_params.get('standard_parallel', 60.0))
        xlonc = float(cf_params.get('straight_vertical_longitude_from_pole', 0.0))

        return MapProjection(
            projType=WPSUtils.Projections.PS,
            startLat=sw_lat, startLon=sw_lon,
            startI=1.0, startJ=1.0,
            deltaLat=0.0, deltaLon=0.0,
            dx=dx_m / 1000.0, dy=dy_m / 1000.0,
            truelat1=truelat1, xlonc=xlonc,
        )

    elif grid_mapping == 'mercator':
        dx_m = float(x[1] - x[0])
        dy_m = float(y[1] - y[0])
        truelat1 = float(cf_params.get('standard_parallel', 0.0))
        xlonc = float(cf_params.get('longitude_of_projection_origin', 0.0))

        return MapProjection(
            projType=WPSUtils.Projections.MERC,
            startLat=sw_lat, startLon=sw_lon,
            startI=1.0, startJ=1.0,
            deltaLat=0.0, deltaLon=0.0,
            dx=dx_m / 1000.0, dy=dy_m / 1000.0,
            truelat1=truelat1, xlonc=xlonc,
        )

    elif grid_mapping == 'latitude_longitude' or crs.to_epsg() == 4326:
        deltalat = float(y[1] - y[0])
        deltalon = float(x[1] - x[0])

        return MapProjection(
            projType=WPSUtils.Projections.LATLON,
            startLat=float(y[0]), startLon=float(x[0]),
            startI=1.0, startJ=1.0,
            deltaLat=deltalat, deltaLon=deltalon,
        )

    else:
        raise ValueError(f'Unsupported CRS grid mapping: {grid_mapping}')


######################################################
# Slab writing


def _write_slab(intfile, slab, xlvl, proj, WPSname, hdate, units, map_source, desc):
    """Write a 2D field slab to a WPS intermediate file."""
    missing_value = -1.0e30
    data = np.squeeze(np.asarray(slab, dtype=np.float64))
    masked = np.ma.array(data, mask=np.isnan(data))
    intfile.write_next_met_field(
        5, masked.shape[1], masked.shape[0], proj.projType, 0.0, xlvl,
        proj.startLat, proj.startLon, proj.startI, proj.startJ,
        proj.deltaLat, proj.deltaLon, proj.dx, proj.dy, proj.xlonc,
        proj.truelat1, proj.truelat2, 6371.229, 0, WPSname,
        hdate, units, map_source, desc, masked.filled(missing_value))


######################################################
# Variable mapping: cfdb_name -> WPS field info

# 3D pressure-level variables: (cfdb_name, WPS_field, units, description)
PRESSURE_LEVEL_VARS = [
    ('air_temp', 'TT', 'K', 'Temperature'),
    ('u_wind', 'UU', 'm s-1', 'U-component of wind'),
    ('v_wind', 'VV', 'm s-1', 'V-component of wind'),
    ('geopotential_height', 'GHT', 'm', 'Geopotential height'),
    ('relative_humidity', 'RH', '%', 'Relative humidity'),
    ('specific_humidity', 'SPECHUMD', 'kg kg-1', 'Specific humidity'),
]

# 2D surface variables: (cfdb_name, WPS_field, xlvl, units, description)
# cfdb_name may have _sfc suffix from conflict resolution
SURFACE_VARS = [
    ('surface_pressure', 'PSFC', 200100.0, 'Pa', 'Surface pressure'),
    ('mslp', 'PMSL', 201300.0, 'Pa', 'Mean sea level pressure'),
    ('soil_temp', 'SKINTEMP', 200100.0, 'K', 'Skin temperature'),
    ('air_temp', 'TT', 200100.0, 'K', 'Temperature'),
    ('air_temp_sfc', 'TT', 200100.0, 'K', 'Temperature'),
    ('u_wind', 'UU', 200100.0, 'm s-1', 'U-component of wind'),
    ('u_wind_sfc', 'UU', 200100.0, 'm s-1', 'U-component of wind'),
    ('v_wind', 'VV', 200100.0, 'm s-1', 'V-component of wind'),
    ('v_wind_sfc', 'VV', 200100.0, 'm s-1', 'V-component of wind'),
    ('dew_temp', 'DEWPT', 200100.0, 'K', 'Dewpoint temperature'),
    ('dew_temp_sfc', 'DEWPT', 200100.0, 'K', 'Dewpoint temperature'),
    ('relative_humidity', 'RH', 200100.0, '%', 'Relative humidity'),
    ('relative_humidity_sfc', 'RH', 200100.0, '%', 'Relative humidity'),
    ('land_sea_mask', 'LANDSEA', 200100.0, '0/1 Flag', 'Land-sea mask'),
    ('sea_ice', 'SEAICE', 200100.0, 'fraction', 'Sea ice fraction'),
    ('sea_surface_temp', 'SST', 200100.0, 'K', 'Sea surface temperature'),
    ('terrain_height', 'SOILHGT', 1.0, 'm', 'Terrain height'),
    ('snow_depth', 'SNOWH', 200100.0, 'm', 'Physical snow depth'),
    ('snow_water_equiv', 'SNOW', 200100.0, 'kg m-2', 'Water equivalent snow depth'),
]


######################################################
# Main conversion


def convert_cfdb_to_int(
    cfdb_path: Union[str, pathlib.Path],
    output_prefix: str = 'WRF',
    start_date: datetime = None,
    end_date: datetime = None,
    hour_interval: int = 6,
):
    """
    Convert a pressure-level cfdb dataset to WPS intermediate files.

    Parameters
    ----------
    cfdb_path : str or Path
        Path to the cfdb dataset.
    output_prefix : str
        Prefix for output intermediate files (e.g., 'WRF' produces 'WRF:YYYY-MM-DD_HH').
    start_date, end_date : datetime
        Date range to convert.
    hour_interval : int
        Interval in hours between output records.
    """
    with cfdb.open_dataset(str(cfdb_path)) as ds:
        proj = _extract_projection(ds)
        available_vars = set(ds.data_var_names)
        map_source = f'cfdb ({pathlib.Path(cfdb_path).name})'

        # Get time coordinate
        times = ds['time'].data  # numpy datetime64 array

        # Build target datetimes
        if start_date is not None:
            start = start_date.replace(minute=0, second=0, microsecond=0)
        else:
            start = times[0].astype('datetime64[h]').astype(datetime)

        if end_date is not None:
            end = end_date.replace(minute=0, second=0, microsecond=0)
        else:
            end = times[-1].astype('datetime64[h]').astype(datetime)

        intv = timedelta(hours=hour_interval)
        target_datetimes = set()
        curr = start
        while curr <= end:
            target_datetimes.add(np.datetime64(curr, 'm'))
            curr += intv

        # Check for pressure coordinate (required for 3D fields)
        has_pressure = 'pressure' in ds.coord_names
        pressure_levels = None
        if has_pressure:
            pressure_levels = ds['pressure'].data

        # Check for depth coordinate (soil fields)
        has_depth = 'depth' in ds.coord_names
        soil_depths = None
        if has_depth:
            soil_depths = ds['depth'].data

        print(f'Start date:     {start}')
        print(f'End date:       {end}')
        print(f'Hour interval:  {hour_interval}')
        if pressure_levels is not None:
            print(f'Pressure levels ({len(pressure_levels)}): {[int(p/100) for p in pressure_levels]} hPa')
        print(f'Available vars: {sorted(available_vars)}')

        for t_idx in range(len(times)):
            t_val = times[t_idx]
            t_val_m = np.datetime64(t_val, 'm')
            if t_val_m not in target_datetimes:
                continue

            # Format hdate for WPS: YYYY-MM-DD_HH:00:00
            dt = t_val_m.astype(datetime)
            hdate = dt.strftime('%Y-%m-%d_%H') + ':00:00'
            datestr = dt.strftime('%Y-%m-%d_%H')

            print(f'  Writing {output_prefix}:{datestr}')
            intfile = WPSUtils.IntermediateFile(output_prefix, datestr)

            # --- 3D pressure-level fields ---
            if has_pressure:
                for cfdb_name, wps_field, units, desc in PRESSURE_LEVEL_VARS:
                    if cfdb_name not in available_vars:
                        continue
                    dv = ds[cfdb_name]
                    for k in range(len(pressure_levels)):
                        xlvl = float(pressure_levels[k])
                        slab = dv[t_idx, k, :, :]
                        _write_slab(intfile, slab, xlvl, proj, wps_field, hdate, units, map_source, desc)

            # --- 2D surface fields ---
            for cfdb_name, wps_field, xlvl, units, desc in SURFACE_VARS:
                if cfdb_name not in available_vars:
                    continue
                dv = ds[cfdb_name]
                # Only process variables that are truly 2D (time, y, x)
                if dv.ndims != 3:
                    continue
                slab = dv[t_idx, :, :]
                _write_slab(intfile, slab, xlvl, proj, wps_field, hdate, units, map_source, desc)

            # --- Soil fields ---
            if has_depth:
                # Depth coordinate contains cumulative bottom boundaries in meters.
                # Layer k has top = bottom of layer k-1 (or 0), bottom = soil_depths[k].
                for d_idx in range(len(soil_depths)):
                    top_cm = 0 if d_idx == 0 else int(round(float(soil_depths[d_idx - 1]) * 100.0))
                    bot_cm = int(round(float(soil_depths[d_idx]) * 100.0))

                    sm_name = f'SM{top_cm:03d}{bot_cm:03d}'
                    st_name = f'ST{top_cm:03d}{bot_cm:03d}'

                    if 'soil_moisture' in available_vars:
                        slab = ds['soil_moisture'][t_idx, d_idx, :, :]
                        _write_slab(intfile, slab, 200100.0, proj, sm_name, hdate, 'm3 m-3', map_source, 'Soil moisture')

                    if 'soil_layer_temp' in available_vars:
                        slab = ds['soil_layer_temp'][t_idx, d_idx, :, :]
                        _write_slab(intfile, slab, 200100.0, proj, st_name, hdate, 'K', map_source, 'Soil temperature')

            intfile.close()

    print('\nDone.')


######################################################
# Standalone CLI

app = typer.Typer()


@app.command()
def main(
    cfdb_path: Annotated[pathlib.Path, typer.Argument(help='Input cfdb dataset path.', exists=True, resolve_path=True)],
    start_date: Annotated[datetime, typer.Option(
        '--start-date', '-s', help='Starting date-time to convert.',
        formats=['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d_%H'],
    )],
    end_date: Annotated[datetime, typer.Option(
        '--end-date', '-e', help='Ending date-time to convert.',
        formats=['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d_%H'],
    )],
    hour_interval: Annotated[int, typer.Option('--hour-interval', '-h', help='Interval in hours between records.')] = 6,
    output_prefix: Annotated[str, typer.Option('--prefix', '-p', help='Output file prefix.')] = 'WRF',
):
    """Convert a cfdb dataset to WPS intermediate files for metgrid.exe."""
    convert_cfdb_to_int(
        cfdb_path=cfdb_path,
        output_prefix=output_prefix,
        start_date=start_date,
        end_date=end_date,
        hour_interval=hour_interval,
    )
