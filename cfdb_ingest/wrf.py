"""
WRF output file converter for cfdb.
"""
import pathlib
from typing import Union, List, Tuple, Dict, Optional

import h5py
import numpy as np
import pyproj

from cfdb_ingest.base import H5Ingest


def _wrf_attr(attrs, key):
    """Extract a scalar value from a WRF HDF5 attribute (may be a 1-element array)."""
    val = attrs[key]
    if hasattr(val, 'item'):
        return val.item()
    if isinstance(val, bytes):
        return val.decode()
    return val


WRF_VARIABLE_MAPPING = {
    # --- Surface variables (height in meters above ground) ---
    'T2': {
        'cfdb_name': 'air_temp',
        'source_vars': ['T2'],
        'transform': None,
        'height': 2.0,
    },
    'PSFC': {
        'cfdb_name': 'surface_pressure',
        'source_vars': ['PSFC'],
        'transform': None,
        'height': 0.0,
    },
    'Q2': {
        'cfdb_name': 'mixing_ratio',
        'source_vars': ['Q2'],
        'transform': None,
        'height': 2.0,
    },
    'Q2_SH': {
        'cfdb_name': 'specific_humidity',
        'source_vars': ['Q2'],
        'transform': 'mixing_ratio_to_specific_humidity_2d',
        'height': 2.0,
    },
    'RAIN': {
        'cfdb_name': 'precip',
        'source_vars': ['RAINNC', 'RAINC'],
        'transform': 'accumulation_increment',
        'height': 0.0,
    },
    'WIND10': {
        'cfdb_name': 'wind_speed',
        'source_vars': ['U10', 'V10'],
        'transform': 'wind_speed',
        'height': 10.0,
    },
    'WIND_DIR10': {
        'cfdb_name': 'wind_direction',
        'source_vars': ['U10', 'V10'],
        'transform': 'wind_direction',
        'height': 10.0,
    },
    'TSK': {
        'cfdb_name': 'soil_temp',
        'source_vars': ['TSK'],
        'transform': None,
        'height': 0.0,
    },
    'SWDOWN': {
        'cfdb_name': 'shortwave_radiation',
        'source_vars': ['SWDOWN'],
        'transform': None,
        'height': 0.0,
    },
    'GLW': {
        'cfdb_name': 'longwave_radiation',
        'source_vars': ['GLW'],
        'transform': None,
        'height': 0.0,
    },
    'SNOWH': {
        'cfdb_name': 'snow_depth',
        'source_vars': ['SNOWH'],
        'transform': None,
        'height': 0.0,
    },
    'HFX': {
        'cfdb_name': 'sensible_heat_flux',
        'source_vars': ['HFX'],
        'transform': None,
        'height': 0.0,
    },
    'QFX': {
        'cfdb_name': 'moisture_flux',
        'source_vars': ['QFX'],
        'transform': None,
        'height': 0.0,
    },
    'ALBEDO': {
        'cfdb_name': 'albedo',
        'source_vars': ['ALBEDO'],
        'transform': None,
        'height': 0.0,
    },
    'EMISS': {
        'cfdb_name': 'emissivity',
        'source_vars': ['EMISS'],
        'transform': None,
        'height': 0.0,
    },
    'LU_INDEX': {
        'cfdb_name': 'land_use_modis',
        'source_vars': ['LU_INDEX'],
        'transform': None,
        'height': 0.0,
    },
    # --- Level-interpolated variables ---
    'T': {
        'cfdb_name': 'air_temp',
        'source_vars': ['T', 'P', 'PB', 'PH', 'PHB'],
        'transform': 'potential_to_actual_temp',
        'height': 'levels',
    },
    'WIND': {
        'cfdb_name': 'wind_speed',
        'source_vars': ['U', 'V', 'PH', 'PHB'],
        'transform': 'wind_speed_3d',
        'height': 'levels',
    },
    'WIND_DIR': {
        'cfdb_name': 'wind_direction',
        'source_vars': ['U', 'V', 'PH', 'PHB'],
        'transform': 'wind_direction_3d',
        'height': 'levels',
    },
    # --- Individual wind components ---
    'U10': {
        'cfdb_name': 'u_wind',
        'source_vars': ['U10', 'V10'],
        'transform': 'u_wind',
        'height': 10.0,
    },
    'V10': {
        'cfdb_name': 'v_wind',
        'source_vars': ['U10', 'V10'],
        'transform': 'v_wind',
        'height': 10.0,
    },
    'U': {
        'cfdb_name': 'u_wind',
        'source_vars': ['U', 'V', 'PH', 'PHB'],
        'transform': 'u_wind_3d',
        'height': 'levels',
    },
    'V': {
        'cfdb_name': 'v_wind',
        'source_vars': ['U', 'V', 'PH', 'PHB'],
        'transform': 'v_wind_3d',
        'height': 'levels',
    },
    # --- 3D moisture ---
    'QVAPOR': {
        'cfdb_name': 'mixing_ratio',
        'source_vars': ['QVAPOR', 'PH', 'PHB'],
        'transform': 'mixing_ratio_3d',
        'height': 'levels',
    },
    'Q_SH': {
        'cfdb_name': 'specific_humidity',
        'source_vars': ['QVAPOR', 'PH', 'PHB'],
        'transform': 'mixing_ratio_to_specific_humidity',
        'height': 'levels',
    },
    # --- Vorticity ---
    'VORT10': {
        'cfdb_name': 'vorticity',
        'source_vars': ['U10', 'V10'],
        'transform': 'vorticity',
        'height': 10.0,
    },
    'VORT': {
        'cfdb_name': 'vorticity',
        'source_vars': ['U', 'V', 'PH', 'PHB'],
        'transform': 'vorticity_3d',
        'height': 'levels',
    },
    # --- Vertical velocity ---
    'W': {
        'cfdb_name': 'vertical_velocity',
        'source_vars': ['W', 'PH', 'PHB'],
        'transform': 'vertical_velocity_3d',
        'height': 'levels',
    },
    # --- Sea level pressure ---
    'SLP': {
        'cfdb_name': 'mslp',
        'source_vars': ['PSFC', 'T2', 'HGT'],
        'transform': 'sea_level_pressure',
        'height': 0.0,
    },
}


def unstagger(data, axis):
    """
    Average adjacent points along a staggered WRF dimension.

    Parameters
    ----------
    data : np.ndarray
        Array with a staggered dimension.
    axis : int
        Axis index of the staggered dimension.

    Returns
    -------
    np.ndarray
        Unstaggered array with size reduced by 1 along the given axis.
    """
    slices_lo = [slice(None)] * data.ndim
    slices_hi = [slice(None)] * data.ndim
    slices_lo[axis] = slice(None, -1)
    slices_hi[axis] = slice(1, None)
    return (data[tuple(slices_lo)] + data[tuple(slices_hi)]) / 2.0


class WrfIngest(H5Ingest):
    """
    Convert WRF output files to cfdb.

    Handles WRF-specific features including CRS extraction from MAP_PROJ
    attributes, wind rotation from grid-relative to earth-relative using
    COSALPHA/SINALPHA, precipitation increment computation from accumulated
    fields, and 3D variable level interpolation from eta to height coordinates.

    Parameters
    ----------
    input_paths : str, Path, or list thereof
        One or more wrfout file paths.
    """

    def _init_metadata(self):
        """
        Override to also load wind rotation fields (COSALPHA, SINALPHA).
        """
        with h5py.File(self.input_paths[0], 'r') as h5:
            self.crs = self._parse_crs(h5)
            spatial = self._parse_spatial_coords(h5)

            # Load wind rotation fields (constant across time)
            if 'COSALPHA' in h5 and 'SINALPHA' in h5:
                self._cosalpha = h5['COSALPHA'][0]
                self._sinalpha = h5['SINALPHA'][0]
            else:
                self._cosalpha = None
                self._sinalpha = None

        self.x = spatial['x']
        self.y = spatial['y']
        self._dx = float(self.x[1] - self.x[0])
        self._dy = float(self.y[1] - self.y[0])

        self._init_time()
        self._init_variables()
        self._compute_bbox_geographic()

    def _parse_crs(self, h5):
        """
        Extract CRS from WRF global attributes.

        Supports MAP_PROJ values:
        - 1: Lambert Conformal Conic
        - 2: Polar Stereographic
        - 3: Mercator
        - 6: Lat-Lon (EPSG:4326)
        """
        attrs = h5.attrs
        map_proj = _wrf_attr(attrs, 'MAP_PROJ')

        if map_proj == 1:
            truelat1 = _wrf_attr(attrs, 'TRUELAT1')
            truelat2 = _wrf_attr(attrs, 'TRUELAT2')
            stand_lon = _wrf_attr(attrs, 'STAND_LON')
            cen_lat = _wrf_attr(attrs, 'CEN_LAT')
            return pyproj.CRS.from_cf({
                'grid_mapping_name': 'lambert_conformal_conic',
                'standard_parallel': [truelat1, truelat2],
                'longitude_of_central_meridian': stand_lon,
                'latitude_of_projection_origin': cen_lat,
                'false_easting': 0.0,
                'false_northing': 0.0,
            })

        elif map_proj == 2:
            truelat1 = _wrf_attr(attrs, 'TRUELAT1')
            stand_lon = _wrf_attr(attrs, 'STAND_LON')
            cen_lat = _wrf_attr(attrs, 'CEN_LAT')
            return pyproj.CRS.from_cf({
                'grid_mapping_name': 'polar_stereographic',
                'straight_vertical_longitude_from_pole': stand_lon,
                'latitude_of_projection_origin': 90.0 if cen_lat > 0 else -90.0,
                'standard_parallel': truelat1,
                'false_easting': 0.0,
                'false_northing': 0.0,
            })

        elif map_proj == 3:
            truelat1 = _wrf_attr(attrs, 'TRUELAT1')
            stand_lon = _wrf_attr(attrs, 'STAND_LON')
            return pyproj.CRS.from_cf({
                'grid_mapping_name': 'mercator',
                'longitude_of_projection_origin': stand_lon,
                'standard_parallel': truelat1,
                'false_easting': 0.0,
                'false_northing': 0.0,
            })

        elif map_proj == 6:
            return pyproj.CRS.from_epsg(4326)

        else:
            raise ValueError(f'Unsupported WRF MAP_PROJ: {map_proj}')

    def _parse_time(self, h5):
        """
        Parse WRF Times character array to datetime64[m].

        WRF stores times as a (n_times, 19) byte array with format
        "YYYY-MM-DD_HH:MM:SS".
        """
        times_raw = h5['Times'][:]
        return np.array([
            np.datetime64(b''.join(row).decode().replace('_', 'T'), 'm')
            for row in times_raw
        ])

    def _parse_spatial_coords(self, h5):
        """
        Compute projected x/y coordinate arrays from WRF grid info.

        Uses DX, DY grid spacing and CEN_LAT, CEN_LON to derive 1D coordinate
        arrays in the projected CRS. For MAP_PROJ=6 (lat-lon), uses XLAT/XLONG
        directly.
        """
        attrs = h5.attrs
        map_proj = _wrf_attr(attrs, 'MAP_PROJ')

        if map_proj == 6:
            xlat = h5['XLAT'][0]
            xlong = h5['XLONG'][0]
            return {
                'x': xlong[0, :].astype('float64'),
                'y': xlat[:, 0].astype('float64'),
            }

        dx = _wrf_attr(attrs, 'DX')
        dy = _wrf_attr(attrs, 'DY')
        cen_lat = _wrf_attr(attrs, 'CEN_LAT')
        cen_lon = _wrf_attr(attrs, 'CEN_LON')

        ny = _wrf_attr(attrs, 'SOUTH-NORTH_PATCH_END_UNSTAG') - _wrf_attr(attrs, 'SOUTH-NORTH_PATCH_START_UNSTAG') + 1
        nx = _wrf_attr(attrs, 'WEST-EAST_PATCH_END_UNSTAG') - _wrf_attr(attrs, 'WEST-EAST_PATCH_START_UNSTAG') + 1

        transformer = pyproj.Transformer.from_crs('EPSG:4326', self.crs, always_xy=True)
        center_x, center_y = transformer.transform(cen_lon, cen_lat)

        center_i = (nx - 1) / 2.0
        center_j = (ny - 1) / 2.0
        x = center_x + (np.arange(nx) - center_i) * dx
        y = center_y + (np.arange(ny) - center_j) * dy

        return {'x': x, 'y': y}

    def _init_variables(self):
        """
        Override to prefer PREC_ACC_C/PREC_ACC_NC (pre-computed hourly precip)
        over RAINC/RAINNC (running accumulations) when available.
        """
        super()._init_variables()
        if 'RAIN' in self.variables:
            with h5py.File(self.input_paths[0], 'r') as h5:
                if 'PREC_ACC_C' in h5 and 'PREC_ACC_NC' in h5:
                    self.variables['RAIN'] = {
                        'cfdb_name': 'precip',
                        'source_vars': ['PREC_ACC_C', 'PREC_ACC_NC'],
                        'transform': 'precip_sum',
                        'height': 0.0,
                    }

    def _get_variable_mapping(self):
        """Return the WRF variable mapping dictionary."""
        return WRF_VARIABLE_MAPPING

    def _read_variable(self, h5, var_key, time_idx, spatial_slice):
        """
        Read and transform a WRF variable for one timestep.
        """
        info = self.variables[var_key]
        transform = info['transform']
        y_sl, x_sl = spatial_slice

        if transform is None:
            src = info['source_vars'][0]
            return h5[src][time_idx, y_sl, x_sl].astype('float32')

        elif transform == 'accumulation_increment':
            return self._read_accumulation_increment(h5, var_key, time_idx, spatial_slice)

        elif transform == 'precip_sum':
            return self._read_precip_sum(h5, var_key, time_idx, spatial_slice)

        elif transform == 'wind_speed':
            u_earth, v_earth = self._read_rotated_wind(h5, time_idx, spatial_slice)
            return np.sqrt(u_earth**2 + v_earth**2).astype('float32')

        elif transform == 'wind_direction':
            u_earth, v_earth = self._read_rotated_wind(h5, time_idx, spatial_slice)
            return ((270.0 - np.degrees(np.arctan2(v_earth, u_earth))) % 360.0).astype('float32')

        elif transform == 'potential_to_actual_temp':
            return self._read_potential_to_actual_temp(h5, time_idx, spatial_slice)

        elif transform == 'wind_speed_3d':
            return self._read_wind_speed_3d(h5, time_idx, spatial_slice)

        elif transform == 'wind_direction_3d':
            return self._read_wind_direction_3d(h5, time_idx, spatial_slice)

        elif transform == 'u_wind':
            return self._read_u_wind(h5, time_idx, spatial_slice)

        elif transform == 'v_wind':
            return self._read_v_wind(h5, time_idx, spatial_slice)

        elif transform == 'u_wind_3d':
            return self._read_u_wind_3d(h5, time_idx, spatial_slice)

        elif transform == 'v_wind_3d':
            return self._read_v_wind_3d(h5, time_idx, spatial_slice)

        elif transform == 'mixing_ratio_to_specific_humidity_2d':
            return self._read_specific_humidity_2d(h5, time_idx, spatial_slice)

        elif transform == 'mixing_ratio_3d':
            return self._read_mixing_ratio_3d(h5, time_idx, spatial_slice)

        elif transform == 'mixing_ratio_to_specific_humidity':
            return self._read_specific_humidity_3d(h5, time_idx, spatial_slice)

        elif transform == 'sea_level_pressure':
            return self._read_sea_level_pressure(h5, time_idx, spatial_slice)

        elif transform == 'vorticity':
            return self._read_vorticity(h5, time_idx, spatial_slice)

        elif transform == 'vorticity_3d':
            return self._read_vorticity_3d(h5, time_idx, spatial_slice)

        elif transform == 'vertical_velocity_3d':
            return self._read_vertical_velocity_3d(h5, time_idx, spatial_slice)

        raise ValueError(f'Unknown transform: {transform!r}')

    def _read_precip_sum(self, h5, var_key, time_idx, spatial_slice):
        """Sum pre-computed hourly precipitation fields (PREC_ACC_C + PREC_ACC_NC)."""
        info = self.variables[var_key]
        y_sl, x_sl = spatial_slice
        total = sum(h5[sv][time_idx, y_sl, x_sl].astype('float64') for sv in info['source_vars'])
        return total.astype('float32')

    def _read_rotated_wind(self, h5, time_idx, spatial_slice):
        """
        Read U10/V10 and rotate from grid-relative to earth-relative.

        Uses COSALPHA/SINALPHA loaded during initialization. If rotation
        fields are not available (e.g., lat-lon grid), returns unrotated values.

        Returns
        -------
        u_earth, v_earth : np.ndarray
            Earth-relative wind components.
        """
        y_sl, x_sl = spatial_slice
        u_grid = h5['U10'][time_idx, y_sl, x_sl].astype('float64')
        v_grid = h5['V10'][time_idx, y_sl, x_sl].astype('float64')

        if self._cosalpha is not None:
            cosa = self._cosalpha[y_sl, x_sl]
            sina = self._sinalpha[y_sl, x_sl]
            u_earth = u_grid * cosa + v_grid * sina
            v_earth = -u_grid * sina + v_grid * cosa
        else:
            u_earth = u_grid
            v_earth = v_grid

        return u_earth, v_earth

    def _compute_geo_height(self, h5, time_idx, spatial_slice):
        """Compute unstaggered geopotential height from PH + PHB."""
        y_sl, x_sl = spatial_slice
        ph = h5['PH'][time_idx, :, y_sl, x_sl].astype('float64')
        phb = h5['PHB'][time_idx, :, y_sl, x_sl].astype('float64')
        return unstagger((ph + phb) / 9.81, axis=0)

    def _read_rotated_wind_3d(self, h5, time_idx, spatial_slice):
        """
        Read 3D U/V, unstagger, and rotate to earth-relative.

        Returns
        -------
        u_earth, v_earth : np.ndarray
            Earth-relative wind components, each shape (nz, ny, nx).
        """
        y_sl, x_sl = spatial_slice
        # U is staggered in x (last dim): shape (nz, ny, nx+1)
        u_raw = h5['U'][time_idx, :, y_sl, :].astype('float64')
        u_unstag = unstagger(u_raw, axis=2)[:, :, x_sl]

        # V is staggered in y (second-to-last dim): shape (nz, ny+1, nx)
        v_raw = h5['V'][time_idx, :, :, x_sl].astype('float64')
        v_unstag = unstagger(v_raw, axis=1)[:, y_sl, :]

        if self._cosalpha is not None:
            cosa = self._cosalpha[y_sl, x_sl]
            sina = self._sinalpha[y_sl, x_sl]
            u_earth = u_unstag * cosa + v_unstag * sina
            v_earth = -u_unstag * sina + v_unstag * cosa
        else:
            u_earth = u_unstag
            v_earth = v_unstag

        return u_earth, v_earth

    def _read_potential_to_actual_temp(self, h5, time_idx, spatial_slice):
        """
        Convert WRF perturbation potential temperature to actual temperature
        and interpolate from eta levels to target height levels.

        WRF T is theta_perturbation = theta - 300 K.
        Actual T = theta * (P_total / P0) ^ (R/Cp) where R/Cp = 0.2854.
        Source levels are geopotential heights derived from (PH + PHB) / g.

        Returns
        -------
        np.ndarray
            Temperature on target height levels, shape (n_levels, ny, nx).
        """
        y_sl, x_sl = spatial_slice

        # Read 3D fields for this timestep
        t_pert = h5['T'][time_idx, :, y_sl, x_sl].astype('float64')
        p = h5['P'][time_idx, :, y_sl, x_sl].astype('float64')
        pb = h5['PB'][time_idx, :, y_sl, x_sl].astype('float64')

        # Convert perturbation potential temp to actual temperature
        theta = t_pert + 300.0
        pressure = p + pb
        t_actual = theta * (pressure / 100000.0) ** 0.2854

        geo_height = self._compute_geo_height(h5, time_idx, spatial_slice)

        # Interpolate to target height levels
        return self._regrid_func(t_actual, geo_height).astype('float32')

    def _read_wind_speed_3d(self, h5, time_idx, spatial_slice):
        """Compute 3D wind speed and interpolate to target height levels."""
        u, v = self._read_rotated_wind_3d(h5, time_idx, spatial_slice)
        speed = np.sqrt(u**2 + v**2)
        geo_height = self._compute_geo_height(h5, time_idx, spatial_slice)
        return self._regrid_func(speed, geo_height).astype('float32')

    def _read_wind_direction_3d(self, h5, time_idx, spatial_slice):
        """Compute 3D wind direction and interpolate to target height levels."""
        u, v = self._read_rotated_wind_3d(h5, time_idx, spatial_slice)
        direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
        geo_height = self._compute_geo_height(h5, time_idx, spatial_slice)
        return self._regrid_func(direction, geo_height).astype('float32')

    def _read_u_wind(self, h5, time_idx, spatial_slice):
        """Read earth-relative U wind component at 10m."""
        u_earth, _ = self._read_rotated_wind(h5, time_idx, spatial_slice)
        return u_earth.astype('float32')

    def _read_v_wind(self, h5, time_idx, spatial_slice):
        """Read earth-relative V wind component at 10m."""
        _, v_earth = self._read_rotated_wind(h5, time_idx, spatial_slice)
        return v_earth.astype('float32')

    def _read_u_wind_3d(self, h5, time_idx, spatial_slice):
        """Read 3D earth-relative U wind and interpolate to target height levels."""
        u, _ = self._read_rotated_wind_3d(h5, time_idx, spatial_slice)
        geo_height = self._compute_geo_height(h5, time_idx, spatial_slice)
        return self._regrid_func(u, geo_height).astype('float32')

    def _read_v_wind_3d(self, h5, time_idx, spatial_slice):
        """Read 3D earth-relative V wind and interpolate to target height levels."""
        _, v = self._read_rotated_wind_3d(h5, time_idx, spatial_slice)
        geo_height = self._compute_geo_height(h5, time_idx, spatial_slice)
        return self._regrid_func(v, geo_height).astype('float32')

    def _read_specific_humidity_2d(self, h5, time_idx, spatial_slice):
        """Convert 2m mixing ratio (Q2) to specific humidity."""
        y_sl, x_sl = spatial_slice
        mixing_ratio = h5['Q2'][time_idx, y_sl, x_sl].astype('float64')
        return (mixing_ratio / (1.0 + mixing_ratio)).astype('float32')

    def _read_mixing_ratio_3d(self, h5, time_idx, spatial_slice):
        """Read 3D mixing ratio and interpolate to target height levels."""
        y_sl, x_sl = spatial_slice
        mixing_ratio = h5['QVAPOR'][time_idx, :, y_sl, x_sl].astype('float64')
        geo_height = self._compute_geo_height(h5, time_idx, spatial_slice)
        return self._regrid_func(mixing_ratio, geo_height).astype('float32')

    def _read_specific_humidity_3d(self, h5, time_idx, spatial_slice):
        """Convert mixing ratio to specific humidity and interpolate to target height levels."""
        y_sl, x_sl = spatial_slice
        mixing_ratio = h5['QVAPOR'][time_idx, :, y_sl, x_sl].astype('float64')
        specific_humidity = mixing_ratio / (1.0 + mixing_ratio)
        geo_height = self._compute_geo_height(h5, time_idx, spatial_slice)
        return self._regrid_func(specific_humidity, geo_height).astype('float32')

    def _read_sea_level_pressure(self, h5, time_idx, spatial_slice):
        """Compute sea level pressure using hypsometric reduction."""
        y_sl, x_sl = spatial_slice
        psfc = h5['PSFC'][time_idx, y_sl, x_sl].astype('float64')
        t2 = h5['T2'][time_idx, y_sl, x_sl].astype('float64')
        hgt = h5['HGT'][time_idx, y_sl, x_sl].astype('float64')

        # Standard hypsometric reduction to sea level
        gamma = 0.0065  # standard lapse rate K/m
        g = 9.81
        rd = 287.05  # dry air gas constant J/(kg·K)
        t_mean = t2 + gamma * hgt / 2.0
        slp = psfc * np.exp(g * hgt / (rd * t_mean))

        return slp.astype('float32')

    def _read_vorticity(self, h5, time_idx, spatial_slice):
        """Compute vertical relative vorticity at 10m from earth-relative wind components."""
        u_earth, v_earth = self._read_rotated_wind(h5, time_idx, spatial_slice)
        dvdx = np.gradient(v_earth, self._dx, axis=1)
        dudy = np.gradient(u_earth, self._dy, axis=0)
        return (dvdx - dudy).astype('float32')

    def _read_vorticity_3d(self, h5, time_idx, spatial_slice):
        """Compute 3D vertical relative vorticity and interpolate to target height levels."""
        u, v = self._read_rotated_wind_3d(h5, time_idx, spatial_slice)
        dvdx = np.gradient(v, self._dx, axis=2)
        dudy = np.gradient(u, self._dy, axis=1)
        vorticity = dvdx - dudy
        geo_height = self._compute_geo_height(h5, time_idx, spatial_slice)
        return self._regrid_func(vorticity, geo_height).astype('float32')

    def _read_vertical_velocity_3d(self, h5, time_idx, spatial_slice):
        """Read W, unstagger vertically, and interpolate to target height levels."""
        y_sl, x_sl = spatial_slice
        w = h5['W'][time_idx, :, y_sl, x_sl].astype('float64')
        w_unstag = unstagger(w, axis=0)
        geo_height = self._compute_geo_height(h5, time_idx, spatial_slice)
        return self._regrid_func(w_unstag, geo_height).astype('float32')

    def _setup_populate(self, var_key, target_levels):
        """Set up level-interpolation regrid function for 3D variables."""
        info = self.variables[var_key]
        if info['height'] == 'levels' and target_levels is not None:
            from geointerp import GridInterpolator
            gi = GridInterpolator()
            self._regrid_func = gi.regrid_levels(
                np.array(target_levels, dtype='float64'), axis=0, method='linear',
            )
