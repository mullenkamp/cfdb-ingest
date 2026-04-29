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
        val = val.item()
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
    'RH2': {
        'cfdb_name': 'relative_humidity',
        'source_vars': ['T2', 'Q2', 'PSFC'],
        'transform': 'relative_humidity_2d',
        'height': 2.0,
    },
    'TD2': {
        'cfdb_name': 'dew_temp',
        'source_vars': ['Q2', 'PSFC'],
        'transform': 'dew_point_2d',
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
    'HGT': {
        'cfdb_name': 'terrain_height',
        'source_vars': ['HGT'],
        'transform': None,
        'height': 0.0,
    },
    'THETA2': {
        'cfdb_name': 'potential_temperature',
        'source_vars': ['T2', 'PSFC'],
        'transform': 'potential_temperature_2d',
        'height': 2.0,
    },
    'THETA_E2': {
        'cfdb_name': 'equivalent_potential_temperature',
        'source_vars': ['T2', 'Q2', 'PSFC'],
        'transform': 'equivalent_potential_temperature_2d',
        'height': 2.0,
    },
    # --- Level-interpolated variables ---
    'T': {
        'cfdb_name': 'air_temp',
        'source_vars': ['T', 'P', 'PB', 'PH', 'PHB'],
        'transform': 'potential_to_actual_temp',
        'height': 'levels',
    },
    'THETA': {
        'cfdb_name': 'potential_temperature',
        'source_vars': ['T', 'PH', 'PHB'],
        'transform': 'potential_temperature_3d',
        'height': 'levels',
    },
    'THETA_E': {
        'cfdb_name': 'equivalent_potential_temperature',
        'source_vars': ['T', 'P', 'PB', 'QVAPOR', 'PH', 'PHB'],
        'transform': 'equivalent_potential_temperature_3d',
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
    'RH': {
        'cfdb_name': 'relative_humidity',
        'source_vars': ['T', 'P', 'PB', 'QVAPOR', 'PH', 'PHB'],
        'transform': 'relative_humidity_3d',
        'height': 'levels',
    },
    'TD': {
        'cfdb_name': 'dew_temp',
        'source_vars': ['QVAPOR', 'P', 'PB', 'PH', 'PHB'],
        'transform': 'dew_point_3d',
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
    # --- Geopotential height (for WPS intermediate files) ---
    'GHT': {
        'cfdb_name': 'geopotential_height',
        'source_vars': ['PH', 'PHB'],
        'transform': 'geopotential_height_3d',
        'height': 'levels',
    },
    # --- Additional surface variables for WPS ---
    'XLAND': {
        'cfdb_name': 'land_sea_mask',
        'source_vars': ['XLAND'],
        'transform': 'land_sea_mask',
        'height': 0.0,
    },
    'SEAICE_VAR': {
        'cfdb_name': 'sea_ice',
        'source_vars': ['SEAICE'],
        'transform': None,
        'height': 0.0,
    },
    'SST_VAR': {
        'cfdb_name': 'sea_surface_temp',
        'source_vars': ['SST'],
        'transform': None,
        'height': 0.0,
    },
    'SNOW_VAR': {
        'cfdb_name': 'snow_water_equiv',
        'source_vars': ['SNOW'],
        'transform': None,
        'height': 0.0,
    },
    # --- Column-integrated variables (3D → 2D) ---
    'PWAT': {
        'cfdb_name': 'pwat',
        'source_vars': ['QVAPOR', 'P', 'PB'],
        'transform': 'precipitable_water',
        'height': 0.0,
    },
    'PWAT_TR': {
        'cfdb_name': 'pwat_tr',
        'source_vars': ['qv_tr', 'P', 'PB'],
        'transform': 'precipitable_water_tracer',
        'height': 0.0,
    },
    'RAIN_TR': {
        'cfdb_name': 'precip_tr',
        'source_vars': ['TR_RAINNC', 'TR_RAINC'],
        'transform': 'accumulation_increment',
        'height': 0.0,
    },
    'VIMF_U': {
        'cfdb_name': 'vimf_u',
        'source_vars': ['QVAPOR', 'U', 'V', 'P', 'PB'],
        'transform': 'vimf_u',
        'height': 0.0,
    },
    'VIMF_V': {
        'cfdb_name': 'vimf_v',
        'source_vars': ['QVAPOR', 'U', 'V', 'P', 'PB'],
        'transform': 'vimf_v',
        'height': 0.0,
    },
    # --- Soil variables ---
    'SMOIS': {
        'cfdb_name': 'soil_moisture',
        'source_vars': ['SMOIS'],
        'transform': 'soil_3d',
        'height': 'soil',
    },
    'TSLB': {
        'cfdb_name': 'soil_layer_temp',
        'source_vars': ['TSLB'],
        'transform': 'soil_3d',
        'height': 'soil',
    },
}

_WRF_DATASET_ATTRS = [
    # Grid
    'GRID_ID', 'DX', 'DY', 'DT',
    # Microphysics
    'MP_PHYSICS',
    # Radiation
    'RA_LW_PHYSICS', 'RA_SW_PHYSICS', 'RADT',
    # PBL
    'BL_PBL_PHYSICS', 'SF_SFCLAY_PHYSICS',
    # Cumulus
    'CU_PHYSICS', 'CUDT', 'SHCU_PHYSICS',
    # Land surface
    'SF_SURFACE_PHYSICS', 'MMINLU', 'NUM_LAND_CAT',
    # Diffusion
    'DIFF_OPT', 'KM_OPT', 'DAMP_OPT',
    # Dynamics
    'HYBRID_OPT', 'MOIST_ADV_OPT', 'USE_THETA_M',
    # Nudging
    'GRID_FDDA', 'GFDDA_INTERVAL_M',
    # Other
    'GWD_OPT', 'SF_LAKE_PHYSICS', 'SF_OCEAN_PHYSICS',
    'SF_URBAN_PHYSICS', 'SST_UPDATE', 'PREC_ACC_DT',
]


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

    file_glob_pattern = 'wrfout*'

    def _init_source_metadata(self):
        """
        Override to also load wind rotation fields and WRF source attributes.
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

            # Extract WRF source info and physics parameters
            self._source_title = _wrf_attr(h5.attrs, 'TITLE').strip()
            self._wrf_params = {}
            for key in _WRF_DATASET_ATTRS:
                if key in h5.attrs:
                    self._wrf_params[key] = _wrf_attr(h5.attrs, key)

        self.x = spatial['x']
        self.y = spatial['y']
        self._heterogeneous_grids = False
        self._dx = float(self.x[1] - self.x[0])
        self._dy = float(self.y[1] - self.y[0])

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
        over RAINC/RAINNC (running accumulations) when available, and also
        allow precipitation when only PREC_ACC_* variables exist.
        """
        super()._init_variables()
        with h5py.File(self.input_paths[0], 'r') as h5:
            has_prec_acc = 'PREC_ACC_C' in h5 and 'PREC_ACC_NC' in h5
        if has_prec_acc:
            self.variables['RAIN'] = {
                'cfdb_name': 'precip',
                'source_vars': ['PREC_ACC_C', 'PREC_ACC_NC'],
                'transform': 'precip_sum',
                'height': 0.0,
            }

    def _get_variable_mapping(self):
        """Return the WRF variable mapping dictionary."""
        return WRF_VARIABLE_MAPPING

    def _get_dataset_attrs(self):
        """Return CF + WRF-specific dataset attributes."""
        attrs = super()._get_dataset_attrs()
        attrs['source'] = self._source_title
        attrs.update(self._wrf_params)
        return attrs

    def _accumulation_source_sum(self, h5, source_vars, time_idx, spatial_slice):
        """
        Sum source_vars with WRF bucket-counter reconstruction.

        When BUCKET_MM > 0 is set on the wrfout file, WRF wraps accumulator
        variables like RAINC/RAINNC periodically and stores the overflow
        count in companion integer variables (I_RAINC/I_RAINNC). The true
        cumulative total is ``<var> + BUCKET_MM * I_<var>``.

        This override adds the bucket term for any source var that has a
        companion ``I_<name>`` in the file. Backward compatible: when the
        bucket is disabled (BUCKET_MM <= 0) or no I_<name> exists, behaves
        identically to the base implementation.
        """
        y_sl, x_sl = spatial_slice

        bucket_mm_attr = h5.attrs.get('BUCKET_MM', -1.0)
        bucket_mm = float(np.asarray(bucket_mm_attr).item())
        use_bucket = bucket_mm > 0.0

        total = None
        for sv in source_vars:
            part = h5[sv][time_idx, y_sl, x_sl].astype('float64')
            if use_bucket:
                i_name = 'I_' + sv
                if i_name in h5:
                    part = part + bucket_mm * h5[i_name][time_idx, y_sl, x_sl].astype('float64')
            total = part if total is None else total + part
        return total

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

        elif transform == 'relative_humidity_2d':
            return self._read_relative_humidity_2d(h5, time_idx, spatial_slice)

        elif transform == 'relative_humidity_3d':
            return self._read_relative_humidity_3d(h5, time_idx, spatial_slice)

        elif transform == 'dew_point_2d':
            return self._read_dew_point_2d(h5, time_idx, spatial_slice)

        elif transform == 'dew_point_3d':
            return self._read_dew_point_3d(h5, time_idx, spatial_slice)

        elif transform == 'sea_level_pressure':
            return self._read_sea_level_pressure(h5, time_idx, spatial_slice)

        elif transform == 'vorticity':
            return self._read_vorticity(h5, time_idx, spatial_slice)

        elif transform == 'vorticity_3d':
            return self._read_vorticity_3d(h5, time_idx, spatial_slice)

        elif transform == 'vertical_velocity_3d':
            return self._read_vertical_velocity_3d(h5, time_idx, spatial_slice)

        elif transform == 'potential_temperature_2d':
            return self._read_potential_temperature_2d(h5, time_idx, spatial_slice)

        elif transform == 'potential_temperature_3d':
            return self._read_potential_temperature_3d(h5, time_idx, spatial_slice)

        elif transform == 'equivalent_potential_temperature_2d':
            return self._read_equivalent_potential_temperature_2d(h5, time_idx, spatial_slice)

        elif transform == 'equivalent_potential_temperature_3d':
            return self._read_equivalent_potential_temperature_3d(h5, time_idx, spatial_slice)

        elif transform == 'geopotential_height_3d':
            return self._read_geopotential_height_3d(h5, time_idx, spatial_slice)

        elif transform == 'land_sea_mask':
            return self._read_land_sea_mask(h5, time_idx, spatial_slice)

        elif transform == 'soil_3d':
            return self._read_soil_3d(h5, var_key, time_idx, spatial_slice)

        elif transform == 'precipitable_water':
            return self._read_precipitable_water(h5, time_idx, spatial_slice)

        elif transform == 'precipitable_water_tracer':
            return self._read_precipitable_water_tracer(h5, time_idx, spatial_slice)

        elif transform == 'vimf_u':
            return self._read_vimf_u(h5, time_idx, spatial_slice)

        elif transform == 'vimf_v':
            return self._read_vimf_v(h5, time_idx, spatial_slice)

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
        cache = getattr(self, '_ts_cache', None)
        if cache is not None and 'wind_2d' in cache:
            return cache['wind_2d']

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

        result = (u_earth, v_earth)
        if cache is not None:
            cache['wind_2d'] = result

        return result

    def _compute_geo_height(self, h5, time_idx, spatial_slice):
        """Compute unstaggered geopotential height from PH + PHB."""
        cache = getattr(self, '_ts_cache', None)
        if cache is not None and 'geo_height' in cache:
            return cache['geo_height']

        y_sl, x_sl = spatial_slice
        ph = h5['PH'][time_idx, :, y_sl, x_sl].astype('float64')
        phb = h5['PHB'][time_idx, :, y_sl, x_sl].astype('float64')
        result = unstagger((ph + phb) / 9.81, axis=0)

        if cache is not None:
            cache['geo_height'] = result

        return result

    def _compute_pressure(self, h5, time_idx, spatial_slice):
        """Compute full pressure (P + PB) on eta levels."""
        cache = getattr(self, '_ts_cache', None)
        if cache is not None and 'pressure' in cache:
            return cache['pressure']

        y_sl, x_sl = spatial_slice
        p = h5['P'][time_idx, :, y_sl, x_sl].astype('float64')
        pb = h5['PB'][time_idx, :, y_sl, x_sl].astype('float64')
        result = p + pb

        if cache is not None:
            cache['pressure'] = result

        return result

    def _get_source_levels(self, h5, time_idx, spatial_slice):
        """Return source levels for vertical interpolation (height or pressure)."""
        if getattr(self, '_vertical_coord', 'height') == 'pressure':
            return self._compute_pressure(h5, time_idx, spatial_slice)
        return self._compute_geo_height(h5, time_idx, spatial_slice)

    def _get_soil_depths(self):
        """
        Return soil depth coordinate values in meters from WRF DZS.

        Returns cumulative bottom boundary depths (ascending). For Noah LSM
        layers [0.1, 0.3, 0.6, 1.0] m, this returns [0.1, 0.4, 1.0, 2.0].
        These can be used to reconstruct WPS layer names (SM000010, SM010040, etc.)
        since the top of layer k is the bottom of layer k-1 (or 0 for k=0).
        """
        with h5py.File(self.input_paths[0], 'r') as h5:
            if 'DZS' not in h5:
                return None
            dzs = h5['DZS'][0, :].astype('float64')

        depths = np.cumsum(dzs)
        return depths

    def _get_qvapor(self, h5, time_idx, spatial_slice):
        cache = getattr(self, '_ts_cache', None)
        if cache is not None and 'qvapor' in cache:
            return cache['qvapor']
        y_sl, x_sl = spatial_slice
        result = h5['QVAPOR'][time_idx, :, y_sl, x_sl].astype('float64')
        if cache is not None:
            cache['qvapor'] = result
        return result

    def _compute_theta(self, h5, time_idx, spatial_slice):
        cache = getattr(self, '_ts_cache', None)
        if cache is not None and 'theta' in cache:
            return cache['theta']
        y_sl, x_sl = spatial_slice
        t_pert = h5['T'][time_idx, :, y_sl, x_sl].astype('float64')
        result = t_pert + 300.0
        if cache is not None:
            cache['theta'] = result
        return result

    def _get_t2(self, h5, time_idx, spatial_slice):
        cache = getattr(self, '_ts_cache', None)
        if cache is not None and 't2' in cache:
            return cache['t2']
        y_sl, x_sl = spatial_slice
        result = h5['T2'][time_idx, y_sl, x_sl].astype('float64')
        if cache is not None:
            cache['t2'] = result
        return result

    def _get_q2(self, h5, time_idx, spatial_slice):
        cache = getattr(self, '_ts_cache', None)
        if cache is not None and 'q2' in cache:
            return cache['q2']
        y_sl, x_sl = spatial_slice
        result = h5['Q2'][time_idx, y_sl, x_sl].astype('float64')
        if cache is not None:
            cache['q2'] = result
        return result

    def _get_psfc(self, h5, time_idx, spatial_slice):
        cache = getattr(self, '_ts_cache', None)
        if cache is not None and 'psfc' in cache:
            return cache['psfc']
        y_sl, x_sl = spatial_slice
        result = h5['PSFC'][time_idx, y_sl, x_sl].astype('float64')
        if cache is not None:
            cache['psfc'] = result
        return result

    def _read_rotated_wind_3d(self, h5, time_idx, spatial_slice):
        """
        Read 3D U/V, unstagger, and rotate to earth-relative.

        Returns
        -------
        u_earth, v_earth : np.ndarray
            Earth-relative wind components, each shape (nz, ny, nx).
        """
        cache = getattr(self, '_ts_cache', None)
        if cache is not None and 'wind_3d' in cache:
            return cache['wind_3d']

        y_sl, x_sl = spatial_slice

        # U is staggered in x (last dim): read only the needed stagger range
        nx_unstag = h5['U'].shape[3] - 1
        x_start, x_stop, _ = x_sl.indices(nx_unstag)
        u_raw = h5['U'][time_idx, :, y_sl, x_start:x_stop + 1].astype('float64')
        u_unstag = unstagger(u_raw, axis=2)

        # V is staggered in y (second-to-last dim): read only the needed stagger range
        ny_unstag = h5['V'].shape[2] - 1
        y_start, y_stop, _ = y_sl.indices(ny_unstag)
        v_raw = h5['V'][time_idx, :, y_start:y_stop + 1, x_sl].astype('float64')
        v_unstag = unstagger(v_raw, axis=1)

        if self._cosalpha is not None:
            cosa = self._cosalpha[y_sl, x_sl]
            sina = self._sinalpha[y_sl, x_sl]
            u_earth = u_unstag * cosa + v_unstag * sina
            v_earth = -u_unstag * sina + v_unstag * cosa
        else:
            u_earth = u_unstag
            v_earth = v_unstag

        result = (u_earth, v_earth)
        if cache is not None:
            cache['wind_3d'] = result

        return result

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
        theta = self._compute_theta(h5, time_idx, spatial_slice)
        pressure = self._compute_pressure(h5, time_idx, spatial_slice)
        t_actual = theta * (pressure / 100000.0) ** 0.2854

        source_levels = self._get_source_levels(h5, time_idx, spatial_slice)
        return self._regrid_func(t_actual, source_levels).astype('float32')

    def _read_wind_speed_3d(self, h5, time_idx, spatial_slice):
        """Compute 3D wind speed and interpolate to target height levels."""
        u, v = self._read_rotated_wind_3d(h5, time_idx, spatial_slice)
        speed = np.sqrt(u**2 + v**2)
        source_levels = self._get_source_levels(h5, time_idx, spatial_slice)
        return self._regrid_func(speed, source_levels).astype('float32')

    def _read_wind_direction_3d(self, h5, time_idx, spatial_slice):
        """Compute 3D wind direction and interpolate to target height levels."""
        u, v = self._read_rotated_wind_3d(h5, time_idx, spatial_slice)
        direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
        source_levels = self._get_source_levels(h5, time_idx, spatial_slice)
        return self._regrid_func(direction, source_levels).astype('float32')

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
        source_levels = self._get_source_levels(h5, time_idx, spatial_slice)
        return self._regrid_func(u, source_levels).astype('float32')

    def _read_v_wind_3d(self, h5, time_idx, spatial_slice):
        """Read 3D earth-relative V wind and interpolate to target height levels."""
        _, v = self._read_rotated_wind_3d(h5, time_idx, spatial_slice)
        source_levels = self._get_source_levels(h5, time_idx, spatial_slice)
        return self._regrid_func(v, source_levels).astype('float32')

    def _read_specific_humidity_2d(self, h5, time_idx, spatial_slice):
        """Convert 2m mixing ratio (Q2) to specific humidity."""
        mixing_ratio = self._get_q2(h5, time_idx, spatial_slice)
        return (mixing_ratio / (1.0 + mixing_ratio)).astype('float32')

    def _read_mixing_ratio_3d(self, h5, time_idx, spatial_slice):
        """Read 3D mixing ratio and interpolate to target height levels."""
        mixing_ratio = self._get_qvapor(h5, time_idx, spatial_slice)
        source_levels = self._get_source_levels(h5, time_idx, spatial_slice)
        return self._regrid_func(mixing_ratio, source_levels).astype('float32')

    def _read_specific_humidity_3d(self, h5, time_idx, spatial_slice):
        """Convert mixing ratio to specific humidity and interpolate to target height levels."""
        mixing_ratio = self._get_qvapor(h5, time_idx, spatial_slice)
        specific_humidity = mixing_ratio / (1.0 + mixing_ratio)
        source_levels = self._get_source_levels(h5, time_idx, spatial_slice)
        return self._regrid_func(specific_humidity, source_levels).astype('float32')

    def _read_relative_humidity_2d(self, h5, time_idx, spatial_slice):
        """Compute 2m relative humidity from T2, Q2, and PSFC."""
        t2 = self._get_t2(h5, time_idx, spatial_slice)
        q2 = self._get_q2(h5, time_idx, spatial_slice)
        psfc = self._get_psfc(h5, time_idx, spatial_slice)

        # Saturation vapor pressure (Bolton 1980) [Pa]
        es = 611.2 * np.exp(17.67 * (t2 - 273.15) / (t2 - 273.15 + 243.5))
        # Actual vapor pressure from mixing ratio [Pa]
        e = q2 * psfc / (0.622 + q2)
        rh = np.clip(e / es, 0.0, 1.0)
        return rh.astype('float32')

    def _read_relative_humidity_3d(self, h5, time_idx, spatial_slice):
        """Compute 3D relative humidity and interpolate to target height levels."""
        theta = self._compute_theta(h5, time_idx, spatial_slice)
        pressure = self._compute_pressure(h5, time_idx, spatial_slice)
        q = self._get_qvapor(h5, time_idx, spatial_slice)

        t_actual = theta * (pressure / 100000.0) ** 0.2854

        es = 611.2 * np.exp(17.67 * (t_actual - 273.15) / (t_actual - 273.15 + 243.5))
        e = q * pressure / (0.622 + q)
        rh = np.clip(e / es, 0.0, 1.0)

        source_levels = self._get_source_levels(h5, time_idx, spatial_slice)
        return self._regrid_func(rh, source_levels).astype('float32')

    def _read_dew_point_2d(self, h5, time_idx, spatial_slice):
        """Compute 2m dew point temperature from Q2 and PSFC."""
        q2 = self._get_q2(h5, time_idx, spatial_slice)
        psfc = self._get_psfc(h5, time_idx, spatial_slice)

        # Actual vapor pressure from mixing ratio [Pa]
        e = q2 * psfc / (0.622 + q2)
        # Inverse Bolton formula for dew point [K]
        # NaN is expected where moisture is zero (e <= 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            ln_ratio = np.log(e / 611.2)
            td = 273.15 + 243.5 * ln_ratio / (17.67 - ln_ratio)
        return td.astype('float32')

    def _read_dew_point_3d(self, h5, time_idx, spatial_slice):
        """Compute 3D dew point temperature and interpolate to target height levels."""
        q = self._get_qvapor(h5, time_idx, spatial_slice)
        pressure = self._compute_pressure(h5, time_idx, spatial_slice)

        e = q * pressure / (0.622 + q)
        with np.errstate(divide='ignore', invalid='ignore'):
            ln_ratio = np.log(e / 611.2)
            td = 273.15 + 243.5 * ln_ratio / (17.67 - ln_ratio)

        source_levels = self._get_source_levels(h5, time_idx, spatial_slice)
        return self._regrid_func(td, source_levels).astype('float32')

    def _read_sea_level_pressure(self, h5, time_idx, spatial_slice):
        """Compute sea level pressure using hypsometric reduction."""
        psfc = self._get_psfc(h5, time_idx, spatial_slice)
        t2 = self._get_t2(h5, time_idx, spatial_slice)
        y_sl, x_sl = spatial_slice
        hgt = h5['HGT'][time_idx, y_sl, x_sl].astype('float64')

        # Standard hypsometric reduction to sea level
        gamma = 0.0065  # standard lapse rate K/m
        g = 9.81
        rd = 287.05  # dry air gas constant J/(kg·K)
        t_mean = t2 + gamma * hgt / 2.0
        slp = psfc * np.exp(g * hgt / (rd * t_mean))

        return slp.astype('float32')

    def _read_potential_temperature_2d(self, h5, time_idx, spatial_slice):
        """Compute 2m potential temperature from T2 and PSFC."""
        t2 = self._get_t2(h5, time_idx, spatial_slice)
        psfc = self._get_psfc(h5, time_idx, spatial_slice)
        theta = t2 * (100000.0 / psfc) ** 0.2854
        return theta.astype('float32')

    def _read_potential_temperature_3d(self, h5, time_idx, spatial_slice):
        """Read WRF potential temperature (T + 300) and interpolate to target height levels."""
        theta = self._compute_theta(h5, time_idx, spatial_slice)
        source_levels = self._get_source_levels(h5, time_idx, spatial_slice)
        return self._regrid_func(theta, source_levels).astype('float32')

    def _read_equivalent_potential_temperature_2d(self, h5, time_idx, spatial_slice):
        """Compute 2m equivalent potential temperature using Bolton (1980)."""
        t2 = self._get_t2(h5, time_idx, spatial_slice)
        q2 = self._get_q2(h5, time_idx, spatial_slice)
        psfc = self._get_psfc(h5, time_idx, spatial_slice)

        # Vapor pressure and dew point for LCL temperature
        # NaN is expected where moisture is zero (e <= 0)
        e = q2 * psfc / (0.622 + q2)
        with np.errstate(divide='ignore', invalid='ignore'):
            ln_ratio = np.log(e / 611.2)
            td = 273.15 + 243.5 * ln_ratio / (17.67 - ln_ratio)

            # LCL temperature (Bolton 1980, eq. 15)
            tl = 1.0 / (1.0 / (td - 56.0) + np.log(t2 / td) / 800.0) + 56.0

            # Bolton (1980) eq. 43
            theta_e = t2 * (100000.0 / psfc) ** (0.2854 * (1.0 - 0.28 * q2)) \
                * np.exp(q2 * (1.0 + 0.81 * q2) * (3376.0 / tl - 2.54))
        return theta_e.astype('float32')

    def _read_equivalent_potential_temperature_3d(self, h5, time_idx, spatial_slice):
        """Compute 3D equivalent potential temperature (Bolton 1980) and interpolate to target levels."""
        theta = self._compute_theta(h5, time_idx, spatial_slice)
        pressure = self._compute_pressure(h5, time_idx, spatial_slice)
        q = self._get_qvapor(h5, time_idx, spatial_slice)

        t_actual = theta * (pressure / 100000.0) ** 0.2854

        # Vapor pressure and dew point for LCL temperature
        # NaN is expected where moisture is zero (e <= 0)
        e = q * pressure / (0.622 + q)
        with np.errstate(divide='ignore', invalid='ignore'):
            ln_ratio = np.log(e / 611.2)
            td = 273.15 + 243.5 * ln_ratio / (17.67 - ln_ratio)

            # LCL temperature (Bolton 1980, eq. 15)
            tl = 1.0 / (1.0 / (td - 56.0) + np.log(t_actual / td) / 800.0) + 56.0

            # Bolton (1980) eq. 43
            theta_e = t_actual * (100000.0 / pressure) ** (0.2854 * (1.0 - 0.28 * q)) \
                * np.exp(q * (1.0 + 0.81 * q) * (3376.0 / tl - 2.54))

        source_levels = self._get_source_levels(h5, time_idx, spatial_slice)
        return self._regrid_func(theta_e, source_levels).astype('float32')

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
        source_levels = self._get_source_levels(h5, time_idx, spatial_slice)
        return self._regrid_func(vorticity, source_levels).astype('float32')

    def _read_vertical_velocity_3d(self, h5, time_idx, spatial_slice):
        """Read W, unstagger vertically, and interpolate to target height levels."""
        y_sl, x_sl = spatial_slice
        w = h5['W'][time_idx, :, y_sl, x_sl].astype('float64')
        w_unstag = unstagger(w, axis=0)
        source_levels = self._get_source_levels(h5, time_idx, spatial_slice)
        return self._regrid_func(w_unstag, source_levels).astype('float32')

    def _read_geopotential_height_3d(self, h5, time_idx, spatial_slice):
        """Compute geopotential height and interpolate to target levels."""
        ght = self._compute_geo_height(h5, time_idx, spatial_slice)
        source_levels = self._get_source_levels(h5, time_idx, spatial_slice)
        return self._regrid_func(ght, source_levels).astype('float32')

    def _read_land_sea_mask(self, h5, time_idx, spatial_slice):
        """Convert XLAND (1=land, 2=water) to (1=land, 0=water)."""
        y_sl, x_sl = spatial_slice
        xland = h5['XLAND'][time_idx, y_sl, x_sl].astype('float64')
        return np.where(xland < 1.5, 1.0, 0.0).astype('float32')

    def _read_soil_3d(self, h5, var_key, time_idx, spatial_slice):
        """Read a 3D soil variable (SMOIS or TSLB) — no vertical interpolation."""
        info = self.variables[var_key]
        src = info['source_vars'][0]
        y_sl, x_sl = spatial_slice
        return h5[src][time_idx, :, y_sl, x_sl].astype('float32')

    def _compute_column_qvapor_dp(self, h5, time_idx, spatial_slice):
        """
        Read QVAPOR and compute pressure layer thickness on eta levels.

        Cached in ``_ts_cache`` so PWAT and VIMF can share the same read.

        Returns
        -------
        qvapor : np.ndarray
            Water vapor mixing ratio, shape (nz, ny, nx).
        dp : np.ndarray
            Pressure thickness of each layer, shape (nz, ny, nx).
        """
        cache = getattr(self, '_ts_cache', None)
        if cache is not None and 'column_qvapor_dp' in cache:
            return cache['column_qvapor_dp']

        qvapor = self._get_qvapor(h5, time_idx, spatial_slice)
        pressure = self._compute_pressure(h5, time_idx, spatial_slice)

        # Compute pressure thickness of each layer using layer midpoints.
        # dp[k] = |p[k-1] - p[k+1]| / 2 for interior levels,
        # half-layers at boundaries.
        dp = np.empty_like(pressure)
        dp[0] = pressure[0] - pressure[1]
        dp[-1] = pressure[-2] - pressure[-1]
        dp[1:-1] = (pressure[:-2] - pressure[2:]) / 2.0

        # Ensure positive dp (pressure decreases with height in WRF)
        dp = np.abs(dp)

        result = (qvapor, dp)
        if cache is not None:
            cache['column_qvapor_dp'] = result
        return result

    def _read_precipitable_water(self, h5, time_idx, spatial_slice):
        """
        Compute total precipitable water by vertically integrating QVAPOR.

        PWAT = (1/g) * sum(q * dp)  over all eta levels.

        Returns
        -------
        np.ndarray
            Precipitable water in kg/m2, shape (ny, nx).
        """
        qvapor, dp = self._compute_column_qvapor_dp(h5, time_idx, spatial_slice)
        pwat = np.sum(qvapor * dp, axis=0) / 9.80665
        return pwat.astype('float32')

    def _read_precipitable_water_tracer(self, h5, time_idx, spatial_slice):
        """
        Compute tracer precipitable water by vertically integrating qv_tr.

        PWAT_TR = (1/g) * sum(qv_tr * dp)  over all eta levels.

        Returns
        -------
        np.ndarray
            Tracer precipitable water in kg/m2, shape (ny, nx).
        """
        _, dp = self._compute_column_qvapor_dp(h5, time_idx, spatial_slice)
        y_sl, x_sl = spatial_slice
        qv_tr = h5['qv_tr'][time_idx, :, y_sl, x_sl].astype('float64')
        pwat_tr = np.sum(qv_tr * dp, axis=0) / 9.80665
        return pwat_tr.astype('float32')

    def _read_vimf_u(self, h5, time_idx, spatial_slice):
        """
        Compute eastward vertically integrated moisture flux.

        VIMF_u = (1/g) * sum(q * u * dp)  over all eta levels.

        Returns
        -------
        np.ndarray
            Eastward VIMF in kg/m/s, shape (ny, nx).
        """
        qvapor, dp = self._compute_column_qvapor_dp(h5, time_idx, spatial_slice)
        u_earth, _ = self._read_rotated_wind_3d(h5, time_idx, spatial_slice)
        vimf_u = np.sum(qvapor * u_earth * dp, axis=0) / 9.80665
        return vimf_u.astype('float32')

    def _read_vimf_v(self, h5, time_idx, spatial_slice):
        """
        Compute northward vertically integrated moisture flux.

        VIMF_v = (1/g) * sum(q * v * dp)  over all eta levels.

        Returns
        -------
        np.ndarray
            Northward VIMF in kg/m/s, shape (ny, nx).
        """
        qvapor, dp = self._compute_column_qvapor_dp(h5, time_idx, spatial_slice)
        _, v_earth = self._read_rotated_wind_3d(h5, time_idx, spatial_slice)
        vimf_v = np.sum(qvapor * v_earth * dp, axis=0) / 9.80665
        return vimf_v.astype('float32')

    # ------------------------------------------------------------------
    # Block-mode (time-batched) transforms.
    # Each takes ``sources`` = {src_name: ndarray of shape (N, ny, nx)} and
    # returns an ndarray of shape (N, ny, nx). They share a per-block
    # ``block_cache`` dict for intermediates that span multiple variables
    # (e.g. earth-relative wind for both wind_speed and wind_direction).
    # ------------------------------------------------------------------

    _BLOCK_TRANSFORMS = {
        'mixing_ratio_to_specific_humidity_2d': '_block_specific_humidity_2d',
        'relative_humidity_2d': '_block_relative_humidity_2d',
        'dew_point_2d': '_block_dew_point_2d',
        'potential_temperature_2d': '_block_potential_temperature_2d',
        'equivalent_potential_temperature_2d': '_block_equivalent_potential_temperature_2d',
        'wind_speed': '_block_wind_speed',
        'wind_direction': '_block_wind_direction',
        'u_wind': '_block_u_wind',
        'v_wind': '_block_v_wind',
        'vorticity': '_block_vorticity',
        'sea_level_pressure': '_block_sea_level_pressure',
        'land_sea_mask': '_block_land_sea_mask',
        'precip_sum': '_block_precip_sum',
        'precipitable_water': '_block_precipitable_water',
        'precipitable_water_tracer': '_block_precipitable_water_tracer',
        'vimf_u': '_block_vimf_u',
        'vimf_v': '_block_vimf_v',
    }

    def _get_block_transform(self, transform_name):
        """Return a bound block-transform method for ``transform_name``, or None."""
        method = self._BLOCK_TRANSFORMS.get(transform_name)
        return getattr(self, method) if method is not None else None

    def _make_source(self, src_name, h5_files, file_lens):
        """
        Override the base virtual-source builder to wrap WRF's staggered
        source variables. ``U`` is staggered on the x axis (last); ``V`` on
        the y axis (second-to-last). Both 4D variables expose unstaggered
        shape to rechunker, with the trapezoidal mean applied on read.
        """
        from cfdb_ingest.base import _ConcatTimeSourceUnstaggered, _ConcatTimeSource
        if src_name == 'U':
            return _ConcatTimeSourceUnstaggered(h5_files, 'U', file_lens, stagger_axis=3)
        if src_name == 'V':
            return _ConcatTimeSourceUnstaggered(h5_files, 'V', file_lens, stagger_axis=2)
        return _ConcatTimeSource(h5_files, src_name, file_lens)

    def _block_rotated_wind_2d(self, sources, y_sl, x_sl, block_cache):
        """
        Earth-relative U/V wind from grid-relative U10/V10. Shape (N, ny, nx).
        Cached per block so wind_speed/wind_direction/u_wind/v_wind/vorticity share work.
        """
        if 'wind_2d' in block_cache:
            return block_cache['wind_2d']
        u_grid = sources['U10'].astype('float64')
        v_grid = sources['V10'].astype('float64')
        if self._cosalpha is not None:
            cosa = self._cosalpha[y_sl, x_sl]
            sina = self._sinalpha[y_sl, x_sl]
            u_earth = u_grid * cosa + v_grid * sina
            v_earth = -u_grid * sina + v_grid * cosa
        else:
            u_earth, v_earth = u_grid, v_grid
        result = (u_earth, v_earth)
        block_cache['wind_2d'] = result
        return result

    def _block_specific_humidity_2d(self, sources, y_sl, x_sl, block_cache):
        q = sources['Q2'].astype('float64')
        return (q / (1.0 + q)).astype('float32')

    def _block_relative_humidity_2d(self, sources, y_sl, x_sl, block_cache):
        t2 = sources['T2'].astype('float64')
        q2 = sources['Q2'].astype('float64')
        psfc = sources['PSFC'].astype('float64')
        es = 611.2 * np.exp(17.67 * (t2 - 273.15) / (t2 - 273.15 + 243.5))
        e = q2 * psfc / (0.622 + q2)
        rh = np.clip(e / es, 0.0, 1.0)
        return rh.astype('float32')

    def _block_dew_point_2d(self, sources, y_sl, x_sl, block_cache):
        q2 = sources['Q2'].astype('float64')
        psfc = sources['PSFC'].astype('float64')
        e = q2 * psfc / (0.622 + q2)
        with np.errstate(divide='ignore', invalid='ignore'):
            ln_ratio = np.log(e / 611.2)
            td = 273.15 + 243.5 * ln_ratio / (17.67 - ln_ratio)
        return td.astype('float32')

    def _block_potential_temperature_2d(self, sources, y_sl, x_sl, block_cache):
        t2 = sources['T2'].astype('float64')
        psfc = sources['PSFC'].astype('float64')
        return (t2 * (100000.0 / psfc) ** 0.2854).astype('float32')

    def _block_equivalent_potential_temperature_2d(self, sources, y_sl, x_sl, block_cache):
        t2 = sources['T2'].astype('float64')
        q2 = sources['Q2'].astype('float64')
        psfc = sources['PSFC'].astype('float64')
        e = q2 * psfc / (0.622 + q2)
        with np.errstate(divide='ignore', invalid='ignore'):
            ln_ratio = np.log(e / 611.2)
            td = 273.15 + 243.5 * ln_ratio / (17.67 - ln_ratio)
            tl = 1.0 / (1.0 / (td - 56.0) + np.log(t2 / td) / 800.0) + 56.0
            theta_e = t2 * (100000.0 / psfc) ** (0.2854 * (1.0 - 0.28 * q2)) \
                * np.exp(q2 * (1.0 + 0.81 * q2) * (3376.0 / tl - 2.54))
        return theta_e.astype('float32')

    def _block_wind_speed(self, sources, y_sl, x_sl, block_cache):
        u, v = self._block_rotated_wind_2d(sources, y_sl, x_sl, block_cache)
        return np.sqrt(u**2 + v**2).astype('float32')

    def _block_wind_direction(self, sources, y_sl, x_sl, block_cache):
        u, v = self._block_rotated_wind_2d(sources, y_sl, x_sl, block_cache)
        return ((270.0 - np.degrees(np.arctan2(v, u))) % 360.0).astype('float32')

    def _block_u_wind(self, sources, y_sl, x_sl, block_cache):
        u, _ = self._block_rotated_wind_2d(sources, y_sl, x_sl, block_cache)
        return u.astype('float32')

    def _block_v_wind(self, sources, y_sl, x_sl, block_cache):
        _, v = self._block_rotated_wind_2d(sources, y_sl, x_sl, block_cache)
        return v.astype('float32')

    def _block_vorticity(self, sources, y_sl, x_sl, block_cache):
        u, v = self._block_rotated_wind_2d(sources, y_sl, x_sl, block_cache)
        # u, v shape: (N, ny, nx). y is axis=1, x is axis=2.
        dvdx = np.gradient(v, self._dx, axis=2)
        dudy = np.gradient(u, self._dy, axis=1)
        return (dvdx - dudy).astype('float32')

    def _block_sea_level_pressure(self, sources, y_sl, x_sl, block_cache):
        psfc = sources['PSFC'].astype('float64')
        t2 = sources['T2'].astype('float64')
        hgt = sources['HGT'].astype('float64')
        gamma = 0.0065
        g = 9.81
        rd = 287.05
        t_mean = t2 + gamma * hgt / 2.0
        return (psfc * np.exp(g * hgt / (rd * t_mean))).astype('float32')

    def _block_land_sea_mask(self, sources, y_sl, x_sl, block_cache):
        xland = sources['XLAND'].astype('float64')
        return np.where(xland < 1.5, 1.0, 0.0).astype('float32')

    def _block_precip_sum(self, sources, y_sl, x_sl, block_cache):
        # Sum the pre-computed hourly precipitation source vars (e.g. PREC_ACC_C + PREC_ACC_NC).
        total = sum(arr.astype('float64') for arr in sources.values())
        return total.astype('float32')

    # ------------------------------------------------------------------
    # Column-integrated block transforms (4D source, 3D output).
    # Sources are aligned on the unstaggered grid; reduction is along axis=1
    # (the eta-level axis). dp is shared between PWAT and VIMF via block_cache.
    # ------------------------------------------------------------------

    def _block_dp(self, sources, block_cache):
        """Pressure layer thickness on eta levels, shape (N, nz, ny, nx)."""
        if 'dp' in block_cache:
            return block_cache['dp']
        p = sources['P'].astype('float64')
        pb = sources['PB'].astype('float64')
        pressure = p + pb
        # Layer thickness via midpoint differences along the z axis (axis=1).
        dp = np.empty_like(pressure)
        dp[:, 0] = pressure[:, 0] - pressure[:, 1]
        dp[:, -1] = pressure[:, -2] - pressure[:, -1]
        dp[:, 1:-1] = (pressure[:, :-2] - pressure[:, 2:]) / 2.0
        dp = np.abs(dp)
        block_cache['dp'] = dp
        return dp

    def _block_rotated_wind_3d(self, sources, y_sl, x_sl, block_cache):
        """
        Earth-relative U/V on the unstaggered grid, shape (N, nz, ny, nx).
        Sources ``U`` and ``V`` are unstaggered upstream by ``_make_source``.
        """
        if 'wind_3d' in block_cache:
            return block_cache['wind_3d']
        u_grid = sources['U'].astype('float64')
        v_grid = sources['V'].astype('float64')
        if self._cosalpha is not None:
            cosa = self._cosalpha[y_sl, x_sl]
            sina = self._sinalpha[y_sl, x_sl]
            u_earth = u_grid * cosa + v_grid * sina
            v_earth = -u_grid * sina + v_grid * cosa
        else:
            u_earth, v_earth = u_grid, v_grid
        result = (u_earth, v_earth)
        block_cache['wind_3d'] = result
        return result

    def _block_precipitable_water(self, sources, y_sl, x_sl, block_cache):
        """PWAT = (1/g) * sum(QVAPOR * dp) over eta levels."""
        q = sources['QVAPOR'].astype('float64')
        dp = self._block_dp(sources, block_cache)
        return (np.sum(q * dp, axis=1) / 9.80665).astype('float32')

    def _block_precipitable_water_tracer(self, sources, y_sl, x_sl, block_cache):
        """PWAT_TR = (1/g) * sum(qv_tr * dp) over eta levels."""
        q = sources['qv_tr'].astype('float64')
        dp = self._block_dp(sources, block_cache)
        return (np.sum(q * dp, axis=1) / 9.80665).astype('float32')

    def _block_vimf_u(self, sources, y_sl, x_sl, block_cache):
        """VIMF_u = (1/g) * sum(QVAPOR * U_earth * dp) over eta levels."""
        q = sources['QVAPOR'].astype('float64')
        u_earth, _ = self._block_rotated_wind_3d(sources, y_sl, x_sl, block_cache)
        dp = self._block_dp(sources, block_cache)
        return (np.sum(q * u_earth * dp, axis=1) / 9.80665).astype('float32')

    def _block_vimf_v(self, sources, y_sl, x_sl, block_cache):
        """VIMF_v = (1/g) * sum(QVAPOR * V_earth * dp) over eta levels."""
        q = sources['QVAPOR'].astype('float64')
        _, v_earth = self._block_rotated_wind_3d(sources, y_sl, x_sl, block_cache)
        dp = self._block_dp(sources, block_cache)
        return (np.sum(q * v_earth * dp, axis=1) / 9.80665).astype('float32')

    def _setup_populate(self, var_key, target_levels):
        """Set up level-interpolation regrid function for 3D variables."""
        info = self.variables[var_key]
        if info['height'] == 'levels' and target_levels is not None:
            levels = np.array(target_levels, dtype='float64')

            if getattr(self, '_vertical_coord', 'height') == 'pressure':
                # Pressure decreases with altitude, but np.interp (used by geointerp)
                # requires ascending source values. Wrap the regrid function to negate
                # both source and target pressure so they become ascending.
                # Negated targets: e.g. [50000, 70000, 90000] -> [-90000, -70000, -50000]
                from geointerp import GridInterpolator
                gi = GridInterpolator()
                neg_targets = np.sort(-levels)
                inner = gi.regrid_levels(neg_targets, axis=0, method='linear')

                def _pressure_regrid(data, source_levels):
                    # Negate source pressure so it's ascending, interpolate,
                    # then reverse output to match original target order (ascending pressure)
                    result = inner(data, -source_levels)
                    return result[::-1]

                self._regrid_func = _pressure_regrid
            else:
                from geointerp import GridInterpolator
                gi = GridInterpolator()
                self._regrid_func = gi.regrid_levels(levels, axis=0, method='linear')
