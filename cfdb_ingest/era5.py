"""
ERA5 NetCDF file converter for cfdb.

Handles ERA5 data from the NCAR S3 archive where each file contains
a single variable. Supports both surface and pressure level products.
"""
import pathlib
from typing import Union, List, Tuple, Dict, Optional

import cfdb
import h5py
import numpy as np
import pyproj
import rechunkit

from cfdb_ingest.base import H5Ingest


# Coordinate variable names in ERA5 NetCDF files (excluded from variable detection)
_ERA5_COORD_VARS = {'latitude', 'longitude', 'time', 'utc_date', 'level'}

# Standard gravity for geopotential → geopotential height conversion
_G = 9.80665


ERA5_VARIABLE_MAPPING = {
    # --- Surface variables at 0m ---
    'SP': {
        'cfdb_name': 'surface_pressure',
        'source_vars': ['SP'],
        'transform': None,
        'height': 0.0,
    },
    'MSL': {
        'cfdb_name': 'mslp',
        'source_vars': ['MSL'],
        'transform': None,
        'height': 0.0,
    },
    'SSTK': {
        'cfdb_name': 'sea_surface_temp',
        'source_vars': ['SSTK'],
        'transform': None,
        'height': 0.0,
    },
    'SKT': {
        'cfdb_name': 'skin_temperature',
        'source_vars': ['SKT'],
        'transform': None,
        'height': 0.0,
    },
    'CI': {
        'cfdb_name': 'sea_ice',
        'source_vars': ['CI'],
        'transform': None,
        'height': 0.0,
    },
    'SD': {
        'cfdb_name': 'snow_water_equiv',
        'source_vars': ['SD'],
        'transform': None,
        'height': 0.0,
    },
    'RSN': {
        'cfdb_name': 'snow_density',
        'source_vars': ['RSN'],
        'transform': None,
        'height': 0.0,
    },
    'ASN': {
        'cfdb_name': 'snow_albedo',
        'source_vars': ['ASN'],
        'transform': None,
        'height': 0.0,
    },
    'TSN': {
        'cfdb_name': 'snow_layer_temperature',
        'source_vars': ['TSN'],
        'transform': None,
        'height': 0.0,
    },
    'FAL': {
        'cfdb_name': 'albedo',
        'source_vars': ['FAL'],
        'transform': None,
        'height': 0.0,
    },
    'STL1': {
        'cfdb_name': 'soil_layer_temp',
        'source_vars': ['STL1'],
        'transform': None,
        'height': 0.0,
    },
    'STL2': {
        'cfdb_name': 'soil_layer_temp',
        'source_vars': ['STL2'],
        'transform': None,
        'height': 0.0,
    },
    'STL3': {
        'cfdb_name': 'soil_layer_temp',
        'source_vars': ['STL3'],
        'transform': None,
        'height': 0.0,
    },
    'STL4': {
        'cfdb_name': 'soil_layer_temp',
        'source_vars': ['STL4'],
        'transform': None,
        'height': 0.0,
    },
    'SWVL1': {
        'cfdb_name': 'soil_moisture',
        'source_vars': ['SWVL1'],
        'transform': None,
        'height': 0.0,
    },
    'SWVL2': {
        'cfdb_name': 'soil_moisture',
        'source_vars': ['SWVL2'],
        'transform': None,
        'height': 0.0,
    },
    'SWVL3': {
        'cfdb_name': 'soil_moisture',
        'source_vars': ['SWVL3'],
        'transform': None,
        'height': 0.0,
    },
    'SWVL4': {
        'cfdb_name': 'soil_moisture',
        'source_vars': ['SWVL4'],
        'transform': None,
        'height': 0.0,
    },
    'ISTL1': {
        'cfdb_name': 'ice_surface_temperature',
        'source_vars': ['ISTL1'],
        'transform': None,
        'height': 0.0,
    },
    'ISTL2': {
        'cfdb_name': 'ice_surface_temperature',
        'source_vars': ['ISTL2'],
        'transform': None,
        'height': 0.0,
    },
    'ISTL3': {
        'cfdb_name': 'ice_surface_temperature',
        'source_vars': ['ISTL3'],
        'transform': None,
        'height': 0.0,
    },
    'ISTL4': {
        'cfdb_name': 'ice_surface_temperature',
        'source_vars': ['ISTL4'],
        'transform': None,
        'height': 0.0,
    },
    'CAPE': {
        'cfdb_name': 'cape',
        'source_vars': ['CAPE'],
        'transform': None,
        'height': 0.0,
    },
    'BLH': {
        'cfdb_name': 'boundary_layer_height',
        'source_vars': ['BLH'],
        'transform': None,
        'height': 0.0,
    },
    'TCC': {
        'cfdb_name': 'total_cloud_cover',
        'source_vars': ['TCC'],
        'transform': None,
        'height': 0.0,
    },
    'LCC': {
        'cfdb_name': 'low_cloud_cover',
        'source_vars': ['LCC'],
        'transform': None,
        'height': 0.0,
    },
    'MCC': {
        'cfdb_name': 'medium_cloud_cover',
        'source_vars': ['MCC'],
        'transform': None,
        'height': 0.0,
    },
    'HCC': {
        'cfdb_name': 'high_cloud_cover',
        'source_vars': ['HCC'],
        'transform': None,
        'height': 0.0,
    },
    'TCW': {
        'cfdb_name': 'total_column_water',
        'source_vars': ['TCW'],
        'transform': None,
        'height': 0.0,
    },
    'TCWV': {
        'cfdb_name': 'pwat',
        'source_vars': ['TCWV'],
        'transform': None,
        'height': 0.0,
    },
    'TCLW': {
        'cfdb_name': 'total_column_liquid_water',
        'source_vars': ['TCLW'],
        'transform': None,
        'height': 0.0,
    },
    'TCIW': {
        'cfdb_name': 'total_column_ice_water',
        'source_vars': ['TCIW'],
        'transform': None,
        'height': 0.0,
    },
    'TCRW': {
        'cfdb_name': 'total_column_rain_water',
        'source_vars': ['TCRW'],
        'transform': None,
        'height': 0.0,
    },
    'TCSW': {
        'cfdb_name': 'total_column_snow_water',
        'source_vars': ['TCSW'],
        'transform': None,
        'height': 0.0,
    },
    'TCO3': {
        'cfdb_name': 'total_column_ozone',
        'source_vars': ['TCO3'],
        'transform': None,
        'height': 0.0,
    },
    'CHNK': {
        'cfdb_name': 'charnock',
        'source_vars': ['CHNK'],
        'transform': None,
        'height': 0.0,
    },
    'SRC': {
        'cfdb_name': 'skin_reservoir_content',
        'source_vars': ['SRC'],
        'transform': None,
        'height': 0.0,
    },
    'FSR': {
        'cfdb_name': 'surface_roughness',
        'source_vars': ['FSR'],
        'transform': None,
        'height': 0.0,
    },
    'FLSR': {
        'cfdb_name': 'surface_roughness_heat',
        'source_vars': ['FLSR'],
        'transform': None,
        'height': 0.0,
    },
    'IEWS': {
        'cfdb_name': 'surface_stress_east',
        'source_vars': ['IEWS'],
        'transform': None,
        'height': 0.0,
    },
    'INSS': {
        'cfdb_name': 'surface_stress_north',
        'source_vars': ['INSS'],
        'transform': None,
        'height': 0.0,
    },
    'ISHF': {
        'cfdb_name': 'sensible_heat_flux',
        'source_vars': ['ISHF'],
        'transform': None,
        'height': 0.0,
    },
    'IE': {
        'cfdb_name': 'moisture_flux',
        'source_vars': ['IE'],
        'transform': None,
        'height': 0.0,
    },
    'ALUVP': {
        'cfdb_name': 'uv_albedo_direct',
        'source_vars': ['ALUVP'],
        'transform': None,
        'height': 0.0,
    },
    'ALUVD': {
        'cfdb_name': 'uv_albedo_diffuse',
        'source_vars': ['ALUVD'],
        'transform': None,
        'height': 0.0,
    },
    'ALNIP': {
        'cfdb_name': 'nir_albedo_direct',
        'source_vars': ['ALNIP'],
        'transform': None,
        'height': 0.0,
    },
    'ALNID': {
        'cfdb_name': 'nir_albedo_diffuse',
        'source_vars': ['ALNID'],
        'transform': None,
        'height': 0.0,
    },
    'LAILV': {
        'cfdb_name': 'leaf_area_index_low',
        'source_vars': ['LAILV'],
        'transform': None,
        'height': 0.0,
    },
    'LAIHV': {
        'cfdb_name': 'leaf_area_index_high',
        'source_vars': ['LAIHV'],
        'transform': None,
        'height': 0.0,
    },
    # Lake variables
    'LBLT': {
        'cfdb_name': 'lake_bottom_temperature',
        'source_vars': ['LBLT'],
        'transform': None,
        'height': 0.0,
    },
    'LTLT': {
        'cfdb_name': 'lake_total_layer_temperature',
        'source_vars': ['LTLT'],
        'transform': None,
        'height': 0.0,
    },
    'LSHF': {
        'cfdb_name': 'lake_shape_factor',
        'source_vars': ['LSHF'],
        'transform': None,
        'height': 0.0,
    },
    'LICT': {
        'cfdb_name': 'lake_ice_temperature',
        'source_vars': ['LICT'],
        'transform': None,
        'height': 0.0,
    },
    'LICD': {
        'cfdb_name': 'lake_ice_depth',
        'source_vars': ['LICD'],
        'transform': None,
        'height': 0.0,
    },

    # --- Surface variables at 2m ---
    'VAR_2T': {
        'cfdb_name': 'air_temperature',
        'source_vars': ['VAR_2T'],
        'transform': None,
        'height': 2.0,
    },
    'VAR_2D': {
        'cfdb_name': 'dew_point_temperature',
        'source_vars': ['VAR_2D'],
        'transform': None,
        'height': 2.0,
    },

    # --- Surface variables at 10m ---
    'VAR_10U': {
        'cfdb_name': 'u_wind',
        'source_vars': ['VAR_10U'],
        'transform': None,
        'height': 10.0,
    },
    'VAR_10V': {
        'cfdb_name': 'v_wind',
        'source_vars': ['VAR_10V'],
        'transform': None,
        'height': 10.0,
    },
    'U10N': {
        'cfdb_name': 'u_wind',
        'source_vars': ['U10N'],
        'transform': None,
        'height': 10.0,
    },
    'V10N': {
        'cfdb_name': 'v_wind',
        'source_vars': ['V10N'],
        'transform': None,
        'height': 10.0,
    },

    # --- Surface variables at 100m ---
    'VAR_100U': {
        'cfdb_name': 'u_wind',
        'source_vars': ['VAR_100U'],
        'transform': None,
        'height': 100.0,
    },
    'VAR_100V': {
        'cfdb_name': 'v_wind',
        'source_vars': ['VAR_100V'],
        'transform': None,
        'height': 100.0,
    },

    # --- Invariant variables (surface, height 0m) ---
    'Z_INV': {
        'cfdb_name': 'terrain_height',
        'source_vars': ['Z'],
        'transform': 'geopotential_to_height',
        'height': 0.0,
    },
    'LSM': {
        'cfdb_name': 'land_sea_mask',
        'source_vars': ['LSM'],
        'transform': None,
        'height': 0.0,
    },
    'CL': {
        'cfdb_name': 'lake_cover',
        'source_vars': ['CL'],
        'transform': None,
        'height': 0.0,
    },
    'DL': {
        'cfdb_name': 'lake_depth',
        'source_vars': ['DL'],
        'transform': None,
        'height': 0.0,
    },
    'CVL': {
        'cfdb_name': 'low_vegetation_cover',
        'source_vars': ['CVL'],
        'transform': None,
        'height': 0.0,
    },
    'CVH': {
        'cfdb_name': 'high_vegetation_cover',
        'source_vars': ['CVH'],
        'transform': None,
        'height': 0.0,
    },
    'TVL': {
        'cfdb_name': 'low_vegetation_type',
        'source_vars': ['TVL'],
        'transform': None,
        'height': 0.0,
    },
    'TVH': {
        'cfdb_name': 'high_vegetation_type',
        'source_vars': ['TVH'],
        'transform': None,
        'height': 0.0,
    },
    'SLT': {
        'cfdb_name': 'soil_type',
        'source_vars': ['SLT'],
        'transform': None,
        'height': 0.0,
    },
    'SDFOR': {
        'cfdb_name': 'std_dev_filtered_orography',
        'source_vars': ['SDFOR'],
        'transform': None,
        'height': 0.0,
    },
    'SDOR': {
        'cfdb_name': 'std_dev_orography',
        'source_vars': ['SDOR'],
        'transform': None,
        'height': 0.0,
    },
    'ISOR': {
        'cfdb_name': 'orography_anisotropy',
        'source_vars': ['ISOR'],
        'transform': None,
        'height': 0.0,
    },
    'ANOR': {
        'cfdb_name': 'orography_angle',
        'source_vars': ['ANOR'],
        'transform': None,
        'height': 0.0,
    },
    'SLOR': {
        'cfdb_name': 'orography_slope',
        'source_vars': ['SLOR'],
        'transform': None,
        'height': 0.0,
    },

    # --- Pressure level variables ---
    'T': {
        'cfdb_name': 'air_temperature',
        'source_vars': ['T'],
        'transform': None,
        'height': 'levels',
    },
    'U': {
        'cfdb_name': 'u_wind',
        'source_vars': ['U'],
        'transform': None,
        'height': 'levels',
    },
    'V': {
        'cfdb_name': 'v_wind',
        'source_vars': ['V'],
        'transform': None,
        'height': 'levels',
    },
    'Z_PL': {
        'cfdb_name': 'geopotential_height',
        'source_vars': ['Z'],
        'transform': 'geopotential_to_height',
        'height': 'levels',
    },
    'Q': {
        'cfdb_name': 'specific_humidity',
        'source_vars': ['Q'],
        'transform': None,
        'height': 'levels',
    },
    'W': {
        'cfdb_name': 'vertical_velocity',
        'source_vars': ['W'],
        'transform': None,
        'height': 'levels',
    },
    'VO': {
        'cfdb_name': 'vorticity',
        'source_vars': ['VO'],
        'transform': None,
        'height': 'levels',
    },
    'D': {
        'cfdb_name': 'divergence',
        'source_vars': ['D'],
        'transform': None,
        'height': 'levels',
    },
    'R': {
        'cfdb_name': 'relative_humidity',
        'source_vars': ['R'],
        'transform': None,
        'height': 'levels',
    },
    'O3': {
        'cfdb_name': 'ozone_mixing_ratio',
        'source_vars': ['O3'],
        'transform': None,
        'height': 'levels',
    },
    'PV': {
        'cfdb_name': 'potential_vorticity',
        'source_vars': ['PV'],
        'transform': None,
        'height': 'levels',
    },
    'CC': {
        'cfdb_name': 'cloud_cover',
        'source_vars': ['CC'],
        'transform': None,
        'height': 'levels',
    },
    'CLWC': {
        'cfdb_name': 'cloud_liquid_water_content',
        'source_vars': ['CLWC'],
        'transform': None,
        'height': 'levels',
    },
    'CIWC': {
        'cfdb_name': 'cloud_ice_water_content',
        'source_vars': ['CIWC'],
        'transform': None,
        'height': 'levels',
    },
    'CRWC': {
        'cfdb_name': 'rain_water_content',
        'source_vars': ['CRWC'],
        'transform': None,
        'height': 'levels',
    },
    'CSWC': {
        'cfdb_name': 'snow_water_content',
        'source_vars': ['CSWC'],
        'transform': None,
        'height': 'levels',
    },
}


def _detect_nc_var(h5):
    """Detect the main data variable name in an ERA5 NetCDF file."""
    for name in h5.keys():
        if name not in _ERA5_COORD_VARS:
            return name
    return None


class Era5Ingest(H5Ingest):
    """
    Convert ERA5 NetCDF files to cfdb.

    Handles ERA5's one-variable-per-file structure. Input files are scanned
    to detect which variable each contains, and a file index is built for
    efficient access during conversion.

    Parameters
    ----------
    input_paths : str, Path, or list thereof
        One or more ERA5 NetCDF file paths, or a directory containing them.
    """

    file_glob_pattern = '*.nc'
    x_coord_name = 'longitude'
    y_coord_name = 'latitude'

    def _init_source_metadata(self):
        """
        Scan all input files to detect variables, build file index, and extract
        spatial coordinates. ERA5 files each contain one variable.
        """
        import concurrent.futures

        def _get_nc_var(path):
            with h5py.File(path, 'r') as h5:
                return _detect_nc_var(h5)

        self._var_file_map = {}  # nc_var_name -> [path, ...]
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            nc_vars = list(executor.map(_get_nc_var, self.input_paths))

        for path, nc_var in zip(self.input_paths, nc_vars):
            if nc_var is not None:
                self._var_file_map.setdefault(nc_var, []).append(path)

        # Sort file lists for consistent ordering
        for nc_var in self._var_file_map:
            self._var_file_map[nc_var].sort()

        # Extract spatial coords and CRS from the first file
        with h5py.File(self.input_paths[0], 'r') as h5:
            self.crs = self._parse_crs(h5)
            spatial = self._parse_spatial_coords(h5)

        self.x = spatial['x']
        self.y = spatial['y']

    def _init_time(self):
        """
        Build unified time index across all input files.

        ERA5 files for the same variable at different time periods are
        concatenated. Files for different variables may have different
        time ranges (sfc=monthly, pl=daily), so we build a union of all times.
        """
        import concurrent.futures

        def _get_times(path):
            with h5py.File(path, 'r') as h5:
                return self._parse_time(h5)

        all_times_list = []
        self._var_time_map = {}  # nc_var -> [(path, times_array), ...]

        for nc_var, paths in self._var_file_map.items():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                times_list = list(executor.map(_get_times, paths))
            
            var_times = []
            for path, times in zip(paths, times_list):
                var_times.append((path, times))
                all_times_list.append(times)
            
            self._var_time_map[nc_var] = var_times

        combined = np.concatenate(all_times_list) if all_times_list else np.array([], dtype='datetime64[m]')
        self.times = np.unique(combined)

        # Build global time -> index lookup
        self._time_to_idx = {t: i for i, t in enumerate(self.times)}

        # Build raw_to_unique for compatibility with base class populate methods
        # (identity mapping since we handle deduplication ourselves)
        self._raw_to_unique = np.arange(len(self.times), dtype='int64')
        self._file_time_map = []  # not used directly, but needed for base class

    def _init_variables(self):
        """
        Determine which mapped variables are available based on detected
        NetCDF variable names in the input files.
        """
        mapping = self._get_variable_mapping()
        available = {}

        # Z_PL and Z_INV both have source_var='Z' — handle separately
        z_keys = {'Z_PL', 'Z_INV'}
        for key, info in mapping.items():
            if key in z_keys:
                continue
            source_var = info['source_vars'][0]
            if source_var in self._var_file_map:
                available[key] = info

        # Handle Z variable disambiguation: Z appears in both pl and invariant
        # products. If files contain pressure-level Z (has 'level' dim), map
        # to Z_PL. If surface-only Z (no 'level' dim), map to Z_INV.
        if 'Z' in self._var_file_map:
            has_pl_z = False
            has_inv_z = False
            for path in self._var_file_map['Z']:
                with h5py.File(path, 'r') as h5:
                    if 'level' in h5:
                        has_pl_z = True
                    else:
                        has_inv_z = True

            if has_pl_z and 'Z_PL' in mapping:
                available['Z_PL'] = mapping['Z_PL']
                # Separate pl Z files into their own time map entry
                pl_z_files = []
                for path in self._var_file_map['Z']:
                    with h5py.File(path, 'r') as h5:
                        if 'level' in h5:
                            pl_z_files.append(path)
                if pl_z_files:
                    var_times = []
                    for path in pl_z_files:
                        with h5py.File(path, 'r') as h5:
                            var_times.append((path, self._parse_time(h5)))
                    self._var_time_map['Z_PL'] = var_times

            if has_inv_z and 'Z_INV' in mapping:
                available['Z_INV'] = mapping['Z_INV']
                # Separate invariant Z files into their own time map entry
                inv_z_files = []
                for path in self._var_file_map['Z']:
                    with h5py.File(path, 'r') as h5:
                        if 'level' not in h5:
                            inv_z_files.append(path)
                if inv_z_files:
                    var_times = []
                    for path in inv_z_files:
                        with h5py.File(path, 'r') as h5:
                            var_times.append((path, self._parse_time(h5)))
                    self._var_time_map['Z_INV'] = var_times

        self.variables = available

    def _parse_crs(self, h5):
        """ERA5 is always on a regular lat-lon grid (EPSG:4326)."""
        return pyproj.CRS.from_epsg(4326)

    def _parse_time(self, h5):
        """
        Extract time coordinate from an ERA5 NetCDF file.

        ERA5 time is stored as hours since 1900-01-01 00:00:00.
        """
        time_var = h5['time']
        hours = time_var[:]
        base = np.datetime64('1900-01-01T00:00', 'h')
        return (base + hours.astype('timedelta64[h]')).astype('datetime64[m]')

    def _parse_spatial_coords(self, h5):
        """
        Extract latitude and longitude arrays.

        ERA5 uses 'latitude' (descending, 90 to -90) and 'longitude' (0 to 359.75).
        For cfdb, we use longitude as x and latitude as y. Latitude is reversed
        to be ascending.
        """
        lat = np.array(h5['latitude'][:], dtype='float64')
        lon = np.array(h5['longitude'][:], dtype='float64')

        # Ensure latitude is ascending (ERA5 stores it descending)
        if lat[0] > lat[-1]:
            lat = lat[::-1]
            self._lat_reversed = True
        else:
            self._lat_reversed = False

        return {'x': lon, 'y': lat}

    def _get_variable_mapping(self):
        return ERA5_VARIABLE_MAPPING

    def _read_variable(self, h5, var_key, time_idx, spatial_slice):
        """
        Read a single variable for one timestep from an ERA5 file.

        For ERA5, h5 is the file for this specific variable (not a shared file).
        """
        info = self.variables[var_key]
        src_var = info['source_vars'][0]
        y_sl, x_sl = spatial_slice
        transform = info.get('transform')

        data = h5[src_var]
        is_3d = data.ndim == 4  # (time, level, lat, lon)

        if is_3d:
            raw = data[time_idx, :, y_sl, x_sl].astype('float32')
        else:
            raw = data[time_idx, y_sl, x_sl].astype('float32')

        # Reverse latitude if needed
        if self._lat_reversed:
            if is_3d:
                raw = raw[:, ::-1, :]
            else:
                raw = raw[::-1, :]

        if transform == 'geopotential_to_height':
            raw = raw / _G

        return raw

    def _get_dataset_attrs(self):
        """Add ERA5-specific attributes."""
        attrs = super()._get_dataset_attrs()
        attrs['source'] = 'ERA5 reanalysis (ECMWF via NCAR)'
        return attrs

    def convert(
        self,
        cfdb_path,
        variables=None,
        start_date=None,
        end_date=None,
        bbox=None,
        target_levels=None,
        vertical_coord='pressure',
        max_mem=2**27,
        chunk_shape=None,
        split=False,
        dataset_type='grid',
        **cfdb_kwargs,
    ):
        """
        Convert ERA5 files to cfdb.

        Parameters
        ----------
        cfdb_path : str or Path
            Output path. For combined mode, a .cfdb file path.
            For split mode, a directory where individual .cfdb files are created.
        variables : list of str or None
            Variable names to convert. None converts all available.
        start_date, end_date : str or None
            Optional time range filter.
        bbox : tuple of 4 floats or None
            Bounding box (min_lon, min_lat, max_lon, max_lat) in WGS84.
        target_levels : list of float or None
            Pressure levels in Pa for pressure-level variables.
            If None and pressure-level variables are requested, levels are
            read from the source files.
        vertical_coord : str
            Vertical coordinate name. Default 'pressure' for ERA5.
        max_mem : int
            Memory budget in bytes for read buffers.
        chunk_shape : tuple or None
            Output chunk shape for 4D variables.
        split : bool
            If True, create one cfdb file per variable.
        dataset_type : str
            Passed to cfdb.open_dataset.
        **cfdb_kwargs
            Extra kwargs for cfdb.open_dataset (e.g., compression).
        """
        var_keys = self.resolve_variables(variables)

        has_levels = any(self.variables[k]['height'] == 'levels' for k in var_keys)

        # Auto-detect pressure levels from source files if not specified
        if has_levels and target_levels is None:
            target_levels = self._detect_pressure_levels(var_keys)

        if split:
            self._convert_split(cfdb_path, var_keys, start_date, end_date,
                                bbox, target_levels, vertical_coord, max_mem,
                                chunk_shape, dataset_type, **cfdb_kwargs)
        else:
            # Use base class convert for combined mode
            super().convert(
                cfdb_path=cfdb_path,
                variables=[k for k in var_keys],
                start_date=start_date,
                end_date=end_date,
                bbox=bbox,
                target_levels=target_levels,
                vertical_coord=vertical_coord,
                max_mem=max_mem,
                chunk_shape=chunk_shape,
                dataset_type=dataset_type,
                **cfdb_kwargs,
            )

    def _detect_pressure_levels(self, var_keys):
        """Read pressure levels from the first pressure-level file."""
        for var_key in var_keys:
            if self.variables[var_key]['height'] != 'levels':
                continue
            src_var = self.variables[var_key]['source_vars'][0]
            if src_var not in self._var_file_map:
                continue
            path = self._var_file_map[src_var][0]
            with h5py.File(path, 'r') as h5:
                if 'level' in h5:
                    # ERA5 levels are in hPa, convert to Pa
                    levels_hpa = np.array(h5['level'][:], dtype='float64')
                    return sorted((levels_hpa * 100).tolist())
        return None

    def _convert_split(self, output_dir, var_keys, start_date, end_date,
                       bbox, target_levels, vertical_coord, max_mem,
                       chunk_shape, dataset_type, **cfdb_kwargs):
        """Create one cfdb file per variable."""

        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for var_key in var_keys:
            info = self.variables[var_key]
            cfdb_name = info['cfdb_name']
            is_levels = info['height'] == 'levels'

            file_name = f'{cfdb_name}.cfdb'
            cfdb_path = output_dir / file_name

            if is_levels:
                vk_levels = target_levels
            else:
                vk_levels = None

            # Use base class convert for each variable individually
            super().convert(
                cfdb_path=cfdb_path,
                variables=[var_key],
                start_date=start_date,
                end_date=end_date,
                bbox=bbox,
                target_levels=vk_levels,
                vertical_coord=vertical_coord,
                max_mem=max_mem,
                chunk_shape=chunk_shape,
                dataset_type=dataset_type,
                **cfdb_kwargs,
            )
            print(f'  Written: {cfdb_path}')

    # ------------------------------------------------------------------
    # Override base class populate to handle ERA5's multi-file structure
    # ------------------------------------------------------------------

    def _get_var_time_entries(self, var_key):
        """Get the (path, times) entries for a variable key."""
        # Try var_key first (for disambiguated keys like Z_PL, Z_INV)
        if var_key in self._var_time_map:
            return self._var_time_map[var_key]
        # Fall back to source var name
        src_var = self.variables[var_key]['source_vars'][0]
        return self._var_time_map.get(src_var, [])

    def _populate_with_rechunkit(self, data_var, var_key, time_mask, spatial_slice, max_mem, vert_indices):
        """
        Override rechunkit populate for ERA5's one-var-per-file structure.

        Instead of iterating over self._file_time_map (which is WRF-oriented),
        we iterate over the files for this specific variable.
        """
        info = self.variables[var_key]
        src_var = info['source_vars'][0]
        y_sl, x_sl = spatial_slice

        entries = self._get_var_time_entries(var_key)
        if not entries:
            return

        # Precompute output time mapping
        output_map = {}
        out_idx = 0
        for global_t in range(len(time_mask)):
            if time_mask[global_t]:
                output_map[global_t] = out_idx
                out_idx += 1

        for path, file_times in entries:
            # Find time bounds for this file to restrict rechunkit reads
            file_mask = []
            for local_t in range(len(file_times)):
                t = file_times[local_t]
                if t in self._time_to_idx:
                    global_t = self._time_to_idx[t]
                    if time_mask[global_t]:
                        file_mask.append((local_t, global_t))

            if not file_mask:
                continue
                
            t_start = file_mask[0][0]
            t_stop = file_mask[-1][0] + 1
            
            # Map local_t to global_t for this file
            local_to_global = {local_t: global_t for local_t, global_t in file_mask}

            with h5py.File(path, 'r') as h5:
                h5_var = h5[src_var]
                source_chunks = h5_var.chunks or rechunkit.guess_chunk_shape(
                    h5_var.shape, h5_var.dtype.itemsize, max_mem
                )

                y_start, y_stop, _ = y_sl.indices(h5_var.shape[1] if h5_var.ndim == 3 else h5_var.shape[2])
                x_start, x_stop, _ = x_sl.indices(h5_var.shape[2] if h5_var.ndim == 3 else h5_var.shape[3])
                ny = y_stop - y_start
                nx = x_stop - x_start
                
                sel_time = slice(t_start, t_stop)
                sel_y = slice(y_start, y_stop)
                sel_x = slice(x_start, x_stop)
                
                if h5_var.ndim == 3:
                    sel = (sel_time, sel_y, sel_x)
                    target_chunks = (1, ny, nx)
                else:
                    sel = (sel_time, slice(None), sel_y, sel_x)
                    target_chunks = (1, h5_var.shape[1], ny, nx)

                for write_slices, data in rechunkit.rechunker(
                    h5_var.__getitem__, h5_var.shape, h5_var.dtype,
                    source_chunks, target_chunks, max_mem, sel=sel,
                ):
                    for i, chunk_t in enumerate(range(write_slices[0].start, write_slices[0].stop)):
                        local_t = t_start + chunk_t
                        if local_t not in local_to_global:
                            continue
                        global_t = local_to_global[local_t]
                        out_t = output_map[global_t]
                        
                        raw = data[i].astype('float32')
                        if self._lat_reversed:
                            if h5_var.ndim == 4:
                                raw = raw[:, ::-1, :]
                            else:
                                raw = raw[::-1, :]

                        if info.get('transform') == 'geopotential_to_height':
                            raw = raw / _G

                        self._write_data_var(data_var, raw, out_t, vert_indices)

    def _populate_per_timestep(self, data_var, var_key, time_mask, spatial_slice, vert_indices, max_mem, is_accumulation):
        """Override per-timestep populate for ERA5's multi-file structure."""
        entries = self._get_var_time_entries(var_key)
        if not entries:
            return

        output_map = {}
        out_idx = 0
        for global_t in range(len(time_mask)):
            if time_mask[global_t]:
                output_map[global_t] = out_idx
                out_idx += 1

        for path, file_times in entries:
            # We also try to use rechunkit here if possible, to align with chunking goals
            # If not, we just iterate with precalculated output_map
            with h5py.File(path, 'r', rdcc_nbytes=max_mem) as h5:
                for local_t in range(len(file_times)):
                    t = file_times[local_t]
                    if t not in self._time_to_idx:
                        continue
                    global_t = self._time_to_idx[t]
                    if not time_mask[global_t]:
                        continue

                    out_t = output_map[global_t]

                    data = self._read_variable(h5, var_key, local_t, spatial_slice)
                    self._write_data_var(data_var, data, out_t, vert_indices)

    def _populate_batch_per_timestep(self, batch_items, time_mask, spatial_slice, max_mem):
        """
        Override batch populate for ERA5's multi-file structure.

        Since ERA5 has one variable per file, there's no benefit to batching
        across variables (each opens a different file). Process individually.
        """
        for var_key, data_var, vert_indices in batch_items:
            self._populate_per_timestep(data_var, var_key, time_mask, spatial_slice, vert_indices, max_mem, is_accumulation=False)
