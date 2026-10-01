"""
WRF output file converter for cfdb.
"""
import pathlib
import warnings
from typing import Union, List, Tuple, Dict, Optional

import h5py
import numpy as np
import pyproj

from cfdb_ingest.base import H5Ingest, resolve_variable_keys


# WRF/WPS map projections are computed on a sphere of this radius (WPS geogrid/src/constants_module.F:
# ``EARTH_RADIUS_M = 6370000.``), which is what XLAT/XLONG were derived on. A CRS without it defaults
# to the WGS84 ellipsoid and misplaces cells by kilometres on NZ domains (cfdb-ingest < 0.6.0).
WPS_EARTH_RADIUS_M = 6370000.0

# Global attributes that must agree across every input file of one conversion (a d02 file slipped into a
# d03 list, or a different bucket/accumulation configuration, would otherwise be read with the first
# file's values). Only keys present in the first file are compared.
_WRF_CONSISTENT_ATTRS = (
    'MAP_PROJ', 'TRUELAT1', 'TRUELAT2', 'STAND_LON', 'MOAD_CEN_LAT', 'CEN_LAT', 'CEN_LON',
    'POLE_LAT', 'POLE_LON', 'DX', 'DY', 'WEST-EAST_GRID_DIMENSION', 'SOUTH-NORTH_GRID_DIMENSION',
    'BUCKET_MM', 'PREC_ACC_DT',
)


def _wrf_attr(attrs, key):
    """Extract a scalar value from a WRF HDF5 attribute (may be a 1-element array)."""
    val = attrs[key]
    if hasattr(val, 'item'):
        val = val.item()
    if isinstance(val, bytes):
        return val.decode()
    return val


def _wrf_run_start(attrs):
    """The run's init from SIMULATION_START_DATE (else START_DATE) as datetime64[m], or None."""
    for key in ('SIMULATION_START_DATE', 'START_DATE'):
        if key in attrs:
            val = _wrf_attr(attrs, key).strip()
            if val:
                return np.datetime64(val.replace('_', 'T'), 'm')
    return None


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
        'nonneg': True,
        'height': 2.0,
    },
    'Q2_SH': {
        'cfdb_name': 'specific_humidity',
        'source_vars': ['Q2'],
        'transform': 'mixing_ratio_to_specific_humidity_2d',
        'nonneg': True,
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
        'accumulated': 'increment',
    },
    # Same quantity as RAIN (precipitation over the output interval) from WRF's own windowed
    # accumulators (namelist prec_acc_dt = history interval): each frame already holds the increment
    # since the previous frame, so it needs no previous-frame state -- the source for an init ingested
    # one file at a time. Lead 0 is 0 (RAIN gives NaN there). Not valid on a two-way-nested PARENT
    # domain, whose PREC_ACC_* the child's feedback overwrites; use it on the innermost domain.
    # Negatives (float noise from WRF's bucket arithmetic, and offline backfills of it) are clipped to 0,
    # as RAIN's increments are (since 0.6.0).
    'PREC_ACC': {
        'cfdb_name': 'precip',
        'source_vars': ['PREC_ACC_NC', 'PREC_ACC_C'],
        'transform': 'sum_nonneg',
        'height': 0.0,
        'accumulated': 'window',
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
    # TSK is WRF's surface skin (radiating) temperature, not a soil layer: stored as skin_temperature since
    # 0.6.2 (it was 'soil_temp' -> soil_temperature before; appending to an older dataset lands it in a NEW
    # variable beside the old one).
    'TSK': {
        'cfdb_name': 'skin_temp',
        'source_vars': ['TSK'],
        'transform': None,
        'height': 0.0,
    },
    'SWDOWN': {
        'cfdb_name': 'shortwave_radiation',
        'source_vars': ['SWDOWN'],
        'transform': None,
        'nonneg': True,
        'height': 0.0,
    },
    'GLW': {
        'cfdb_name': 'longwave_radiation',
        'source_vars': ['GLW'],
        'transform': None,
        'nonneg': True,
        'height': 0.0,
    },
    'SNOWH': {
        'cfdb_name': 'snow_depth',
        'source_vars': ['SNOWH'],
        'transform': None,
        'nonneg': True,
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
    # Native passthrough from WRF image >= wrf-auto-runs-intel-wvt:1.12, with
    # fallback to the hypsometric computation for older wrfouts that only
    # have PSFC/T2/HGT.
    'SLP': {
        'cfdb_name': 'mslp',
        'source_vars': ['SLP'],
        'transform': None,
        'fallback_source_vars': ['PSFC', 'T2', 'HGT'],
        'fallback_transform': 'sea_level_pressure',
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
        'nonneg': True,
        'height': 0.0,
    },
    # --- Column-integrated variables (native 2D if WRF >= 1.12, else 3D→2D) ---
    'PWAT': {
        'cfdb_name': 'pwat',
        'source_vars': ['PWAT'],
        'transform': None,
        'nonneg': True,
        'fallback_source_vars': ['QVAPOR', 'P', 'PB'],
        'fallback_transform': 'precipitable_water',
        'height': 0.0,
    },
    'PWAT_TR': {
        'cfdb_name': 'pwat_tr',
        'source_vars': ['PWAT_TR'],
        'transform': None,
        'fallback_source_vars': ['qv_tr', 'P', 'PB'],
        'fallback_transform': 'precipitable_water_tracer',
        'height': 0.0,
        'region_aware': True,
    },
    'RAIN_TR': {
        'cfdb_name': 'precip_tr',
        'source_vars': ['TR_RAINNC', 'TR_RAINC'],
        'transform': 'accumulation_increment',
        'height': 0.0,
        'region_aware': True,
        'accumulated': 'increment',
    },
    'VIMF_U': {
        'cfdb_name': 'vimf_u',
        'source_vars': ['VIMF_U'],
        'transform': None,
        'fallback_source_vars': ['QVAPOR', 'U', 'V', 'P', 'PB'],
        'fallback_transform': 'vimf_u',
        'height': 0.0,
    },
    'VIMF_V': {
        'cfdb_name': 'vimf_v',
        'source_vars': ['VIMF_V'],
        'transform': None,
        'fallback_source_vars': ['QVAPOR', 'U', 'V', 'P', 'PB'],
        'fallback_transform': 'vimf_v',
        'height': 0.0,
    },
    # --- Tracer moisture flux and IVT magnitude (native only, WRF >= 1.12) ---
    'VIMF_TR_U': {
        'cfdb_name': 'vimf_tr_u',
        'source_vars': ['VIMF_TR_U'],
        'transform': None,
        'height': 0.0,
        'region_aware': True,
    },
    'VIMF_TR_V': {
        'cfdb_name': 'vimf_tr_v',
        'source_vars': ['VIMF_TR_V'],
        'transform': None,
        'height': 0.0,
        'region_aware': True,
    },
    'IVT': {
        'cfdb_name': 'ivt',
        'source_vars': ['IVT'],
        'transform': None,
        'nonneg': True,
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
    'SF_URBAN_PHYSICS', 'SST_UPDATE',
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


def grid_rotation(map_proj, truelat1, truelat2, stand_lon, xlong, pole_lat=90.0):
    """
    Grid->earth wind rotation of a WRF grid, analytically: ``(cos_alpha, sin_alpha, validated)`` on the
    shape of ``xlong`` (degrees), in the sense of WRF's ``COSALPHA``/``SINALPHA`` and of every rotation in
    this module: ``u_earth = u*cos + v*sin``, ``v_earth = -u*sin + v*cos``.

    Lambert conformal: ``alpha = cone * hemi * (lon - stand_lon)`` (the difference wrapped to [-180, 180),
    so a grid across the antimeridian is continuous), the cone constant as WPS computes it
    (``module_map_utils`` ``lc_cone``, tangent or secant) and ``hemi`` the sign of TRUELAT1. Polar
    stereographic: the same with cone 1. Mercator and a true lat-lon grid: no rotation.

    ``validated`` is True only for the branches measured against WRF's own ``COSALPHA`` on real output:
    the tangent-cone Lambert (TRUELAT1 == TRUELAT2) and the unrotated lat-lon grid -- measured
    2026-09-28 on real SH d01/d02 wrfouts: max 2.7e-7 rad, a flipped sign 91-106 deg off. The secant
    cone, polar stereographic and Mercator are WPS's formulas, not yet checked against a real file;
    callers must cross-check them against a wrfout's COSALPHA/SINALPHA. A rotated-pole lat-lon grid
    (``POLE_LAT != 90``) has a rotation this function does not model and raises.
    """
    xlong = np.asarray(xlong, dtype='float64')
    if map_proj == 6:
        if not np.isclose(pole_lat, 90.0):
            raise ValueError(f'MAP_PROJ=6 with POLE_LAT={pole_lat}: a rotated-pole grid, whose wind rotation is not '
                             f'modelled here')
        return np.ones_like(xlong), np.zeros_like(xlong), True
    if map_proj == 3:
        return np.ones_like(xlong), np.zeros_like(xlong), False
    rad = np.pi / 180.0
    hemi = -1.0 if truelat1 < 0 else 1.0
    if map_proj == 1:
        tangent = abs(truelat1 - truelat2) <= 0.1   # WPS's own tangent test (lc_cone)
        if tangent:
            cone = np.sin(abs(truelat1) * rad)
        else:
            cone = ((np.log10(np.cos(truelat1 * rad)) - np.log10(np.cos(truelat2 * rad)))
                    / (np.log10(np.tan((45.0 - abs(truelat1) / 2.0) * rad))
                       - np.log10(np.tan((45.0 - abs(truelat2) / 2.0) * rad))))
        validated = bool(tangent)
    elif map_proj == 2:
        cone = 1.0
        validated = False
    else:
        raise ValueError(f'Unsupported WRF MAP_PROJ: {map_proj}')
    dlon = (xlong - stand_lon + 180.0) % 360.0 - 180.0
    alpha = cone * hemi * dlon * rad
    return np.cos(alpha), np.sin(alpha), validated


# Transforms that turn grid-relative U/V into earth-relative components. Without a known rotation their
# output would be grid-relative data under an earth-relative name, so they are withheld (speed and
# vorticity are frame-invariant and stay). Keyed on the ACTIVE transform: native VIMF_U/V (transform None)
# is written earth-relative by WRF; only the 3-D fallback rotates.
_ROTATING_TRANSFORMS = frozenset({
    'wind_direction', 'u_wind', 'v_wind', 'wind_direction_3d', 'u_wind_3d', 'v_wind_3d', 'vimf_u', 'vimf_v',
    'u_wind_pl', 'v_wind_pl',
})


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
    static_path : str or Path, optional
        A full wrfout or a ``wrfinput`` of the same domain, used only when the input files carry no
        COSALPHA/SINALPHA (a wrfout pruned of them): its grid attributes and XLAT/XLONG must match, its rotation
        must agree with the projection formula, and WRF's own values from it then rotate U10/V10/WIND_DIR10
        (since 0.8.0). Without it those keys are withheld.
    """

    file_glob_pattern = 'wrfout*'

    # Largest accepted distance (m) between the fitted x/y lattice and the file's own XLAT/XLONG, in
    # both the lattice residual and the round trip back to lat/lon. Calibrated 2026-09-24 on 114 real
    # Lambert wrfout files (DX 1-27 km, every local run): healthy 1.6-4.7 m (float32 XLAT/XLONG; worst on
    # 12 km d01), so 25 m leaves ~5x headroom; a WGS84 construction is off by 1-8 km.
    xy_tolerance_m = 25.0

    def __init__(self, input_paths, static_path=None):
        # A companion file of the same domain (a full wrfout, or wrfinput) lending its COSALPHA/SINALPHA when the
        # input files carry none (since 0.8.0). Set before the base __init__, which reads the metadata.
        self._static_path = pathlib.Path(static_path) if static_path is not None else None
        super().__init__(input_paths)

    def _init_source_metadata(self):
        """
        Override to also load wind rotation fields and WRF source attributes.

        The grid, CRS and physics attributes come from the first file; every other file's header is
        checked against it (``_WRF_CONSISTENT_ATTRS``) and its run start is recorded per file, because
        one conversion may span several independent runs (a stitched hindcast).
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

            self._map_proj = int(_wrf_attr(h5.attrs, 'MAP_PROJ'))
            self._pole_lat = float(_wrf_attr(h5.attrs, 'POLE_LAT')) if 'POLE_LAT' in h5.attrs else 90.0
        self._rotation_note = None
        self._rotation_source = None
        if getattr(self, '_static_path', None) is not None:
            if self._cosalpha is None:
                # The files carry no rotation: take the companion's, after its grid and the formula agree with it.
                self._init_rotation(fallback_to_formula=False)
            else:
                # The files' own rotation is used; a companion of another grid is still a caller error.
                self._check_companion_grid()
        with h5py.File(self.input_paths[0], 'r') as h5:
            # Extract WRF source info and physics parameters
            self._source_title = _wrf_attr(h5.attrs, 'TITLE').strip()
            # The run's init, for forecast mode: SIMULATION_START_DATE is the true init of a
            # restarted run; START_DATE is the start of this segment.
            self._simulation_start = _wrf_run_start(h5.attrs)
            self._wrf_params = {}
            for key in _WRF_DATASET_ATTRS:
                if key in h5.attrs:
                    self._wrf_params[key] = _wrf_attr(h5.attrs, key)
            ref_attrs = {key: _wrf_attr(h5.attrs, key) for key in _WRF_CONSISTENT_ATTRS if key in h5.attrs}
            self._prec_acc_dt = float(ref_attrs['PREC_ACC_DT']) if 'PREC_ACC_DT' in ref_attrs else None

        # Per-file run start (aligned with self.input_paths) and a header consistency check.
        self._file_run_starts = [self._simulation_start]
        for path in self.input_paths[1:]:
            with h5py.File(path, 'r') as h5:
                self._file_run_starts.append(_wrf_run_start(h5.attrs))
                extra = sorted(k for k in _WRF_CONSISTENT_ATTRS if k in h5.attrs and k not in ref_attrs)
                if extra:
                    raise ValueError(
                        f'{path.name}: global attribute(s) {extra} are absent from {self.input_paths[0].name}; '
                        f'all files of one conversion must share one grid and configuration')
                for key, ref in ref_attrs.items():
                    val = _wrf_attr(h5.attrs, key) if key in h5.attrs else None
                    same = val is not None and (val == ref if isinstance(ref, str) else np.isclose(val, ref, rtol=1e-6, atol=0.0))
                    if not same:
                        raise ValueError(
                            f'{path.name}: global attribute {key}={val!r} differs from {ref!r} in '
                            f'{self.input_paths[0].name}; all files of one conversion must share one grid '
                            f'and configuration (a file from another domain or run setup?)'
                        )

        self._path_run_start = {str(p): r for p, r in zip(self.input_paths, self._file_run_starts)}

        self.x = spatial['x']
        self.y = spatial['y']
        self._heterogeneous_grids = False
        self._dx = float(self.x[1] - self.x[0])
        self._dy = float(self.y[1] - self.y[0])

    # convert(extend=..., time_label='start', squeeze_height=...) are implemented for WRF only.
    _supports_grid_extend = True

    # Keys withheld because their transform rotates winds and the rotation is unknown (see _init_variables).
    _unrotatable = ()

    def _read_companion(self, attrs, xlat, xlong, need_rotation):
        """
        Check that ``static_path`` is the same grid as the inputs (grid attributes, XLAT/XLONG); return its
        ``(COSALPHA, SINALPHA)`` as float64 when ``need_rotation`` (refused if it has none), else ``(None, None)``.
        """
        with h5py.File(self._static_path, 'r') as st:
            for key, ref in attrs.items():
                val = _wrf_attr(st.attrs, key) if key in st.attrs else None
                same = val is not None and (val == ref if isinstance(ref, str) else
                                            np.isclose(val, ref, rtol=1e-6, atol=1e-6))
                if not same:
                    raise ValueError(f'static_path {self._static_path.name}: {key}={val!r} differs from {ref!r} '
                                     f'in {self.input_paths[0].name}; not the same grid')
            if st['XLAT'].shape[-2:] != xlat.shape:
                raise ValueError(f'static_path {self._static_path.name}: XLAT shape {st["XLAT"].shape[-2:]} differs '
                                 f'from {xlat.shape}; not the same grid')
            off = max(float(np.max(np.abs(st['XLAT'][0] - xlat))),
                      float(np.max(np.abs((st['XLONG'][0] - xlong + 180.0) % 360.0 - 180.0))))
            if off > _COMPANION_XY_ATOL_DEG:
                raise ValueError(f'static_path {self._static_path.name}: XLAT/XLONG differ by up to {off:.2e} '
                                 f'deg from {self.input_paths[0].name}; not the same grid')
            if not need_rotation:
                return None, None
            if 'COSALPHA' not in st or 'SINALPHA' not in st:
                raise ValueError(f'static_path {self._static_path.name} has no COSALPHA/SINALPHA')
            return st['COSALPHA'][0].astype('float64'), st['SINALPHA'][0].astype('float64')

    def _check_companion_grid(self):
        """``static_path`` given while the inputs carry their own rotation: still refuse a companion of another grid."""
        with h5py.File(self.input_paths[0], 'r') as h5:
            attrs = {k: _wrf_attr(h5.attrs, k) for k in _WRF_GRID_ATTRS if k in h5.attrs}
            xlat = h5['XLAT'][0].astype('float64')
            xlong = h5['XLONG'][0].astype('float64')
        self._read_companion(attrs, xlat, xlong, need_rotation=False)

    def _init_rotation(self, fallback_to_formula=True):
        """
        Set ``_cosalpha``/``_sinalpha`` (2-D) from the formula, cross-checked against ``static_path`` if given (then
        WRF's own values from it are used). Without ``static_path``: the formula where it is validated, if
        ``fallback_to_formula`` (pressure-level files; ``WrfIngest`` calls this only with a ``static_path``).
        """
        with h5py.File(self.input_paths[0], 'r') as h5:
            attrs = {k: _wrf_attr(h5.attrs, k) for k in _WRF_GRID_ATTRS if k in h5.attrs}
            xlat = h5['XLAT'][0].astype('float64')
            xlong = h5['XLONG'][0].astype('float64')
        self._rotation_note = None
        try:
            truelat1 = float(attrs.get('TRUELAT1', 0.0))
            cosa, sina, validated = grid_rotation(
                self._map_proj, truelat1, float(attrs.get('TRUELAT2', truelat1)), float(attrs.get('STAND_LON', 0.0)),
                xlong, pole_lat=self._pole_lat)
        except ValueError as e:
            self._cosalpha = self._sinalpha = None
            self._rotation_note = str(e)
            return

        if self._static_path is not None:
            f_cos, f_sin = self._read_companion(attrs, xlat, xlong, need_rotation=True)
            diff = np.abs(np.angle(np.exp(1j * (np.arctan2(f_sin, f_cos) - np.arctan2(sina, cosa)))))
            worst = float(np.max(diff))
            if worst > ROTATION_TOLERANCE_RAD:
                raise ValueError(
                    f'wind rotation from the projection attributes disagrees with COSALPHA/SINALPHA of '
                    f'{self._static_path.name} by up to {worst:.2e} rad (tolerance {ROTATION_TOLERANCE_RAD:g}); '
                    f'the formula does not describe this grid (MAP_PROJ={self._map_proj}) -- refusing')
            # WRF's own values, now shown to match the formula.
            self._cosalpha, self._sinalpha = f_cos, f_sin
            self._rotation_source = f'COSALPHA/SINALPHA of {self._static_path.name} (formula agrees to {worst:.1e} rad)'
        elif validated and fallback_to_formula:
            self._cosalpha, self._sinalpha = cosa, sina
            self._rotation_source = 'analytic from TRUELAT1/TRUELAT2/STAND_LON/XLONG'
        else:
            self._cosalpha = self._sinalpha = None
            self._rotation_note = (f'the rotation formula for MAP_PROJ={self._map_proj} '
                                   f'(TRUELAT1={attrs.get("TRUELAT1")}, TRUELAT2={attrs.get("TRUELAT2")}) is not '
                                   f'validated against real WRF output')

    def _rotation_known(self):
        """True when grid->earth rotation is available: COSALPHA/SINALPHA loaded, or a true lat-lon grid."""
        return self._cosalpha is not None or (self._map_proj == 6 and np.isclose(self._pole_lat, 90.0))

    def _rotation_remedy(self):
        if self._rotation_note:
            return self._rotation_note
        return ('the input files carry no COSALPHA/SINALPHA (an auxiliary output stream, or a wrfout pruned '
                'without them); convert from the wrfout history files instead, or pass static_path=<a wrfout or '
                'wrfinput of the same domain>')

    def _init_variables(self):
        """
        The base availability check, then withhold every key whose ACTIVE transform rotates winds when the
        rotation is unknown (since 0.7.0). Before, the rotation fell back to identity on any projection and
        stored grid-relative components as ``eastward_wind``/``northward_wind`` -- measured 13 m/s wrong on a
        real auxiliary-stream file (U10/V10 are written to the aux streams, COSALPHA is not).
        """
        super()._init_variables()
        if self._rotation_known():
            return
        self._variables_before_rotation_check = dict(self.variables)
        dropped = sorted(k for k, info in self.variables.items() if info.get('transform') in _ROTATING_TRANSFORMS)
        if dropped:
            for k in dropped:
                del self.variables[k]
            self._unrotatable = tuple(dropped)
            warnings.warn(f'{self.input_paths[0].name}: {dropped} are unavailable: {self._rotation_remedy()}. '
                          f'Frame-invariant fields (wind speed, vorticity) are still available.')

    def resolve_variables(self, variables):
        """
        As the base, but a name that would have resolved to a withheld (rotation-dependent) key is refused with
        the reason -- rather than reported unknown, or worse resolved by source name to a different field
        (``'U10'`` would otherwise fall through to ``WIND10``, the speed).
        """
        if variables is not None and self._unrotatable:
            before = resolve_variable_keys(self._variables_before_rotation_check, variables)
            blocked = [k for k in before if k in self._unrotatable]
            if blocked:
                raise ValueError(f'{blocked} need the grid->earth wind rotation, which is unknown: '
                                 f'{self._rotation_remedy()}')
        elif variables is None and self._unrotatable:
            # "Everything available" silently lacks the withheld winds otherwise (the init warning is easy to miss
            # by the time convert runs).
            warnings.warn(f'converting every available variable EXCEPT {list(self._unrotatable)}: '
                          f'{self._rotation_remedy()}')
        return super().resolve_variables(variables)

    def _label_shift(self, var_keys):
        """
        The interval to subtract from WRF's frame times to label each accumulation by its START.

        WRF stamps an accumulation over (t - dt, t] at t. PREC_ACC's dt is PREC_ACC_DT (checked equal
        across files) and must equal the frame spacing, so each frame is one output interval; RAIN's
        increments span the frame spacing. Only accumulations can be relabelled: every field of one
        conversion shares one time axis.
        """
        kinds = {k: self.variables[k].get('accumulated') for k in var_keys}
        others = sorted(k for k, a in kinds.items() if not a)
        if others:
            raise ValueError(f"time_label='start' applies to accumulations only; {others} are instantaneous "
                             f'and share the one time axis -- convert them in a separate call/dataset')
        # The frame spacing is the smallest step between frames: a missing file leaves a larger gap,
        # which is not a different spacing (extend mode refuses a window with missing frames itself).
        diffs = np.diff(self.times)
        step = int(diffs.min() / np.timedelta64(1, 'm')) if len(diffs) else None
        shifts = set()
        for key, kind in kinds.items():
            if kind == 'window':
                dt = self._prec_acc_dt
                if dt is None or dt <= 0:
                    raise ValueError(f'{key}: PREC_ACC_DT is missing or not positive; the accumulation window is unknown')
                if step is not None and int(dt) != step:
                    raise ValueError(f'PREC_ACC_DT={dt:g} min but frames are {step} min apart; each PREC_ACC frame '
                                     f'must be one output interval')
                shifts.add(int(dt))
            else:
                if step is None:
                    raise ValueError(f'{key}: one frame only, so its accumulation interval is unknown')
                shifts.add(step)
        if len(shifts) != 1:
            raise ValueError(f'the requested accumulations have different intervals {sorted(shifts)} min')
        return np.timedelta64(shifts.pop(), 'm')

    def _time_run_starts(self):
        """The run start (SIMULATION_START_DATE) of the file supplying each of ``self.times`` (NaT if unknown)."""
        out = np.full(len(self.times), np.datetime64('NaT'), dtype='datetime64[m]')
        for (path, times, g0), run in zip(self._file_time_map, self._file_run_starts):
            if run is None:
                continue
            u = self._raw_to_unique[g0:g0 + len(times)]
            out[u[u >= 0]] = run
        return out

    def _label_frames(self, shift, var_keys):
        """
        Interval-start labels for ``self.times`` and a mask of the frames that can carry one.

        A frame whose window starts before ITS OWN file's run start is a lead-0 frame (WRF's
        zero-initialised accumulator, or an increment with no prior) and is dropped; PREC_ACC frames must
        lie on their own run's ``SIMULATION_START_DATE + k * dt`` grid, else the window WRF reset does
        not match the label. Evaluated per file: one band of a stitched hindcast spans several runs.
        """
        labels = self.times - shift
        run = self._time_run_starts()
        known = ~np.isnat(run)
        keep = ~(known & (labels < run))
        if any(self.variables[k].get('accumulated') == 'window' for k in var_keys):
            dt = int(shift / np.timedelta64(1, 'm'))
            since = ((self.times - run) / np.timedelta64(1, 'm'))
            off = known & keep & (np.mod(np.where(known, since, 0), dt) != 0)
            if off.any():
                bad = self.times[off][:3]
                raise ValueError(f'frames {[str(b) for b in bad]} are not on their run\'s SIMULATION_START_DATE + k*{dt} '
                                 f'min grid; the accumulation window does not match an interval label')
        return labels, keep

    def _default_forecast_reference_time(self, filtered_times):
        """Forecast mode: the run's init from SIMULATION_START_DATE / START_DATE, else the first timestep."""
        if self._simulation_start is not None:
            return self._simulation_start
        return super()._default_forecast_reference_time(filtered_times)

    def _parse_crs(self, h5):
        """
        Extract the CRS from WRF global attributes, on the WPS sphere.

        Supports MAP_PROJ values:
        - 1: Lambert Conformal Conic
        - 2: Polar Stereographic
        - 3: Mercator
        - 6: Lat-Lon (EPSG:4326)

        Projections 1-3 use the WPS projection parameters (TRUELAT1/2, STAND_LON) on a sphere of
        ``WPS_EARTH_RADIUS_M``, the earth model XLAT/XLONG were computed on. The Lambert latitude of
        origin is a convention: WPS anchors its grid at a known corner and has no such parameter, and
        any value gives the same geometry with shifted y. MOAD_CEN_LAT (the outermost domain's centre)
        is used so that every nest of one run shares one CRS. The hemisphere of the polar projection
        follows the sign of TRUELAT1, as WPS ``map_set`` does.
        """
        attrs = h5.attrs
        map_proj = _wrf_attr(attrs, 'MAP_PROJ')

        if map_proj == 1:
            truelat1 = _wrf_attr(attrs, 'TRUELAT1')
            truelat2 = _wrf_attr(attrs, 'TRUELAT2')
            stand_lon = _wrf_attr(attrs, 'STAND_LON')
            if 'MOAD_CEN_LAT' in attrs:
                lat_0 = _wrf_attr(attrs, 'MOAD_CEN_LAT')
            else:
                lat_0 = _wrf_attr(attrs, 'CEN_LAT')
                warnings.warn('MOAD_CEN_LAT absent: using CEN_LAT as the Lambert latitude of origin, so '
                              'nests of this run will not share one CRS')
            return pyproj.CRS.from_cf({
                'grid_mapping_name': 'lambert_conformal_conic',
                'standard_parallel': [truelat1, truelat2],
                'longitude_of_central_meridian': stand_lon,
                'latitude_of_projection_origin': lat_0,
                'false_easting': 0.0,
                'false_northing': 0.0,
                'earth_radius': WPS_EARTH_RADIUS_M,
            })

        elif map_proj == 2:
            truelat1 = _wrf_attr(attrs, 'TRUELAT1')
            stand_lon = _wrf_attr(attrs, 'STAND_LON')
            return pyproj.CRS.from_cf({
                'grid_mapping_name': 'polar_stereographic',
                'straight_vertical_longitude_from_pole': stand_lon,
                # pyproj takes the pole from the sign of standard_parallel (lat_ts) whatever this says;
                # set consistently for readers of the CF attributes.
                'latitude_of_projection_origin': -90.0 if truelat1 < 0 else 90.0,
                'standard_parallel': truelat1,
                'false_easting': 0.0,
                'false_northing': 0.0,
                'earth_radius': WPS_EARTH_RADIUS_M,
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
                'earth_radius': WPS_EARTH_RADIUS_M,
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

        For MAP_PROJ=6 (lat-lon), uses XLAT/XLONG directly. Otherwise the file's own cell centres
        (XLAT/XLONG) are projected through ``self.crs`` and a regular DX/DY lattice is fitted to them,
        then checked against them both ways (lattice residual, and the lattice projected back to
        lat/lon) within ``xy_tolerance_m``; a failure means the projection attributes do not describe
        the grid, and is refused rather than written. Without XLAT/XLONG the lattice is laid around
        CEN_LAT/CEN_LON (unverified, with a warning).
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

        dx = float(_wrf_attr(attrs, 'DX'))
        dy = float(_wrf_attr(attrs, 'DY'))
        to_xy = pyproj.Transformer.from_crs('EPSG:4326', self.crs, always_xy=True)

        if 'XLAT' not in h5 or 'XLONG' not in h5:
            warnings.warn(f'{pathlib.Path(h5.filename).name}: XLAT/XLONG absent, so x/y are laid out '
                          f'around CEN_LAT/CEN_LON and not verified against the cell centres')
            ny = _wrf_attr(attrs, 'SOUTH-NORTH_PATCH_END_UNSTAG') - _wrf_attr(attrs, 'SOUTH-NORTH_PATCH_START_UNSTAG') + 1
            nx = _wrf_attr(attrs, 'WEST-EAST_PATCH_END_UNSTAG') - _wrf_attr(attrs, 'WEST-EAST_PATCH_START_UNSTAG') + 1
            center_x, center_y = to_xy.transform(_wrf_attr(attrs, 'CEN_LON'), _wrf_attr(attrs, 'CEN_LAT'))
            x = center_x + (np.arange(nx) - (nx - 1) / 2.0) * dx
            y = center_y + (np.arange(ny) - (ny - 1) / 2.0) * dy
            return {'x': x, 'y': y}

        xlat_ds, xlong_ds = h5['XLAT'], h5['XLONG']
        if xlat_ds.ndim == 3:
            xlat = xlat_ds[0].astype('float64')
            xlong = xlong_ds[0].astype('float64')
            if xlat_ds.shape[0] > 1 and not (np.allclose(xlat_ds[-1], xlat, atol=1e-4) and
                                             np.allclose(xlong_ds[-1], xlong, atol=1e-4)):
                raise ValueError(f'{pathlib.Path(h5.filename).name}: XLAT/XLONG vary in time (a moving '
                                 f'nest), which cfdb-ingest does not support')
        else:
            xlat = xlat_ds[:].astype('float64')
            xlong = xlong_ds[:].astype('float64')

        ny, nx = xlat.shape
        xp, yp = to_xy.transform(xlong, xlat)
        ii = np.arange(nx)[None, :]
        jj = np.arange(ny)[:, None]
        x0 = float(np.mean(xp - ii * dx))
        y0 = float(np.mean(yp - jj * dy))
        x = x0 + np.arange(nx) * dx
        y = y0 + np.arange(ny) * dy

        residual = max(float(np.max(np.abs(xp - x[None, :]))), float(np.max(np.abs(yp - y[:, None]))))
        lon_back, lat_back = pyproj.Transformer.from_crs(self.crs, 'EPSG:4326', always_xy=True).transform(
            np.broadcast_to(x[None, :], (ny, nx)), np.broadcast_to(y[:, None], (ny, nx)))
        round_trip = float(np.max(pyproj.Geod(a=WPS_EARTH_RADIUS_M, b=WPS_EARTH_RADIUS_M).inv(
            lon_back, lat_back, xlong, xlat)[2]))
        err = max(residual, round_trip)
        if not np.isfinite(err) or err > self.xy_tolerance_m:
            raise ValueError(
                f'WRF grid check failed for {pathlib.Path(h5.filename).name}: a DX={dx:g}/DY={dy:g} lattice '
                f'in the WPS-sphere MAP_PROJ={map_proj} CRS is {err:.1f} m from XLAT/XLONG (lattice residual '
                f'{residual:.1f} m, round trip {round_trip:.1f} m; tolerance {self.xy_tolerance_m:g} m). The '
                f"file's projection attributes do not describe its grid; refusing rather than writing "
                f'misplaced coordinates.'
            )
        self._xy_fit_error_m = err

        return {'x': x, 'y': y}

    def _get_variable_mapping(self):
        """Return the WRF variable mapping dictionary."""
        return WRF_VARIABLE_MAPPING

    def _get_dataset_attrs(self):
        """Return CF + WRF-specific dataset attributes."""
        attrs = super()._get_dataset_attrs()
        attrs['source'] = self._source_title
        attrs.update(self._wrf_params)
        if self._cosalpha is not None and getattr(self, '_rotation_source', None):
            attrs['wind_rotation'] = self._rotation_source
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
        # Soil layers are read as-is; registering them sends SMOIS/TSLB through the cross-file rechunker, so each
        # (time, depth, y, x) chunk is written once -- not once per input file and layer (0.8.0).
        'soil_3d': '_block_soil_3d',
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
        'precipitable_water': '_block_precipitable_water',
        'precipitable_water_tracer': '_block_precipitable_water_tracer',
        'vimf_u': '_block_vimf_u',
        'vimf_v': '_block_vimf_v',
        'sum': '_block_sum',
        'sum_nonneg': '_block_sum_nonneg',
    }

    def _check_var_keys(self, var_keys):
        if 'RAIN' in var_keys and 'PREC_ACC' in var_keys:
            raise ValueError(
                "'RAIN' and 'PREC_ACC' both write cfdb 'precip'; request one of them "
                "(PREC_ACC for inits ingested file by file, RAIN for a whole run)"
            )

    def _check_window(self, var_keys, time_mask):
        """
        Differenced running totals (RAIN, RAIN_TR) are only valid within one run: at a cold-start seam the
        totals restart from zero, the difference goes negative and is clipped to 0, so the first interval
        of every later run would be stored as 0 mm (measured: 99.9 % of wet cells at a real seam). Refused
        when the frames of THIS window come from more than one run (files outside the window do not
        matter). Restart-chained runs keep one SIMULATION_START_DATE and are unaffected.
        """
        increments = [k for k in var_keys if self.variables.get(k, {}).get('transform') == 'accumulation_increment']
        if not increments:
            return
        runs = self._time_run_starts()[time_mask]
        runs = sorted({str(r) for r in runs[~np.isnat(runs)]})
        if len(runs) > 1:
            raise ValueError(
                f'{increments} difference accumulated totals, which restart at every cold start, but the '
                f'requested window spans {len(runs)} runs (SIMULATION_START_DATE {runs[0]} .. {runs[-1]}); the '
                f'first interval of each later run would be stored as 0. Use PREC_ACC (WRF\'s windowed '
                f'accumulator), or convert one run at a time.'
            )

    def _run_start_of_path(self, path):
        """SIMULATION_START_DATE of an input file (None if unknown)."""
        return self._path_run_start.get(str(path))

    def _block_sum(self, sources, y_sl, x_sl, block_cache):
        """Elementwise sum of every source (e.g. PREC_ACC_NC + PREC_ACC_C). Shape (N, ny, nx)."""
        total = None
        for name in sorted(sources):
            arr = np.asarray(sources[name], dtype='float32')
            total = arr.copy() if total is None else total + arr
        return total

    def _block_sum_nonneg(self, sources, y_sl, x_sl, block_cache):
        """``_block_sum`` with negatives set to 0 (NaN kept): precipitation accumulators cannot be negative."""
        total = self._block_sum(sources, y_sl, x_sl, block_cache)
        np.maximum(total, 0.0, out=total, where=~np.isnan(total))
        return total

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

    def _block_soil_3d(self, sources, y_sl, x_sl, block_cache):
        """A soil block (time, layer, y, x) as stored: the same values ``_read_soil_3d`` returns per frame."""
        return next(iter(sources.values())).astype('float32', copy=False)

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


# --------------------------------------------------------------------------------------------------------
# Pressure-level diagnostics (namelist &diags p_lev_diags = 1, output stream auxhist23, 'wrfplevels' files)
# --------------------------------------------------------------------------------------------------------

# WRF interpolates these to the namelist press_levels (Registry/registry.diags, phys/module_diag_pld.F) and
# writes them to their own stream with P_PL, the levels in Pa. U_PL/V_PL are GRID-relative (WRF <= 4.7.1
# averages the staggered winds without rotating), Q_PL is a MIXING RATIO and RH_PL is in percent.
WRF_PLEV_VARIABLE_MAPPING = {
    'GHT_PL': {'cfdb_name': 'geopotential_height', 'source_vars': ['GHT_PL'], 'transform': None, 'height': 'levels'},
    'T_PL': {'cfdb_name': 'air_temp', 'source_vars': ['T_PL'], 'transform': None, 'height': 'levels'},
    'Q_PL': {'cfdb_name': 'mixing_ratio', 'source_vars': ['Q_PL'], 'transform': None, 'nonneg': True,
             'height': 'levels'},
    'U_PL': {'cfdb_name': 'u_wind', 'source_vars': ['U_PL', 'V_PL'], 'transform': 'u_wind_pl', 'height': 'levels'},
    'V_PL': {'cfdb_name': 'v_wind', 'source_vars': ['U_PL', 'V_PL'], 'transform': 'v_wind_pl', 'height': 'levels'},
    # WRF's RH_PL is q/qs*100 without a cap, so it exceeds 100 % in supersaturated cells; clipped to the
    # 0-1 fraction cfdb stores, as ERA5's R is.
    'RH_PL': {'cfdb_name': 'relative_humidity', 'source_vars': ['RH_PL'], 'transform': 'percent_to_fraction',
              'height': 'levels'},
    'TD_PL': {'cfdb_name': 'dew_temp', 'source_vars': ['TD_PL'], 'transform': None, 'height': 'levels'},
    'S_PL': {'cfdb_name': 'wind_speed', 'source_vars': ['S_PL'], 'transform': None, 'height': 'levels'},
}

# The grid attributes a companion wrfout must share to lend its COSALPHA/SINALPHA (the accumulation
# attributes in _WRF_CONSISTENT_ATTRS may legitimately differ between runs of one domain).
_WRF_GRID_ATTRS = ('MAP_PROJ', 'TRUELAT1', 'TRUELAT2', 'STAND_LON', 'MOAD_CEN_LAT', 'CEN_LAT', 'CEN_LON',
                   'POLE_LAT', 'POLE_LON', 'DX', 'DY', 'WEST-EAST_GRID_DIMENSION', 'SOUTH-NORTH_GRID_DIMENSION')

# Largest accepted disagreement (rad) between the analytic rotation and a companion wrfout's
# COSALPHA/SINALPHA. Measured healthy: 2.7e-7 rad (float32 storage of COSALPHA) on a real SH tangent-cone
# d02; a sign error is ~1 rad. ~40x headroom over the measured value.
ROTATION_TOLERANCE_RAD = 1e-5

# Largest accepted XLAT/XLONG difference (deg) between a companion wrfout and the pressure-level files --
# the moving-nest tolerance of _parse_spatial_coords, ~10 m, far below any grid spacing.
_COMPANION_XY_ATOL_DEG = 1e-4


class WrfPlevIngest(WrfIngest):
    """
    Convert WRF pressure-level diagnostics (``wrfplevels`` files, stream auxhist23) to cfdb.

    Variables are stored on a ``pressure`` coordinate taken from the files' own ``P_PL`` -- no
    interpolation. WRF's missing value (namelist ``p_lev_missing``, default -999; written below the lowest
    model level when ``extrap_below_grnd = 1``, and above the model top in every mode) becomes NaN before
    any transform, and the count per variable and level is returned by ``convert`` as ``masked_cells``
    (the files do not record the extrapolation mode, so this is how a caller tells).

    ``U_PL``/``V_PL`` are grid-relative in the file and are rotated to earth-relative. The auxiliary stream
    carries no COSALPHA/SINALPHA, so the rotation is computed from the projection attributes
    (:func:`grid_rotation`). For projections whose formula is not validated against real WRF output
    (secant-cone Lambert, polar stereographic, Mercator) pass ``static_path`` -- a wrfout of the same
    domain -- whose COSALPHA/SINALPHA must agree with the formula; without it the winds are unavailable.

    Parameters
    ----------
    input_paths : str, Path, or list thereof
        ``wrfplevels`` files, or a directory holding them.
    static_path : str or Path, optional
        A wrfout of the same domain, to cross-check (and, for unvalidated projections, enable) the rotation.
    missing_value : float
        WRF's ``p_lev_missing``.
    """

    file_glob_pattern = 'wrfplevels*'
    _supports_grid_extend = False
    _BLOCK_TRANSFORMS = {
        **WrfIngest._BLOCK_TRANSFORMS,
        'u_wind_pl': '_block_u_wind_pl',
        'v_wind_pl': '_block_v_wind_pl',
        'percent_to_fraction': '_block_percent_to_fraction',
    }

    def __init__(self, input_paths, static_path=None, missing_value: float = -999.0):
        self._missing_value = float(missing_value)
        if not np.isfinite(self._missing_value) or self._missing_value == 0.0:
            # 0 is a real value of every field here (and the value of the zero pad rows the rechunk blocks carry),
            # so it cannot mark 'missing'.
            raise ValueError(f'missing_value={missing_value!r}: must be a finite, non-zero sentinel '
                             f'(WRF p_lev_missing)')
        self._masked = {}
        super().__init__(input_paths, static_path=static_path)

    def _get_variable_mapping(self):
        return WRF_PLEV_VARIABLE_MAPPING

    def _init_source_metadata(self):
        """
        The WRF grid/header checks, then the pressure axis (identical in every frame of every file, and the
        level length of every ``*_PL`` field) and the wind rotation. Metadata reads only.
        """
        super()._init_source_metadata()

        levels = None
        for path in self.input_paths:
            with h5py.File(path, 'r') as h5:
                if 'P_PL' not in h5:
                    raise ValueError(f'{path.name}: no P_PL -- not a WRF pressure-level (p_lev_diags) file')
                p = np.asarray(h5['P_PL'][:], dtype='float64')
                frames = p.reshape(-1, p.shape[-1])
                if not (frames == frames[0]).all():
                    raise ValueError(f'{path.name}: P_PL differs between frames')
                if levels is None:
                    levels = frames[0]
                elif not np.array_equal(frames[0], levels):
                    raise ValueError(f'{path.name}: P_PL {frames[0].tolist()} differs from {levels.tolist()} in '
                                     f'{self.input_paths[0].name}; one conversion needs one set of pressure levels')
                for name in h5:
                    if name.endswith('_PL') and name != 'P_PL':
                        shape = h5[name].shape
                        if len(shape) != 4 or shape[1] != len(levels):
                            raise ValueError(
                                f'{path.name}: {name} has shape {shape}, but P_PL has {len(levels)} levels; every '
                                f'*_PL field must be (time, level, y, x) on the P_PL axis (a file split or pruned '
                                f'per level is not supported)')
        if len(np.unique(levels)) != len(levels):
            raise ValueError(f'P_PL has repeated levels: {levels.tolist()}')
        if not (np.diff(levels) < 0).all():
            warnings.warn(f'P_PL {levels.tolist()} is not in descending order: WRF searches the levels in namelist '
                          f'order and leaves later levels missing when press_levels is not descending '
                          f'(module_diag_pld.F); expect whole levels of NaN')
        self._plevels = levels

        self._init_rotation()

    def _rotation_remedy(self):
        note = self._rotation_note or 'no grid->earth rotation is known'
        return f'{note}; pass static_path=<a wrfout of the same domain> to use its COSALPHA/SINALPHA'

    def _native_level_values(self):
        """``P_PL`` (Pa) in file order: WRF writes press_levels in namelist order, normally descending."""
        return self._plevels

    def _setup_populate(self, var_key, target_levels):
        """Native levels only: no vertical interpolation to set up."""
        return None

    def _get_dataset_attrs(self):
        attrs = super()._get_dataset_attrs()
        attrs['p_lev_missing'] = self._missing_value
        if self._cosalpha is not None:
            attrs['wind_rotation'] = self._rotation_source
        return attrs

    def _post_block_transform(self, block, var_key, source_ndim):
        """
        WRF's missing value -> NaN in every ``*_PL`` block, before any transform (so a rotated sentinel cannot
        leave as a plausible wind), counted per output level. Blocks reach here already level-picked into
        output order (base: ``_populate_with_rechunkit`` / ``_populate_with_multi_rechunker_group``), and the
        batch path, which skips this hook, is refused for native-level variables.
        """
        if not var_key.endswith('_PL') or var_key == 'P_PL':
            return block
        miss = block == self._missing_value
        counts = miss.sum(axis=(0, 2, 3)) if block.ndim == 4 else None
        if counts is not None:
            prev = self._masked.get(var_key)
            self._masked[var_key] = counts if prev is None else prev + counts
        if not miss.any():
            return block
        return np.where(miss, np.array(np.nan, dtype=block.dtype), block)

    def _block_rotated_wind_pl(self, sources, y_sl, x_sl, block_cache):
        """Earth-relative U/V from grid-relative U_PL/V_PL, ``(N, nz, ny, nx)``, in float32 (one block of each)."""
        if 'wind_pl' in block_cache:
            return block_cache['wind_pl']
        cosa = self._cosalpha[y_sl, x_sl].astype('float32')
        sina = self._sinalpha[y_sl, x_sl].astype('float32')
        u = sources['U_PL']
        v = sources['V_PL']
        # u_e = u*cos + v*sin, v_e = -u*sin + v*cos, each built in place: one block-sized temporary at a time
        # instead of three (the block is every requested level of chunk_t frames).
        u_e = u * cosa
        u_e += v * sina
        v_e = v * cosa
        v_e -= u * sina
        result = (u_e, v_e)
        block_cache['wind_pl'] = result
        return result

    def _block_u_wind_pl(self, sources, y_sl, x_sl, block_cache):
        return self._block_rotated_wind_pl(sources, y_sl, x_sl, block_cache)[0].astype('float32', copy=False)

    def _block_v_wind_pl(self, sources, y_sl, x_sl, block_cache):
        return self._block_rotated_wind_pl(sources, y_sl, x_sl, block_cache)[1].astype('float32', copy=False)

    def _block_percent_to_fraction(self, sources, y_sl, x_sl, block_cache):
        r = next(iter(sources.values()))
        return np.clip(r / np.float32(100.0), 0.0, 1.0).astype('float32', copy=False)

    def convert(self, cfdb_path, variables=None, start_date=None, end_date=None, bbox=None, target_levels=None,
                vertical_coord='pressure', max_mem=2**27, chunk_shape=None, dataset_type='grid', extend=False,
                time_label='end', squeeze_height=False, clip_nonneg=False, **cfdb_kwargs):
        """
        Convert to a cfdb on a ``pressure`` coordinate (Pa). ``target_levels`` selects a subset of the
        files' levels (default: all of them); interpolation is not offered. Grid mode only: ``extend``,
        ``time_label='start'``, ``squeeze_height`` and forecast mode are refused.

        Returns the base result plus ``masked_cells`` -- ``{source variable: {pressure: cells set to NaN}}``
        -- and warns for a level that is missing everywhere (e.g. press_levels not descending, or a level
        above the model top).
        """
        if vertical_coord != 'pressure':
            raise ValueError(f"pressure-level files have a pressure axis; vertical_coord={vertical_coord!r} is not "
                             f"supported (no interpolation)")
        refused = [name for name, on in (('extend', extend), ("time_label='start'", time_label != 'end'),
                                         ('squeeze_height', squeeze_height),
                                         (f'dataset_type={dataset_type!r}', dataset_type != 'grid')) if on]
        refused += [k for k in ('forecast_reference_time', 'forecast_step_minutes', 'overwrite', 'leads',
                                'mark_complete') if k in cfdb_kwargs]
        if refused:
            raise ValueError(f'{refused} are not implemented for WRF pressure-level files')
        levels = sorted(self._plevels.tolist()) if target_levels is None else [float(p) for p in target_levels]
        self._masked = {}
        result = super().convert(cfdb_path, variables=variables, start_date=start_date, end_date=end_date,
                                 bbox=bbox, target_levels=levels, vertical_coord='pressure', max_mem=max_mem,
                                 chunk_shape=chunk_shape, dataset_type='grid', clip_nonneg=clip_nonneg, **cfdb_kwargs)
        sorted_levels = sorted(levels)
        masked = {sv: {float(p): int(c) for p, c in zip(sorted_levels, counts)} for sv, counts in self._masked.items()}
        result['masked_cells'] = masked

        if bbox is not None:
            _, _, fx, fy = self._bbox_to_indices(bbox)
            n_cells = result['n_times'] * len(fx) * len(fy)
        else:
            n_cells = result['n_times'] * len(self.x) * len(self.y)
        whole = [(sv, p) for sv, per in masked.items() for p, c in per.items() if n_cells and c >= n_cells]
        if whole:
            warnings.warn(f'levels missing everywhere (all {n_cells} cells NaN): {whole}. press_levels not in '
                          f'descending order, or a level above the model top?')
        return result
