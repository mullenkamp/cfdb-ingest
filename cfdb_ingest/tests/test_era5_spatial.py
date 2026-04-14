#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for ERA5 ingest spatial alignment.

Verifies that Era5Ingest.convert() produces correct lat/lon alignment
when input NetCDF files have different spatial extents (e.g., 2015 files
with 128 lat points vs 2023 files with 145 lat points).
"""
import shutil
from pathlib import Path

import cfdb
import h5py
import numpy as np
import pytest


def _make_era5_nc(path, var_name, times_h, lat, lon, fill_func):
    """
    Create a minimal ERA5-style NetCDF (HDF5) file.

    Parameters
    ----------
    path : Path
        Output file path.
    var_name : str
        Variable name (e.g., 'MSL').
    times_h : np.ndarray
        Time values as hours since 1900-01-01.
    lat : np.ndarray
        Latitude values (descending, matching ERA5 convention).
    lon : np.ndarray
        Longitude values (ascending).
    fill_func : callable
        f(lat_2d, lon_2d) -> data[ny, nx]. Called per timestep.
    """
    nt = len(times_h)
    ny = len(lat)
    nx = len(lon)

    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing='ij')

    with h5py.File(path, 'w') as h5:
        h5.create_dataset('time', data=times_h, dtype='int32')
        h5.create_dataset('latitude', data=lat, dtype='float64')
        h5.create_dataset('longitude', data=lon, dtype='float64')
        ds = h5.create_dataset(var_name, shape=(nt, ny, nx), dtype='float32')
        for t in range(nt):
            ds[t] = fill_func(lat_2d, lon_2d).astype('float32')


def _hours_since_1900(date_str, n_hours):
    """Return array of hours-since-1900-01-01 for n_hours starting at date_str."""
    base = np.datetime64('1900-01-01T00:00', 'h')
    start = np.datetime64(date_str, 'h')
    offset = int((start - base) / np.timedelta64(1, 'h'))
    return np.arange(offset, offset + n_hours, dtype='int32')


def _mslp_pattern(lat_2d, lon_2d):
    """
    Recognizable MSLP pattern: 100000 + lat*100 + lon*10.

    This makes each grid cell uniquely identifiable from its value.
    """
    return 100000.0 + lat_2d * 100.0 + lon_2d * 10.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def work_dir(tmp_path):
    """Provide a temporary working directory."""
    return tmp_path


@pytest.fixture()
def lon():
    """Common longitude grid (ascending)."""
    return np.arange(170.0, 175.25, 0.25)


@pytest.fixture()
def lat_narrow():
    """Narrower latitude grid: -46.75 to -20.0 (descending for ERA5)."""
    return np.arange(-20.0, -47.0, -0.25)  # descending


@pytest.fixture()
def lat_wide():
    """Wider latitude grid: -51.0 to -15.0 (descending for ERA5)."""
    return np.arange(-15.0, -51.25, -0.25)  # descending


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSameGrid:
    """All input files share the same spatial grid — baseline correctness."""

    def test_single_file(self, work_dir, lon, lat_wide):
        """Single file converts with correct lat alignment."""
        nc_path = work_dir / 'msl_2023.nc'
        times = _hours_since_1900('2023-01-01', 3)
        _make_era5_nc(nc_path, 'MSL', times, lat_wide, lon, _mslp_pattern)

        cfdb_path = work_dir / 'out.cfdb'
        from cfdb_ingest import Era5Ingest
        era5 = Era5Ingest([nc_path])
        era5.convert(cfdb_path, variables=['MSL'])

        with cfdb.open_dataset(cfdb_path) as ds:
            lats = ds['latitude'].data
            lons = ds['longitude'].data
            data = ds['mslp'][0, 0, :, :].data.squeeze()

            # Verify several grid cells
            for lat_target in [-45.0, -35.0, -25.0]:
                lat_i = int(np.argmin(np.abs(lats - lat_target)))
                lon_i = int(np.argmin(np.abs(lons - 172.0)))
                expected = _mslp_pattern(
                    np.array([[lats[lat_i]]]), np.array([[lons[lon_i]]])
                )[0, 0]
                assert abs(data[lat_i, lon_i] - expected) < 1.0, (
                    f'At lat={lats[lat_i]:.2f}, lon={lons[lon_i]:.2f}: '
                    f'got {data[lat_i, lon_i]:.0f}, expected {expected:.0f}'
                )

    def test_two_files_same_grid(self, work_dir, lon, lat_wide):
        """Two files with identical grids produce correct alignment."""
        nc_a = work_dir / 'msl_2023a.nc'
        nc_b = work_dir / 'msl_2023b.nc'
        _make_era5_nc(nc_a, 'MSL', _hours_since_1900('2023-01-01', 3),
                      lat_wide, lon, _mslp_pattern)
        _make_era5_nc(nc_b, 'MSL', _hours_since_1900('2023-01-04', 3),
                      lat_wide, lon, _mslp_pattern)

        cfdb_path = work_dir / 'out.cfdb'
        from cfdb_ingest import Era5Ingest
        era5 = Era5Ingest([nc_a, nc_b])
        era5.convert(cfdb_path, variables=['MSL'])

        with cfdb.open_dataset(cfdb_path) as ds:
            lats = ds['latitude'].data
            lons = ds['longitude'].data

            for ti in [0, 3]:  # first timestep from each file
                data = ds['mslp'][ti, 0, :, :].data.squeeze()
                for lat_t in [-45.0, -30.0, -20.0]:
                    lat_i = int(np.argmin(np.abs(lats - lat_t)))
                    lon_i = int(np.argmin(np.abs(lons - 172.0)))
                    expected = _mslp_pattern(
                        np.array([[lats[lat_i]]]), np.array([[lons[lon_i]]])
                    )[0, 0]
                    assert abs(data[lat_i, lon_i] - expected) < 1.0


class TestDifferentGrids:
    """Input files with different spatial extents — the bug scenario."""

    def test_narrow_then_wide_lat(self, work_dir, lon, lat_narrow, lat_wide):
        """
        First file has narrow lat range, second has wider.

        This reproduces the ERA5 bug: 2015 files with 128 lat points
        followed by 2023 files with 145 lat points. The second file's
        data must still land at the correct latitude positions.
        """
        nc_narrow = work_dir / 'msl_2015.nc'
        nc_wide = work_dir / 'msl_2023.nc'

        _make_era5_nc(nc_narrow, 'MSL', _hours_since_1900('2015-01-01', 3),
                      lat_narrow, lon, _mslp_pattern)
        _make_era5_nc(nc_wide, 'MSL', _hours_since_1900('2023-01-01', 3),
                      lat_wide, lon, _mslp_pattern)

        cfdb_path = work_dir / 'out.cfdb'
        from cfdb_ingest import Era5Ingest
        era5 = Era5Ingest([nc_narrow, nc_wide])
        era5.convert(cfdb_path, variables=['MSL'])

        with cfdb.open_dataset(cfdb_path) as ds:
            lats = ds['latitude'].data
            lons = ds['longitude'].data
            times = ds['time'].data

            # --- Check data from the WIDE file (2023) ---
            t_wide = int(np.searchsorted(times, np.datetime64('2023-01-01T00:00')))
            data_wide = ds['mslp'][t_wide, 0, :, :].data.squeeze()

            for lat_t in [-48.0, -40.0, -30.0, -20.0]:
                lat_i = int(np.argmin(np.abs(lats - lat_t)))
                lon_i = int(np.argmin(np.abs(lons - 172.0)))
                expected = _mslp_pattern(
                    np.array([[lats[lat_i]]]), np.array([[lons[lon_i]]])
                )[0, 0]
                assert abs(data_wide[lat_i, lon_i] - expected) < 1.0, (
                    f'Wide file data at lat={lats[lat_i]:.2f}: '
                    f'got {data_wide[lat_i, lon_i]:.0f}, expected {expected:.0f}'
                )

            # --- Check data from the NARROW file (2015) ---
            t_narrow = int(np.searchsorted(times, np.datetime64('2015-01-01T00:00')))
            data_narrow = ds['mslp'][t_narrow, 0, :, :].data.squeeze()

            for lat_t in [-45.0, -35.0, -25.0]:
                lat_i = int(np.argmin(np.abs(lats - lat_t)))
                lon_i = int(np.argmin(np.abs(lons - 172.0)))
                expected = _mslp_pattern(
                    np.array([[lats[lat_i]]]), np.array([[lons[lon_i]]])
                )[0, 0]
                assert abs(data_narrow[lat_i, lon_i] - expected) < 1.0, (
                    f'Narrow file data at lat={lats[lat_i]:.2f}: '
                    f'got {data_narrow[lat_i, lon_i]:.0f}, expected {expected:.0f}'
                )

    def test_wide_then_narrow_lat(self, work_dir, lon, lat_narrow, lat_wide):
        """
        First file has wide lat range, second has narrower.

        Reversed ordering: ensures the coordinate grid accommodates both
        and data from the narrower file is correctly placed.
        """
        nc_wide = work_dir / 'msl_2015.nc'
        nc_narrow = work_dir / 'msl_2023.nc'

        _make_era5_nc(nc_wide, 'MSL', _hours_since_1900('2015-01-01', 3),
                      lat_wide, lon, _mslp_pattern)
        _make_era5_nc(nc_narrow, 'MSL', _hours_since_1900('2023-01-01', 3),
                      lat_narrow, lon, _mslp_pattern)

        cfdb_path = work_dir / 'out.cfdb'
        from cfdb_ingest import Era5Ingest
        era5 = Era5Ingest([nc_wide, nc_narrow])
        era5.convert(cfdb_path, variables=['MSL'])

        with cfdb.open_dataset(cfdb_path) as ds:
            lats = ds['latitude'].data
            lons = ds['longitude'].data
            times = ds['time'].data

            # Check narrow file data at overlapping lats
            t_narrow = int(np.searchsorted(times, np.datetime64('2023-01-01T00:00')))
            data_narrow = ds['mslp'][t_narrow, 0, :, :].data.squeeze()

            for lat_t in [-45.0, -35.0, -25.0]:
                lat_i = int(np.argmin(np.abs(lats - lat_t)))
                lon_i = int(np.argmin(np.abs(lons - 172.0)))
                expected = _mslp_pattern(
                    np.array([[lats[lat_i]]]), np.array([[lons[lon_i]]])
                )[0, 0]
                assert abs(data_narrow[lat_i, lon_i] - expected) < 1.0, (
                    f'Narrow file data at lat={lats[lat_i]:.2f}: '
                    f'got {data_narrow[lat_i, lon_i]:.0f}, expected {expected:.0f}'
                )

    def test_offset_grids(self, work_dir, lon):
        """
        Two files with same step but offset start points.

        File A: lat -50.0 to -20.0 (descending)
        File B: lat -50.25 to -20.25 (descending, offset by 0.25)
        """
        lat_a = np.arange(-20.0, -50.25, -0.25)
        lat_b = np.arange(-20.25, -50.50, -0.25)

        nc_a = work_dir / 'msl_a.nc'
        nc_b = work_dir / 'msl_b.nc'

        _make_era5_nc(nc_a, 'MSL', _hours_since_1900('2015-01-01', 3),
                      lat_a, lon, _mslp_pattern)
        _make_era5_nc(nc_b, 'MSL', _hours_since_1900('2023-01-01', 3),
                      lat_b, lon, _mslp_pattern)

        cfdb_path = work_dir / 'out.cfdb'
        from cfdb_ingest import Era5Ingest
        era5 = Era5Ingest([nc_a, nc_b])
        era5.convert(cfdb_path, variables=['MSL'])

        with cfdb.open_dataset(cfdb_path) as ds:
            lats = ds['latitude'].data
            lons = ds['longitude'].data
            times = ds['time'].data

            # Data from file B should be at file B's lat positions
            t_b = int(np.searchsorted(times, np.datetime64('2023-01-01T00:00')))
            data_b = ds['mslp'][t_b, 0, :, :].data.squeeze()

            for lat_t in [-40.25, -30.25, -25.25]:
                lat_i = int(np.argmin(np.abs(lats - lat_t)))
                if abs(lats[lat_i] - lat_t) > 0.13:
                    continue  # skip if not in the grid
                lon_i = int(np.argmin(np.abs(lons - 172.0)))
                expected = _mslp_pattern(
                    np.array([[lats[lat_i]]]), np.array([[lons[lon_i]]])
                )[0, 0]
                assert abs(data_b[lat_i, lon_i] - expected) < 1.0, (
                    f'Offset grid data at lat={lats[lat_i]:.2f}: '
                    f'got {data_b[lat_i, lon_i]:.0f}, expected {expected:.0f}'
                )

    def test_bbox_with_different_grids(self, work_dir, lon, lat_narrow, lat_wide):
        """
        Convert with a bbox that clips both files differently.
        Verifies the bbox + heterogeneous grid interaction.
        """
        nc_narrow = work_dir / 'msl_2015.nc'
        nc_wide = work_dir / 'msl_2023.nc'

        _make_era5_nc(nc_narrow, 'MSL', _hours_since_1900('2015-01-01', 3),
                      lat_narrow, lon, _mslp_pattern)
        _make_era5_nc(nc_wide, 'MSL', _hours_since_1900('2023-01-01', 3),
                      lat_wide, lon, _mslp_pattern)

        cfdb_path = work_dir / 'out.cfdb'
        from cfdb_ingest import Era5Ingest
        era5 = Era5Ingest([nc_narrow, nc_wide])

        # Bbox that clips within both files' range
        bbox = (170.0, -45.0, 175.0, -22.0)
        era5.convert(cfdb_path, variables=['MSL'], bbox=bbox)

        with cfdb.open_dataset(cfdb_path) as ds:
            lats = ds['latitude'].data
            lons = ds['longitude'].data
            times = ds['time'].data

            assert lats[0] >= -45.0
            assert lats[-1] <= -22.0

            # Check wide file data within bbox
            t_wide = int(np.searchsorted(times, np.datetime64('2023-01-01T00:00')))
            data = ds['mslp'][t_wide, 0, :, :].data.squeeze()

            for lat_t in [-40.0, -35.0, -25.0]:
                lat_i = int(np.argmin(np.abs(lats - lat_t)))
                lon_i = int(np.argmin(np.abs(lons - 172.0)))
                expected = _mslp_pattern(
                    np.array([[lats[lat_i]]]), np.array([[lons[lon_i]]])
                )[0, 0]
                assert abs(data[lat_i, lon_i] - expected) < 1.0


class TestCoordinateCoverage:
    """Verify the output coordinate grid covers the union of all inputs."""

    def test_union_lat_coverage(self, work_dir, lon, lat_narrow, lat_wide):
        """Output lat range should be the union of all input lat ranges."""
        nc_narrow = work_dir / 'msl_2015.nc'
        nc_wide = work_dir / 'msl_2023.nc'

        _make_era5_nc(nc_narrow, 'MSL', _hours_since_1900('2015-01-01', 3),
                      lat_narrow, lon, _mslp_pattern)
        _make_era5_nc(nc_wide, 'MSL', _hours_since_1900('2023-01-01', 3),
                      lat_wide, lon, _mslp_pattern)

        cfdb_path = work_dir / 'out.cfdb'
        from cfdb_ingest import Era5Ingest
        era5 = Era5Ingest([nc_narrow, nc_wide])
        era5.convert(cfdb_path, variables=['MSL'])

        lat_narrow_asc = np.sort(lat_narrow)
        lat_wide_asc = np.sort(lat_wide)
        expected_union = np.union1d(lat_narrow_asc, lat_wide_asc)

        with cfdb.open_dataset(cfdb_path) as ds:
            lats = ds['latitude'].data
            # The output should cover at least the union range
            assert lats[0] <= expected_union[0] + 0.01
            assert lats[-1] >= expected_union[-1] - 0.01

    def test_narrow_file_nan_in_extended_region(self, work_dir, lon, lat_narrow, lat_wide):
        """
        Positions outside the narrow file's coverage should be NaN for
        that file's timesteps.
        """
        nc_narrow = work_dir / 'msl_2015.nc'
        nc_wide = work_dir / 'msl_2023.nc'

        _make_era5_nc(nc_narrow, 'MSL', _hours_since_1900('2015-01-01', 3),
                      lat_narrow, lon, _mslp_pattern)
        _make_era5_nc(nc_wide, 'MSL', _hours_since_1900('2023-01-01', 3),
                      lat_wide, lon, _mslp_pattern)

        cfdb_path = work_dir / 'out.cfdb'
        from cfdb_ingest import Era5Ingest
        era5 = Era5Ingest([nc_narrow, nc_wide])
        era5.convert(cfdb_path, variables=['MSL'])

        lat_narrow_min = float(np.min(lat_narrow))  # most negative
        lat_narrow_max = float(np.max(lat_narrow))  # least negative

        with cfdb.open_dataset(cfdb_path) as ds:
            lats = ds['latitude'].data
            ds_lons = ds['longitude'].data
            times = ds['time'].data

            t_narrow = int(np.searchsorted(times, np.datetime64('2015-01-01T00:00')))
            data = ds['mslp'][t_narrow, 0, :, :].data.squeeze()

            # Check a position outside narrow file's range
            for lat_t in [lat_narrow_min - 2.0, lat_narrow_max + 2.0]:
                lat_i = int(np.argmin(np.abs(lats - lat_t)))
                if abs(lats[lat_i] - lat_t) < 0.5:
                    # This position exists in the grid but not in the narrow file.
                    # Should be fill value (NaN for float, or large int for encoded types)
                    val = data[lat_i, 0]
                    expected_val = _mslp_pattern(
                        np.array([[lats[lat_i]]]), np.array([[ds_lons[0]]])
                    )[0, 0]
                    is_fill = np.isnan(val) or abs(val - expected_val) > 1e6
                    assert is_fill, (
                        f'Expected fill value at lat={lats[lat_i]:.2f} for narrow file timestep, '
                        f'got {val:.0f} (expected pattern would be {expected_val:.0f})'
                    )
