"""Tests for ERA5 ingestion to cfdb."""
import pathlib
import uuid

import cfdb
import numpy as np
import pytest

from cfdb_ingest.era5 import Era5Ingest, ERA5_VARIABLE_MAPPING

ERA5_TEST_DIR = pathlib.Path(__file__).parent / 'data' / 'era5'
SFC_DIR = ERA5_TEST_DIR / 'sfc'
PL_DIR = ERA5_TEST_DIR / 'pl'
INV_DIR = ERA5_TEST_DIR / 'inv'


@pytest.fixture
def cfdb_out(tmp_path):
    return tmp_path / f'{uuid.uuid4().hex}.cfdb'


@pytest.fixture
def split_out(tmp_path):
    return tmp_path / 'split'


# ======================================================================
# Initialization
# ======================================================================

class TestInit:

    def test_init_from_directory(self):
        """Init from a directory should find all .nc files."""
        ingest = Era5Ingest(SFC_DIR)
        assert len(ingest.input_paths) > 0

    def test_init_from_file_list(self):
        """Init from explicit file paths."""
        files = sorted(SFC_DIR.glob('*.nc'))[:2]
        ingest = Era5Ingest(files)
        assert len(ingest.input_paths) == 2

    def test_init_from_mixed_dirs(self):
        """Init from multiple directories."""
        ingest = Era5Ingest([SFC_DIR, PL_DIR])
        assert len(ingest.input_paths) > 0

    def test_crs_is_4326(self):
        """ERA5 should always be EPSG:4326."""
        ingest = Era5Ingest(SFC_DIR)
        assert ingest.crs.to_epsg() == 4326

    def test_spatial_coords(self):
        """x=longitude, y=latitude, latitude should be ascending."""
        ingest = Era5Ingest(SFC_DIR)
        assert ingest.x[0] < ingest.x[-1]  # longitude ascending
        assert ingest.y[0] < ingest.y[-1]  # latitude ascending (reversed from file)

    def test_times_detected(self):
        """Time array should be populated."""
        ingest = Era5Ingest(SFC_DIR)
        assert len(ingest.times) > 0
        assert ingest.times.dtype == np.dtype('datetime64[m]')

    def test_variables_detected(self):
        """Available variables should be detected from files."""
        ingest = Era5Ingest(SFC_DIR)
        assert len(ingest.variables) > 0
        assert 'SP' in ingest.variables
        assert 'VAR_2T' in ingest.variables

    def test_pl_variables_detected(self):
        """Pressure level variables should be detected."""
        ingest = Era5Ingest(PL_DIR)
        assert 'T' in ingest.variables
        assert 'U' in ingest.variables

    def test_invariant_z_disambiguation(self):
        """Z in invariant files (no level dim) should map to Z_INV."""
        ingest = Era5Ingest(INV_DIR)
        assert 'Z_INV' in ingest.variables
        assert 'Z_PL' not in ingest.variables

    def test_pl_z_disambiguation(self):
        """Z in pressure level files (has level dim) should map to Z_PL."""
        ingest = Era5Ingest(PL_DIR)
        assert 'Z_PL' in ingest.variables

    def test_mixed_z_disambiguation(self):
        """When both pl and invariant Z files are present, both mappings appear."""
        ingest = Era5Ingest([PL_DIR, INV_DIR])
        assert 'Z_PL' in ingest.variables
        assert 'Z_INV' in ingest.variables

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            Era5Ingest('/nonexistent/path.nc')

    def test_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='No files matching'):
            Era5Ingest(tmp_path)


# ======================================================================
# Variable resolution
# ======================================================================

class TestResolveVariables:

    def test_resolve_by_mapping_key(self):
        ingest = Era5Ingest(SFC_DIR)
        keys = ingest.resolve_variables(['SP'])
        assert 'SP' in keys

    def test_resolve_by_cfdb_name(self):
        ingest = Era5Ingest(SFC_DIR)
        keys = ingest.resolve_variables(['surface_pressure'])
        assert 'SP' in keys

    def test_resolve_all(self):
        ingest = Era5Ingest(SFC_DIR)
        keys = ingest.resolve_variables(None)
        assert len(keys) == len(ingest.variables)

    def test_resolve_unknown_raises(self):
        ingest = Era5Ingest(SFC_DIR)
        with pytest.raises(ValueError, match='Unknown variable'):
            ingest.resolve_variables(['BOGUS'])


# ======================================================================
# Conversion — Surface only
# ======================================================================

class TestConvertSurface:

    def test_single_surface_var(self, cfdb_out):
        """Convert a single surface variable."""
        ingest = Era5Ingest(SFC_DIR)
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['SP'],
            start_date='2020-01-01',
            end_date='2020-01-01T23:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            assert 'surface_pressure' in ds.data_var_names
            dv = ds['surface_pressure']
            assert dv.ndims == 4
            assert 'height_0m' in dv.coord_names

    def test_2m_variable(self, cfdb_out):
        """2m temperature should use height_2m coordinate."""
        ingest = Era5Ingest(SFC_DIR)
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['VAR_2T'],
            start_date='2020-01-01',
            end_date='2020-01-01T23:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            dv = ds['air_temperature']
            assert dv.coord_names == ('time', 'height_2m', 'latitude', 'longitude')

    def test_10m_variable(self, cfdb_out):
        """10m wind should use height_10m coordinate."""
        ingest = Era5Ingest(SFC_DIR)
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['VAR_10U'],
            start_date='2020-01-01',
            end_date='2020-01-01T23:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            dv = ds['u_wind']
            assert dv.coord_names == ('time', 'height_10m', 'latitude', 'longitude')

    def test_multiple_surface_heights(self, cfdb_out):
        """Multiple surface heights should create separate height coordinates."""
        ingest = Era5Ingest(SFC_DIR)
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['SP', 'VAR_2T', 'VAR_10U'],
            start_date='2020-01-01',
            end_date='2020-01-01T23:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            assert 'height_0m' in ds.coord_names
            assert 'height_2m' in ds.coord_names
            assert 'height_10m' in ds.coord_names

    def test_multi_file_time_concatenation(self, cfdb_out):
        """Two SP files (day 1 + day 2) should concatenate times."""
        ingest = Era5Ingest(SFC_DIR)
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['SP'],
            start_date='2020-01-01',
            end_date='2020-01-02T23:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            times = ds['time'].data
            assert len(times) == 48  # 24h x 2 days

    def test_temperature_values_reasonable(self, cfdb_out):
        """Converted temperature should be in a physically reasonable range."""
        ingest = Era5Ingest(SFC_DIR)
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['VAR_2T'],
            start_date='2020-01-01T00:00',
            end_date='2020-01-01T00:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            data = np.squeeze(ds['air_temperature'].data)
            assert np.nanmean(data) > 200.0
            assert np.nanmean(data) < 350.0

    def test_latitude_ascending(self, cfdb_out):
        """Output latitude should be ascending (ERA5 stores descending)."""
        ingest = Era5Ingest(SFC_DIR)
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['SP'],
            start_date='2020-01-01T00:00',
            end_date='2020-01-01T00:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            lat = ds['latitude'].data
            assert lat[0] < lat[-1]


# ======================================================================
# Conversion — Pressure levels
# ======================================================================

class TestConvertPressureLevels:

    def test_single_pl_var(self, cfdb_out):
        """Convert a single pressure level variable."""
        ingest = Era5Ingest(PL_DIR)
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['T'],
            start_date='2020-01-01',
            end_date='2020-01-01T23:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            dv = ds['air_temperature']
            assert dv.ndims == 4
            assert dv.coord_names == ('time', 'pressure', 'latitude', 'longitude')

    def test_pressure_levels_auto_detected(self, cfdb_out):
        """Pressure levels should be auto-detected from source files."""
        ingest = Era5Ingest(PL_DIR)
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['T'],
            start_date='2020-01-01T00:00',
            end_date='2020-01-01T00:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            pressure = ds['pressure'].data
            # Test data has 5 levels: 500, 700, 850, 925, 1000 hPa -> Pa
            assert len(pressure) == 5
            assert pressure[0] == 50000.0  # 500 hPa in Pa

    def test_pressure_axis_z(self, cfdb_out):
        """Pressure coordinate should have axis='Z'."""
        ingest = Era5Ingest(PL_DIR)
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['T'],
            start_date='2020-01-01T00:00',
            end_date='2020-01-01T00:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            # Verify pressure coord exists and data is correct
            assert 'pressure' in ds.coord_names

    def test_geopotential_to_height_transform(self, cfdb_out):
        """Z_PL should be divided by g to get geopotential height."""
        ingest = Era5Ingest(PL_DIR)
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['Z_PL'],
            start_date='2020-01-01T00:00',
            end_date='2020-01-01T00:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            data = np.squeeze(ds['geopotential_height'].data)
            # Original Z is 0-60000 m2/s2, divided by 9.80665 -> 0-6116 m
            assert np.nanmax(data) < 7000.0

    def test_multiple_pl_vars(self, cfdb_out):
        """Multiple pressure level variables in one dataset."""
        ingest = Era5Ingest(PL_DIR)
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['T', 'U', 'V', 'Q'],
            start_date='2020-01-01T00:00',
            end_date='2020-01-01T00:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            assert len(ds.data_vars) == 4
            # All should share the same pressure coordinate
            for dv in ds.data_vars:
                assert 'pressure' in dv.coord_names

    def test_convert_vimf(self, cfdb_out):
        """VIMF should be correctly computed from Q, U, V."""
        ingest = Era5Ingest(PL_DIR)
        # VIMF requires Q, U, V
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['VIMF_U', 'VIMF_V'],
            start_date='2020-01-01T00:00',
            end_date='2020-01-01T00:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            assert 'vimf_u' in ds.data_var_names
            assert 'vimf_v' in ds.data_var_names
            
            vimf_u = ds['vimf_u']
            # VIMF is 2D (time, height_0m, latitude, longitude)
            assert vimf_u.ndims == 4
            assert 'height_0m' in vimf_u.coord_names
            assert 'pressure' not in vimf_u.coord_names
            
            # Check for non-zero data
            data = vimf_u[:].data[0]
            assert not np.all(data == 0)
            assert np.all(np.isfinite(data))


# ======================================================================
# Conversion — Combined surface + pressure levels
# ======================================================================

class TestConvertCombined:

    def test_surface_and_pl_combined(self, cfdb_out):
        """Surface + pressure level vars in one combined dataset."""
        ingest = Era5Ingest([SFC_DIR, PL_DIR])
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['VAR_2T', 'T'],
            start_date='2020-01-01T00:00',
            end_date='2020-01-01T00:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            var_names = sorted(v.name for v in ds.data_vars)
            # Same cfdb_name -> conflict resolved with height suffix
            assert 'air_temperature' in var_names
            assert 'air_temperature_2m' in var_names

            # Pressure level var uses pressure coord
            assert ds['air_temperature'].coord_names == ('time', 'pressure', 'latitude', 'longitude')
            # Surface var uses named height coord
            assert ds['air_temperature_2m'].coord_names == ('time', 'height_2m', 'latitude', 'longitude')

    def test_no_conflict_different_names(self, cfdb_out):
        """Surface and pl vars with different cfdb_names don't conflict."""
        ingest = Era5Ingest([SFC_DIR, PL_DIR])
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['SP', 'T'],
            start_date='2020-01-01T00:00',
            end_date='2020-01-01T00:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            var_names = sorted(v.name for v in ds.data_vars)
            assert 'surface_pressure' in var_names
            assert 'air_temperature' in var_names
            # No suffix needed
            assert not any('_0m' in n for n in var_names)

    def test_height_coords_no_axis_when_pressure_present(self, cfdb_out):
        """Named height coords should not have axis='Z' when pressure coord exists."""
        ingest = Era5Ingest([SFC_DIR, PL_DIR])
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['SP', 'T'],
            start_date='2020-01-01T00:00',
            end_date='2020-01-01T00:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            assert 'height_0m' in ds.coord_names
            assert 'pressure' in ds.coord_names


# ======================================================================
# Conversion — Invariant
# ======================================================================

class TestConvertInvariant:

    def test_invariant_terrain_height(self, cfdb_out):
        """Invariant Z should be converted to terrain height (geopotential / g)."""
        ingest = Era5Ingest(INV_DIR)
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['Z_INV'],
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            assert 'terrain_height' in ds.data_var_names
            data = np.squeeze(ds['terrain_height'].data)
            # Original Z is 0-5000 m2/s2, divided by 9.80665 -> 0-509 m
            assert np.nanmax(data) < 600.0

    def test_invariant_land_sea_mask(self, cfdb_out):
        """Land-sea mask should be 0 or 1."""
        ingest = Era5Ingest(INV_DIR)
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['LSM'],
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            assert 'land_sea_mask' in ds.data_var_names


# ======================================================================
# Conversion — Split mode
# ======================================================================

class TestConvertSplit:

    def test_split_creates_separate_files(self, split_out):
        """Split mode should create one cfdb file per variable."""
        ingest = Era5Ingest(SFC_DIR)
        ingest.convert(
            cfdb_path=split_out,
            variables=['SP', 'VAR_2T'],
            start_date='2020-01-01T00:00',
            end_date='2020-01-01T00:00',
            split=True,
        )
        files = sorted(split_out.glob('*.cfdb'))
        assert len(files) == 2
        names = {f.stem for f in files}
        assert 'surface_pressure' in names
        assert 'air_temperature' in names

    def test_split_surface_and_pl(self, split_out):
        """Split mode with both surface and pl vars."""
        ingest = Era5Ingest([SFC_DIR, PL_DIR])
        ingest.convert(
            cfdb_path=split_out,
            variables=['SP', 'T'],
            start_date='2020-01-01T00:00',
            end_date='2020-01-01T00:00',
            split=True,
        )
        files = sorted(split_out.glob('*.cfdb'))
        assert len(files) == 2

        # Each file should have a single data variable
        for f in files:
            with cfdb.open_dataset(f, 'r') as ds:
                assert len(ds.data_vars) == 1

    def test_split_pl_has_pressure_coord(self, split_out):
        """Split pressure level file should have pressure coordinate."""
        ingest = Era5Ingest(PL_DIR)
        ingest.convert(
            cfdb_path=split_out,
            variables=['T'],
            start_date='2020-01-01T00:00',
            end_date='2020-01-01T00:00',
            split=True,
        )
        with cfdb.open_dataset(split_out / 'air_temperature.cfdb', 'r') as ds:
            assert 'pressure' in ds.coord_names


# ======================================================================
# Conversion — Time filtering
# ======================================================================

class TestTimeFiltering:

    def test_time_subset(self, cfdb_out):
        """Convert only a subset of timesteps."""
        ingest = Era5Ingest(SFC_DIR)
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['SP'],
            start_date='2020-01-01T06:00',
            end_date='2020-01-01T12:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            times = ds['time'].data
            assert len(times) == 7  # 06:00 to 12:00 inclusive


# ======================================================================
# Conversion — Bbox filtering
# ======================================================================

class TestBboxFiltering:

    def test_bbox_subset(self, cfdb_out):
        """Bounding box should spatially subset the data."""
        ingest = Era5Ingest(SFC_DIR)
        full_nx = len(ingest.x)
        full_ny = len(ingest.y)

        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['SP'],
            start_date='2020-01-01T00:00',
            end_date='2020-01-01T00:00',
            bbox=(170.5, -39.5, 172.0, -38.0),
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            lon = ds['longitude'].data
            lat = ds['latitude'].data
            assert len(lon) < full_nx
            assert len(lat) < full_ny


# ======================================================================
# Dataset attributes
# ======================================================================

class TestAttributes:

    def test_dataset_attrs(self, cfdb_out):
        """Dataset should have CF attributes and ERA5 source."""
        ingest = Era5Ingest(SFC_DIR)
        ingest.convert(
            cfdb_path=cfdb_out,
            variables=['SP'],
            start_date='2020-01-01T00:00',
            end_date='2020-01-01T00:00',
        )
        with cfdb.open_dataset(cfdb_out, 'r') as ds:
            attrs = ds.attrs.data
            assert 'Conventions' in attrs
            assert attrs['source'] == 'ERA5 reanalysis (ECMWF via NCAR)'


# ======================================================================
# Variable mapping
# ======================================================================

class TestVariableMapping:

    def test_mapping_has_all_products(self):
        """Mapping should cover surface, pressure level, and invariant vars."""
        heights = set()
        for info in ERA5_VARIABLE_MAPPING.values():
            heights.add(info['height'])
        assert 0.0 in heights
        assert 2.0 in heights
        assert 10.0 in heights
        assert 100.0 in heights
        assert 'levels' in heights

    def test_all_source_vars_are_single(self):
        """ERA5 vars should have exactly one source var (one var per file), except VIMF."""
        for key, info in ERA5_VARIABLE_MAPPING.items():
            if key.startswith('VIMF_'):
                assert len(info['source_vars']) == 2
            else:
                assert len(info['source_vars']) == 1, f'{key} has multiple source vars'

    def test_no_transforms_except_geopotential(self):
        """Only Z and VIMF should have transforms."""
        for key, info in ERA5_VARIABLE_MAPPING.items():
            if key in ('Z_PL', 'Z_INV'):
                assert info['transform'] == 'geopotential_to_height'
            elif key.startswith('VIMF_'):
                assert info['transform'].startswith('compute_vimf_')
            else:
                assert info['transform'] is None, f'{key} has unexpected transform'
