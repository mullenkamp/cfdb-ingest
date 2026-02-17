import pathlib

import pytest


WRF_TEST_DIR = pathlib.Path('/home/mike/data/wrf/tests/nudge_tests/test_d03_no_nudge')

WRF_FILE_1 = WRF_TEST_DIR / 'wrfout_d03_2015-10-26_00:00:00.nc'
WRF_FILE_2 = WRF_TEST_DIR / 'wrfout_d03_2015-10-27_00:00:00.nc'

wrf_files_available = WRF_FILE_1.exists() and WRF_FILE_2.exists()

requires_wrf_files = pytest.mark.skipif(
    not wrf_files_available,
    reason='WRF test files not found',
)


@pytest.fixture
def wrf_file_1():
    return WRF_FILE_1


@pytest.fixture
def wrf_file_2():
    return WRF_FILE_2


@pytest.fixture
def wrf_single(wrf_file_1):
    from cfdb_ingest.wrf import WrfIngest
    return WrfIngest(wrf_file_1)


@pytest.fixture
def wrf_multi(wrf_file_1, wrf_file_2):
    from cfdb_ingest.wrf import WrfIngest
    return WrfIngest([wrf_file_1, wrf_file_2])
