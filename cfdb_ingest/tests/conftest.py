import pathlib
import uuid

import pytest


WRF_TEST_DIR = pathlib.Path(__file__).parent / 'data'

WRF_FILE_1 = WRF_TEST_DIR / 'wrfout_d01_2023-02-12_00:00:00.nc'
WRF_FILE_2 = WRF_TEST_DIR / 'wrfout_d01_2023-02-13_00:00:00.nc'


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


@pytest.fixture
def cfdb_out(tmp_path):
    """Unique output path for each test to avoid collisions in parallel CI."""
    return tmp_path / f'{uuid.uuid4().hex}.cfdb'
