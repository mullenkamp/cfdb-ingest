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
def wrf_overlap(wrf_file_1, wrf_file_2):
    """Two files passed as [file1, file2, file1] so all of file1's times are duplicated."""
    from cfdb_ingest.wrf import WrfIngest
    return WrfIngest([wrf_file_1, wrf_file_2, wrf_file_1])


@pytest.fixture
def cfdb_out(tmp_path):
    """Unique output path for each test to avoid collisions in parallel CI."""
    return tmp_path / f'{uuid.uuid4().hex}.cfdb'


# ---------------------------------------------------------------- IFS (GRIB2) fixtures



def _write_cycle(tmp_path_factory, init, **kw):
    pytest.importorskip('eccodes')
    from cfdb_ingest.tests.create_ifs_test_data import write_cycle
    out = tmp_path_factory.mktemp('ifs') / init.replace('-', '').replace('T', '')
    write_cycle(out, init=init, **kw)
    return out


@pytest.fixture(scope='session')
def ifs_cycle_dir(tmp_path_factory):
    """The reference synthetic cycle (init 2026-09-13T00, steps 0/3/6, levels 1000/850/500 hPa).

    Generated per session by create_ifs_test_data.write_cycle rather than committed: the generator
    is deterministic and sub-second, and a committed copy can drift from it.
    """
    return _write_cycle(tmp_path_factory, '2026-09-13T00')


@pytest.fixture(scope='session')
def ifs_cycle_b(tmp_path_factory):
    return _write_cycle(tmp_path_factory, '2026-09-13T12')


@pytest.fixture(scope='session')
def ifs_cycle_c(tmp_path_factory):
    return _write_cycle(tmp_path_factory, '2026-09-14T00')


@pytest.fixture(scope='session')
def ifs_cycle_irregular(tmp_path_factory):
    return _write_cycle(tmp_path_factory, '2026-09-15T00', steps=(0, 3, 6, 12))
