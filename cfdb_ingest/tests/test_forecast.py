"""Tests for cfdb_ingest.forecast -- the shared grid_forecast axis rules and the ForecastWriter."""

import cfdb
import numpy as np
import pytest
from cfdb import dtypes

from cfdb_ingest import forecast as fc

LON = np.arange(150.0, 152.0, 0.5)  # 4
LAT = np.arange(-45.0, -43.5, 0.5)  # 3
LEADS = np.array([0, 3, 6], dtype='int32')
INIT_A = np.datetime64('2026-09-13T00', 'm')
INIT_B = np.datetime64('2026-09-13T12', 'm')
INIT_C = np.datetime64('2026-09-14T00', 'm')
STEP = 360


def _make(path, init=INIT_A, leads=LEADS, chunk=None):
    ds = cfdb.open_dataset(str(path), 'n', dataset_type='grid_forecast')
    fc.create_forecast_coords(ds, init, leads, step_minutes=STEP)
    ds.create.coord.generic('longitude', data=LON, axis='x', step=True)
    ds.create.coord.generic('latitude', data=LAT, axis='y', step=True)
    ds.create.coord.generic('height_2m', data=np.array([2.0]), axis=None)
    ds.create.data_var.generic(
        'air_temperature_2m',
        (fc.FRT, fc.LEAD, 'height_2m', 'latitude', 'longitude'),
        dtype=dtypes.dtype('float32'),
        chunk_shape=chunk or (1, len(leads), 1, len(LAT), len(LON)),
    )
    return ds


# ---------------------------------------------------------------- chunk shape


def test_chunk_shape_full_extent_when_small():
    assert fc.forecast_chunk_shape(49, 160, 200) == (1, 49, 1, 160, 200)


def test_chunk_shape_tiles_when_large():
    shape = fc.forecast_chunk_shape(145, 537, 317)
    _, n_lead, _, ty, tx = shape
    assert n_lead == 145 and ty < 537 and tx < 317
    assert 145 * ty * tx * 4 <= 8 * 2**20
    assert 145 * ty * tx * 4 >= 2 * 2**20  # not shredded into slivers
    assert -(-537 // ty) * ty >= 537 and -(-317 // tx) * tx >= 317


# ---------------------------------------------------------------- axis rules


def test_lead_step():
    assert fc.lead_step([0, 3, 6]) == 3
    assert fc.lead_step([0]) is None
    with pytest.raises(ValueError, match='not regularly spaced'):
        fc.lead_step([0, 3, 6, 12])


def test_check_init_on_grid():
    fc.check_init_on_grid(INIT_B, INIT_A, STEP)
    with pytest.raises(ValueError, match='not on the declared grid'):
        fc.check_init_on_grid(np.datetime64('2026-09-13T01', 'm'), INIT_A, STEP)


def test_create_coords_sets_step_and_units(tmp_path):
    with _make(tmp_path / 'a.cfdb') as ds:
        assert ds[fc.FRT].step == STEP
        assert ds[fc.LEAD].attrs['units'] == 'h'
        assert ds[fc.LEAD].step == 3
        assert ds.dataset_type == 'grid_forecast'


def test_lead_index_map(tmp_path):
    with _make(tmp_path / 'a.cfdb') as ds:
        np.testing.assert_array_equal(fc.lead_index_map(ds, [3, 6]), [1, 2])
        np.testing.assert_array_equal(fc.lead_index_map(ds, [0, 3, 6]), [0, 1, 2])
        with pytest.raises(ValueError, match=r'leads \[9\] are not on the stored'):
            fc.lead_index_map(ds, [6, 9])
        with pytest.raises(ValueError, match='not on the stored'):
            fc.lead_index_map(ds, [1])


def test_valid_times_units(tmp_path):
    with _make(tmp_path / 'a.cfdb') as ds:
        np.testing.assert_array_equal(
            fc.valid_times(ds, 0),
            np.array(['2026-09-13T00:00', '2026-09-13T03:00', '2026-09-13T06:00'], dtype='datetime64[m]'),
        )
        ds[fc.LEAD].attrs['units'] = 'min'
        assert fc.valid_times(ds, 0)[1] == np.datetime64('2026-09-13T00:03', 'm')
        del ds[fc.LEAD].attrs['units']
        with pytest.raises(ValueError, match='units'):
            fc.valid_times(ds, 0)


def test_place_init_lifecycle(tmp_path):
    with _make(tmp_path / 'a.cfdb') as ds:
        fc.mark_init_complete(ds, INIT_A)
        # a gap of 24 h at a 6 h step auto-fills three slots
        placed = fc.place_init(ds, INIT_C, step_minutes=STEP)
        assert placed == {'index': 4, 'status': 'new', 'autofilled': 3}
        assert len(ds[fc.FRT].data) == 5
        # back-fill the auto-filled 12z slot
        placed = fc.place_init(ds, INIT_B, step_minutes=STEP)
        assert placed == {'index': 2, 'status': 'backfill', 'autofilled': 0}
        # a complete init is immutable ...
        with pytest.raises(ValueError, match='immutable'):
            fc.place_init(ds, INIT_A, step_minutes=STEP)
        # ... unless overwrite is explicit
        assert fc.place_init(ds, INIT_A, step_minutes=STEP, overwrite=True)['status'] == 'overwrite'
        # an incomplete (crashed) init is a backfill, not a refusal
        assert fc.place_init(ds, INIT_C, step_minutes=STEP)['status'] == 'backfill'
        with pytest.raises(ValueError, match='predates'):
            fc.place_init(ds, np.datetime64('2026-09-12T12', 'm'), step_minutes=STEP)
        with pytest.raises(ValueError, match='not on the declared grid'):
            fc.place_init(ds, np.datetime64('2026-09-14T07', 'm'), step_minutes=STEP)
        assert fc.complete_inits(ds) == ['2026-09-13T00:00']
        fc.unmark_init_complete(ds, INIT_A)
        assert fc.complete_inits(ds) == []


def test_place_init_without_marker_probes_chunks(tmp_path):
    with _make(tmp_path / 'a.cfdb') as ds:
        assert fc.COMPLETE_INITS_ATTR not in ds.attrs.data
        # slot exists but holds no chunk -> backfill
        assert fc.place_init(ds, INIT_A, step_minutes=STEP)['status'] == 'backfill'
        ds['air_temperature_2m'][0, :, 0, :, :] = np.ones((3, 3, 4), dtype='float32')
        with pytest.raises(ValueError, match='immutable'):
            fc.place_init(ds, INIT_A, step_minutes=STEP)


# ---------------------------------------------------------------- writer


def test_writer_writes_each_chunk_row_once(tmp_path):
    with _make(tmp_path / 'a.cfdb') as ds:
        dv = ds['air_temperature_2m']
        w = fc.ForecastWriter(ds, init_idx=0, lead_index=fc.lead_index_map(ds, LEADS), ny=3, nx=4)
        blk = np.arange(3 * 3 * 4, dtype='float32').reshape(3, 3, 4)
        w.put(dv, 0, slice(0, 2), blk[:2])  # first two leads
        assert w.writes == 0  # nothing written until the row is complete
        w.put(dv, 0, 2, blk[2])  # last lead, as a single-index write
        assert w.writes == 1
        w.close()
        assert w.writes == 1
        np.testing.assert_array_equal(np.squeeze(np.asarray(dv[0, :, 0, :, :].data), axis=(0, 2)), blk)
        assert ds.prune() == 0  # written once: nothing to reclaim


def test_writer_places_by_lead_value(tmp_path):
    # stored axis 0..6 step 3; the source provides leads 3 and 6 only (e.g. a start_date filter)
    with _make(tmp_path / 'a.cfdb') as ds:
        dv = ds['air_temperature_2m']
        w = fc.ForecastWriter(ds, init_idx=0, lead_index=fc.lead_index_map(ds, [3, 6]), ny=3, nx=4)
        w.put(dv, 0, slice(0, 2), np.full((2, 3, 4), 7.0, dtype='float32'))
        assert w.writes == 1
        out = np.squeeze(np.asarray(dv[0, :, 0, :, :].data), axis=(0, 2))
        assert np.isnan(out[0]).all()
        assert (out[1:] == 7.0).all()


def test_writer_close_flushes_partial_rows_once(tmp_path):
    with _make(tmp_path / 'a.cfdb') as ds:
        dv = ds['air_temperature_2m']
        w = fc.ForecastWriter(ds, init_idx=0, lead_index=fc.lead_index_map(ds, LEADS), ny=3, nx=4)
        w.put(dv, 0, 0, np.full((3, 4), 1.0, dtype='float32'))
        w.close()
        assert w.writes == 1
        out = np.squeeze(np.asarray(dv[0, :, 0, :, :].data), axis=(0, 2))
        assert (out[0] == 1.0).all() and np.isnan(out[1:]).all()


# ---------------------------------------------------------------- targets


def test_open_target_path_creates_then_appends(tmp_path):
    p = tmp_path / 'a.cfdb'
    with fc.open_target(p) as (ds, created):
        assert created and ds.dataset_type == 'grid_forecast'
        fc.create_forecast_coords(ds, INIT_A, LEADS, step_minutes=STEP)
    with fc.open_target(p) as (ds, created):
        assert not created and fc.FRT in ds.coord_names


def test_open_target_handle_is_not_closed(tmp_path):
    ds = _make(tmp_path / 'a.cfdb')
    with fc.open_target(ds) as (same, created):
        assert same is ds and not created
    assert ds.coord_names  # still usable
    ds.close()


def test_open_target_refuses_remote_backed_path(tmp_path):
    p = tmp_path / 'a.cfdb'
    with _make(p) as ds:
        ds._sys_meta.remote = True
        ds.sync()
    with pytest.raises(ValueError, match='remote-backed'):
        with fc.open_target(p):
            pass


def test_validate_target(tmp_path):
    with _make(tmp_path / 'a.cfdb') as ds:
        fc.validate_target(ds, x_name='longitude', y_name='latitude', x=LON, y=LAT)
        with pytest.raises(ValueError, match='longitude'):
            fc.validate_target(ds, x_name='longitude', y_name='latitude', x=LON + 1, y=LAT)
        with pytest.raises(ValueError, match="lacks the 'pressure'"):
            fc.validate_target(ds, x_name='longitude', y_name='latitude', x=LON, y=LAT, levels=[100000.0])
    with cfdb.open_dataset(str(tmp_path / 'g.cfdb'), 'n') as g:
        g.create.coord.time(data=np.array(['2026-01-01'], dtype='datetime64[m]'))
        with pytest.raises(ValueError, match="expected 'grid_forecast'"):
            fc.validate_target(g, x_name='longitude', y_name='latitude', x=LON, y=LAT)


def test_writer_refuses_spatially_partial_blocks(tmp_path):
    with _make(tmp_path / 'p.cfdb') as ds:
        w = fc.ForecastWriter(ds, 0, fc.lead_index_map(ds, LEADS), len(LAT), len(LON))
        with pytest.raises(NotImplementedError, match='full-extent'):
            w.put(ds['air_temperature_2m'], 0, 0, np.zeros((len(LAT), 2), 'float32'), slice(None), slice(0, 2))
