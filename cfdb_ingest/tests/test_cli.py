"""
Tests for cfdb_ingest.cli.
"""
import pathlib
import uuid

import numpy as np
import pytest
from typer.testing import CliRunner

from cfdb_ingest.cli import app
from cfdb_ingest.tests.conftest import WRF_FILE_1

runner = CliRunner()


class TestCliHelp:
    def test_help_exits_zero(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_help_mentions_wrf(self):
        result = runner.invoke(app, ["--help"])
        assert "wrfout" in result.output.lower() or "wrf" in result.output.lower()


class TestCliArgParsing:
    def test_no_args_exits_nonzero(self):
        result = runner.invoke(app, ["wrf"])
        assert result.exit_code != 0

    def test_missing_cfdb_path_exits_nonzero(self):
        result = runner.invoke(app, ["wrf", "/nonexistent/file.nc"])
        assert result.exit_code != 0


class TestCliConvert:
    def test_basic_conversion(self, tmp_path):
        """CLI converts a single variable from a real WRF file."""
        import cfdb

        out = tmp_path / f"{uuid.uuid4().hex}.cfdb"
        result = runner.invoke(app, [
            "wrf",
            str(WRF_FILE_1),
            str(out),
            "-v", "T2",
            "-s", "2023-02-12T12:00",
            "-e", "2023-02-12T12:00",
        ])
        assert result.exit_code == 0, result.output

        with cfdb.open_dataset(out, "r") as ds:
            times = np.array(ds["time"][:])
            assert len(times) == 1
            var_names = [v.name for v in ds.data_vars]
            assert "air_temperature" in var_names

    def test_multiple_variables(self, tmp_path):
        """CLI handles comma-separated variable list."""
        import cfdb

        out = tmp_path / f"{uuid.uuid4().hex}.cfdb"
        result = runner.invoke(app, [
            "wrf",
            str(WRF_FILE_1),
            str(out),
            "-v", "T2,WIND10",
            "-s", "2023-02-12T12:00",
            "-e", "2023-02-12T12:00",
        ])
        assert result.exit_code == 0, result.output

        with cfdb.open_dataset(out, "r") as ds:
            var_names = [v.name for v in ds.data_vars]
            assert "air_temperature" in var_names
            assert "wind_speed" in var_names

    def test_bbox_option(self, tmp_path):
        """CLI parses bbox option correctly."""
        import cfdb

        out = tmp_path / f"{uuid.uuid4().hex}.cfdb"
        result = runner.invoke(app, [
            "wrf",
            str(WRF_FILE_1),
            str(out),
            "-v", "T2",
            "-s", "2023-02-12T12:00",
            "-e", "2023-02-12T12:00",
            "-b", "165.0,-47.0,175.0,-40.0",
        ])
        assert result.exit_code == 0, result.output

        with cfdb.open_dataset(out, "r") as ds:
            x = np.array(ds["x"][:])
            assert len(x) < 99  # Less than full domain

    def test_target_levels(self, tmp_path):
        """CLI parses target-levels for 3D variable conversion."""
        import cfdb

        out = tmp_path / f"{uuid.uuid4().hex}.cfdb"
        result = runner.invoke(app, [
            "wrf",
            str(WRF_FILE_1),
            str(out),
            "-v", "T",
            "-s", "2023-02-12T12:00",
            "-e", "2023-02-12T12:00",
            "-b", "165.0,-47.0,175.0,-40.0",
            "-l", "100.0,500.0,1000.0",
        ])
        assert result.exit_code == 0, result.output

        with cfdb.open_dataset(out, "r") as ds:
            height = np.array(ds["height"][:])
            np.testing.assert_array_equal(height, [100.0, 500.0, 1000.0])

    def test_chunk_shape(self, tmp_path):
        """CLI parses --chunk-shape and applies it to 4D output."""
        import cfdb

        out = tmp_path / f"{uuid.uuid4().hex}.cfdb"
        result = runner.invoke(app, [
            "wrf",
            str(WRF_FILE_1),
            str(out),
            "-v", "T",
            "-s", "2023-02-12T12:00",
            "-e", "2023-02-12T12:00",
            "-b", "165.0,-47.0,175.0,-40.0",
            "-l", "100.0,500.0",
            "-c", "1,1,50,50",
        ])
        assert result.exit_code == 0, result.output

        with cfdb.open_dataset(out, "r") as ds:
            cs = ds["air_temperature"].chunk_shape
            assert cs == (1, 1, 50, 50)

    def test_nonexistent_input_file(self, tmp_path):
        """CLI should fail gracefully for missing input files."""
        out = tmp_path / f"{uuid.uuid4().hex}.cfdb"
        result = runner.invoke(app, [
            "wrf",
            "/nonexistent/wrfout_d03_2015-10-26_00:00:00.nc",
            str(out),
        ])
        assert result.exit_code != 0


class TestCliPreset:
    def test_preset_wps(self, tmp_path):
        """--preset wps selects WPS variables and sets pressure coord."""
        import cfdb

        out = tmp_path / f"{uuid.uuid4().hex}.cfdb"
        result = runner.invoke(app, [
            "wrf",
            str(WRF_FILE_1),
            str(out),
            "--preset", "wps",
            "-s", "2023-02-12T12:00",
            "-e", "2023-02-12T12:00",
            "-b", "165.0,-47.0,175.0,-40.0",
        ])
        assert result.exit_code == 0, result.output

        with cfdb.open_dataset(out, "r") as ds:
            # Should have pressure coordinate, not height
            assert 'pressure' in ds.coord_names
            assert 'height' not in ds.coord_names

            # Should have multiple data variables
            var_names = [v.name for v in ds.data_vars]
            assert len(var_names) > 5

    def test_preset_wps_with_extra_vars(self, tmp_path):
        """--preset wps with -v adds extra variables."""
        import cfdb

        out = tmp_path / f"{uuid.uuid4().hex}.cfdb"
        result = runner.invoke(app, [
            "wrf",
            str(WRF_FILE_1),
            str(out),
            "--preset", "wps",
            "-v", "WIND10",
            "-s", "2023-02-12T12:00",
            "-e", "2023-02-12T12:00",
            "-b", "165.0,-47.0,175.0,-40.0",
        ])
        assert result.exit_code == 0, result.output

        with cfdb.open_dataset(out, "r") as ds:
            var_names = [v.name for v in ds.data_vars]
            # WIND10 should be included (not in default WPS preset but added via -v)
            assert 'wind_speed' in var_names or 'wind_speed_sfc' in var_names

    def test_preset_wps_custom_levels(self, tmp_path):
        """--preset wps with -l overrides default pressure levels."""
        import cfdb

        out = tmp_path / f"{uuid.uuid4().hex}.cfdb"
        result = runner.invoke(app, [
            "wrf",
            str(WRF_FILE_1),
            str(out),
            "--preset", "wps",
            "-l", "90000,70000,50000",
            "-s", "2023-02-12T12:00",
            "-e", "2023-02-12T12:00",
            "-b", "165.0,-47.0,175.0,-40.0",
        ])
        assert result.exit_code == 0, result.output

        with cfdb.open_dataset(out, "r") as ds:
            pressure = np.array(ds['pressure'][:])
            np.testing.assert_array_equal(pressure, [50000.0, 70000.0, 90000.0])

    def test_preset_unknown_exits_nonzero(self, tmp_path):
        """Unknown preset name should fail."""
        out = tmp_path / f"{uuid.uuid4().hex}.cfdb"
        result = runner.invoke(app, [
            "wrf",
            str(WRF_FILE_1),
            str(out),
            "--preset", "nonexistent",
        ])
        assert result.exit_code != 0


class TestCliVerticalCoord:
    def test_vertical_coord_pressure(self, tmp_path):
        """--vertical-coord pressure creates pressure coordinate."""
        import cfdb

        out = tmp_path / f"{uuid.uuid4().hex}.cfdb"
        result = runner.invoke(app, [
            "wrf",
            str(WRF_FILE_1),
            str(out),
            "-v", "T",
            "-s", "2023-02-12T12:00",
            "-e", "2023-02-12T12:00",
            "-b", "165.0,-47.0,175.0,-40.0",
            "-l", "90000,70000,50000",
            "--vertical-coord", "pressure",
        ])
        assert result.exit_code == 0, result.output

        with cfdb.open_dataset(out, "r") as ds:
            assert 'pressure' in ds.coord_names
            assert 'height' not in ds.coord_names
            assert ds['air_temperature'].coord_names == ('time', 'pressure', 'y', 'x')
