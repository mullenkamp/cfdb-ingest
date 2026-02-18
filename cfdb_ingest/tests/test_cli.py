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
        result = runner.invoke(app, [])
        assert result.exit_code != 0

    def test_missing_cfdb_path_exits_nonzero(self):
        result = runner.invoke(app, ["/nonexistent/file.nc"])
        assert result.exit_code != 0


class TestCliConvert:
    def test_basic_conversion(self, tmp_path):
        """CLI converts a single variable from a real WRF file."""
        import cfdb

        out = tmp_path / f"{uuid.uuid4().hex}.cfdb"
        result = runner.invoke(app, [
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

    def test_nonexistent_input_file(self, tmp_path):
        """CLI should fail gracefully for missing input files."""
        out = tmp_path / f"{uuid.uuid4().hex}.cfdb"
        result = runner.invoke(app, [
            "/nonexistent/wrfout_d03_2015-10-26_00:00:00.nc",
            str(out),
        ])
        assert result.exit_code != 0
