"""
CLI for cfdb-ingest.
"""
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated

app = typer.Typer(help="Convert file formats to cfdb.")

# Variables needed for WPS intermediate file export via cfdb-to-int
WPS_PRESET_VARS = [
    'T', 'U', 'V', 'GHT', 'RH', 'Q_SH',           # 3D pressure-level
    'PSFC', 'SLP', 'TSK', 'T2', 'U10', 'V10',       # surface
    'TD2', 'RH2', 'XLAND', 'HGT', 'SNOWH',          # surface
    'SST_VAR', 'SEAICE_VAR', 'SNOW_VAR',             # surface (optional)
    'SMOIS', 'TSLB',                                  # soil
]

# Default pressure levels in Pa for WPS preset (same as wrf_to_int defaults)
WPS_DEFAULT_PRESSURE_LEVELS_PA = [
    100000, 97500, 95000, 92500, 90000, 85000, 80000, 75000, 70000, 65000,
    60000, 55000, 50000, 45000, 40000, 35000, 30000, 25000, 20000, 15000,
    10000, 7000, 5000, 3000, 2000, 1000,
]


@app.command()
def wrf(
    input_paths: Annotated[List[Path], typer.Argument(help="One or more wrfout file paths.")],
    cfdb_path: Annotated[Path, typer.Argument(help="Output cfdb file path.")],
    variables: Annotated[Optional[str], typer.Option("--variables", "-v", help="Comma-separated variable names.")] = None,
    preset: Annotated[Optional[str], typer.Option("--preset", help="Variable preset: 'wps' selects all variables needed for cfdb-to-int export.")] = None,
    start_date: Annotated[Optional[str], typer.Option("--start-date", "-s", help="Start date (ISO format).")] = None,
    end_date: Annotated[Optional[str], typer.Option("--end-date", "-e", help="End date (ISO format).")] = None,
    bbox: Annotated[Optional[str], typer.Option("--bbox", "-b", help="Bounding box: min_lon,min_lat,max_lon,max_lat")] = None,
    target_levels: Annotated[Optional[str], typer.Option("--target-levels", "-l", help="Comma-separated target levels (meters for height, Pa for pressure).")] = None,
    vertical_coord: Annotated[str, typer.Option("--vertical-coord", help="Vertical coordinate: 'height' (meters) or 'pressure' (Pa).")] = 'height',
    max_mem: Annotated[int, typer.Option(help="Read buffer size in bytes.")] = 2**29,
    chunk_shape: Annotated[Optional[str], typer.Option("--chunk-shape", "-c", help="Output chunk shape as time,z,y,x (e.g. 1,1,50,50); init,lead,z,y,x with --forecast.")] = None,
    compression: Annotated[Optional[str], typer.Option(help="Compression: zstd or lz4.")] = None,
    forecast: Annotated[bool, typer.Option("--forecast", help="Treat the files as ONE forecast run and write a grid_forecast dataset (appends to an existing one).")] = False,
    init: Annotated[Optional[str], typer.Option("--init", help="Forecast mode: the run's init (ISO). Default: SIMULATION_START_DATE / START_DATE.")] = None,
    forecast_step_minutes: Annotated[int, typer.Option("--forecast-step-minutes", help="Forecast mode: init step baked into a NEW dataset.")] = 360,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Forecast mode: replace an init the dataset already holds complete.")] = False,
):
    """Convert WRF output files to cfdb."""
    from cfdb_ingest.wrf import WrfIngest

    ingest = WrfIngest(input_paths)

    bbox_tuple = tuple(float(x) for x in bbox.split(",")) if bbox else None
    levels = [float(x) for x in target_levels.split(",")] if target_levels else None
    chunks = tuple(int(x) for x in chunk_shape.split(",")) if chunk_shape else None

    if preset is not None:
        if preset.lower() == 'wps':
            var_list = [v for v in WPS_PRESET_VARS if v in ingest.variables]
            # Append any extra --variables on top of the preset
            if variables:
                for v in variables.split(","):
                    v = v.strip()
                    if v not in var_list:
                        var_list.append(v)
            if vertical_coord != 'pressure':
                print("Note: --preset wps implies --vertical-coord pressure")
                vertical_coord = 'pressure'
            if levels is None:
                levels = [float(x) for x in WPS_DEFAULT_PRESSURE_LEVELS_PA]
        else:
            print(f"Error: Unknown preset '{preset}'. Available presets: wps", file=__import__('sys').stderr)
            raise typer.Exit(code=1)
    else:
        var_list = [v.strip() for v in variables.split(",")] if variables else None

    cfdb_kwargs = {}
    if compression is not None:
        cfdb_kwargs["compression"] = compression

    if forecast:
        cfdb_kwargs.update(dataset_type='grid_forecast', forecast_reference_time=init,
                           forecast_step_minutes=forecast_step_minutes, overwrite=overwrite)

    result = ingest.convert(
        cfdb_path=cfdb_path,
        variables=var_list,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox_tuple,
        target_levels=levels,
        vertical_coord=vertical_coord,
        max_mem=max_mem,
        chunk_shape=chunks,
        **cfdb_kwargs,
    )
    if result:
        print(result)


@app.command()
def ifs(
    input_paths: Annotated[List[Path], typer.Argument(help="One forecast cycle's GRIB2 files, or directories of them.")],
    target: Annotated[Path, typer.Argument(help="Output cfdb path (created, or appended to).")],
    variables: Annotated[Optional[str], typer.Option("--variables", "-v", help="Comma-separated mapping keys, GRIB shortNames or cfdb names.")] = None,
    preset: Annotated[Optional[str], typer.Option("--preset", help="Variable preset: 'wps' selects the fields cfdb-to-int needs for WRF.")] = None,
    bbox: Annotated[Optional[str], typer.Option("--bbox", "-b", help="Bounding box: min_lon,min_lat,max_lon,max_lat (may cross the dateline).")] = None,
    target_levels: Annotated[Optional[str], typer.Option("--target-levels", "-l", help="Comma-separated pressure levels (Pa) to keep; default all.")] = None,
    max_lead_hours: Annotated[Optional[int], typer.Option("--max-lead-hours", help="Drop leads beyond this (144 for IFS 00/12z cycles: 3-hourly only to 144 h).")] = None,
    chunk_shape: Annotated[Optional[str], typer.Option("--chunk-shape", "-c", help="Chunk shape as init,lead,z,y,x.")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace an init the dataset already holds complete.")] = False,
    compression: Annotated[Optional[str], typer.Option(help="Compression: zstd or lz4 (new datasets only).")] = None,
):
    """Ingest one ECMWF IFS open-data forecast cycle (GRIB2) into a cfdb grid_forecast dataset."""
    from cfdb_ingest.ifs import IFS_WPS_PRESET_KEYS, IfsIngest

    ingest = IfsIngest(input_paths)

    bbox_tuple = tuple(float(x) for x in bbox.split(",")) if bbox else None
    levels = [float(x) for x in target_levels.split(",")] if target_levels else None
    chunks = tuple(int(x) for x in chunk_shape.split(",")) if chunk_shape else None

    if preset is not None:
        if preset.lower() == 'wps':
            var_list = [v for v in IFS_WPS_PRESET_KEYS if v in ingest.variables]
            if variables:
                for v in variables.split(","):
                    v = v.strip()
                    if v not in var_list:
                        var_list.append(v)
        else:
            print(f"Error: Unknown preset '{preset}'. Available presets: wps", file=__import__('sys').stderr)
            raise typer.Exit(code=1)
    else:
        var_list = [v.strip() for v in variables.split(",")] if variables else None

    cfdb_kwargs = {}
    if compression is not None:
        cfdb_kwargs["compression"] = compression

    result = ingest.convert(
        target,
        variables=var_list,
        bbox=bbox_tuple,
        target_levels=levels,
        max_lead_hours=max_lead_hours,
        chunk_shape=chunks,
        overwrite=overwrite,
        **cfdb_kwargs,
    )
    print(result)


@app.command()
def era5(
    input_paths: Annotated[List[Path], typer.Argument(help="ERA5 NetCDF files or directory.")],
    output_path: Annotated[Path, typer.Argument(help="Output cfdb path (file for combined, directory for split).")],
    split: Annotated[bool, typer.Option("--split", help="Create one cfdb file per variable.")] = False,
    variables: Annotated[Optional[str], typer.Option("--variables", "-v", help="Comma-separated variable names (mapping keys, source names, or cfdb names).")] = None,
    start_date: Annotated[Optional[str], typer.Option("--start-date", "-s", help="Start date (ISO format).")] = None,
    end_date: Annotated[Optional[str], typer.Option("--end-date", "-e", help="End date (ISO format).")] = None,
    bbox: Annotated[Optional[str], typer.Option("--bbox", "-b", help="Bounding box: min_lon,min_lat,max_lon,max_lat")] = None,
    target_levels: Annotated[Optional[str], typer.Option("--target-levels", "-l", help="Comma-separated pressure levels in Pa. Auto-detected from files if omitted.")] = None,
    chunk_shape: Annotated[Optional[str], typer.Option("--chunk-shape", "-c", help="Output chunk shape as time,z,y,x (e.g. 1,1,50,50).")] = None,
    compression: Annotated[Optional[str], typer.Option(help="Compression: zstd or lz4.")] = None,
):
    """Convert ERA5 NetCDF files to cfdb."""
    from cfdb_ingest.era5 import Era5Ingest

    ingest = Era5Ingest(input_paths)

    bbox_tuple = tuple(float(x) for x in bbox.split(",")) if bbox else None
    levels = [float(x) for x in target_levels.split(",")] if target_levels else None
    chunks = tuple(int(x) for x in chunk_shape.split(",")) if chunk_shape else None
    var_list = [v.strip() for v in variables.split(",")] if variables else None

    cfdb_kwargs = {}
    if compression is not None:
        cfdb_kwargs["compression"] = compression

    ingest.convert(
        cfdb_path=output_path,
        variables=var_list,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox_tuple,
        target_levels=levels,
        split=split,
        chunk_shape=chunks,
        **cfdb_kwargs,
    )


@app.command()
def cfdb_to_int(
    cfdb_path: Annotated[Path, typer.Argument(help="Input cfdb dataset path.")],
    start_date: Annotated[Optional[datetime], typer.Option(
        "--start-date", "-s", help="First valid time to convert (default: the first available).",
        formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H", "%Y-%m-%d_%H"],
    )] = None,
    end_date: Annotated[Optional[datetime], typer.Option(
        "--end-date", "-e", help="Last valid time to convert (default: the last available).",
        formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H", "%Y-%m-%d_%H"],
    )] = None,
    hour_interval: Annotated[int, typer.Option("--hour-interval", "-h", help="Interval in hours between records.")] = 6,
    prefix: Annotated[str, typer.Option("--prefix", "-p", help="Output file prefix (may include a directory).")] = 'WRF',
    init: Annotated[Optional[datetime], typer.Option(
        "--init", "-i", help="grid_forecast datasets: the forecast init to export (required).",
        formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H", "%Y-%m-%d_%H"],
    )] = None,
):
    """Convert a cfdb dataset to WPS intermediate files for metgrid.exe."""
    from cfdb_ingest.cfdb_to_int import convert_cfdb_to_int

    convert_cfdb_to_int(
        cfdb_path=cfdb_path,
        output_prefix=prefix,
        start_date=start_date,
        end_date=end_date,
        hour_interval=hour_interval,
        init=init,
    )
