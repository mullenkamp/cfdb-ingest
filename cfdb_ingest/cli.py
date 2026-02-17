"""
CLI for cfdb-ingest.
"""
from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated

app = typer.Typer(help="Convert file formats to cfdb.")


@app.command()
def wrf(
    input_paths: Annotated[List[Path], typer.Argument(help="One or more wrfout file paths.")],
    cfdb_path: Annotated[Path, typer.Argument(help="Output cfdb file path.")],
    variables: Annotated[Optional[str], typer.Option("--variables", "-v", help="Comma-separated variable names.")] = None,
    start_date: Annotated[Optional[str], typer.Option("--start-date", "-s", help="Start date (ISO format).")] = None,
    end_date: Annotated[Optional[str], typer.Option("--end-date", "-e", help="End date (ISO format).")] = None,
    bbox: Annotated[Optional[str], typer.Option("--bbox", "-b", help="Bounding box: min_lon,min_lat,max_lon,max_lat")] = None,
    target_levels: Annotated[Optional[str], typer.Option("--target-levels", "-l", help="Comma-separated height levels in meters.")] = None,
    max_mem: Annotated[int, typer.Option(help="Read buffer size in bytes.")] = 2**27,
    compression: Annotated[Optional[str], typer.Option(help="Compression: zstd or lz4.")] = None,
):
    """Convert WRF output files to cfdb."""
    from cfdb_ingest.wrf import WrfIngest

    ingest = WrfIngest(input_paths)

    var_list = [v.strip() for v in variables.split(",")] if variables else None
    bbox_tuple = tuple(float(x) for x in bbox.split(",")) if bbox else None
    levels = [float(x) for x in target_levels.split(",")] if target_levels else None

    cfdb_kwargs = {}
    if compression is not None:
        cfdb_kwargs["compression"] = compression

    ingest.convert(
        cfdb_path=cfdb_path,
        variables=var_list,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox_tuple,
        target_levels=levels,
        max_mem=max_mem,
        **cfdb_kwargs,
    )
