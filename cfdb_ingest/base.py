"""
Base class for HDF5/netCDF4 ingestion to cfdb via h5py.
"""
import pathlib
from typing import Union, List, Tuple, Dict, Optional

import h5py
import numpy as np
import pyproj
import rechunkit
import cfdb


class H5Ingest:
    """
    Abstract base class for converting HDF5/netCDF4 files to cfdb.

    Subclasses must implement the abstract methods to provide source-specific
    parsing of CRS, time, spatial coordinates, variable mappings, and data reading.

    Initialization derives all metadata from the input files and exposes it
    as attributes for inspection before calling convert().

    Parameters
    ----------
    input_paths : str, Path, or list thereof
        One or more source HDF5/netCDF4 file paths.
    """

    def __init__(self, input_paths: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]]):
        if isinstance(input_paths, (str, pathlib.Path)):
            input_paths = [input_paths]
        self.input_paths = sorted(pathlib.Path(p) for p in input_paths)

        for p in self.input_paths:
            if not p.exists():
                raise FileNotFoundError(f'File not found: {p}')

        self._init_metadata()

    def _init_metadata(self):
        """
        Derive all metadata from the source files by calling subclass methods.
        """
        with h5py.File(self.input_paths[0], 'r') as h5:
            self.crs = self._parse_crs(h5)
            spatial = self._parse_spatial_coords(h5)

        self.x = spatial['x']
        self.y = spatial['y']

        self._init_time()
        self._init_variables()
        self._compute_bbox_geographic()

    def _init_time(self):
        """
        Build self.times and self._file_time_map from all input files.
        """
        file_time_map = []
        all_times = []
        global_idx = 0

        for path in self.input_paths:
            with h5py.File(path, 'r') as h5:
                times = self._parse_time(h5)
            file_time_map.append((path, times, global_idx))
            all_times.append(times)
            global_idx += len(times)

        combined = np.concatenate(all_times)
        _, unique_idx = np.unique(combined, return_index=True)
        unique_idx.sort()
        self.times = combined[unique_idx]
        self._file_time_map = file_time_map

    def _init_variables(self):
        """
        Determine which mapped variables are available in the source files.
        """
        mapping = self._get_variable_mapping()

        available = {}
        with h5py.File(self.input_paths[0], 'r') as h5:
            source_vars = set(h5.keys())
            for key, info in mapping.items():
                if all(sv in source_vars for sv in info['source_vars']):
                    available[key] = info

        self.variables = available

    def _compute_bbox_geographic(self):
        """
        Compute the geographic bounding box (WGS84) from x/y coords and CRS.
        """
        transformer = pyproj.Transformer.from_crs(self.crs, 'EPSG:4326', always_xy=True)

        corners_x = [self.x[0], self.x[-1], self.x[0], self.x[-1]]
        corners_y = [self.y[0], self.y[0], self.y[-1], self.y[-1]]
        lons, lats = transformer.transform(corners_x, corners_y)

        self.bbox_geographic = (min(lons), min(lats), max(lons), max(lats))

    # ------------------------------------------------------------------
    # Abstract methods — subclasses must implement
    # ------------------------------------------------------------------

    def _parse_crs(self, h5: h5py.File) -> pyproj.CRS:
        """Extract CRS from a source file."""
        raise NotImplementedError

    def _parse_time(self, h5: h5py.File) -> np.ndarray:
        """Extract time coordinate array (datetime64) from a source file."""
        raise NotImplementedError

    def _parse_spatial_coords(self, h5: h5py.File) -> Dict[str, np.ndarray]:
        """
        Return spatial coordinate arrays from a source file.

        Must return a dict with at least 'x' and 'y' keys mapping to 1D arrays.
        """
        raise NotImplementedError

    def _get_variable_mapping(self) -> Dict[str, dict]:
        """
        Return the full variable mapping dict.

        Each key is a variable identifier, and each value is a dict with at least:
        - 'cfdb_name': str — cfdb short name (e.g., 'air_temp')
        - 'source_vars': list of str — source variable names needed
        - 'transform': str or None — name of a transform method
        - 'height': float or 'levels' — height above ground in meters,
          or 'levels' for variables that require vertical interpolation
          to user-specified target_levels
        """
        raise NotImplementedError

    def _read_variable(self, h5: h5py.File, var_key: str, time_idx: int,
                       spatial_slice: Tuple[slice, ...]) -> np.ndarray:
        """
        Read and transform a single variable for one timestep.

        Parameters
        ----------
        h5 : h5py.File
            Open source file.
        var_key : str
            Key into the variable mapping dict.
        time_idx : int
            Time index within this file.
        spatial_slice : tuple of slice
            Spatial subset slices (y_slice, x_slice).

        Returns
        -------
        np.ndarray
            2D array (ny, nx) for single-height vars, or
            3D array (n_levels, ny, nx) for level-interpolated vars.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def resolve_variables(self, variables: Optional[List[str]]) -> List[str]:
        """
        Resolve user-provided variable names to mapping keys.

        Accepts mapping keys, source variable names, or cfdb short names.
        Returns a list of mapping keys. If variables is None, returns all
        available mapping keys.

        When a cfdb_name maps to multiple keys (e.g., both surface and
        level-interpolated variants of air_temp), all matching keys are returned.
        """
        if variables is None:
            return list(self.variables.keys())

        mapping = self.variables
        cfdb_name_to_keys = {}
        source_var_to_key = {}
        for key, info in mapping.items():
            cfdb_name_to_keys.setdefault(info['cfdb_name'], []).append(key)
            for sv in info['source_vars']:
                source_var_to_key[sv] = key

        resolved = []
        seen = set()
        for name in variables:
            if name in mapping:
                keys = [name]
            elif name in cfdb_name_to_keys:
                keys = cfdb_name_to_keys[name]
            elif name in source_var_to_key:
                keys = [source_var_to_key[name]]
            else:
                raise ValueError(f'Unknown variable: {name!r}. '
                                 f'Available: {list(mapping.keys())}')
            for key in keys:
                if key not in seen:
                    resolved.append(key)
                    seen.add(key)

        return resolved

    def convert(
        self,
        cfdb_path: Union[str, pathlib.Path],
        variables: Optional[List[str]] = None,
        start_date: Union[str, np.datetime64, None] = None,
        end_date: Union[str, np.datetime64, None] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        target_levels: Optional[List[float]] = None,
        max_mem: int = 2**27,
        chunk_shape: Optional[Tuple[int, int, int, int]] = None,
        dataset_type: str = 'grid',
        **cfdb_kwargs,
    ):
        """
        Convert source files to a cfdb dataset.

        All output data variables have dimensions (time, height, y, x).
        Surface variables are placed at their physical height above ground
        (e.g., 2 m for T2, 10 m for U10/V10, 0 m for surface fluxes).
        Variables with height='levels' are interpolated to target_levels.
        Variables sharing a cfdb_name are merged into one data variable
        spanning all their height levels.

        Parameters
        ----------
        cfdb_path : str or Path
            Output cfdb file path.
        variables : list of str or None
            Variable names to convert (mapping keys, source names, or cfdb names).
            None converts all available mapped variables.
        start_date, end_date : str, np.datetime64, or None
            Optional time range filter.
        bbox : tuple of 4 floats or None
            Bounding box as (min_lon, min_lat, max_lon, max_lat) in WGS84.
        target_levels : list of float or None
            Target height levels (meters) for level-interpolated variables.
        max_mem : int
            Memory budget in bytes for rechunkit read buffers.
        chunk_shape : tuple of 4 ints or None
            Output chunk shape as (time, z, y, x), following CF conventions.
            Defaults to (1, 1, ny, nx) — one full spatial slab per timestep
            per height level.
        dataset_type : str
            Passed to cfdb.open_dataset.
        **cfdb_kwargs
            Extra kwargs for cfdb.open_dataset (e.g., compression).
        """
        var_keys = self.resolve_variables(variables)

        has_level_interp = any(self.variables[k]['height'] == 'levels' for k in var_keys)
        if has_level_interp and target_levels is None:
            raise ValueError('target_levels is required when converting level-interpolated variables.')

        # Filter time
        time_mask, filtered_times = self._filter_time(start_date, end_date)

        # Filter space
        if bbox is not None:
            x_slice, y_slice, filtered_x, filtered_y = self._bbox_to_indices(bbox)
        else:
            x_slice = slice(None)
            y_slice = slice(None)
            filtered_x = self.x
            filtered_y = self.y

        spatial_slice = (y_slice, x_slice)
        ny = len(filtered_y)
        nx = len(filtered_x)

        # Group variables by cfdb_name (variables sharing a name are merged)
        cfdb_var_groups = {}
        all_heights = set()
        for var_key in var_keys:
            info = self.variables[var_key]
            cfdb_name = info['cfdb_name']
            height_spec = info['height']

            if height_spec == 'levels':
                heights = [float(h) for h in target_levels]
            else:
                heights = [float(height_spec)]

            if cfdb_name not in cfdb_var_groups:
                cfdb_var_groups[cfdb_name] = []
            cfdb_var_groups[cfdb_name].append((var_key, heights))
            all_heights.update(heights)

        sorted_heights = np.array(sorted(all_heights), dtype='float64')
        height_to_idx = {h: i for i, h in enumerate(sorted_heights)}

        coord_names = ('time', 'height', 'y', 'x')
        if chunk_shape is None:
            chunk_shape = (1, 1, ny, nx)

        with cfdb.open_dataset(cfdb_path, 'n', dataset_type=dataset_type, **cfdb_kwargs) as ds:
            # Create coordinates
            ds.create.coord.time(data=filtered_times)
            ds.create.coord.height(data=sorted_heights)
            ds.create.coord.y(data=filtered_y.astype('float32'))
            ds.create.coord.x(data=filtered_x.astype('float32'))

            # Set CRS
            ds.create.crs.from_user_input(self.crs, x_coord='x', y_coord='y')

            # Create and populate data variables
            for cfdb_name, var_group in cfdb_var_groups.items():
                data_var = self._create_cfdb_data_var(ds, cfdb_name, coord_names, chunk_shape)

                for var_key, heights in var_group:
                    height_indices = [height_to_idx[h] for h in heights]
                    self._populate_data_var(
                        data_var, var_key, time_mask, spatial_slice, max_mem,
                        height_indices, target_levels,
                    )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_time(self, start_date, end_date):
        """
        Return a boolean mask and the filtered times array.
        """
        mask = np.ones(len(self.times), dtype=bool)

        if start_date is not None:
            start = np.datetime64(start_date)
            mask &= self.times >= start

        if end_date is not None:
            end = np.datetime64(end_date)
            mask &= self.times <= end

        return mask, self.times[mask]

    def _bbox_to_indices(self, bbox):
        """
        Transform a WGS84 bbox to source CRS and return index slices + subsetted arrays.

        Parameters
        ----------
        bbox : tuple
            (min_lon, min_lat, max_lon, max_lat) in WGS84.

        Returns
        -------
        x_slice, y_slice, filtered_x, filtered_y
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        transformer = pyproj.Transformer.from_crs('EPSG:4326', self.crs, always_xy=True)

        # Transform corners and edge midpoints for curved projections
        sample_lons = [min_lon, max_lon, min_lon, max_lon, (min_lon + max_lon) / 2, (min_lon + max_lon) / 2, min_lon, max_lon]
        sample_lats = [min_lat, min_lat, max_lat, max_lat, min_lat, max_lat, (min_lat + max_lat) / 2, (min_lat + max_lat) / 2]
        proj_x, proj_y = transformer.transform(sample_lons, sample_lats)

        x_min, x_max = min(proj_x), max(proj_x)
        y_min, y_max = min(proj_y), max(proj_y)

        x_mask = (self.x >= x_min) & (self.x <= x_max)
        y_mask = (self.y >= y_min) & (self.y <= y_max)

        x_indices = np.where(x_mask)[0]
        y_indices = np.where(y_mask)[0]

        if len(x_indices) == 0 or len(y_indices) == 0:
            raise ValueError('Bounding box does not overlap with the dataset domain.')

        x_slice = slice(int(x_indices[0]), int(x_indices[-1]) + 1)
        y_slice = slice(int(y_indices[0]), int(y_indices[-1]) + 1)

        return x_slice, y_slice, self.x[x_slice], self.y[y_slice]

    def _create_cfdb_data_var(self, ds, cfdb_name, coord_names, chunk_shape):
        """
        Create a cfdb data variable using the template method for cfdb_name.

        Template methods (e.g., ds.create.data_var.air_temp) auto-set the
        appropriate dtype, encoding, and CF attributes from cfdb's defaults.
        """
        creator = ds.create.data_var
        template = getattr(creator, cfdb_name, None)
        if template is None:
            raise TypeError(f'{cfdb_name} does not exist in cfdb')

        return template(coord_names, chunk_shape=chunk_shape)

        # return creator.generic(cfdb_name, coord_names, dtype='float32', chunk_shape=chunk_shape)

    def _setup_populate(self, var_key, target_levels):
        """Hook called before populating a data variable. Override as needed."""
        pass

    def _read_accumulation_increment(self, h5, var_key, time_idx, spatial_slice):
        """
        Compute increment from accumulated source variables.

        Sums all source_vars at current and previous timestep, returns the
        difference. Uses self._prev_accum_total for cross-file boundaries.
        Returns NaN for the very first overall timestep.
        """
        info = self.variables[var_key]
        y_sl, x_sl = spatial_slice

        total = sum(h5[sv][time_idx, y_sl, x_sl].astype('float64') for sv in info['source_vars'])

        if time_idx > 0:
            prev = sum(h5[sv][time_idx - 1, y_sl, x_sl].astype('float64') for sv in info['source_vars'])
            result = total - prev
        elif self._prev_accum_total is not None:
            result = total - self._prev_accum_total
        else:
            result = np.full_like(total, np.nan)

        return result.astype('float32')

    def _populate_data_var(self, data_var, var_key, time_mask, spatial_slice, max_mem, height_indices, target_levels):
        """
        Dispatch data population to the appropriate strategy.

        Simple variables (no transform, single source var) use rechunkit for
        optimal HDF5 reads. Transform variables (accumulation, wind, 3D temp)
        use per-timestep iteration.
        """
        info = self.variables[var_key]
        transform = info.get('transform')
        is_simple = transform is None and len(info['source_vars']) == 1
        is_accumulation = transform == 'accumulation_increment'

        self._setup_populate(var_key, target_levels)

        if is_accumulation:
            self._prev_accum_total = None

        if is_simple:
            self._populate_with_rechunkit(data_var, var_key, time_mask, spatial_slice, max_mem, height_indices)
        else:
            self._populate_per_timestep(data_var, var_key, time_mask, spatial_slice, height_indices, is_accumulation)

    def _populate_with_rechunkit(self, data_var, var_key, time_mask, spatial_slice, max_mem, height_indices):
        """
        Populate a simple (no-transform, single source var) data variable using
        rechunkit for optimized HDF5 chunk reads.
        """
        src_var = self.variables[var_key]['source_vars'][0]
        y_sl, x_sl = spatial_slice
        h_idx = height_indices[0]

        # Pre-compute global_time_idx -> output_time_idx mapping
        output_map = {}
        out_idx = 0
        for g_idx in range(len(time_mask)):
            if time_mask[g_idx]:
                output_map[g_idx] = out_idx
                out_idx += 1

        global_time_idx = 0
        for path, file_times, _ in self._file_time_map:
            n_file_times = len(file_times)

            with h5py.File(path, 'r') as h5:
                h5_var = h5[src_var]
                source_chunks = h5_var.chunks or rechunkit.guess_chunk_shape(
                    h5_var.shape, h5_var.dtype.itemsize, max_mem
                )

                # Build explicit sel (rechunkit requires non-None start/stop)
                y_start, y_stop, _ = y_sl.indices(h5_var.shape[1])
                x_start, x_stop, _ = x_sl.indices(h5_var.shape[2])
                ny = y_stop - y_start
                nx = x_stop - x_start
                sel = (slice(0, n_file_times), slice(y_start, y_stop), slice(x_start, x_stop))

                target_chunks = (1, ny, nx)

                for write_slices, data in rechunkit.rechunker(
                    h5_var.__getitem__, h5_var.shape, h5_var.dtype,
                    source_chunks, target_chunks, max_mem, sel=sel,
                ):
                    local_t = write_slices[0].start
                    g_idx = global_time_idx + local_t
                    if g_idx not in output_map:
                        continue

                    data_var[(output_map[g_idx], h_idx, slice(None), slice(None))] = data[0].astype('float32')

            global_time_idx += n_file_times

    def _populate_per_timestep(self, data_var, var_key, time_mask, spatial_slice, height_indices, is_accumulation):
        """
        Populate a data variable using per-timestep iteration.

        Used for transform variables (accumulation, wind, 3D temp) that need
        per-timestep logic. Handles accumulation cross-file caching.
        """
        global_time_idx = 0
        output_time_idx = 0

        for path, file_times, _ in self._file_time_map:
            n_file_times = len(file_times)

            with h5py.File(path, 'r') as h5:
                for local_t in range(n_file_times):
                    if not time_mask[global_time_idx]:
                        global_time_idx += 1
                        continue

                    data = self._read_variable(h5, var_key, local_t, spatial_slice)

                    if len(height_indices) == 1:
                        data_var[(output_time_idx, height_indices[0], slice(None), slice(None))] = data
                    else:
                        for lev_i, h_idx in enumerate(height_indices):
                            data_var[(output_time_idx, h_idx, slice(None), slice(None))] = data[lev_i]

                    output_time_idx += 1
                    global_time_idx += 1

                # Cache last accumulated total for cross-file boundary
                if is_accumulation:
                    info = self.variables[var_key]
                    y_sl, x_sl = spatial_slice
                    total = sum(
                        h5[sv][n_file_times - 1, y_sl, x_sl].astype('float64')
                        for sv in info['source_vars']
                    )
                    self._prev_accum_total = total.astype('float32')

    def _get_file_for_global_time(self, global_idx):
        """
        Return (path, local_time_idx) for a given global time index.
        """
        for path, file_times, start_idx in self._file_time_map:
            if global_idx < start_idx + len(file_times):
                return path, global_idx - start_idx
        raise IndexError(f'Global time index {global_idx} out of range.')
