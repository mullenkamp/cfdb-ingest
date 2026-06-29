"""
Base class for HDF5/netCDF4 ingestion to cfdb via h5py.
"""
import datetime
import pathlib
from contextlib import ExitStack
from typing import Union, List, Tuple, Dict, Optional
import concurrent.futures
import h5py
import numpy as np
import pyproj
import rechunkit
import cfdb
import cfdb_ingest


class _ConcatTimeSource:
    """
    Adapter that presents N already-open HDF5 datasets as one virtual array
    concatenated along the time axis (axis 0), exposing a callable interface
    suitable for ``rechunkit.rechunker``.

    All files must contain ``var_name`` with identical shape on axes 1+ and
    identical dtype/chunk layout. ``file_lens[i]`` is the time-axis length of
    file ``i``; the cumulative sum gives the global time index range each file
    covers.
    """

    def __init__(self, h5_files: list, var_name: str, file_lens: list):
        self._datasets = [h5[var_name] for h5 in h5_files]
        self._cum = np.cumsum([0] + list(file_lens)).astype('int64')
        first = self._datasets[0]
        self.shape = (int(self._cum[-1]),) + tuple(first.shape[1:])
        self.dtype = first.dtype
        # Inherit source chunks from the first file. Caller supplies a fallback
        # via guess_chunk_shape when this is None.
        self.source_chunks = first.chunks

    def __call__(self, slices):
        t = slices[0]
        rest = slices[1:]
        parts = []
        for i, ds in enumerate(self._datasets):
            f0 = int(self._cum[i])
            f1 = int(self._cum[i + 1])
            if f1 <= t.start or f0 >= t.stop:
                continue
            local = (slice(max(0, t.start - f0), min(f1 - f0, t.stop - f0)),) + rest
            parts.append(ds[local])
        if not parts:
            return np.empty((0,) + self.shape[1:], dtype=self.dtype)
        return np.concatenate(parts, axis=0)


class _ConcatTimeSourceUnstaggered(_ConcatTimeSource):
    """
    Variant of ``_ConcatTimeSource`` that exposes a virtual unstaggered view
    of a source variable that is staggered on a single spatial axis (e.g.
    WRF's U is staggered on the x axis, V on the y axis).

    The exposed shape is the file shape with ``stagger_axis`` reduced by 1.
    On read, the wrapper extends the slice on ``stagger_axis`` by 1 and applies
    the trapezoidal-mean unstagger.
    """

    def __init__(self, h5_files: list, var_name: str, file_lens: list, stagger_axis: int):
        super().__init__(h5_files, var_name, file_lens)
        if stagger_axis <= 0 or stagger_axis >= len(self.shape):
            raise ValueError(f'stagger_axis must be a spatial axis: got {stagger_axis}')
        self._stagger_axis = stagger_axis
        # Unstaggered shape is shape with stagger_axis - 1.
        self.shape = (
            self.shape[:stagger_axis]
            + (self.shape[stagger_axis] - 1,)
            + self.shape[stagger_axis + 1:]
        )
        # Source chunks aren't critical for correctness; let the caller fall
        # back to ``rechunkit.guess_chunk_shape`` when None.
        self.source_chunks = None

    def __call__(self, slices):
        # Extend the staggered axis slice by 1 to read the staggered range,
        # then unstagger the result along that axis.
        ax = self._stagger_axis
        s = slices[ax]
        extended = slice(s.start, s.stop + 1)
        modified = slices[:ax] + (extended,) + slices[ax + 1:]
        raw = super().__call__(modified)
        lo = (slice(None),) * ax + (slice(None, -1),)
        hi = (slice(None),) * ax + (slice(1, None),)
        return (raw[lo] + raw[hi]) / 2.0


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

    file_glob_pattern = '*'
    """Glob pattern for finding source files in directories. Override in subclasses."""

    def __init__(self, input_paths: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]]):
        if isinstance(input_paths, (str, pathlib.Path)):
            input_paths = [input_paths]

        expanded = []
        for p in input_paths:
            p = pathlib.Path(p)
            if p.is_dir():
                found = sorted(p.glob(self.file_glob_pattern))
                if not found:
                    raise FileNotFoundError(
                        f'No files matching {self.file_glob_pattern!r} found in directory: {p}'
                    )
                expanded.extend(found)
            else:
                expanded.append(p)
        self.input_paths = sorted(expanded)

        for p in self.input_paths:
            if not p.exists():
                raise FileNotFoundError(f'File not found: {p}')

        self._init_metadata()

    def _init_metadata(self):
        """
        Derive all metadata from the source files by calling subclass methods.
        """
        # Region (multi-region WVT) state. Initialized here so it always exists
        # for every subclass — including those that override _init_variables
        # (e.g. ERA5) and never populate it. Populated by the base
        # _init_variables for region_aware fields; a no-op elsewhere.
        self._field_region_size = {}
        self._n_wvt_regions = None

        self._init_source_metadata()
        self._init_time()
        self._init_variables()
        self._compute_bbox_geographic()

    def _init_source_metadata(self):
        """
        Extract CRS and spatial coordinates from source files.

        Default implementation opens the first file and calls _parse_crs and
        _parse_spatial_coords. Override for sources with non-standard layouts
        (e.g., one variable per file).

        When input files have different spatial extents, self.x and self.y
        are set to the union grid and self._heterogeneous_grids is set True.
        """
        with h5py.File(self.input_paths[0], 'r') as h5:
            self.crs = self._parse_crs(h5)
            spatial_first = self._parse_spatial_coords(h5)

        self.x = spatial_first['x']
        self.y = spatial_first['y']
        self._heterogeneous_grids = False

        if len(self.input_paths) > 1:
            with h5py.File(self.input_paths[-1], 'r') as h5:
                spatial_last = self._parse_spatial_coords(h5)
            if not (np.array_equal(self.y, spatial_last['y']) and
                    np.array_equal(self.x, spatial_last['x'])):
                self.y = np.union1d(self.y, spatial_last['y'])
                self.x = np.union1d(self.x, spatial_last['x'])
                self._heterogeneous_grids = True

    def _init_time(self):
        """
        Build self.times, self._file_time_map, and self._raw_to_unique from all input files.

        When files have overlapping timesteps, _raw_to_unique maps each raw
        timestep index to its position in the deduplicated self.times array,
        or -1 for duplicates that should be skipped.
        """

        def _get_times(path):
            with h5py.File(path, 'r') as h5:
                return self._parse_time(h5)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            times_list = list(executor.map(_get_times, self.input_paths))

        file_time_map = []
        all_times = []
        global_idx = 0

        for path, times in zip(self.input_paths, times_list):
            file_time_map.append((path, times, global_idx))
            all_times.append(times)
            global_idx += len(times)

        combined = np.concatenate(all_times)
        _, unique_idx = np.unique(combined, return_index=True)
        unique_idx.sort()
        self.times = combined[unique_idx]
        self._file_time_map = file_time_map

        # Map raw timestep indices to unique time indices (-1 = duplicate)
        unique_set = set(unique_idx.tolist())
        time_to_unique = {t: i for i, t in enumerate(self.times)}
        self._raw_to_unique = np.full(len(combined), -1, dtype='int64')
        for raw_i in unique_set:
            self._raw_to_unique[raw_i] = time_to_unique[combined[raw_i]]

    def _init_variables(self):
        """
        Determine which mapped variables are available in the source files.

        Mapping entries with a primary ``source_vars`` are checked first. If
        all primary source vars exist in the wrfout, the entry is used as-is.
        Otherwise, if the entry declares ``fallback_source_vars``, those are
        checked and (if satisfied) promoted into the active source set with
        the optional ``fallback_transform``. This lets the same cfdb output
        name be served either by a native passthrough field or by an offline
        computation from 3D source fields, depending on which the wrfout has.
        """
        mapping = self._get_variable_mapping()

        available = {}
        with h5py.File(self.input_paths[0], 'r') as h5:
            source_vars = set(h5.keys())
            for key, info in mapping.items():
                if all(sv in source_vars for sv in info['source_vars']):
                    available[key] = info
                elif 'fallback_source_vars' in info and \
                        all(sv in source_vars for sv in info['fallback_source_vars']):
                    promoted = dict(info)
                    promoted['source_vars'] = info['fallback_source_vars']
                    # Defensive .get() on both: tolerates a future entry that
                    # omits 'transform' or 'fallback_transform' entirely.
                    promoted['transform'] = info.get('fallback_transform', info.get('transform'))
                    available[key] = promoted

            # Detect region-dimensioned (multi-region WVT) fields, using the
            # still-open file. A region_aware field is region-dimensioned iff its
            # NATIVE (unpromoted) primary source var carries a `wvt_regions`
            # named HDF5 dimension at axis 1. Probe the native primary from
            # ``mapping`` (not ``available``): after fallback promotion the
            # active primary may differ (e.g. PWAT_TR -> qv_tr).
            for key, info in available.items():
                if not info.get('region_aware'):
                    continue
                native_primary = mapping[key]['source_vars'][0]
                if native_primary not in source_vars:
                    continue
                n = self._region_axis_size(h5[native_primary])
                if n is not None:
                    self._field_region_size[key] = n

        self.variables = available

        if self._field_region_size:
            sizes = set(self._field_region_size.values())
            if len(sizes) != 1:
                raise ValueError(
                    f'Inconsistent WVT region counts across fields: {self._field_region_size}'
                )
            self._n_wvt_regions = sizes.pop()

    @staticmethod
    def _region_axis_size(h5ds):
        """
        Return the region count N if ``h5ds`` carries a ``wvt_regions`` named
        HDF5 dimension scale at axis 1, else None.

        Detection is by dimension NAME, not ndim: a future natively-3D
        region_aware field would also be 4D in a single-region run, where a bare
        ndim test would misread the vertical level count as the region count.
        WRF/netCDF always attaches the dimension scale (axis path
        ``/wvt_regions``).
        """
        if h5ds.ndim < 2:
            return None
        dim = h5ds.dims[1]
        if len(dim) == 0:
            return None
        name = dim[0].name.rsplit('/', 1)[-1]
        if name != 'wvt_regions':
            return None
        # Region fields are (time, wvt_regions, y, x). Corroborate the layout.
        assert h5ds.ndim == 4, (
            f'{h5ds.name!r} has a wvt_regions axis but ndim={h5ds.ndim} (expected 4)'
        )
        return int(h5ds.shape[1])

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
    # Overridable coordinate naming
    # ------------------------------------------------------------------

    x_coord_name = 'x'
    y_coord_name = 'y'

    def _create_spatial_coords(self, ds, filtered_x, filtered_y):
        """
        Create spatial coordinates on the cfdb dataset.

        Override in subclasses to use different coordinate names
        (e.g., 'latitude'/'longitude' instead of 'x'/'y').
        """
        creator = ds.create.coord
        creator.generic(self.y_coord_name, data=filtered_y.astype('float64'), axis='y', step=True)
        creator.generic(self.x_coord_name, data=filtered_x.astype('float64'), axis='x', step=True)

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
        - 'height': float, 'levels', or 'soil' — height above ground in meters,
          'levels' for variables requiring vertical interpolation to target_levels,
          or 'soil' for soil-layer variables using a depth coordinate
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

    def _get_dataset_attrs(self) -> Dict[str, object]:
        """
        Return dataset-level CF attributes for the cfdb output.

        Subclasses should override and call super() to add source-specific
        attributes (e.g., model version, physics parameters).
        """

        filenames = ', '.join(p.name for p in self.input_paths)
        now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

        return {
            'Conventions': 'CF-1.11',
            'history': f'{now} Converted using cfdb-ingest {cfdb_ingest.__version__}',
            'source_files': filenames,
        }

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

    def _get_soil_depths(self) -> Optional[np.ndarray]:
        """
        Return soil depth coordinate values in meters (ascending), or None
        if the source has no soil data. Override in subclasses that support soil.
        """
        return None

    def convert(
        self,
        cfdb_path: Union[str, pathlib.Path],
        variables: Optional[List[str]] = None,
        start_date: Union[str, np.datetime64, None] = None,
        end_date: Union[str, np.datetime64, None] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        target_levels: Optional[List[float]] = None,
        vertical_coord: str = 'height',
        max_mem: int = 2**27,
        chunk_shape: Optional[Tuple[int, ...]] = None,
        dataset_type: str = 'grid',
        **cfdb_kwargs,
    ):
        """
        Convert source files to a cfdb dataset.

        Variables are stored with coordinates appropriate to their type:
        - Surface variables (height is a float): (time, y, x)
        - Level-interpolated variables (height='levels'): (time, <vertical_coord>, y, x)
        - Soil variables (height='soil'): (time, depth, y, x)

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
            Target levels for level-interpolated variables. Interpretation
            depends on vertical_coord: height in meters or pressure in Pa.
        vertical_coord : str
            Name of the vertical coordinate for level-interpolated variables.
            'height' (default) or 'pressure'.
        max_mem : int
            Memory budget in bytes for rechunkit read buffers.
        chunk_shape : tuple of ints or None
            Output chunk shape. For 4D variables: (time, z, y, x).
            For 3D surface variables: (1, ny, nx) is used automatically.
            Defaults to (1, 1, ny, nx) for 4D.
        dataset_type : str
            Passed to cfdb.open_dataset.
        **cfdb_kwargs
            Extra kwargs for cfdb.open_dataset (e.g., compression).
        """
        self._vertical_coord = vertical_coord

        var_keys = self.resolve_variables(variables)

        has_level_interp = any(self.variables[k]['height'] == 'levels' for k in var_keys)
        if has_level_interp and target_levels is None:
            raise ValueError('target_levels is required when converting level-interpolated variables.')

        has_soil = any(self.variables[k]['height'] == 'soil' for k in var_keys)
        soil_depths = None
        if has_soil:
            soil_depths = self._get_soil_depths()
            if soil_depths is None:
                raise ValueError('Soil variables requested but source has no soil depth data.')

        # Region (multi-region WVT) guards. region_keys = requested region_aware
        # fields that resolved to a region-dimensioned native source.
        region_keys = [k for k in var_keys
                       if self.variables[k].get('region_aware') and self._field_region_size.get(k)]
        if region_keys:
            if self._heterogeneous_grids:
                # The per-file / per-timestep fallbacks index [time, y, x] and
                # would treat the region axis as y — silent corruption. WVT runs
                # are single-domain, so this never triggers in practice.
                raise NotImplementedError(
                    'Multi-region WVT fields are not supported with heterogeneous grids.'
                )
            # A requested region_aware field that did NOT resolve to a region
            # source (e.g. fell back to its 3D fallback) while others did would
            # silently drop the region axis — refuse rather than mislead.
            dropped = [k for k in var_keys
                       if self.variables[k].get('region_aware') and not self._field_region_size.get(k)]
            if dropped:
                raise ValueError(
                    f'Region-aware field(s) {dropped} lack a native wvt_regions axis while other '
                    f'fields are region-dimensioned (N={self._n_wvt_regions}); native region '
                    f'outputs are required (the fallback path cannot reconstruct per-region values).'
                )

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

        # Classify variables into coordinate groups
        level_vars = {}      # cfdb_name -> [var_key, ...]
        surface_vars = {}    # cfdb_name -> [var_key, ...]  (vars at a fixed height)
        soil_vars = {}       # cfdb_name -> [var_key, ...]
        region_vars = {}     # cfdb_name -> [var_key, ...]  (multi-region WVT, region axis)

        sorted_levels = np.array(sorted(target_levels), dtype='float64') if target_levels else None

        for var_key in var_keys:
            info = self.variables[var_key]
            cfdb_name = info['cfdb_name']
            height_spec = info['height']

            # Region-dimensioned fields take priority over their nominal height:
            # they are 2D-surface quantities (height 0.0) carrying an extra
            # wvt_regions axis. Gated on a recorded region size, so a region_aware
            # field in a single-region (3D) file falls through to surface_vars
            # (current behavior, byte-identical).
            if self._field_region_size.get(var_key):
                region_vars.setdefault(cfdb_name, []).append(var_key)
            elif height_spec == 'levels':
                level_vars.setdefault(cfdb_name, []).append(var_key)
            elif height_spec == 'soil':
                soil_vars.setdefault(cfdb_name, []).append(var_key)
            else:
                surface_vars.setdefault(cfdb_name, []).append(var_key)

        # Resolve cfdb_name conflicts between surface and level/soil groups.
        # When a cfdb_name appears in both groups, suffix the surface variant
        # with its height to disambiguate (e.g. air_temperature_2m).
        conflicting = set(surface_vars) & (set(level_vars) | set(soil_vars))
        for name in conflicting:
            var_key_list = surface_vars.pop(name)
            h = float(self.variables[var_key_list[0]]['height'])
            new_name = f'{name}_{int(h)}m'
            surface_vars[new_name] = var_key_list

        # Collect unique fixed heights needed for surface variables
        fixed_heights = set()
        for cfdb_name, var_key_list in surface_vars.items():
            h = float(self.variables[var_key_list[0]]['height'])
            fixed_heights.add(h)

        # Default chunk shape
        chunk_4d = chunk_shape if chunk_shape is not None else (1, 1, ny, nx)

        has_multi_level = sorted_levels is not None and level_vars

        with cfdb.open_dataset(cfdb_path, 'n', dataset_type=dataset_type, **cfdb_kwargs) as ds:
            # Create coordinates
            ds.create.coord.time(data=filtered_times, step=True)
            self._create_spatial_coords(ds, filtered_x, filtered_y)

            if has_multi_level:
                if vertical_coord == 'pressure':
                    ds.create.coord.pressure(data=sorted_levels)
                else:
                    ds.create.coord.height(data=sorted_levels)

            if soil_depths is not None and soil_vars:
                ds.create.coord.depth(data=soil_depths, axis=None)

            # Create named height coordinates for fixed-height surface variables.
            # Each distinct height gets its own length-1 coordinate (e.g. height_2m).
            # If there's also a multi-level vertical coord, these get axis=None
            # to avoid conflicting with the axis='Z' on pressure/height.
            for h in sorted(fixed_heights):
                coord_name = f'height_{int(h)}m'
                axis = None if (has_multi_level or len(fixed_heights) > 1) else 'z'
                ds.create.coord.generic(
                    coord_name,
                    data=np.array([h], dtype='float64'),
                    axis=axis,
                )

            # Region (multi-region WVT) coordinate: integer source-region index
            # 1..N. axis=None (categorical, not a physical Z). Created before the
            # region data-var loop (cfdb requires named coords to pre-exist).
            if region_vars:
                region_coord = ds.create.coord.generic(
                    'wvt_region',
                    data=np.arange(1, self._n_wvt_regions + 1, dtype='int64'),
                    axis=None,
                )
                region_coord.attrs['long_name'] = 'Water Vapour Tracer Source Region'

            # Set CRS
            ds.create.crs.from_user_input(self.crs, x_coord=self.x_coord_name, y_coord=self.y_coord_name)

            # Set dataset attributes
            for key, value in self._get_dataset_attrs().items():
                ds.attrs[key] = value

            # Classify into processing groups
            rechunkit_items = []
            accumulation_items = []
            multi_rechunkit_items = []
            batch_items = []

            def _classify(item, info):
                transform = info.get('transform')
                if transform == 'accumulation_increment':
                    accumulation_items.append(item)
                elif len(info['source_vars']) == 1 and (
                    transform is None or self._get_block_transform(transform) is not None
                ):
                    # Single source: simple rechunker handles both no-transform
                    # and registered block-transform cases. The transform (if any)
                    # is applied per yielded block inside _populate_with_rechunkit.
                    rechunkit_items.append(item)
                elif self._get_block_transform(transform) is not None:
                    # Multi-source with registered block transform.
                    multi_rechunkit_items.append(item)
                else:
                    batch_items.append(item)

            # Level-interpolated variables: (time, <vertical_coord>, y, x)
            level_coord_names = ('time', vertical_coord, self.y_coord_name, self.x_coord_name)
            for cfdb_name, var_key_list in level_vars.items():
                data_var = self._create_cfdb_data_var(ds, cfdb_name, level_coord_names, chunk_4d)
                for var_key in var_key_list:
                    level_indices = list(range(len(sorted_levels)))
                    _classify((var_key, data_var, level_indices), self.variables[var_key])

            # Surface variables: (time, height_Xm, y, x)
            for cfdb_name, var_key_list in surface_vars.items():
                h = float(self.variables[var_key_list[0]]['height'])
                coord_name = f'height_{int(h)}m'
                surface_coord_names = ('time', coord_name, self.y_coord_name, self.x_coord_name)
                data_var = self._create_cfdb_data_var(ds, cfdb_name, surface_coord_names, chunk_4d)
                for var_key in var_key_list:
                    # index 0 of the length-1 height coord
                    _classify((var_key, data_var, [0]), self.variables[var_key])

            # Soil variables: (time, depth, y, x)
            soil_coord_names = ('time', 'depth', self.y_coord_name, self.x_coord_name)
            for cfdb_name, var_key_list in soil_vars.items():
                data_var = self._create_cfdb_data_var(ds, cfdb_name, soil_coord_names, chunk_4d)
                for var_key in var_key_list:
                    depth_indices = list(range(len(soil_depths)))
                    _classify((var_key, data_var, depth_indices), self.variables[var_key])

            # Region (multi-region WVT) variables: (time, wvt_region, y, x).
            # Mechanically identical to level/soil — the region axis is written
            # via vert_indices=range(N) by the existing populate paths.
            region_coord_names = ('time', 'wvt_region', self.y_coord_name, self.x_coord_name)
            for cfdb_name, var_key_list in region_vars.items():
                data_var = self._create_cfdb_data_var(ds, cfdb_name, region_coord_names, chunk_4d)
                for var_key in var_key_list:
                    region_indices = list(range(self._field_region_size[var_key]))
                    _classify((var_key, data_var, region_indices), self.variables[var_key])

            # Map requested (sorted, ascending) levels onto source native level indices so
            # the rechunkit path can subset/reorder the level axis. None keeps the original
            # all-native-levels path untouched (computed only when the request differs).
            source_level_sel = None
            if has_multi_level:
                native_levels = self._native_level_values()
                if native_levels is not None and not (
                    len(native_levels) == len(sorted_levels)
                    and np.allclose(np.sort(native_levels), sorted_levels)
                ):
                    source_level_sel = []
                    for lev in sorted_levels:
                        idx = np.where(np.isclose(native_levels, lev))[0]
                        if len(idx) == 0:
                            raise ValueError(
                                f'Requested level {lev} is not among the source native levels '
                                f'{np.sort(native_levels).tolist()}; interpolation is not supported '
                                f'for non-transform pressure-level variables.'
                            )
                        source_level_sel.append(int(idx[0]))

            # Process each group with its optimal strategy
            for var_key, data_var, vert_indices in rechunkit_items:
                self._setup_populate(var_key, target_levels)
                # Region passthrough fields share the 4D rechunkit branch with
                # native pressure-level vars; their middle axis is regions, not
                # levels, so the level-subset selector must not apply (it is
                # always None for WRF today, but be explicit).
                sls = None if self._field_region_size.get(var_key) else source_level_sel
                self._populate_with_rechunkit(data_var, var_key, time_mask, spatial_slice, max_mem, vert_indices,
                                              chunk_4d=chunk_4d, source_level_sel=sls,
                                              filtered_y=filtered_y, filtered_x=filtered_x)

            for var_key, data_var, vert_indices in accumulation_items:
                self._setup_populate(var_key, target_levels)
                self._populate_with_accumulation(data_var, var_key, time_mask, spatial_slice, chunk_4d, max_mem, vert_indices,
                                                  filtered_y=filtered_y, filtered_x=filtered_x)

            if multi_rechunkit_items:
                for var_key, _, _ in multi_rechunkit_items:
                    self._setup_populate(var_key, target_levels)
                self._populate_with_multi_rechunker(multi_rechunkit_items, time_mask, spatial_slice, chunk_4d, max_mem,
                                                    filtered_y=filtered_y, filtered_x=filtered_x)

            if batch_items:
                for var_key, _, _ in batch_items:
                    self._setup_populate(var_key, target_levels)
                self._populate_batch_per_timestep(batch_items, time_mask, spatial_slice, max_mem,
                                                  chunk_4d=chunk_4d,
                                                  filtered_y=filtered_y, filtered_x=filtered_x)

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

    def _get_file_spatial_mapping(self, h5, target_y, target_x):
        """
        Compute per-file spatial slice and target offset for heterogeneous grids.

        The returned slices index into the file's parsed (ascending) coordinate
        arrays, matching how _bbox_to_indices works. Subclasses that reverse
        axes (e.g., ERA5 lat) must convert to raw HDF5 indices themselves.

        Returns
        -------
        file_y_slice, file_x_slice : slice
            Slices into the file's parsed (ascending) spatial arrays.
        y_offset, x_offset : int
            Starting index in the target (cfdb) grid where this file's data goes.
        """
        file_spatial = self._parse_spatial_coords(h5)
        file_y, file_x = file_spatial['y'], file_spatial['x']

        # Overlap between file grid and target grid (round to avoid float mismatch)
        target_y_r = np.round(target_y, 8)
        target_x_r = np.round(target_x, 8)
        file_y_r = np.round(file_y, 8)
        file_x_r = np.round(file_x, 8)

        y_mask = np.isin(file_y_r, target_y_r)
        x_mask = np.isin(file_x_r, target_x_r)

        y_idx = np.where(y_mask)[0]
        x_idx = np.where(x_mask)[0]

        if len(y_idx) == 0 or len(x_idx) == 0:
            return None, None, 0, 0

        file_y_slice = slice(int(y_idx[0]), int(y_idx[-1]) + 1)
        file_x_slice = slice(int(x_idx[0]), int(x_idx[-1]) + 1)

        y_offset = int(np.searchsorted(target_y_r, file_y_r[y_idx[0]]))
        x_offset = int(np.searchsorted(target_x_r, file_x_r[x_idx[0]]))

        return file_y_slice, file_x_slice, y_offset, x_offset

    def _get_block_transform(self, transform_name):
        """
        Return a callable that applies a block-mode (time-batched) transform,
        or None if no such transform is registered.

        Block transforms have signature ``fn(sources, y_sl, x_sl, block_cache) -> ndarray``,
        where ``sources`` is a dict mapping source variable name to an ndarray of
        shape ``(N, ny, nx)``. Subclasses register transforms via ``_BLOCK_TRANSFORMS``.
        """
        return None

    def _files_for_var(self, var_key):
        """
        Return a time-ordered list of source file paths contributing to
        ``var_key``. Default: every input file (correct for WRF, where each
        wrfout file contains every variable). Subclasses with one-var-per-file
        layouts (e.g. ERA5) override this.
        """
        return list(self.input_paths)

    def _make_source(self, src_name, h5_files, file_lens):
        """
        Build a virtual rechunker source for ``src_name`` over ``h5_files``.

        Default returns a plain ``_ConcatTimeSource``. Subclasses override to
        wrap staggered source variables (e.g. WRF U is x-staggered, V is
        y-staggered) so the rechunker sees the unstaggered shape.
        """
        return _ConcatTimeSource(h5_files, src_name, file_lens)

    def _post_block_transform(self, block, var_key, source_ndim):
        """
        Apply source-format-specific transforms to a rechunker-yielded block
        before it is written to cfdb. Default: identity. ERA5 overrides for
        latitude reversal and ``geopotential_to_height``. Phase-2 work will
        unify this with the ``_BLOCK_TRANSFORMS`` registry.
        """
        return block

    def _create_cfdb_data_var(self, ds, cfdb_name, coord_names, chunk_shape):
        """
        Create a cfdb data variable using the template method for cfdb_name.

        Template methods (e.g., ds.create.data_var.air_temp) auto-set the
        appropriate dtype, encoding, and CF attributes from cfdb's defaults.
        Falls back to generic float32 for names without a cfdb template.
        """
        creator = ds.create.data_var
        template = getattr(creator, cfdb_name, None)
        if template is not None:
            return template(coord_names, chunk_shape=chunk_shape)

        return creator.generic(cfdb_name, coord_names, dtype='float32', chunk_shape=chunk_shape)

    def _setup_populate(self, var_key, target_levels):
        """Hook called before populating a data variable. Override as needed."""
        pass

    def _native_level_values(self):
        """
        Native vertical level values (same units as ``target_levels``) in source-file
        axis order, or None when the source has no native level axis (e.g. transform-based
        vertical interpolation, as in WRF). Subclasses with on-disk pressure levels
        override this so the rechunkit path can subset/reorder to requested levels.
        """
        return None

    def _accumulation_source_sum(self, h5, source_vars, time_idx, spatial_slice):
        """
        Sum source_vars at a given timestep.

        Hook for subclasses to apply source-specific corrections (e.g., WRF
        reconstructs wrapped accumulators from bucket counters when
        ``BUCKET_MM > 0``). Base implementation is the plain sum.
        """
        y_sl, x_sl = spatial_slice
        return sum(h5[sv][time_idx, y_sl, x_sl].astype('float64') for sv in source_vars)

    def _read_accumulation_increment(self, h5, var_key, time_idx, spatial_slice):
        """
        Compute increment from accumulated source variables.

        Sums all source_vars at current and previous timestep, returns the
        difference. Uses self._prev_accum_total for cross-file boundaries.
        Returns NaN for the very first overall timestep.

        Negative increments are clipped to 0: the only current consumers are
        precipitation accumulators which are physically non-negative, and
        small negative deltas occasionally appear when WRF nudging or
        two-way feedback retroactively reduces a parent-grid accumulator.
        """
        info = self.variables[var_key]
        source_vars = info['source_vars']

        total = self._accumulation_source_sum(h5, source_vars, time_idx, spatial_slice)

        if time_idx > 0:
            prev = self._accumulation_source_sum(h5, source_vars, time_idx - 1, spatial_slice)
            result = total - prev
        elif self._prev_accum_total is not None:
            result = total - self._prev_accum_total
        else:
            result = np.full_like(total, np.nan)

        np.maximum(result, 0.0, out=result, where=~np.isnan(result))
        return result.astype('float32')

    def _populate_with_rechunkit(self, data_var, var_key, time_mask, spatial_slice, max_mem, vert_indices,
                                 chunk_4d=None, filtered_y=None, filtered_x=None, source_level_sel=None):
        """
        Populate a simple (no-transform, single source var) data variable using
        a single rechunkit call across all input files.

        Opens every file contributing to ``var_key`` at once via ``ExitStack``
        and presents them to ``rechunkit.rechunker`` as one virtual time-major
        array (``_ConcatTimeSource``). Rechunker accumulates source reads
        until it has a full ``target_chunks`` block to yield, so each cfdb
        chunk is written exactly once regardless of source file granularity.

        ``chunk_4d``, when provided, sets the time dim of rechunkit's
        ``target_chunks`` so blocks align with the cfdb output chunk shape.

        Heterogeneous-grid runs fall back to the per-file path until the
        cross-file path supports per-file spatial remapping.
        """
        if self._heterogeneous_grids:
            return self._populate_with_rechunkit_per_file(
                data_var, var_key, time_mask, spatial_slice, max_mem, vert_indices,
                chunk_4d=chunk_4d, filtered_y=filtered_y, filtered_x=filtered_x,
                source_level_sel=source_level_sel,
            )

        src_var = self.variables[var_key]['source_vars'][0]
        target_t = chunk_4d[0] if chunk_4d is not None else 1
        paths = self._files_for_var(var_key)
        if not paths:
            return

        # output_map: unique_time_idx -> output_time_idx (compacted by time_mask)
        output_map = {}
        out_idx = 0
        for u_idx in range(len(time_mask)):
            if time_mask[u_idx]:
                output_map[u_idx] = out_idx
                out_idx += 1

        # raw_to_unique: maps each raw global time index (concatenated across
        # files) to a unique time index, or -1 for duplicates from overlapping
        # files. Reuses ``self._raw_to_unique`` when paths match input_paths.
        raw_to_unique = self._get_raw_to_unique(paths)

        with ExitStack() as stack:
            h5_files = [stack.enter_context(h5py.File(p, 'r')) for p in paths]
            file_lens = [int(h5[src_var].shape[0]) for h5 in h5_files]
            source = self._make_source(src_var, h5_files, file_lens)
            source_chunks = source.source_chunks or rechunkit.guess_chunk_shape(
                source.shape, source.dtype.itemsize, max_mem,
            )

            y_sl, x_sl = spatial_slice
            spatial_axes = source.shape[1:]
            level_pick = None  # in-block level reindexing for subset/reorder (4D only)
            if len(spatial_axes) == 2:
                y_start, y_stop, _ = y_sl.indices(spatial_axes[0])
                x_start, x_stop, _ = x_sl.indices(spatial_axes[1])
                sel = (slice(0, source.shape[0]),
                       slice(y_start, y_stop), slice(x_start, x_stop))
                target_chunks = (target_t, y_stop - y_start, x_stop - x_start)
            elif len(spatial_axes) == 3:
                # 4D source: (T, Z, Y, X)
                nz = spatial_axes[0]
                y_start, y_stop, _ = y_sl.indices(spatial_axes[1])
                x_start, x_stop, _ = x_sl.indices(spatial_axes[2])
                if source_level_sel is not None:
                    # Honor target_levels: read only the contiguous bounding span of the
                    # requested source levels, then pick (and reorder) within each block.
                    # Handles single levels (one slice) and non-contiguous subsets.
                    lo, hi = min(source_level_sel), max(source_level_sel) + 1
                    z_slice = slice(lo, hi)
                    level_pick = [i - lo for i in source_level_sel]
                    nz_read = hi - lo
                else:
                    z_slice = slice(0, nz)
                    nz_read = nz
                sel = (slice(0, source.shape[0]), z_slice,
                       slice(y_start, y_stop), slice(x_start, x_stop))
                target_chunks = (target_t, nz_read, y_stop - y_start, x_stop - x_start)
            else:
                raise ValueError(f'Unsupported source ndim {len(source.shape)} for {var_key!r}')

            # Optional block transform for single-source-with-transform vars
            # (e.g. ERA5 ``geopotential_to_height``). Applied per yielded block,
            # after layout corrections from ``_post_block_transform``.
            transform_name = self.variables[var_key].get('transform')
            block_fn = self._get_block_transform(transform_name)

            for write_slices, data in rechunkit.rechunker(
                source, source.shape, source.dtype,
                source_chunks, target_chunks, max_mem, sel=sel,
            ):
                block = self._post_block_transform(data, src_var, len(source.shape))
                if block_fn is not None:
                    block = block_fn({src_var: block}, y_sl, x_sl, {})
                if level_pick is not None:
                    block = block[:, level_pick]
                n_block = write_slices[0].stop - write_slices[0].start
                t_outs = [None] * n_block
                for i in range(n_block):
                    raw = write_slices[0].start + i
                    u = int(raw_to_unique[raw])
                    if u != -1 and u in output_map:
                        t_outs[i] = output_map[u]
                self._write_block_from_t_outs(
                    data_var, block, t_outs, vert_indices, None, None,
                )

    def _get_raw_to_unique(self, paths):
        """
        Map raw global time indices (concatenated across ``paths``) to unique
        time indices in ``self.times``, or -1 for duplicates from overlapping
        files. Reuses ``self._raw_to_unique`` when ``paths == self.input_paths``;
        otherwise rebuilds for the given paths (cheap — only Times reads).
        """
        if hasattr(self, '_raw_to_unique') and list(paths) == list(self.input_paths):
            return self._raw_to_unique

        all_times = []
        for path in paths:
            with h5py.File(path, 'r') as h5:
                all_times.append(self._parse_time(h5))
        combined = np.concatenate(all_times)
        time_to_unique = {t: i for i, t in enumerate(self.times)}
        result = np.full(len(combined), -1, dtype='int64')
        seen = set()
        for raw_i, t in enumerate(combined):
            u = time_to_unique.get(t, -1)
            if u == -1 or u in seen:
                continue
            seen.add(u)
            result[raw_i] = u
        return result

    def _populate_with_rechunkit_per_file(self, data_var, var_key, time_mask, spatial_slice,
                                          max_mem, vert_indices, chunk_4d=None,
                                          filtered_y=None, filtered_x=None, source_level_sel=None):
        """
        Legacy per-file rechunkit populate. Retained for the heterogeneous-grid
        case until the cross-file path supports per-file spatial remapping.

        ``source_level_sel`` (target-level subsetting) is accepted for signature
        parity but not yet supported here: this path is 3D-only and is reached only
        for heterogeneous grids, which ERA5 pressure-level conversion never uses.
        """
        src_var = self.variables[var_key]['source_vars'][0]
        y_sl, x_sl = spatial_slice

        output_map = {}
        out_idx = 0
        for u_idx in range(len(time_mask)):
            if time_mask[u_idx]:
                output_map[u_idx] = out_idx
                out_idx += 1

        target_t = chunk_4d[0] if chunk_4d is not None else 1

        raw_offset = 0
        for path, file_times, _ in self._file_time_map:
            n_file_times = len(file_times)

            file_mask = []
            for local_t in range(n_file_times):
                u_idx = self._raw_to_unique[raw_offset + local_t]
                if u_idx != -1 and u_idx in output_map:
                    file_mask.append(local_t)

            if not file_mask:
                raw_offset += n_file_times
                continue

            t_start = file_mask[0]
            t_stop = file_mask[-1] + 1

            with h5py.File(path, 'r') as h5:
                if self._heterogeneous_grids and filtered_y is not None:
                    fy_sl, fx_sl, y_off, x_off = self._get_file_spatial_mapping(
                        h5, filtered_y, filtered_x)
                    if fy_sl is None:
                        raw_offset += n_file_times
                        continue
                else:
                    fy_sl, fx_sl = y_sl, x_sl
                    y_off, x_off = 0, 0

                h5_var = h5[src_var]
                source_chunks = h5_var.chunks or rechunkit.guess_chunk_shape(
                    h5_var.shape, h5_var.dtype.itemsize, max_mem
                )

                y_start, y_stop, _ = fy_sl.indices(h5_var.shape[1])
                x_start, x_stop, _ = fx_sl.indices(h5_var.shape[2])
                ny = y_stop - y_start
                nx = x_stop - x_start
                sel = (slice(t_start, t_stop), slice(y_start, y_stop), slice(x_start, x_stop))
                target_chunks = (target_t, ny, nx)

                y_write = slice(y_off, y_off + ny) if y_off > 0 or self._heterogeneous_grids else None
                x_write = slice(x_off, x_off + nx) if x_off > 0 or self._heterogeneous_grids else None

                for write_slices, data in rechunkit.rechunker(
                    h5_var.__getitem__, h5_var.shape, h5_var.dtype,
                    source_chunks, target_chunks, max_mem, sel=sel,
                ):
                    self._write_block(
                        data_var, data, write_slices, t_start, raw_offset, output_map,
                        vert_indices, y_write, x_write,
                    )

            raw_offset += n_file_times

    def _write_block(self, data_var, data, write_slices, t_start, raw_offset, output_map,
                     vert_indices, y_write, x_write):
        """
        Write a rechunkit-yielded block to ``data_var``, coalescing into a single
        cfdb call when the block's timesteps map to a contiguous output range,
        and falling back to per-timestep writes when there are gaps (rare —
        only happens at file overlaps). Maps via ``_raw_to_unique`` (WRF-style).
        """
        n_block = write_slices[0].stop - write_slices[0].start
        t_outs = [None] * n_block
        for i, chunk_t in enumerate(range(write_slices[0].start, write_slices[0].stop)):
            local_t = t_start + chunk_t
            u_idx = self._raw_to_unique[raw_offset + local_t]
            if u_idx == -1 or u_idx not in output_map:
                continue
            t_outs[i] = output_map[u_idx]
        self._write_block_from_t_outs(data_var, data, t_outs, vert_indices, y_write, x_write)

    def _write_block_from_t_outs(self, data_var, block, t_outs, vert_indices, y_write, x_write):
        """
        Write a block of data given a per-timestep mapping of output indices.

        ``t_outs[i]`` is the output time index for ``block[i]``, or ``None`` to
        skip that timestep. Fast-paths a single coalesced cfdb call when every
        index is set and they form a contiguous run; falls back to per-timestep
        writes otherwise.
        """
        n_block = len(t_outs)
        if n_block == 0:
            return
        if t_outs[0] is not None and all(
            t_outs[i] is not None and t_outs[i] == t_outs[0] + i
            for i in range(n_block)
        ):
            self._write_data_var_block(
                data_var, block, slice(t_outs[0], t_outs[0] + n_block),
                vert_indices, y_write, x_write,
            )
            return
        for i, t_out in enumerate(t_outs):
            if t_out is None:
                continue
            self._write_data_var(data_var, block[i], t_out, vert_indices, y_write, x_write)

    @staticmethod
    def _write_data_var(data_var, data, output_time_idx, vert_indices, y_write=None, x_write=None):
        """
        Write data for a single timestep at the correct indices.

        Branches on ``data.ndim`` (not ``len(vert_indices)``): 2D data is a
        no-middle-axis surface field; 3D data carries a middle axis (levels /
        soil / wvt_region) and is written one slice per ``vert_indices`` entry.
        Equivalent to the old len(vert_indices) test for surface/level/soil, and
        additionally correct for a length-1 region axis (3D data, single index).
        """
        ys = y_write if y_write is not None else slice(None)
        xs = x_write if x_write is not None else slice(None)
        if data.ndim == 2:
            data_var[(output_time_idx, vert_indices[0], ys, xs)] = data[np.newaxis, np.newaxis, ...]
        else:
            for lev_i, v_idx in enumerate(vert_indices):
                data_var[(output_time_idx, v_idx, ys, xs)] = data[lev_i][np.newaxis, np.newaxis, ...]

    @staticmethod
    def _write_data_var_block(data_var, block, time_slice, vert_indices, y_write=None, x_write=None):
        """
        Coalesced write of a multi-timestep block.

        Branches on ``block.ndim``:
            - ``(N, ny, nx)`` (ndim 3): no middle axis — surface variables.
            - ``(N, K, ny, nx)`` (ndim 4): middle axis (levels / soil /
              wvt_region), written one slice per ``vert_indices`` entry
              (``K == len(vert_indices)``).

        Discriminating on ndim (not ``len(vert_indices)``) keeps surface/level/
        soil behavior identical and additionally handles a length-1 region axis,
        where a 4D ``(N, 1, ny, nx)`` block has a single vert index.
        """
        ys = y_write if y_write is not None else slice(None)
        xs = x_write if x_write is not None else slice(None)
        if block.ndim == 3:
            data_var[(time_slice, vert_indices[0], ys, xs)] = block
        else:
            for lev_i, v_idx in enumerate(vert_indices):
                data_var[(time_slice, v_idx, ys, xs)] = block[:, lev_i]

    def _multi_rechunker(self, sources, shape, dtype, source_chunks, target_chunks, max_mem, sel):
        """
        Generic synchronized multivariable rechunker.
        
        Parameters
        ----------
        sources : dict
            {label: callable} source functions for rechunkit.
        shape, dtype, source_chunks, target_chunks, max_mem, sel : 
            Arguments passed to rechunkit.rechunker. max_mem is the total budget.
            
        Yields
        ------
        slices : tuple
            The output write slices.
        data_blocks : dict
            {label: ndarray} synchronized data blocks.
        """

        per_var_mem = max_mem // len(sources)
        labels = sorted(list(sources.keys()))
        
        generators = [
            rechunkit.rechunker(sources[label], shape, dtype, source_chunks, target_chunks, per_var_mem, sel=sel)
            for label in labels
        ]
        
        for outputs in zip(*generators):
            # All slices should be identical due to rechunkit determinism
            write_slices = outputs[0][0]
            data_blocks = {label: out[1] for label, out in zip(labels, outputs)}
            yield write_slices, data_blocks

    def _populate_with_multi_rechunker(self, items, time_mask, spatial_slice, chunk_4d, max_mem,
                                       filtered_y=None, filtered_x=None):
        """
        Populate one or more transform variables using a single
        ``_multi_rechunker`` call across all input files.

        Opens every file contributing to the union of source variables at once
        via ``ExitStack``, presents each source variable's files as a
        ``_ConcatTimeSource``, and feeds them to ``_multi_rechunker``. Each
        cfdb chunk is written exactly once.

        Each item in ``items`` is ``(var_key, data_var, vert_indices)``. Every
        ``var_key`` must have a registered block transform (see
        ``_get_block_transform``); single-source no-transform items go through
        ``_populate_with_rechunkit`` instead.

        Assumes all source variables share the same shape, dtype, and HDF5
        chunk layout (true for WRF 2D surface fields and ERA5 pressure-level
        VIMF sources). Heterogeneous-grid runs fall back to the per-file path.
        """
        if not items:
            return
        if self._heterogeneous_grids:
            return self._populate_with_multi_rechunker_per_file(
                items, time_mask, spatial_slice, chunk_4d, max_mem,
                filtered_y=filtered_y, filtered_x=filtered_x,
            )

        # Group items by their (source-paths, spatial-shape) signature so each
        # multi_rechunker call sees a consistent shape across sources. Items
        # whose source vars all share the same spatial shape and resolve to the
        # same files can co-batch — e.g. all WRF 2D-surface transforms together,
        # all 3D-source column-integrated together. ERA5's Z_PL/Z_INV split into
        # separate groups because their 'Z' source resolves to different files.
        groups = self._group_items_for_multi_rechunker(items)
        for group in groups:
            self._populate_with_multi_rechunker_group(
                group, time_mask, spatial_slice, chunk_4d, max_mem,
                filtered_y=filtered_y, filtered_x=filtered_x,
            )

    def _group_items_for_multi_rechunker(self, items):
        """
        Partition items into groups whose source vars share spatial shape, so
        each group reads its union of sources in a single ``_multi_rechunker``
        call. Items with overlapping but distinct source-var sets (e.g. WRF
        PWAT / PWAT_TR / VIMF_U / VIMF_V — all 3D) batch together so common
        sources (QVAPOR, P, PB) are read once instead of once per item.

        Per-source-var path consistency is implicit: WRF has every variable in
        every input file, and ERA5 routes its ambiguous-source items
        (Z_PL/Z_INV both with source 'Z') through the single-source path.
        """
        shape_cache = {}

        def _src_spatial(sv):
            if sv in shape_cache:
                return shape_cache[sv]
            paths = self._files_for_var(sv)
            with h5py.File(paths[0], 'r') as h5:
                src = self._make_source(sv, [h5], [int(h5[sv].shape[0])])
                shape_cache[sv] = src.shape[1:]
            return shape_cache[sv]

        groups = {}
        for item in items:
            var_key, _, _ = item
            info = self.variables[var_key]
            spatials = {_src_spatial(sv) for sv in info['source_vars']}
            if len(spatials) != 1:
                raise ValueError(
                    f'{var_key!r}: source variables have inconsistent spatial shapes {spatials}'
                )
            key = next(iter(spatials))
            groups.setdefault(key, []).append(item)
        return list(groups.values())

    def _populate_with_multi_rechunker_group(self, items, time_mask, spatial_slice,
                                              chunk_4d, max_mem,
                                              filtered_y=None, filtered_x=None):
        """Run a single ``_multi_rechunker`` call for a shape-compatible group."""
        # Union of source variable names across all items in this group.
        all_sources = set()
        for var_key, _, _ in items:
            for src in self.variables[var_key]['source_vars']:
                all_sources.add(src)
        all_sources = sorted(all_sources)

        # Per-source-var paths via the same hook used for the single-source
        # path. WRF: every wrfout has every var → all paths. ERA5: var-specific.
        paths_per_src = {sv: self._files_for_var(sv) for sv in all_sources}
        for sv, paths in paths_per_src.items():
            if not paths:
                raise ValueError(f'No source files found for {sv!r}')

        # Pre-compute unique_time_idx -> output_time_idx mapping.
        output_map = {}
        out_idx = 0
        for u_idx in range(len(time_mask)):
            if time_mask[u_idx]:
                output_map[u_idx] = out_idx
                out_idx += 1

        target_t = chunk_4d[0] if chunk_4d is not None else 1

        # Resolve block transform callables once per item.
        item_transforms = []
        for var_key, data_var, vert_indices in items:
            transform_name = self.variables[var_key].get('transform')
            fn = self._get_block_transform(transform_name)
            if fn is None:
                raise RuntimeError(
                    f'No block transform registered for {var_key!r} '
                    f'(transform={transform_name!r}); should not be in multi_rechunker_items.'
                )
            item_transforms.append((var_key, data_var, vert_indices, fn))

        # raw_to_unique mapping for the time axis. All source vars are assumed
        # to have synchronised time coverage (validated below by shape match).
        ref_paths = paths_per_src[all_sources[0]]
        raw_to_unique = self._get_raw_to_unique(ref_paths)

        y_sl, x_sl = spatial_slice

        # Open each unique path once, then point each source var's
        # ``_ConcatTimeSource`` at the relevant subset.
        unique_paths = []
        for sv in all_sources:
            for p in paths_per_src[sv]:
                if p not in unique_paths:
                    unique_paths.append(p)

        with ExitStack() as stack:
            h5_by_path = {p: stack.enter_context(h5py.File(p, 'r')) for p in unique_paths}

            sources = {}
            ref_shape = None
            ref_dtype = None
            ref_source_chunks = None
            for sv in all_sources:
                h5_files = [h5_by_path[p] for p in paths_per_src[sv]]
                file_lens = [int(h5[sv].shape[0]) for h5 in h5_files]
                src = self._make_source(sv, h5_files, file_lens)
                sources[sv] = src
                if ref_shape is None:
                    ref_shape = src.shape
                    ref_dtype = src.dtype
                    ref_source_chunks = src.source_chunks
                elif src.shape != ref_shape:
                    raise ValueError(
                        f'Source-var shape mismatch in multi-rechunker: {sv} has {src.shape}, '
                        f'expected {ref_shape} (all sources must share shape including the time axis)'
                    )

            if ref_source_chunks is None:
                ref_source_chunks = rechunkit.guess_chunk_shape(
                    ref_shape, ref_dtype.itemsize, max_mem,
                )

            spatial_axes = ref_shape[1:]
            if len(spatial_axes) == 2:
                y_start, y_stop, _ = y_sl.indices(spatial_axes[0])
                x_start, x_stop, _ = x_sl.indices(spatial_axes[1])
                sel = (slice(0, ref_shape[0]),
                       slice(y_start, y_stop), slice(x_start, x_stop))
                target_chunks = (target_t, y_stop - y_start, x_stop - x_start)
            elif len(spatial_axes) == 3:
                # 4D source: (T, Z, Y, X)
                nz = spatial_axes[0]
                y_start, y_stop, _ = y_sl.indices(spatial_axes[1])
                x_start, x_stop, _ = x_sl.indices(spatial_axes[2])
                sel = (slice(0, ref_shape[0]), slice(0, nz),
                       slice(y_start, y_stop), slice(x_start, x_stop))
                target_chunks = (target_t, nz, y_stop - y_start, x_stop - x_start)
            else:
                raise ValueError(f'Unsupported source ndim {len(ref_shape)}')

            # Wrap virtual sources as callables for ``_multi_rechunker``.
            source_callables = {sv: src.__call__ for sv, src in sources.items()}

            for write_slices, data_blocks in self._multi_rechunker(
                source_callables, ref_shape, ref_dtype, ref_source_chunks,
                target_chunks, max_mem, sel,
            ):
                # Apply per-source-var post-block transform (e.g. lat reversal)
                # before any block transforms run.
                data_blocks_post = {
                    sv: self._post_block_transform(b, sv, len(ref_shape))
                    for sv, b in data_blocks.items()
                }

                n_block = write_slices[0].stop - write_slices[0].start
                t_outs = [None] * n_block
                for i in range(n_block):
                    raw = write_slices[0].start + i
                    u = int(raw_to_unique[raw])
                    if u != -1 and u in output_map:
                        t_outs[i] = output_map[u]

                block_cache = {}
                for var_key, data_var, vert_indices, fn in item_transforms:
                    item_sources = {
                        sv: data_blocks_post[sv]
                        for sv in self.variables[var_key]['source_vars']
                    }
                    block_data = fn(item_sources, y_sl, x_sl, block_cache)
                    self._write_block_from_t_outs(
                        data_var, block_data, t_outs, vert_indices, None, None,
                    )

    def _populate_with_multi_rechunker_per_file(self, items, time_mask, spatial_slice, chunk_4d, max_mem,
                                                filtered_y=None, filtered_x=None):
        """
        Legacy per-file multi-rechunker. Retained for the heterogeneous-grid
        case until the cross-file path supports per-file spatial remapping.
        """
        if not items:
            return

        all_sources = set()
        for var_key, _, _ in items:
            for src in self.variables[var_key]['source_vars']:
                all_sources.add(src)
        all_sources = sorted(all_sources)

        output_map = {}
        out_idx = 0
        for u_idx in range(len(time_mask)):
            if time_mask[u_idx]:
                output_map[u_idx] = out_idx
                out_idx += 1

        target_t = chunk_4d[0] if chunk_4d is not None else 1

        item_transforms = []
        for var_key, data_var, vert_indices in items:
            transform_name = self.variables[var_key].get('transform')
            fn = self._get_block_transform(transform_name)
            if fn is None:
                raise RuntimeError(
                    f'No block transform registered for {var_key!r} '
                    f'(transform={transform_name!r}); should not be in multi_rechunker_items.'
                )
            item_transforms.append((var_key, data_var, vert_indices, fn))

        y_sl, x_sl = spatial_slice
        raw_offset = 0
        for path, file_times, _ in self._file_time_map:
            n_file_times = len(file_times)

            file_mask = []
            for local_t in range(n_file_times):
                u_idx = self._raw_to_unique[raw_offset + local_t]
                if u_idx != -1 and u_idx in output_map:
                    file_mask.append(local_t)

            if not file_mask:
                raw_offset += n_file_times
                continue

            t_start = file_mask[0]
            t_stop = file_mask[-1] + 1

            with h5py.File(path, 'r') as h5:
                if self._heterogeneous_grids and filtered_y is not None:
                    fy_sl, fx_sl, y_off, x_off = self._get_file_spatial_mapping(
                        h5, filtered_y, filtered_x)
                    if fy_sl is None:
                        raw_offset += n_file_times
                        continue
                else:
                    fy_sl, fx_sl = y_sl, x_sl
                    y_off, x_off = 0, 0

                ref = h5[all_sources[0]]
                source_chunks = ref.chunks or rechunkit.guess_chunk_shape(
                    ref.shape, ref.dtype.itemsize, max_mem,
                )
                y_start, y_stop, _ = fy_sl.indices(ref.shape[1])
                x_start, x_stop, _ = fx_sl.indices(ref.shape[2])
                ny = y_stop - y_start
                nx = x_stop - x_start
                sel = (slice(t_start, t_stop), slice(y_start, y_stop), slice(x_start, x_stop))
                target_chunks = (target_t, ny, nx)

                y_write = slice(y_off, y_off + ny) if y_off > 0 or self._heterogeneous_grids else None
                x_write = slice(x_off, x_off + nx) if x_off > 0 or self._heterogeneous_grids else None

                sources = {name: h5[name].__getitem__ for name in all_sources}

                for write_slices, data_blocks in self._multi_rechunker(
                    sources, ref.shape, ref.dtype, source_chunks, target_chunks, max_mem, sel,
                ):
                    block_cache = {}
                    for var_key, data_var, vert_indices, fn in item_transforms:
                        item_sources = {
                            sv: data_blocks[sv]
                            for sv in self.variables[var_key]['source_vars']
                        }
                        block_data = fn(item_sources, fy_sl, fx_sl, block_cache)
                        self._write_block(
                            data_var, block_data, write_slices, t_start, raw_offset, output_map,
                            vert_indices, y_write, x_write,
                        )

            raw_offset += n_file_times

    def _populate_with_accumulation(self, data_var, var_key, time_mask, spatial_slice,
                                    chunk_4d, max_mem, vert_indices,
                                    filtered_y=None, filtered_x=None):
        """
        Populate an ``accumulation_increment`` variable (e.g. WRF RAINNC+RAINC)
        using a single cross-file rechunker call, with stateful prev-block
        tracking so that each cfdb chunk is written exactly once.

        Sources are summed (with WRF bucket-counter reconstruction when
        ``BUCKET_MM > 0``) into a per-block cumulative ``total``; the increment
        is ``np.diff`` along the time axis, with the previous block's last
        cumulative carried forward via ``self._prev_accum_total``. The very
        first timestep of the entire conversion is NaN (no prior).
        """
        if self._heterogeneous_grids:
            # Fall back to legacy per-timestep path for heterogeneous grids.
            self._prev_accum_total = None
            self._populate_per_timestep(
                data_var, var_key, time_mask, spatial_slice, vert_indices, max_mem,
                is_accumulation=True, filtered_y=filtered_y, filtered_x=filtered_x,
            )
            return

        info = self.variables[var_key]
        source_vars = list(info['source_vars'])
        target_t = chunk_4d[0] if chunk_4d is not None else 1

        paths = self._files_for_var(var_key)
        if not paths:
            return

        # Probe the first file for WRF bucket counter setup. If BUCKET_MM > 0
        # and ``I_<name>`` exists for any source, the cumulative reconstruction
        # is ``<name> + BUCKET_MM * I_<name>`` per source (see
        # ``_accumulation_source_sum`` for the per-timestep equivalent).
        bucket_mm = 0.0
        bucket_vars = []
        with h5py.File(paths[0], 'r') as h5:
            attr = h5.attrs.get('BUCKET_MM', -1.0)
            try:
                bucket_mm = float(np.asarray(attr).item())
            except Exception:
                bucket_mm = -1.0
            if bucket_mm > 0.0:
                for sv in source_vars:
                    if 'I_' + sv in h5:
                        bucket_vars.append('I_' + sv)
        use_bucket = bucket_mm > 0.0 and len(bucket_vars) > 0
        all_sources = source_vars + bucket_vars

        output_map = {}
        out_idx = 0
        for u_idx in range(len(time_mask)):
            if time_mask[u_idx]:
                output_map[u_idx] = out_idx
                out_idx += 1

        raw_to_unique = self._get_raw_to_unique(paths)

        y_sl, x_sl = spatial_slice
        self._prev_accum_total = None

        with ExitStack() as stack:
            h5_files = [stack.enter_context(h5py.File(p, 'r')) for p in paths]

            sources = {}
            ref_shape = None
            ref_dtype = None
            ref_source_chunks = None
            for sv in all_sources:
                file_lens = [int(h5[sv].shape[0]) for h5 in h5_files]
                src = self._make_source(sv, h5_files, file_lens)
                sources[sv] = src
                if ref_shape is None:
                    ref_shape = src.shape
                    ref_dtype = src.dtype
                    ref_source_chunks = src.source_chunks

            if ref_source_chunks is None:
                ref_source_chunks = rechunkit.guess_chunk_shape(
                    ref_shape, ref_dtype.itemsize, max_mem,
                )

            spatial_axes = ref_shape[1:]
            if len(spatial_axes) == 2:
                # Ordinary 2D-spatial source: (T, Y, X).
                y_start, y_stop, _ = y_sl.indices(spatial_axes[0])
                x_start, x_stop, _ = x_sl.indices(spatial_axes[1])
                sel = (slice(0, ref_shape[0]),
                       slice(y_start, y_stop), slice(x_start, x_stop))
                target_chunks = (target_t, y_stop - y_start, x_stop - x_start)
            elif len(spatial_axes) == 3:
                # Region-dimensioned source: (T, N, Y, X). The block math below is
                # elementwise / time-axis-0, so it works unchanged on (n, N, y, x);
                # the write uses vert_indices=range(N). The region axis is read whole.
                nreg = spatial_axes[0]
                y_start, y_stop, _ = y_sl.indices(spatial_axes[1])
                x_start, x_stop, _ = x_sl.indices(spatial_axes[2])
                sel = (slice(0, ref_shape[0]), slice(0, nreg),
                       slice(y_start, y_stop), slice(x_start, x_stop))
                target_chunks = (target_t, nreg, y_stop - y_start, x_stop - x_start)
            else:
                raise ValueError(
                    f'accumulation_increment expects 2D- or 3D-spatial sources, got {ref_shape}'
                )

            source_callables = {sv: src.__call__ for sv, src in sources.items()}

            for write_slices, data_blocks in self._multi_rechunker(
                source_callables, ref_shape, ref_dtype, ref_source_chunks,
                target_chunks, max_mem, sel,
            ):
                # Apply post-block transform (e.g. ERA5 lat reversal — currently
                # no ERA5 var has accumulation_increment, but be consistent).
                blocks = {
                    sv: self._post_block_transform(b, sv, len(ref_shape))
                    for sv, b in data_blocks.items()
                }

                # Cumulative total for this block, with optional bucket-counter
                # reconstruction.
                total = sum(blocks[sv].astype('float64') for sv in source_vars)
                if use_bucket:
                    bucket_sum = sum(blocks[bv].astype('float64') for bv in bucket_vars)
                    total = total + bucket_mm * bucket_sum

                # Increment along time axis. First timestep of the very first
                # block has no prior → NaN; otherwise diff against the previous
                # block's saved last timestep. Negative increments are clipped
                # to 0 (precip can't go negative; nudging/feedback occasionally
                # reduces parent-grid accumulators).
                n_block = total.shape[0]
                diff = np.empty_like(total)
                if self._prev_accum_total is None:
                    diff[0] = np.nan
                else:
                    diff[0] = total[0] - self._prev_accum_total
                if n_block > 1:
                    diff[1:] = total[1:] - total[:-1]
                self._prev_accum_total = total[-1].copy()
                np.maximum(diff, 0.0, out=diff, where=~np.isnan(diff))

                t_outs = [None] * n_block
                for i in range(n_block):
                    raw = write_slices[0].start + i
                    u = int(raw_to_unique[raw])
                    if u != -1 and u in output_map:
                        t_outs[i] = output_map[u]

                self._write_block_from_t_outs(
                    data_var, diff.astype('float32'), t_outs, vert_indices, None, None,
                )

    def _populate_per_timestep(self, data_var, var_key, time_mask, spatial_slice, vert_indices, max_mem, is_accumulation,
                               filtered_y=None, filtered_x=None):
        """
        Populate a data variable using per-timestep iteration.

        Used for transform variables (accumulation, wind, 3D temp) that need
        per-timestep logic. Handles accumulation cross-file caching.

        vert_indices is [0] for surface variables with a length-1 height coordinate.

        Per-timestep arrays are buffered within a single source file and flushed
        as a coalesced block write so that we do not rewrite a multi-timestep
        cfdb chunk once per timestep.
        """
        raw_offset = 0
        output_time_idx = 0

        # We calculate the required max memory for the HDF5 chunk cache to prevent chunk-thrashing.
        # This implicitly acts as our "block read" by letting the C-library manage the block buffer
        # optimally across multiple source variables without breaking axis-dependent transform logic.
        chunk_cache_mem = max_mem if 'max_mem' in locals() else 2**27

        for path, file_times, _ in self._file_time_map:
            n_file_times = len(file_times)

            with h5py.File(path, 'r', rdcc_nbytes=chunk_cache_mem) as h5:
                # Per-file spatial mapping
                if self._heterogeneous_grids and filtered_y is not None:
                    fy_sl, fx_sl, y_off, x_off = self._get_file_spatial_mapping(
                        h5, filtered_y, filtered_x)
                    if fy_sl is None:
                        raw_offset += n_file_times
                        continue
                    file_spatial_slice = (fy_sl, fx_sl)
                    ny = fy_sl.stop - fy_sl.start
                    nx = fx_sl.stop - fx_sl.start
                    y_write = slice(y_off, y_off + ny)
                    x_write = slice(x_off, x_off + nx)
                else:
                    file_spatial_slice = spatial_slice
                    y_write, x_write = None, None

                buffer = []
                buffer_t_outs = []
                for local_t in range(n_file_times):
                    u_idx = self._raw_to_unique[raw_offset + local_t]
                    if u_idx == -1 or not time_mask[u_idx]:
                        continue

                    data = self._read_variable(h5, var_key, local_t, file_spatial_slice)
                    buffer.append(data)
                    buffer_t_outs.append(output_time_idx)
                    output_time_idx += 1

                self._flush_buffered(data_var, buffer, buffer_t_outs, vert_indices, y_write, x_write)

                # Cache last accumulated total for cross-file boundary
                if is_accumulation:
                    info = self.variables[var_key]
                    total = self._accumulation_source_sum(
                        h5, info['source_vars'], n_file_times - 1, file_spatial_slice
                    )
                    self._prev_accum_total = total.astype('float32')

            raw_offset += n_file_times

    def _populate_batch_per_timestep(self, batch_items, time_mask, spatial_slice, max_mem,
                                     chunk_4d=None, filtered_y=None, filtered_x=None):
        """
        Populate multiple transform variables with timestep-outer, variable-inner
        loop order, enabling per-timestep caching of shared intermediates
        (e.g., geo_height, rotated winds).

        ``chunk_4d`` is accepted for signature compatibility with subclasses that
        forward it to a block-mode helper; the base implementation does not use
        it (it always per-timestep reads + flushes one buffered block per file).

        Per-variable arrays are buffered within a single source file and flushed
        as coalesced block writes per variable so that multi-timestep cfdb chunks
        are not rewritten once per timestep.
        """
        raw_offset = 0
        output_time_idx = 0

        # Delegate block caching to HDF5 C-level chunk cache to prevent chunk thrashing
        chunk_cache_mem = max_mem if 'max_mem' in locals() else 2**27

        for path, file_times, _ in self._file_time_map:
            n_file_times = len(file_times)

            with h5py.File(path, 'r', rdcc_nbytes=chunk_cache_mem) as h5:
                # Per-file spatial mapping
                if self._heterogeneous_grids and filtered_y is not None:
                    fy_sl, fx_sl, y_off, x_off = self._get_file_spatial_mapping(
                        h5, filtered_y, filtered_x)
                    if fy_sl is None:
                        raw_offset += n_file_times
                        continue
                    file_spatial_slice = (fy_sl, fx_sl)
                    ny = fy_sl.stop - fy_sl.start
                    nx = fx_sl.stop - fx_sl.start
                    y_write = slice(y_off, y_off + ny)
                    x_write = slice(x_off, x_off + nx)
                else:
                    file_spatial_slice = spatial_slice
                    y_write, x_write = None, None

                buffers = {id(item[1]): [] for item in batch_items}
                buffer_t_outs = []

                for local_t in range(n_file_times):
                    u_idx = self._raw_to_unique[raw_offset + local_t]
                    if u_idx == -1 or not time_mask[u_idx]:
                        continue

                    self._ts_cache = {}

                    for var_key, data_var, vert_indices in batch_items:
                        data = self._read_variable(h5, var_key, local_t, file_spatial_slice)
                        buffers[id(data_var)].append(data)

                    self._ts_cache = None

                    buffer_t_outs.append(output_time_idx)
                    output_time_idx += 1

                for var_key, data_var, vert_indices in batch_items:
                    self._flush_buffered(
                        data_var, buffers[id(data_var)], buffer_t_outs,
                        vert_indices, y_write, x_write,
                    )

            raw_offset += n_file_times

    def _flush_buffered(self, data_var, buffer, buffer_t_outs, vert_indices, y_write, x_write):
        """
        Write a list of per-timestep arrays as a single coalesced block when the
        output time indices are contiguous, falling back to per-timestep writes
        when there are gaps (rare).
        """
        if not buffer:
            return
        if len(buffer) == 1:
            self._write_data_var(data_var, buffer[0], buffer_t_outs[0], vert_indices, y_write, x_write)
            return
        contiguous = all(
            buffer_t_outs[i + 1] - buffer_t_outs[i] == 1 for i in range(len(buffer_t_outs) - 1)
        )
        if contiguous:
            block = np.stack(buffer, axis=0)
            self._write_data_var_block(
                data_var, block,
                slice(buffer_t_outs[0], buffer_t_outs[-1] + 1),
                vert_indices, y_write, x_write,
            )
        else:
            for i, t_out in enumerate(buffer_t_outs):
                self._write_data_var(data_var, buffer[i], t_out, vert_indices, y_write, x_write)

    def _get_file_for_global_time(self, global_idx):
        """
        Return (path, local_time_idx) for a given global time index.
        """
        for path, file_times, start_idx in self._file_time_map:
            if global_idx < start_idx + len(file_times):
                return path, global_idx - start_idx
        raise IndexError(f'Global time index {global_idx} out of range.')
