"""
Base class for HDF5/netCDF4 ingestion to cfdb via h5py.
"""
import contextlib
import datetime
import pathlib
import re
from contextlib import ExitStack
from typing import Union, List, Tuple, Dict, Optional
import concurrent.futures
import h5py
import numpy as np
import pyproj
import rechunkit
import cfdb
from cfdb.utils import get_var_params
from cfdb_vars import short_name_map
import cfdb_ingest
from cfdb_ingest import forecast as _fc
from cfdb_ingest import grid as _grid
from cfdb_ingest.checks import check_encodable

_HEIGHT_SUFFIX = re.compile(r'_(\d+)m$')
_FULL_TO_SHORT = {full: short for short, full in short_name_map.items()}

# Relative humidity is stored as a FRACTION (0-1) in every cfdb-ingest source (cfdb-vars >= 0.2.4:
# precision 3, units '1').


######################################################
# Helpers shared by every source (h5py-based or not)


def resolve_variable_keys(mapping: Dict[str, dict], variables: Optional[List[str]]) -> List[str]:
    """
    Resolve user-provided variable names to mapping keys.

    Accepts mapping keys, source variable names, or cfdb short names. When a cfdb_name maps to
    multiple keys (e.g. both surface and level-interpolated variants of air_temp), all matching keys
    are returned. ``None`` returns every key.
    """
    if variables is None:
        return list(mapping.keys())

    cfdb_name_to_keys = {}
    source_var_to_key = {}
    for key, info in mapping.items():
        cfdb_name_to_keys.setdefault(info['cfdb_name'], []).append(key)
        for sv in info['source_vars']:
            # a source name resolves to its passthrough entry when one exists (``2t`` -> T2, not the
            # derived RH2 that also reads 2t); otherwise to the first entry that reads it
            single = len(info['source_vars']) == 1
            if sv not in source_var_to_key or (single and len(mapping[source_var_to_key[sv]]['source_vars']) > 1):
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


def full_cfdb_name(cfdb_name: str) -> str:
    """The name a variable is stored under: the cfdb-vars full name when ``cfdb_name`` is a short name."""
    return short_name_map.get(cfdb_name, cfdb_name)


def split_height_suffix(name: str) -> Tuple[str, str]:
    """``'air_temperature_2m'`` -> ``('air_temperature', '_2m')``; names without a suffix return ``''``."""
    m = _HEIGHT_SUFFIX.search(name)
    if m is None:
        return name, ''
    return name[:m.start()], m.group(0)


def group_variables(mapping: Dict[str, dict], var_keys: List[str],
                    region_size: Optional[Dict[str, int]] = None) -> Tuple[dict, dict, dict, dict]:
    """
    Classify mapping keys into coordinate groups.

    Returns ``(level_vars, surface_vars, soil_vars, region_vars)``. ``level_vars`` / ``soil_vars`` /
    ``region_vars`` map ``cfdb_name -> [var_key, ...]``. ``surface_vars`` maps
    ``stored_name -> (height, [var_key, ...])`` where fixed-height surface fields are grouped by
    (name, height): a name that also exists as a level/soil field, OR that appears at more than one
    height, gets a ``_<h>m`` suffix on its FULL cfdb-vars name (``air_temperature_2m``,
    ``u_wind_10m`` and ``u_wind_100m`` coexist). Otherwise the surface field keeps the bare name.
    """
    region_size = region_size or {}
    level_vars = {}
    soil_vars = {}
    region_vars = {}
    by_name_height = {}   # cfdb_name -> {height: [var_key, ...]}

    for var_key in var_keys:
        info = mapping[var_key]
        cfdb_name = info['cfdb_name']
        height_spec = info['height']
        # Region-dimensioned fields take priority over their nominal height: they are 2D surface
        # quantities carrying an extra wvt_regions axis. Gated on a recorded region size, so a
        # region_aware field in a single-region (3D) file falls through to the surface group.
        if region_size.get(var_key):
            region_vars.setdefault(cfdb_name, []).append(var_key)
        elif height_spec == 'levels':
            level_vars.setdefault(cfdb_name, []).append(var_key)
        elif height_spec == 'soil':
            soil_vars.setdefault(cfdb_name, []).append(var_key)
        else:
            by_name_height.setdefault(cfdb_name, {}).setdefault(float(height_spec), []).append(var_key)

    surface_vars = {}
    for cfdb_name, heights in by_name_height.items():
        conflicting = cfdb_name in level_vars or cfdb_name in soil_vars or len(heights) > 1
        for h, keys in heights.items():
            stored = f'{full_cfdb_name(cfdb_name)}_{int(h)}m' if conflicting else cfdb_name
            surface_vars[stored] = (h, keys)

    return level_vars, surface_vars, soil_vars, region_vars


def _resolve_var_template(cfdb_name: str, chunk_shape, dtype=None):
    """``(stored_name, var_params, template_attrs)`` for ``cfdb_name`` -- the naming of ``create_cfdb_data_var``."""
    base, suffix = split_height_suffix(cfdb_name)
    base = _FULL_TO_SHORT.get(base, base)     # accept either spelling; templates are keyed by short name
    kwargs = {'chunk_shape': chunk_shape}
    if dtype is not None:
        kwargs['dtype'] = dtype

    if base in short_name_map:
        stored_base, var_params, template_attrs = get_var_params(base, kwargs)
    else:
        stored_base, var_params, template_attrs = base, dict(kwargs), {}
        var_params.setdefault('dtype', 'float32')
    return stored_base + suffix, var_params, template_attrs


def stored_var_name(cfdb_name: str) -> str:
    """The name ``create_cfdb_data_var`` stores ``cfdb_name`` under."""
    return _resolve_var_template(cfdb_name, None)[0]


def create_cfdb_data_var(ds, cfdb_name: str, coord_names: Tuple[str, ...], chunk_shape: Tuple[int, ...],
                         dtype=None, attrs: Optional[dict] = None, strict: bool = False):
    """
    Create (or, in append mode, reuse) a cfdb data variable.

    The cfdb-vars template -- dtype encoding and CF attributes -- is looked up by the BASE name
    (height suffix stripped, short or full spelling), and the variable is stored under
    ``full_name + suffix``. ``dtype`` overrides the template encoding (attrs are kept); ``attrs``
    are applied on top (and are the only attrs for names cfdb-vars does not know).

    If the resolved name already exists (appending an init to an existing forecast dataset) the
    existing variable is returned after checking its coordinates match -- and with ``strict`` (grid
    extend mode) its stored encoding too, so a changed template cannot mix two encodings in one variable.
    """
    name, var_params, template_attrs = _resolve_var_template(cfdb_name, chunk_shape, dtype)
    if attrs:
        template_attrs = {**template_attrs, **attrs}

    if name in ds.data_var_names:
        existing = ds[name]
        if tuple(existing.coord_names) != tuple(coord_names):
            raise ValueError(
                f'existing variable {name!r} has coords {existing.coord_names}, expected {tuple(coord_names)}'
            )
        if strict:
            want = cfdb.dtypes.dtype(var_params['dtype']).to_dict()
            have = existing.dtype.to_dict()
            if want != have:
                raise ValueError(f'existing variable {name!r} is encoded as {have}; incoming data would be '
                                 f'{want}. Rebuild the target, or pass the stored dtype.')
        return existing

    data_var = ds.create.data_var.generic(name, coord_names, **var_params)
    if template_attrs:
        data_var.attrs.update(template_attrs)
    return data_var


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


def _check_existing_var(data_var, chunk_shape) -> None:
    """Extend mode: an existing variable must keep its chunk shape (when one is requested)."""
    if chunk_shape is not None and tuple(data_var.chunk_shape) != tuple(chunk_shape):
        raise ValueError(f'existing {data_var.name!r}: chunk_shape {tuple(data_var.chunk_shape)} != requested '
                         f'{tuple(chunk_shape)}; extend with the target\'s chunk shape (or pass chunk_shape=None)')


class _TimeMappedSource:
    """
    A virtual time axis over a concatenated source, for a rechunker whose blocks must line up with the
    OUTPUT dataset's time chunks.

    Row ``v`` of this source is raw row ``raw_of[v]`` of ``inner`` (the files concatenated), or a pad row
    of zeros when ``raw_of[v] == -1``. ``raw_of`` lists the kept frames in output order, preceded by
    ``pad`` pad rows, where ``pad`` is the output's first absolute time index modulo the chunk length --
    so a rechunker block of ``chunk_t`` rows starting at a multiple of ``chunk_t`` covers exactly one
    output chunk. Pad rows are never written, and frames outside the requested window are never read.
    """

    def __init__(self, inner, raw_of):
        self._inner = inner
        self._raw_of = np.asarray(raw_of, dtype='int64')
        self.shape = (len(self._raw_of),) + tuple(inner.shape[1:])
        self.dtype = inner.dtype
        # Keep the source's own chunking, time included: a time-chunked source (ERA5: 12-24 frames per
        # HDF5 chunk) is then read a whole chunk at a time, not one frame (= one decompression) per call.
        sc = inner.source_chunks
        self.source_chunks = None if sc is None else tuple(sc)

    def __call__(self, slices):
        t = slices[0]
        rest = tuple(slices[1:])
        rows = self._raw_of[t.start:t.stop]
        out = np.zeros((len(rows),) + tuple(s.stop - s.start for s in rest), dtype=self.dtype)
        i = 0
        while i < len(rows):
            if rows[i] < 0:
                i += 1
                continue
            j = i + 1
            while j < len(rows) and rows[j] == rows[j - 1] + 1:
                j += 1
            out[i:j] = self._inner((slice(int(rows[i]), int(rows[j - 1]) + 1),) + rest)
            i = j
        return out


@contextlib.contextmanager
def _grid_target(cfdb_path, dataset_type, cfdb_kwargs):
    """Grid mode: always a fresh file (unchanged behaviour)."""
    with cfdb.open_dataset(cfdb_path, 'n', dataset_type=dataset_type, **cfdb_kwargs) as ds:
        yield ds, True


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
    _forecast_writer = None
    # convert(extend=..., time_label='start', squeeze_height=...) -- opt-in, implemented per source.
    _supports_grid_extend = False
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
        Resolve user-provided variable names to mapping keys (see ``resolve_variable_keys``).
        """
        return resolve_variable_keys(self.variables, variables)

    def _default_forecast_reference_time(self, filtered_times):
        """Forecast mode: the run's init when the source records it; the first timestep otherwise."""
        return np.datetime64(filtered_times[0], 'm')

    def _forecast_axes(self, filtered_times, forecast_reference_time):
        """
        (init, leads) for forecast mode: leads are whole hours since ``init``, must be non-negative
        and regularly spaced (``forecast.lead_step``).
        """
        if forecast_reference_time is None:
            init = self._default_forecast_reference_time(filtered_times)
        else:
            init = np.datetime64(forecast_reference_time, 'm')
        delta_min = (np.asarray(filtered_times).astype('datetime64[m]') - init).astype('int64')
        if (delta_min < 0).any():
            raise ValueError(f'timesteps before the forecast_reference_time {init} cannot be leads')
        if (delta_min % 60).any():
            raise ValueError('forecast leads must be whole hours after the forecast_reference_time')
        leads = (delta_min // 60).astype('int32')
        _fc.lead_step(leads)
        return init, leads

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
        forecast_reference_time=None,
        forecast_step_minutes: int = 360,
        overwrite: bool = False,
        leads=None,
        mark_complete: bool = True,
        extend: bool = False,
        time_label: str = 'end',
        squeeze_height: bool = False,
        **cfdb_kwargs,
    ):
        """
        Convert source files to a cfdb dataset.

        Variables are stored with coordinates appropriate to their type:
        - Surface variables (height is a float): (time, height_Xm, y, x)
        - Level-interpolated variables (height='levels'): (time, <vertical_coord>, y, x)
        - Soil variables (height='soil'): (time, depth, y, x)

        Forecast mode (``dataset_type='grid_forecast'``): the source is ONE forecast run and ``time``
        becomes the pair ``(forecast_reference_time, forecast_period)`` -- see ``cfdb_ingest.forecast``.
        ``cfdb_path`` may then be an existing dataset (the init is appended) or an open
        Dataset / EDataset handle (never closed here; the caller pushes). Every (init, level) chunk
        is written exactly once through a ``ForecastWriter``; that writer buffers one chunk-row
        ``(n_lead, ny, nx)`` per live (variable, level), so keep forecast-mode ingests of per-file
        sources to 2-D variables.

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
            'grid' (default) or 'grid_forecast'.
        forecast_reference_time : str, np.datetime64 or None
            Forecast mode only: the run's init. Default: the source's own init
            (``_default_forecast_reference_time``), else the first timestep.
        forecast_step_minutes : int
            Forecast mode only: the ``forecast_reference_time`` step baked into a NEW dataset.
        overwrite : bool
            Forecast mode only: replace an init that is already complete in the target.
        leads : sequence of int or None
            Forecast mode only: the FULL ``forecast_period`` axis (hours) to bake into a NEW dataset,
            when this call supplies only part of the run (an init built up file by file). Default:
            this call's leads. On an existing dataset it must equal the stored axis.
        mark_complete : bool
            Forecast mode only: record the init in ``complete_inits`` at the end (default). Pass
            ``False`` for a partial call; the init then stays a back-fill target until the caller has
            checked ``forecast.missing_chunks`` and marks it.
        extend : bool
            Grid mode only (WRF; since 0.6.0): write into an existing dataset (or create it) instead of
            replacing it -- a long record built in time bands. The window may extend either end of the
            stored axis (across a gap: cfdb auto-fills placeholder slots) or overwrite stored times
            (placeholders, re-runs); ``cfdb_path`` may be a path or an open Dataset / EDataset handle.
            Every output chunk is written once per call. A window whose frames are only partly present
            in the input files is refused. See ``cfdb_ingest.grid``.
        time_label : {'end', 'start'}
            ``'end'`` (default): WRF's frame times. ``'start'`` (grid mode, accumulations only; since
            0.6.0): label each accumulated interval by its start (``frame - PREC_ACC_DT`` for PREC_ACC).
            ``start_date``/``end_date`` then select these labels; a run's lead-0 frame is dropped.
        squeeze_height : bool
            Grid mode only (since 0.6.0): store surface fields as ``(time, y, x)`` without the length-1
            ``height_Xm`` axis (the height moves to a ``height`` attribute). ``chunk_shape`` may then be
            3-D ``(time, y, x)``. Surface variables only.
        **cfdb_kwargs
            Extra kwargs for cfdb.open_dataset (e.g., compression).
        """
        self._vertical_coord = vertical_coord
        forecast = dataset_type == 'grid_forecast'
        self._forecast_writer = None
        self._grid_t_offset = 0
        self._grid_abs_t0 = 0
        leads_arg = leads
        if not forecast and (leads is not None or not mark_complete):
            raise ValueError("'leads' and 'mark_complete' apply to dataset_type='grid_forecast' only")

        var_keys = self.resolve_variables(variables)
        self._check_var_keys(var_keys)

        if time_label not in ('end', 'start'):
            raise ValueError(f"time_label must be 'end' or 'start', got {time_label!r}")
        grid_opts = [name for name, on in (('extend', extend), ("time_label='start'", time_label == 'start'),
                                           ('squeeze_height', squeeze_height)) if on]
        if grid_opts and forecast:
            raise ValueError(f'{grid_opts} apply to grid mode only (forecast leads are valid times)')
        if grid_opts and not self._supports_grid_extend:
            raise ValueError(f'{grid_opts} are implemented for WRF sources only ({type(self).__name__})')
        label_shift = self._label_shift(var_keys) if time_label == 'start' else None

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

        # Filter time. Interval-start mode selects on the labels (lead-0 frames dropped).
        if label_shift is not None:
            labels, keep = self._label_frames(label_shift, var_keys)
            time_mask, filtered_times = self._filter_time(start_date, end_date, labels=labels)
            time_mask &= keep
            filtered_times = labels[time_mask]
        else:
            time_mask, filtered_times = self._filter_time(start_date, end_date)

        self._check_window(var_keys, time_mask)
        all_labels = labels if label_shift is not None else self.times

        # Output slots. Default: the kept frames, compacted. Extend mode: every step of the window, so a frame
        # missing from the input is an unwritten slot (read as missing; filled by re-running the window once
        # the file exists) and is reported in the result, rather than shifting later frames forward.
        step_minutes = None
        missing_frames = np.array([], dtype='datetime64[m]')
        if extend:
            step_minutes = self._grid_step_minutes(label_shift)
            filtered_times, missing_frames = self._window_slots(filtered_times, start_date, end_date, step_minutes)
            if missing_frames.size and any(
                    self.variables[k].get('transform') == 'accumulation_increment' for k in var_keys):
                raise ValueError(
                    f'{missing_frames.size} frames of the window are missing from the input (first '
                    f'{[str(m) for m in missing_frames[:3]]}); a differenced accumulation (RAIN) across the hole '
                    f'would span several intervals. Supply the files, or split the window at the hole.')
        self._set_output_slots(time_mask, all_labels, filtered_times)

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

        # Classify variables into coordinate groups (surface fields keyed by stored name -> (height, keys))
        sorted_levels = np.array(sorted(target_levels), dtype='float64') if target_levels else None
        level_vars, surface_vars, soil_vars, region_vars = group_variables(
            self.variables, var_keys, self._field_region_size
        )

        # Collect unique fixed heights needed for surface variables
        fixed_heights = {h for h, _ in surface_vars.values()}

        # Default chunk shape. In forecast mode the storage chunk is 5-D (one init, all leads, one
        # level, a spatial tile); the populate paths keep seeing a 4-D "time block" view whose time
        # extent is the whole lead axis, so they hand the writer whole-lead blocks.
        if forecast:
            init, leads = self._forecast_axes(filtered_times, forecast_reference_time)
            axis_leads = leads if leads_arg is None else _fc.check_axis_leads(leads_arg, leads)
            storage_chunk = (tuple(chunk_shape) if chunk_shape is not None
                             else _fc.forecast_chunk_shape(len(axis_leads), ny, nx))
            if len(storage_chunk) != 5:
                raise ValueError(f'forecast-mode chunk_shape must be 5-D (init, lead, z, y, x), got {storage_chunk}')
            chunk_4d = (storage_chunk[1], storage_chunk[2], storage_chunk[3], storage_chunk[4])
            time_coord_names = (_fc.FRT, _fc.LEAD)
            target_cm = _fc.open_target(cfdb_path, dataset_type=dataset_type, **cfdb_kwargs)
        else:
            if squeeze_height:
                if level_vars or soil_vars or region_vars:
                    raise ValueError('squeeze_height applies to surface variables only; convert level, soil and '
                                     'region fields in a separate call')
                if chunk_shape is None:
                    storage_chunk = (1, ny, nx)
                elif len(chunk_shape) == 3:
                    storage_chunk = tuple(chunk_shape)
                elif len(chunk_shape) == 4:
                    storage_chunk = (chunk_shape[0], chunk_shape[2], chunk_shape[3])
                else:
                    raise ValueError(f'chunk_shape must be 3-D (time, y, x) with squeeze_height, got {chunk_shape}')
                chunk_4d = (storage_chunk[0], 1, storage_chunk[1], storage_chunk[2])
            else:
                if chunk_shape is not None and len(chunk_shape) != 4:
                    raise ValueError(f'grid-mode chunk_shape must be 4-D (time, z, y, x) '
                                     f'(3-D needs squeeze_height=True), got {chunk_shape}')
                chunk_4d = chunk_shape if chunk_shape is not None else (1, 1, ny, nx)
                storage_chunk = chunk_4d
            time_coord_names = ('time',)
            if extend:
                target_cm = _fc.open_target(cfdb_path, dataset_type=dataset_type, **cfdb_kwargs)
            else:
                target_cm = _grid_target(cfdb_path, dataset_type, cfdb_kwargs)

        has_multi_level = sorted_levels is not None and level_vars

        with target_cm as (ds, created):
            if created:
                # Create coordinates
                if forecast:
                    _fc.create_forecast_coords(ds, init, axis_leads, step_minutes=forecast_step_minutes)
                elif extend:
                    # An explicit numeric step, never step=True: see cfdb_ingest.grid.
                    ds.create.coord.time(data=filtered_times, step=int(step_minutes))
                else:
                    ds.create.coord.time(data=filtered_times, step=True)
                if label_shift is not None:
                    ds['time'].attrs[_grid.TIME_LABEL_ATTR] = 'interval_start'
                    ds['time'].attrs['interval_minutes'] = int(label_shift / np.timedelta64(1, 'm'))
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
                for h in (() if squeeze_height else sorted(fixed_heights)):
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
            elif forecast:
                _fc.validate_target(
                    ds, x_name=self.x_coord_name, y_name=self.y_coord_name, x=filtered_x, y=filtered_y,
                    levels=sorted_levels if has_multi_level else None,
                    depths=soil_depths if (soil_depths is not None and soil_vars) else None,
                    crs=self.crs,
                )
                _fc.append_history(ds, f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} '
                                       f'{type(self).__name__} appended init {np.datetime64(init, "m")}')
            else:
                _grid.validate_target(
                    ds, crs=self.crs, x_name=self.x_coord_name, y_name=self.y_coord_name, x=filtered_x,
                    y=filtered_y, levels=sorted_levels if has_multi_level else None,
                    depths=soil_depths if (soil_depths is not None and soil_vars) else None,
                    step_minutes=step_minutes, time_label=time_label,
                )

            if extend:
                if created:
                    placed = {'status': 'new', 'index': 0, 'n_new': len(filtered_times), 'gap_filled': 0,
                              'abs_start': 0}
                else:
                    # Refuse a variable mismatch before the time axis is touched.
                    for cfdb_name in surface_vars:
                        if stored_var_name(cfdb_name) in ds.data_var_names:
                            coords = ((*time_coord_names, self.y_coord_name, self.x_coord_name) if squeeze_height else
                                      (*time_coord_names, f'height_{int(surface_vars[cfdb_name][0])}m',
                                       self.y_coord_name, self.x_coord_name))
                            existing = self._create_cfdb_data_var(ds, cfdb_name, coords, storage_chunk, strict=True)
                            _check_existing_var(existing, storage_chunk if chunk_shape is not None else None)
                    placed = _grid.place_times(ds, filtered_times, step_minutes=step_minutes)
                    _fc.append_history(
                        ds, f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} {type(self).__name__} '
                            f'{placed["status"]} {filtered_times[0]}..{filtered_times[-1]} '
                            f'({self.input_paths[0].name} .. {self.input_paths[-1].name})')
                self._grid_t_offset = placed['index']
                self._grid_abs_t0 = placed['abs_start']

            if forecast:
                if created:
                    placed = {'index': 0, 'status': 'new', 'autofilled': 0}
                else:
                    if leads_arg is not None:
                        stored = np.asarray(ds[_fc.LEAD].data)
                        if len(stored) != len(axis_leads) or not np.array_equal(stored, axis_leads):
                            raise ValueError(
                                f'leads={np.asarray(leads_arg).tolist()} does not match the stored {_fc.LEAD} '
                                f'axis {stored.tolist()}; the lead axis is fixed at creation'
                            )
                    placed = _fc.place_init(ds, init, step_minutes=forecast_step_minutes, overwrite=overwrite)
                _fc.unmark_init_complete(ds, init)
                self._forecast_writer = _fc.ForecastWriter(
                    ds, placed['index'], _fc.lead_index_map(ds, leads), ny, nx
                )

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
            level_coord_names = (*time_coord_names, vertical_coord, self.y_coord_name, self.x_coord_name)
            for cfdb_name, var_key_list in level_vars.items():
                data_var = self._create_cfdb_data_var(ds, cfdb_name, level_coord_names, storage_chunk)
                for var_key in var_key_list:
                    level_indices = list(range(len(sorted_levels)))
                    _classify((var_key, data_var, level_indices), self.variables[var_key])

            # Surface variables: (time, height_Xm, y, x), or (time, y, x) with squeeze_height
            for cfdb_name, (h, var_key_list) in surface_vars.items():
                coord_name = f'height_{int(h)}m'
                if squeeze_height:
                    surface_coord_names = (*time_coord_names, self.y_coord_name, self.x_coord_name)
                else:
                    surface_coord_names = (*time_coord_names, coord_name, self.y_coord_name, self.x_coord_name)
                existed = stored_var_name(cfdb_name) in ds.data_var_names
                data_var = self._create_cfdb_data_var(ds, cfdb_name, surface_coord_names, storage_chunk,
                                                      strict=extend)
                if extend and existed:
                    _check_existing_var(data_var, storage_chunk if chunk_shape is not None else None)
                if extend:
                    # Align every write to the variable's REAL time chunk (stored one when it existed).
                    chunk_4d = (data_var.chunk_shape[0],) + tuple(chunk_4d[1:])
                if not existed:
                    if squeeze_height:
                        data_var.attrs['height'] = f'{h:g} m'
                    if label_shift is not None:
                        data_var.attrs['cell_methods'] = (
                            f'time: sum (interval: {int(label_shift / np.timedelta64(1, "m"))} minutes)')
                for var_key in var_key_list:
                    # index 0 of the length-1 height coord; None = no middle axis (squeeze_height)
                    _classify((var_key, data_var, None if squeeze_height else [0]), self.variables[var_key])

            # Soil variables: (time, depth, y, x)
            soil_coord_names = (*time_coord_names, 'depth', self.y_coord_name, self.x_coord_name)
            for cfdb_name, var_key_list in soil_vars.items():
                data_var = self._create_cfdb_data_var(ds, cfdb_name, soil_coord_names, storage_chunk)
                for var_key in var_key_list:
                    depth_indices = list(range(len(soil_depths)))
                    _classify((var_key, data_var, depth_indices), self.variables[var_key])

            # Region (multi-region WVT) variables: (time, wvt_region, y, x).
            # Mechanically identical to level/soil — the region axis is written
            # via vert_indices=range(N) by the existing populate paths.
            region_coord_names = (*time_coord_names, 'wvt_region', self.y_coord_name, self.x_coord_name)
            for cfdb_name, var_key_list in region_vars.items():
                data_var = self._create_cfdb_data_var(ds, cfdb_name, region_coord_names, storage_chunk)
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

            if forecast:
                self._forecast_writer.close()
                chunk_writes = self._forecast_writer.writes
                self._forecast_writer = None
                if mark_complete:
                    _fc.mark_init_complete(ds, init)
                return {'init': str(np.datetime64(init, 'm')), 'init_index': placed['index'],
                        'status': placed['status'], 'autofilled': placed['autofilled'],
                        'n_leads': int(len(leads)), 'variables': list(var_keys),
                        'chunk_writes': int(chunk_writes), 'complete': bool(mark_complete)}

            result = {'status': placed['status'] if extend else 'new',
                      'time_index': (int(self._grid_t_offset), int(self._grid_t_offset) + len(filtered_times)),
                      'n_times': int(len(filtered_times)),
                      'n_new': int(placed['n_new']) if extend else int(len(filtered_times)),
                      'gap_filled': int(placed['gap_filled']) if extend else 0,
                      'missing_frames': [str(m) for m in missing_frames],
                      'variables': list(var_keys)}
            self._grid_t_offset = 0
            self._grid_abs_t0 = 0
            return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_time(self, start_date, end_date, labels=None):
        """
        Return a boolean mask and the filtered times array. ``labels`` (aligned with ``self.times``)
        are the output time values when they differ from the frame times (interval-start labels).
        """
        times = self.times if labels is None else labels
        mask = np.ones(len(times), dtype=bool)

        if start_date is not None:
            start = np.datetime64(start_date)
            mask &= times >= start

        if end_date is not None:
            end = np.datetime64(end_date)
            mask &= times <= end

        return mask, times[mask]

    def _check_window(self, var_keys, time_mask) -> None:
        """Source-specific refusals that depend on the frames of the requested window (override)."""
        return None

    def _run_start_of_path(self, path):
        """The run (init) an input file belongs to, where the source records one (override); None if unknown."""
        return None

    def _grid_step_minutes(self, label_shift):
        """
        Extend mode: the time step (minutes) of the output axis -- the accumulation interval with
        interval-start labels, else the smallest spacing of ALL input frames (a missing file only widens
        some gaps). Never inferred from the window alone: two frames around a hole would read as one step.
        """
        if label_shift is not None:
            return int(label_shift / np.timedelta64(1, 'm'))
        d = np.diff(self.times)
        if len(d):
            return int(d.min() / np.timedelta64(1, 'm'))
        raise ValueError('cannot determine the time step from one frame; pass more frames '
                         "(or use time_label='start' for accumulations)")

    @staticmethod
    def _window_slots(filtered_times, start_date, end_date, step_minutes):
        """
        Extend mode: every time of the window [start_date, end_date] (default: first..last frame present)
        at the step, and those absent from the input. Refused: no frame at all, or frames off the step grid
        of the window.
        """
        if len(filtered_times) == 0:
            raise ValueError(f'no input frames in the window {start_date} .. {end_date}')
        step = np.timedelta64(int(step_minutes), 'm')
        present = np.asarray(filtered_times).astype('datetime64[m]')
        lo = np.datetime64(start_date, 'm') if start_date is not None else present[0]
        hi = np.datetime64(end_date, 'm') if end_date is not None else present[-1]
        window = np.arange(lo, hi + step, step)
        off = np.setdiff1d(present, window)
        if off.size:
            raise ValueError(f'frames {[str(o) for o in off[:3]]} are off the {step_minutes}-min grid of the '
                             f'window starting {lo}')
        return window, np.setdiff1d(window, present)

    def _set_output_slots(self, time_mask, labels, out_times):
        """
        Map each kept unique time to its output slot (``self._out_index``, -1 = not written) and each slot to
        its unique time (``self._slot_u``, -1 = no frame). Every populate path writes through these.
        """
        kept = np.flatnonzero(time_mask)
        out_times = np.asarray(out_times).astype('datetime64[m]')
        pos = np.searchsorted(out_times, np.asarray(labels)[kept].astype('datetime64[m]'))
        if kept.size and (pos.max() >= len(out_times) or not np.array_equal(out_times[pos], np.asarray(labels)[kept])):
            raise RuntimeError('kept frames do not map onto the output time slots')
        out_index = np.full(len(time_mask), -1, dtype='int64')
        out_index[kept] = pos
        slot_u = np.full(len(out_times), -1, dtype='int64')
        slot_u[pos] = kept
        self._out_index, self._slot_u = out_index, slot_u

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

    def _check_var_keys(self, var_keys) -> None:
        """Source-specific refusals of variable combinations (override; base accepts everything)."""
        return None

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

    def _create_cfdb_data_var(self, ds, cfdb_name, coord_names, chunk_shape, dtype=None, attrs=None, strict=False):
        """
        Create a cfdb data variable (see the module-level ``create_cfdb_data_var``).
        """
        return create_cfdb_data_var(ds, cfdb_name, coord_names, chunk_shape, dtype=dtype, attrs=attrs, strict=strict)

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

        # raw_to_unique: maps each raw global time index (concatenated across
        # files) to a unique time index, or -1 for duplicates from overlapping
        # files. Reuses ``self._raw_to_unique`` when paths match input_paths.
        raw_to_unique = self._get_raw_to_unique(paths)
        raw_of, pad = self._time_plan(raw_to_unique, time_mask, target_t)

        with ExitStack() as stack:
            h5_files = [stack.enter_context(h5py.File(p, 'r')) for p in paths]
            file_lens = [int(h5[src_var].shape[0]) for h5 in h5_files]
            source = _TimeMappedSource(self._make_source(src_var, h5_files, file_lens), raw_of)
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
                t_outs = self._plan_t_outs(raw_of, pad, write_slices[0].start, write_slices[0].stop)
                self._write_block_from_t_outs(
                    data_var, block, t_outs, vert_indices, None, None,
                )

    # Absolute index (in the target's time-chunk grid) of output time 0, and the write offset of output
    # time 0 into the target's time axis. Both 0 for a fresh dataset; set by grid extend mode.
    _grid_abs_t0 = 0
    _grid_t_offset = 0

    def _time_plan(self, raw_to_unique, time_mask, chunk_t):
        """
        Virtual time rows for a cross-file rechunk, aligned to the target's chunk grid.

        Returns ``(raw_of, pad)``: ``raw_of[v]`` is the raw (concatenated-files) row of virtual row ``v``,
        or -1 for a pad row / a kept time absent from these files; output time ``k`` is virtual row
        ``pad + k``. ``pad`` puts output time 0 at its position within its target chunk, so every
        rechunker block (``chunk_t`` rows from a multiple of ``chunk_t``) is exactly one output chunk
        and each chunk is written once -- also when leading frames are filtered out or the output is
        written at an offset into an existing axis (possibly with a negative origin after a prepend).
        """
        unique_to_raw = np.full(len(time_mask), -1, dtype='int64')
        raws = np.flatnonzero(raw_to_unique >= 0)
        unique_to_raw[raw_to_unique[raws]] = raws
        slot_u = self._slot_u
        raw_slots = np.where(slot_u >= 0, unique_to_raw[np.maximum(slot_u, 0)], -1)
        pad = int(self._grid_abs_t0 % chunk_t) if chunk_t > 1 else 0
        raw_of = np.concatenate([np.full(pad, -1, dtype='int64'), raw_slots])
        return raw_of, pad

    @staticmethod
    def _plan_t_outs(raw_of, pad, v_start, v_stop):
        """Output time index for each virtual row in [v_start, v_stop), or None (pad / absent frame)."""
        return [(v - pad) if (v >= pad and raw_of[v] >= 0) else None for v in range(v_start, v_stop)]

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

        output_map = {int(u): int(self._out_index[u]) for u in np.flatnonzero(time_mask)}

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
        i = 0
        while i < n_block:
            if t_outs[i] is None:
                i += 1
                continue
            j = i + 1
            while j < n_block and t_outs[j] is not None and t_outs[j] == t_outs[j - 1] + 1:
                j += 1
            # One coalesced write per run of consecutive output indices: a block aligned to the output
            # chunk grid (see _time_plan) then touches each output chunk exactly once, including a
            # leading pad (the run starts part-way into the chunk).
            if j - i == 1:
                self._write_data_var(data_var, block[i], t_outs[i], vert_indices, y_write, x_write)
            else:
                self._write_data_var_block(
                    data_var, block[i:j], slice(t_outs[i], t_outs[i] + (j - i)),
                    vert_indices, y_write, x_write,
                )
            i = j

    def _write_data_var(self, data_var, data, output_time_idx, vert_indices, y_write=None, x_write=None):
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
        writer = self._forecast_writer
        if writer is not None:
            if data.ndim == 2:
                writer.put(data_var, vert_indices[0], int(output_time_idx), data, ys, xs)
            else:
                for lev_i, v_idx in enumerate(vert_indices):
                    writer.put(data_var, v_idx, int(output_time_idx), data[lev_i], ys, xs)
            return
        check_encodable(data_var, data)
        output_time_idx = output_time_idx + self._grid_t_offset
        if vert_indices is None:
            data_var[(output_time_idx, ys, xs)] = data[np.newaxis, ...]
        elif data.ndim == 2:
            data_var[(output_time_idx, vert_indices[0], ys, xs)] = data[np.newaxis, np.newaxis, ...]
        else:
            for lev_i, v_idx in enumerate(vert_indices):
                data_var[(output_time_idx, v_idx, ys, xs)] = data[lev_i][np.newaxis, np.newaxis, ...]

    def _write_data_var_block(self, data_var, block, time_slice, vert_indices, y_write=None, x_write=None):
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
        writer = self._forecast_writer
        if writer is not None:
            if block.ndim == 3:
                writer.put(data_var, vert_indices[0], time_slice, block, ys, xs)
            else:
                for lev_i, v_idx in enumerate(vert_indices):
                    writer.put(data_var, v_idx, time_slice, block[:, lev_i], ys, xs)
            return
        check_encodable(data_var, block)
        if self._grid_t_offset:
            time_slice = slice(time_slice.start + self._grid_t_offset, time_slice.stop + self._grid_t_offset)
        if vert_indices is None:
            data_var[(time_slice, ys, xs)] = block
        elif block.ndim == 3:
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
        # Each source var maps output times to ITS OWN files' rows (ERA5 has per-variable file sets, which
        # need not sort in the same order); all must cover the same times, or the combined value is undefined.
        plans = {sv: self._time_plan(self._get_raw_to_unique(paths_per_src[sv]), time_mask, target_t)
                 for sv in all_sources}
        raw_of, pad = plans[all_sources[0]]
        for sv in all_sources[1:]:
            if not np.array_equal(plans[sv][0] >= 0, raw_of >= 0):
                missing = self.times[self._slot_u[(plans[sv][0][pad:] >= 0) != (raw_of[pad:] >= 0)]][:3]
                raise ValueError(
                    f'source variables {all_sources[0]!r} and {sv!r} cover different times (e.g. {missing.tolist()}); '
                    f'{[k for k, _, _ in items]} combine them frame by frame, so every source needs every frame'
                )

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
                src = _TimeMappedSource(self._make_source(sv, h5_files, file_lens), plans[sv][0])
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

                t_outs = self._plan_t_outs(raw_of, pad, write_slices[0].start, write_slices[0].stop)

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

        output_map = {int(u): int(self._out_index[u]) for u in np.flatnonzero(time_mask)}

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

        raw_to_unique = self._get_raw_to_unique(paths)
        raw_of, pad = self._time_plan(raw_to_unique, time_mask, target_t)

        y_sl, x_sl = spatial_slice
        self._prev_accum_total = None

        with ExitStack() as stack:
            h5_files = [stack.enter_context(h5py.File(p, 'r')) for p in paths]

            sources = {}
            raw_sources = {}
            ref_shape = None
            ref_dtype = None
            ref_source_chunks = None
            for sv in all_sources:
                file_lens = [int(h5[sv].shape[0]) for h5 in h5_files]
                raw_sources[sv] = self._make_source(sv, h5_files, file_lens)
                src = _TimeMappedSource(raw_sources[sv], raw_of)
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

            def _cumulative(blocks):
                total = sum(blocks[sv].astype('float64') for sv in source_vars)
                if use_bucket:
                    total = total + bucket_mm * sum(blocks[bv].astype('float64') for bv in bucket_vars)
                return total

            # Seed the increment of the first kept frame from the frame before the window (read on its
            # own: frames outside the window are otherwise never read). None -> NaN, as before, when
            # the first kept frame is the first frame of the input.
            real_rows = raw_of[raw_of >= 0]
            cum = np.cumsum([0] + [int(h5[source_vars[0]].shape[0]) for h5 in h5_files])
            file_of = lambda r: paths[int(np.searchsorted(cum, r, side='right')) - 1]  # noqa: E731
            if (len(real_rows) and real_rows[0] > 0 and self._run_start_of_path(file_of(real_rows[0] - 1))
                    == self._run_start_of_path(file_of(real_rows[0]))):
                r_prev = int(real_rows[0]) - 1
                prev_blocks = {
                    sv: self._post_block_transform(
                        raw_sources[sv]((slice(r_prev, r_prev + 1),) + tuple(sel[1:])), sv, len(ref_shape))
                    for sv in all_sources
                }
                self._prev_accum_total = _cumulative(prev_blocks)[0]

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
                total = _cumulative(blocks)

                # Increment between consecutive kept frames (pad rows skipped). The first kept frame
                # of the input has no prior -> NaN; otherwise diff against the previous kept frame,
                # carried across blocks. Negative increments are clipped to 0 (precip can't go
                # negative; nudging/feedback occasionally reduces parent-grid accumulators).
                v0, v1 = write_slices[0].start, write_slices[0].stop
                real = raw_of[v0:v1] >= 0
                diff = np.zeros_like(total)
                if real.any():
                    t_real = total[real]
                    d = np.empty_like(t_real)
                    d[0] = np.nan if self._prev_accum_total is None else t_real[0] - self._prev_accum_total
                    if len(t_real) > 1:
                        d[1:] = t_real[1:] - t_real[:-1]
                    self._prev_accum_total = t_real[-1].copy()
                    np.maximum(d, 0.0, out=d, where=~np.isnan(d))
                    diff[real] = d

                t_outs = self._plan_t_outs(raw_of, pad, v0, v1)

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
                    buffer_t_outs.append(int(self._out_index[u_idx]))

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

                    buffer_t_outs.append(int(self._out_index[u_idx]))

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
