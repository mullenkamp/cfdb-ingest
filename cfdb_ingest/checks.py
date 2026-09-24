"""
Pre-write checks shared by every write path (grid, forecast; WRF, ERA5, IFS).
"""
import warnings

import numpy as np


def check_encodable(data_var, values) -> None:
    """
    Refuse values a packed cfdb variable cannot store.

    cfdb's packed dtypes (a float stored as a scaled integer, e.g. the ``precipitation`` template: uint16,
    precision 2, offset -1) map a finite value whose scaled integer falls outside the encoded width, or
    onto the reserved fill code, to the fill -- it reads back as NaN, silently. For precipitation that is
    anything above 654.35 mm. A value outside the template's range means the template or the data is
    wrong, so this raises instead (since 0.6.0).

    The arithmetic mirrors ``cfdb.dtypes.DTypeTranscoder.encode`` exactly (same dtype, same rounding),
    applied to the block's finite min and max: encoding is monotonic, so the extremes decide, unless the
    fill code sits strictly inside the encoded range, when every value is checked.
    """
    dt = data_var.dtype
    enc = getattr(dt, 'dtype_encoded', None)
    factor = getattr(dt, '_factor', None)
    offset = getattr(dt, 'offset', None)
    if enc is None or np.dtype(enc).kind not in 'ui' or getattr(dt, 'kind', None) != 'f':
        return
    values = np.asarray(values)
    if values.size == 0 or values.dtype.kind != 'f':
        return
    # cfdb casts to the decoded dtype before encoding (a float64 654.35497 becomes float32 654.355 -> fill),
    # so check in that dtype; no copy when the block already has it (every WRF path: float32).
    decoded = getattr(dt, 'dtype_decoded', None)
    if decoded is not None and np.dtype(decoded).kind == 'f':
        values = values.astype(decoded, copy=False)
    info = np.iinfo(enc)
    fill = 0 if dt.fillvalue is None else dt.fillvalue
    off = 0 if offset is None else offset

    def codes(v):
        return (v - off) if factor is None else np.round((v - off) * factor)

    def bad(c):
        # NaN compares False everywhere, so a missing value is never "bad"; +/-inf is.
        return (c < info.min) | (c > info.max) | (c == fill)

    # nanmin/nanmax reduce without copying the block (it can be hundreds of MB); +/-inf propagate, and an
    # all-NaN block reduces to NaN (nothing to store, nothing to check).
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        vmin, vmax = np.nanmin(values), np.nanmax(values)
    if np.isnan(vmin):
        return
    if not (info.min < fill < info.max):
        extremes = np.array([vmin, vmax], dtype=values.dtype)
        with np.errstate(all='ignore'):
            if not bad(codes(extremes)).any():
                return
    with np.errstate(all='ignore'):
        n = int(np.count_nonzero(bad(codes(values))))
    if n:
        lo_code = info.min + (1 if fill == info.min else 0)
        hi_code = info.max - (1 if fill == info.max else 0)
        scale = 1 if factor is None else factor
        rng = (lo_code / scale + off, hi_code / scale + off)
        raise ValueError(
            f'{data_var.name!r}: {n} value(s) outside the storable range {rng[0]:g} .. {rng[1]:g} of its packed '
            f'dtype (block min {float(vmin):g}, max {float(vmax):g}); cfdb would store them as missing. '
            f'Refusing rather than losing data -- check the source values or widen the variable template.'
        )
