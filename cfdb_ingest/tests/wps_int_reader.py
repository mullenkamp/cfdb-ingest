"""
Minimal reader for WPS intermediate-format files (big-endian Fortran sequential records, version 5),
the inverse of ``wrf_to_int.IntermediateFile.write_next_met_field``. Test support only.
"""

import struct

import numpy as np

_PROJ_NFLOATS = {0: 5, 1: 6, 3: 8, 4: 5, 5: 7}  # LATLON, MERC, LC, GAUSS, PS


def _records(data):
    pos = 0
    while pos < len(data):
        (n,) = struct.unpack('>i', data[pos : pos + 4])
        payload = data[pos + 4 : pos + 4 + n]
        (n2,) = struct.unpack('>i', data[pos + 4 + n : pos + 8 + n])
        if n2 != n:
            raise ValueError(f'record length mismatch at byte {pos}: {n} vs {n2}')
        pos += 8 + n
        yield payload


def read_int_file(path):
    """Return a list of dicts, one per field: field, xlvl, hdate, units, desc, nx, ny, iproj, proj, slab (ny, nx)."""
    with open(path, 'rb') as f:
        data = f.read()
    recs = _records(data)
    fields = []
    while True:
        try:
            version = struct.unpack('>i', next(recs))[0]
        except StopIteration:
            return fields
        if version != 5:
            raise ValueError(f'unsupported intermediate-format version {version}')
        hdr = next(recs)
        hdate = hdr[0:24].decode().strip()
        (xfcst,) = struct.unpack('>f', hdr[24:28])
        map_source = hdr[28:60].decode().strip()
        field = hdr[60:69].decode().strip()
        units = hdr[69:94].decode().strip()
        desc = hdr[94:140].decode().strip()
        xlvl, nx, ny, iproj = struct.unpack('>fiii', hdr[140:156])
        proj_rec = next(recs)
        nfloats = _PROJ_NFLOATS[iproj]
        startloc = proj_rec[0:8].decode().strip()
        floats = struct.unpack('>' + 'f' * nfloats, proj_rec[8 : 8 + 4 * nfloats])
        proj = {'iproj': iproj, 'startloc': startloc, 'floats': floats}
        if iproj == 0:
            proj.update(dict(zip(('startlat', 'startlon', 'deltalat', 'deltalon', 'earth_radius'), floats)))
        elif iproj == 3:
            proj.update(
                dict(zip(('startlat', 'startlon', 'dx', 'dy', 'xlonc', 'truelat1', 'truelat2', 'earth_radius'), floats))
            )
        (is_wind_grid_rel,) = struct.unpack('>i', next(recs))
        slab = np.frombuffer(next(recs), dtype='>f4').reshape(ny, nx)  # C order: x fastest
        fields.append(
            {
                'field': field,
                'xlvl': xlvl,
                'hdate': hdate,
                'xfcst': xfcst,
                'map_source': map_source,
                'units': units,
                'desc': desc,
                'nx': nx,
                'ny': ny,
                'iproj': iproj,
                'proj': proj,
                'is_wind_grid_rel': is_wind_grid_rel,
                'slab': slab.astype('float32'),
            }
        )


def fields_by_name(records):
    """{(field, xlvl): slab} for quick lookups in tests."""
    return {(r['field'], round(float(r['xlvl']), 1)): r['slab'] for r in records}
