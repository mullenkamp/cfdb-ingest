"""
Moist-thermodynamic diagnostics shared by the ingest sources and the WPS exporter.

Relative humidity is a FRACTION (0-1) everywhere in cfdb-ingest (the cfdb-vars
``relative_humidity`` convention); the WPS exporter multiplies by 100 on the way out.
"""

import numpy as np

# Thompson-microphysics RSLF polynomial (liquid saturation vapour pressure, Pa),
# as used by NCAR's era5_to_int so that IFS- and ERA5-forced runs share one formula.
_RSLF_COEFFS = (
    0.611583699e03,
    0.444606896e02,
    0.143177157e01,
    0.264224321e-1,
    0.299291081e-3,
    0.203154182e-5,
    0.702620698e-8,
    0.379534310e-11,
    -0.321582393e-13,
)
_T0 = 273.16
_LV = 2.5e6  # J kg-1
_RV = 461.5  # J kg-1 K-1
_EPS = 0.622


def saturation_vapour_pressure_liquid(t):
    """
    Saturation vapour pressure over liquid water [Pa] from temperature [K] (Thompson RSLF).
    """
    t = np.asarray(t, dtype='float64')
    tc = np.maximum(-80.0, t - _T0)
    es = np.zeros_like(tc) + _RSLF_COEFFS[-1]
    for c in reversed(_RSLF_COEFFS[:-1]):
        es = c + tc * es
    return es


def saturation_mixing_ratio_liquid(t, p):
    """
    Saturation mixing ratio [kg kg-1] over liquid water at temperature ``t`` [K] and pressure ``p`` [Pa].

    ``es`` is capped at 15 % of ``p`` (era5_to_int's guard: even at 1050 hPa and 55 C the
    saturation vapour pressure is only ~15 % of the total).
    """
    p = np.asarray(p, dtype='float64')
    es = np.minimum(saturation_vapour_pressure_liquid(t), p * 0.15)
    return _EPS * es / (p - es)


def rh_from_q_t_p(q, t, p):
    """
    Relative humidity FRACTION from specific humidity ``q`` [kg kg-1], temperature ``t`` [K] and
    pressure ``p`` [Pa], clipped to 0-1. Mirrors era5_to_int's RHDiags.
    """
    q = np.asarray(q, dtype='float64')
    w = q / (1.0 - q)
    rh = w / saturation_mixing_ratio_liquid(t, p)
    return np.clip(rh, 0.0, 1.0).astype('float32')


def rh_from_t_td(t, td):
    """
    Relative humidity FRACTION from temperature ``t`` [K] and dew point ``td`` [K] (Clausius-Clapeyron,
    as era5_to_int's RH2mDiags), clipped to 0-1.
    """
    t = np.asarray(t, dtype='float64')
    td = np.asarray(td, dtype='float64')
    rh = np.exp(_LV / _RV * (1.0 / t - 1.0 / td))
    return np.clip(rh, 0.0, 1.0).astype('float32')
