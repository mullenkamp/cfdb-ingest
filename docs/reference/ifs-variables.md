# IFS Variables

ECMWF IFS open-data forecast fields (GRIB2 `shortName`s) mapped to cfdb-vars names by `IfsIngest`
(`cfdb_ingest/ifs.py::IFS_VARIABLE_MAPPING`). This page is checked against the mapping by
`scripts/check_var_docs.py`.

## Coordinate types

Every IFS dataset is a `grid_forecast`:

- **Pressure-level variables**: `(forecast_reference_time, forecast_period, pressure, latitude, longitude)`
  -- the native levels (1000 ... 10 hPa, stored in Pa, ascending)
- **Surface variables**: `(forecast_reference_time, forecast_period, height_Xm, latitude, longitude)`
  -- a name that also exists on levels, or at more than one height, is suffixed (`air_temperature_2m`,
  `u_wind_10m`, `u_wind_100m`)
- **Soil variables**: `(forecast_reference_time, forecast_period, depth, latitude, longitude)` with
  `depth` = [0.07, 0.28, 1.0, 2.89] m (cumulative layer bottoms of the 0-7 / 7-28 / 28-100 / 100-289 cm layers)

All variables are stored as **unpacked float32** (the GRIB is already 12-16-bit quantised; the packed
cfdb-vars templates would floor `terrain_height` / `geopotential_height` at -0.9 m). Relative humidity is
a 0-1 fraction; `sea_ice` is a 0/1 flag derived from sea-ice thickness (the open data carries no
fraction); `land_sea_mask` is the IFS fraction; `sea_surface_temp` is skin temperature over water
(the open data carries no SST field and the IFS is ocean-coupled). Accumulated fields (`tp`, `ssrd`,
`strd`) become per-lead increments / mean fluxes with lead 0 missing.

## Pressure-level variables

| Key | cfdb Name | Source Var(s) | Transform |
|-----|-----------|---------------|-----------|
| `T` | `air_temp` | t | direct |
| `U` | `u_wind` | u | direct |
| `V` | `v_wind` | v | direct |
| `Q` | `specific_humidity` | q | direct |
| `GH` | `geopotential_height` | gh | direct |
| `RH` | `relative_humidity` | q, t | rh from q t p |

## Surface variables

`Z_SFC` (orography) exists only in the 0 h message set and is broadcast to every lead.

| Key | cfdb Name | Height | Source Var(s) | Transform |
|-----|-----------|--------|---------------|-----------|
| `T2` | `air_temp` | 2 m | 2t | direct |
| `TD2` | `dew_temp` | 2 m | 2d | direct |
| `RH2` | `relative_humidity` | 2 m | 2t, 2d | rh from t td |
| `U10` | `u_wind` | 10 m | 10u | direct |
| `V10` | `v_wind` | 10 m | 10v | direct |
| `U100` | `u_wind` | 100 m | 100u | direct |
| `V100` | `v_wind` | 100 m | 100v | direct |
| `FG10` | `wind_gust` | 10 m | 10fg | direct |
| `MSL` | `mslp` | 0 m | msl | direct |
| `SP` | `surface_pressure` | 0 m | sp | direct |
| `SKT` | `skin_temp` | 0 m | skt | direct |
| `SST` | `sea_surface_temp` | 0 m | skt, lsm | skt over water |
| `LSM` | `land_sea_mask` | 0 m | lsm | direct |
| `SEAICE` | `sea_ice` | 0 m | sithick | thickness to flag |
| `SD` | `snow_water_equiv` | 0 m | sd | m to kg m2 |
| `SNOWH` | `snow_depth` | 0 m | sd, rsn | snow physical depth |
| `RSN` | `snow_density` | 0 m | rsn | direct |
| `TP` | `precip` | 0 m | tp | accumulation increment m to mm |
| `SSRD` | `shortwave_radiation` | 0 m | ssrd | accumulation to mean flux |
| `STRD` | `longwave_radiation` | 0 m | strd | accumulation to mean flux |
| `TCWV` | `pwat` | 0 m | tcwv | direct |
| `CAPE` | `cape` | 0 m | mucape | direct |
| `Z_SFC` | `terrain_height` | 0 m | z | geopotential to height |

## Soil variables

| Key | cfdb Name | Source Var(s) | Transform |
|-----|-----------|---------------|-----------|
| `SOT` | `soil_layer_temp` | sot | direct |
| `VSW` | `soil_moisture` | vsw | direct |

## WPS preset

`cfdb-ingest ifs --preset wps` selects the rows `cfdb-to-int` needs for WRF forcing:
`T, U, V, Q, GH, RH, T2, TD2, RH2, U10, V10, MSL, SP, SKT, SST, LSM, SEAICE, SD, SNOWH, Z_SFC, SOT, VSW`.
