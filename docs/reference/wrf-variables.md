# WRF Variables

## Coordinate types

Variables are stored with coordinates appropriate to their type:

- **Surface variables**: `(time, y, x)` -- no vertical dimension
- **Level-interpolated variables**: `(time, height, y, x)` or `(time, pressure, y, x)`
- **Soil variables**: `(time, depth, y, x)`

When a surface variable shares a cfdb name with a 3D level-interpolated variable (e.g., `T2` and `T` both map to `air_temp`), the surface variant is automatically suffixed with `_sfc` (e.g., `air_temp_sfc`) to avoid conflicts.

## Surface variables

Fixed height above ground, stored as `(time, y, x)`:

| Key | cfdb Name | Height | Source Vars | Transform |
|-----|-----------|--------|-------------|-----------|
| `T2` | `air_temp` | 2 m | T2 | direct |
| `PSFC` | `surface_pressure` | 0 m | PSFC | direct |
| `Q2` | `mixing_ratio` | 2 m | Q2 | direct |
| `Q2_SH` | `specific_humidity` | 2 m | Q2 | mixing ratio to SH |
| `RH2` | `relative_humidity` | 2 m | T2, Q2, PSFC | Bolton (1980) |
| `TD2` | `dew_temp` | 2 m | Q2, PSFC | inverse Bolton |
| `RAIN` | `precip` | 0 m | RAINNC, RAINC | accumulation increment |
| `WIND10` | `wind_speed` | 10 m | U10, V10 | wind rotation |
| `WIND_DIR10` | `wind_direction` | 10 m | U10, V10 | wind rotation |
| `U10` | `u_wind` | 10 m | U10, V10 | wind rotation |
| `V10` | `v_wind` | 10 m | U10, V10 | wind rotation |
| `TSK` | `soil_temp` | 0 m | TSK | direct |
| `SWDOWN` | `shortwave_radiation` | 0 m | SWDOWN | direct |
| `GLW` | `longwave_radiation` | 0 m | GLW | direct |
| `SNOWH` | `snow_depth` | 0 m | SNOWH | direct |
| `HGT` | `terrain_height` | 0 m | HGT | direct |
| `SLP` | `mslp` | 0 m | PSFC, T2, HGT | hypsometric reduction |
| `XLAND` | `land_sea_mask` | 0 m | XLAND | 1/2 to 1/0 conversion |
| `SEAICE_VAR` | `sea_ice` | 0 m | SEAICE | direct |
| `SST_VAR` | `sea_surface_temp` | 0 m | SST | direct |
| `SNOW_VAR` | `snow_water_equiv` | 0 m | SNOW | direct |

## 3D level-interpolated variables

Interpolated to user-specified `target_levels` on height or pressure coordinate:

| Key | cfdb Name | Source Vars | Transform |
|-----|-----------|-------------|-----------|
| `T` | `air_temp` | T, P, PB, PH, PHB | potential to actual temperature |
| `WIND` | `wind_speed` | U, V, PH, PHB | unstagger + rotation |
| `WIND_DIR` | `wind_direction` | U, V, PH, PHB | unstagger + rotation |
| `U` | `u_wind` | U, V, PH, PHB | unstagger + rotation |
| `V` | `v_wind` | U, V, PH, PHB | unstagger + rotation |
| `QVAPOR` | `mixing_ratio` | QVAPOR, PH, PHB | level interpolation |
| `Q_SH` | `specific_humidity` | QVAPOR, PH, PHB | mixing ratio to specific humidity |
| `RH` | `relative_humidity` | T, P, PB, QVAPOR, PH, PHB | Bolton (1980) |
| `TD` | `dew_temp` | QVAPOR, P, PB, PH, PHB | inverse Bolton |
| `GHT` | `geopotential_height` | PH, PHB | unstagger z, /9.81 |
| `W` | `vertical_velocity` | W, PH, PHB | unstagger z |

## Soil variables

Stored on a `depth` coordinate derived from WRF's DZS (soil layer thicknesses):

| Key | cfdb Name | Source Vars | Transform |
|-----|-----------|-------------|-----------|
| `SMOIS` | `soil_moisture` | SMOIS | direct (per layer) |
| `TSLB` | `soil_layer_temp` | TSLB | direct (per layer) |
