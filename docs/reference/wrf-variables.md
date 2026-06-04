# WRF Variables

## Coordinate types

Variables are stored with coordinates appropriate to their type:

- **Surface variables**: `(time, height_Xm, y, x)` -- named height coordinate (e.g. `height_0m`, `height_2m`, `height_10m`)
- **Level-interpolated variables**: `(time, height, y, x)` or `(time, pressure, y, x)`
- **Soil variables**: `(time, depth, y, x)`

When a surface variable shares a cfdb name with a 3D level-interpolated variable (e.g., `T2` and `T` both map to `air_temp`), the surface variant is suffixed with its height (e.g., `air_temp_2m`) to avoid conflicts.

## Surface variables

Fixed height above ground, stored as `(time, height_Xm, y, x)`:

| Key | cfdb Name | Height | Source Vars | Transform |
|-----|-----------|--------|-------------|-----------|
| `T2` | `air_temp` | 2 m | T2 | direct |
| `THETA2` | `potential_temperature` | 2 m | T2, PSFC | potential temperature |
| `THETA_E2` | `equivalent_potential_temperature` | 2 m | T2, Q2, PSFC | equivalent potential temperature |
| `Q2` | `mixing_ratio` | 2 m | Q2 | direct |
| `Q2_SH` | `specific_humidity` | 2 m | Q2 | mixing ratio to SH |
| `RH2` | `relative_humidity` | 2 m | T2, Q2, PSFC | Bolton (1980) |
| `TD2` | `dew_temp` | 2 m | Q2, PSFC | inverse Bolton |
| `PSFC` | `surface_pressure` | 0 m | PSFC | direct |
| `SLP` | `mslp` | 0 m | SLP _(fallback: PSFC, T2, HGT)_ | native; fallback = hypsometric reduction |
| `WIND10` | `wind_speed` | 10 m | U10, V10 | wind rotation |
| `WIND_DIR10` | `wind_direction` | 10 m | U10, V10 | wind rotation |
| `U10` | `u_wind` | 10 m | U10, V10 | wind rotation |
| `V10` | `v_wind` | 10 m | U10, V10 | wind rotation |
| `VORT10` | `vorticity` | 10 m | U10, V10 | relative vorticity |
| `RAIN` | `precip` | 0 m | RAINNC, RAINC | accumulation increment |
| `RAIN_TR` | `precip_tr` | 0 m | TR_RAINNC, TR_RAINC | accumulation increment |
| `SWDOWN` | `shortwave_radiation` | 0 m | SWDOWN | direct |
| `GLW` | `longwave_radiation` | 0 m | GLW | direct |
| `HFX` | `sensible_heat_flux` | 0 m | HFX | direct |
| `QFX` | `moisture_flux` | 0 m | QFX | direct |
| `ALBEDO` | `albedo` | 0 m | ALBEDO | direct |
| `EMISS` | `emissivity` | 0 m | EMISS | direct |
| `TSK` | `soil_temp` | 0 m | TSK | direct |
| `SNOWH` | `snow_depth` | 0 m | SNOWH | direct |
| `HGT` | `terrain_height` | 0 m | HGT | direct |
| `LU_INDEX` | `land_use_modis` | 0 m | LU_INDEX | direct |
| `XLAND` | `land_sea_mask` | 0 m | XLAND | 1/2 to 1/0 conversion |
| `SEAICE_VAR` | `sea_ice` | 0 m | SEAICE | direct |
| `SST_VAR` | `sea_surface_temp` | 0 m | SST | direct |
| `SNOW_VAR` | `snow_water_equiv` | 0 m | SNOW | direct |

## Column-integrated and moisture-transport variables

Vertically integrated quantities, stored as surface variables at `height_0m`. On recent WRF builds
(`wrf-auto-runs-intel-wvt:1.12`+) these are emitted directly and read as a native passthrough; on
older `wrfout` files the same cfdb output is reconstructed from 3D fields via the fallback shown in
parentheses. See [Native passthrough vs. computed fallback](../guide/wrf-ingestion.md#native-passthrough-vs-computed-fallback)
in the WRF guide for how resolution works.

| Key | cfdb Name | Height | Source Vars | Transform |
|-----|-----------|--------|-------------|-----------|
| `PWAT` | `pwat` | 0 m | PWAT _(fallback: QVAPOR, P, PB)_ | native; fallback = precipitable water |
| `PWAT_TR` | `pwat_tr` | 0 m | PWAT_TR _(fallback: qv_tr, P, PB)_ | native; fallback = tracer precipitable water |
| `VIMF_U` | `vimf_u` | 0 m | VIMF_U _(fallback: QVAPOR, U, V, P, PB)_ | native; fallback = (1/g)∫ q·u dp |
| `VIMF_V` | `vimf_v` | 0 m | VIMF_V _(fallback: QVAPOR, U, V, P, PB)_ | native; fallback = (1/g)∫ q·v dp |
| `VIMF_TR_U` | `vimf_tr_u` | 0 m | VIMF_TR_U | native only (WRF ≥ 1.12) |
| `VIMF_TR_V` | `vimf_tr_v` | 0 m | VIMF_TR_V | native only (WRF ≥ 1.12) |
| `IVT` | `ivt` | 0 m | IVT | native only (WRF ≥ 1.12) |

## 3D level-interpolated variables

Interpolated to user-specified `target_levels` on height or pressure coordinate:

| Key | cfdb Name | Source Vars | Transform |
|-----|-----------|-------------|-----------|
| `T` | `air_temp` | T, P, PB, PH, PHB | potential to actual temperature |
| `THETA` | `potential_temperature` | T, PH, PHB | potential temperature |
| `THETA_E` | `equivalent_potential_temperature` | T, P, PB, QVAPOR, PH, PHB | equivalent potential temperature |
| `WIND` | `wind_speed` | U, V, PH, PHB | unstagger + rotation |
| `WIND_DIR` | `wind_direction` | U, V, PH, PHB | unstagger + rotation |
| `U` | `u_wind` | U, V, PH, PHB | unstagger + rotation |
| `V` | `v_wind` | U, V, PH, PHB | unstagger + rotation |
| `QVAPOR` | `mixing_ratio` | QVAPOR, PH, PHB | level interpolation |
| `Q_SH` | `specific_humidity` | QVAPOR, PH, PHB | mixing ratio to specific humidity |
| `RH` | `relative_humidity` | T, P, PB, QVAPOR, PH, PHB | Bolton (1980) |
| `TD` | `dew_temp` | QVAPOR, P, PB, PH, PHB | inverse Bolton |
| `VORT` | `vorticity` | U, V, PH, PHB | relative vorticity (unstagger + rotation) |
| `GHT` | `geopotential_height` | PH, PHB | unstagger z, /9.81 |
| `W` | `vertical_velocity` | W, PH, PHB | unstagger z |

## Soil variables

Stored on a `depth` coordinate derived from WRF's DZS (soil layer thicknesses):

| Key | cfdb Name | Source Vars | Transform |
|-----|-----------|-------------|-----------|
| `SMOIS` | `soil_moisture` | SMOIS | direct (per layer) |
| `TSLB` | `soil_layer_temp` | TSLB | direct (per layer) |
