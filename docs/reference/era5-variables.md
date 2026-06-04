# ERA5 Variables

## Coordinate types

Variables are stored with coordinates appropriate to their type:

- **Surface variables**: `(time, height_Xm, latitude, longitude)` -- named height coordinate (e.g. `height_0m`, `height_2m`, `height_10m`, `height_100m`)
- **Pressure level variables**: `(time, pressure, latitude, longitude)` -- pressure coordinate with `axis='Z'`

When a variable name conflicts between surface and pressure levels, the surface variant is suffixed with its height (e.g. `air_temperature_2m`).

## Surface variables at 0m

| Key | cfdb Name | Source Var | Description |
|-----|-----------|-----------|-------------|
| `SP` | `surface_pressure` | SP | Surface pressure |
| `MSL` | `mslp` | MSL | Mean sea level pressure |
| `SSTK` | `sea_surface_temp` | SSTK | Sea surface temperature |
| `SKT` | `skin_temperature` | SKT | Skin temperature |
| `CI` | `sea_ice` | CI | Sea-ice cover |
| `SD` | `snow_water_equiv` | SD | Snow depth (water equiv) |
| `RSN` | `snow_density` | RSN | Snow density |
| `ASN` | `snow_albedo` | ASN | Snow albedo |
| `TSN` | `snow_layer_temperature` | TSN | Temperature of snow layer |
| `FAL` | `albedo` | FAL | Forecast albedo |
| `STL1-4` | `soil_layer_temp` | STL1-4 | Soil temperature layers 1-4 |
| `SWVL1-4` | `soil_moisture` | SWVL1-4 | Soil water volume layers 1-4 |
| `ISTL1-4` | `ice_surface_temperature` | ISTL1-4 | Ice surface temperature layers 1-4 |
| `CAPE` | `cape` | CAPE | Convective available potential energy |
| `BLH` | `boundary_layer_height` | BLH | Boundary layer height |
| `TCC` | `total_cloud_cover` | TCC | Total cloud cover |
| `LCC` | `low_cloud_cover` | LCC | Low cloud cover |
| `MCC` | `medium_cloud_cover` | MCC | Medium cloud cover |
| `HCC` | `high_cloud_cover` | HCC | High cloud cover |
| `TCW` | `total_column_water` | TCW | Total column water |
| `TCWV` | `pwat` | TCWV | Total column water vapour |
| `TCLW` | `total_column_liquid_water` | TCLW | Total column liquid water |
| `TCIW` | `total_column_ice_water` | TCIW | Total column ice water |
| `TCRW` | `total_column_rain_water` | TCRW | Total column rain water |
| `TCSW` | `total_column_snow_water` | TCSW | Total column snow water |
| `TCO3` | `total_column_ozone` | TCO3 | Total column ozone |
| `CHNK` | `charnock` | CHNK | Charnock parameter |
| `SRC` | `skin_reservoir_content` | SRC | Skin reservoir content |
| `FSR` | `surface_roughness` | FSR | Forecast surface roughness |
| `FLSR` | `surface_roughness_heat` | FLSR | Forecast log of surface roughness for heat |
| `IEWS` | `surface_stress_east` | IEWS | Instantaneous eastward turbulent surface stress |
| `INSS` | `surface_stress_north` | INSS | Instantaneous northward turbulent surface stress |
| `ISHF` | `sensible_heat_flux` | ISHF | Instantaneous surface sensible heat flux |
| `IE` | `moisture_flux` | IE | Instantaneous moisture flux |
| `ALUVP` | `uv_albedo_direct` | ALUVP | UV visible albedo for direct radiation |
| `ALUVD` | `uv_albedo_diffuse` | ALUVD | UV visible albedo for diffuse radiation |
| `ALNIP` | `nir_albedo_direct` | ALNIP | Near IR albedo for direct radiation |
| `ALNID` | `nir_albedo_diffuse` | ALNID | Near IR albedo for diffuse radiation |
| `LAILV` | `leaf_area_index_low` | LAILV | Leaf area index, low vegetation |
| `LAIHV` | `leaf_area_index_high` | LAIHV | Leaf area index, high vegetation |
| `LBLT` | `lake_bottom_temperature` | LBLT | Lake bottom temperature |
| `LTLT` | `lake_total_layer_temperature` | LTLT | Lake total layer temperature |
| `LSHF` | `lake_shape_factor` | LSHF | Lake shape factor |
| `LICT` | `lake_ice_temperature` | LICT | Lake ice temperature |
| `LICD` | `lake_ice_depth` | LICD | Lake ice depth |

## Surface variables at 2m

| Key | cfdb Name | Source Var | Description |
|-----|-----------|-----------|-------------|
| `VAR_2T` | `air_temperature` | VAR_2T | 2 metre temperature |
| `VAR_2D` | `dew_point_temperature` | VAR_2D | 2 metre dewpoint temperature |

## Surface variables at 10m

| Key | cfdb Name | Source Var | Description |
|-----|-----------|-----------|-------------|
| `VAR_10U` | `u_wind` | VAR_10U | 10 metre U wind component |
| `VAR_10V` | `v_wind` | VAR_10V | 10 metre V wind component |
| `U10N` | `u_wind` | U10N | Neutral wind at 10m U-component |
| `V10N` | `v_wind` | V10N | Neutral wind at 10m V-component |

## Surface variables at 100m

| Key | cfdb Name | Source Var | Description |
|-----|-----------|-----------|-------------|
| `VAR_100U` | `u_wind` | VAR_100U | 100 metre U wind component |
| `VAR_100V` | `v_wind` | VAR_100V | 100 metre V wind component |

## Column-integrated moisture flux (height 0m)

Derived by vertically integrating the pressure-level humidity and wind fields. See the
[VIMF computation](../guide/era5-ingestion.md#vimf-computation) section of the ERA5 guide for details.

| Key | cfdb Name | Source Vars | Transform | Description |
|-----|-----------|-----------|-----------|-------------|
| `VIMF_U` | `vimf_u` | Q, U | (1/g)∫ q·u dp | Vertically integrated eastward moisture flux |
| `VIMF_V` | `vimf_v` | Q, V | (1/g)∫ q·v dp | Vertically integrated northward moisture flux |

## Invariant variables (height 0m)

| Key | cfdb Name | Source Var | Transform | Description |
|-----|-----------|-----------|-----------|-------------|
| `Z_INV` | `terrain_height` | Z | geopotential / g | Geopotential at the surface |
| `LSM` | `land_sea_mask` | LSM | | Land-sea mask |
| `CL` | `lake_cover` | CL | | Lake cover |
| `DL` | `lake_depth` | DL | | Lake depth |
| `CVL` | `low_vegetation_cover` | CVL | | Low vegetation cover |
| `CVH` | `high_vegetation_cover` | CVH | | High vegetation cover |
| `TVL` | `low_vegetation_type` | TVL | | Type of low vegetation |
| `TVH` | `high_vegetation_type` | TVH | | Type of high vegetation |
| `SLT` | `soil_type` | SLT | | Soil type |
| `SDFOR` | `std_dev_filtered_orography` | SDFOR | | Std dev of filtered subgrid orography |
| `SDOR` | `std_dev_orography` | SDOR | | Standard deviation of orography |
| `ISOR` | `orography_anisotropy` | ISOR | | Anisotropy of sub-gridscale orography |
| `ANOR` | `orography_angle` | ANOR | | Angle of sub-gridscale orography |
| `SLOR` | `orography_slope` | SLOR | | Slope of sub-gridscale orography |

## Pressure level variables

| Key | cfdb Name | Source Var | Transform | Description |
|-----|-----------|-----------|-----------|-------------|
| `T` | `air_temperature` | T | | Temperature |
| `U` | `u_wind` | U | | U component of wind |
| `V` | `v_wind` | V | | V component of wind |
| `Z_PL` | `geopotential_height` | Z | geopotential / g | Geopotential height |
| `Q` | `specific_humidity` | Q | | Specific humidity |
| `W` | `vertical_velocity` | W | | Vertical velocity |
| `VO` | `vorticity` | VO | | Vorticity (relative) |
| `D` | `divergence` | D | | Divergence |
| `R` | `relative_humidity` | R | | Relative humidity |
| `O3` | `ozone_mixing_ratio` | O3 | | Ozone mass mixing ratio |
| `PV` | `potential_vorticity` | PV | | Potential vorticity |
| `CC` | `cloud_cover` | CC | | Cloud cover |
| `CLWC` | `cloud_liquid_water_content` | CLWC | | Specific cloud liquid water content |
| `CIWC` | `cloud_ice_water_content` | CIWC | | Specific cloud ice water content |
| `CRWC` | `rain_water_content` | CRWC | | Specific rain water content |
| `CSWC` | `snow_water_content` | CSWC | | Specific snow water content |
