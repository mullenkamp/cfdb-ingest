# WRF Pressure-Level Variables

`WrfPlevIngest` (since 0.7.0) reads WRF's pressure-level diagnostics: namelist `&diags` `p_lev_diags = 1`,
written to output stream auxhist23 (`wrfplevels_d0N_*` files). WRF interpolates to the namelist
`press_levels` itself (`phys/module_diag_pld.F`); cfdb-ingest stores those levels as they are.

```python
from cfdb_ingest import WrfPlevIngest

ing = WrfPlevIngest('/path/to/run')                  # a directory globs wrfplevels*
result = ing.convert('plevels.cfdb', variables=['GHT_PL', 'T_PL', 'U_PL', 'V_PL'],
                     chunk_shape=(24, 1, len(ing.y), len(ing.x)))
result['masked_cells']   # {source variable: {pressure (Pa): cells set to NaN}}
```

## Coordinates

Every variable is `(time, pressure, y, x)`. `pressure` (Pa, ascending) is the files' own `P_PL`, which
must be identical in every frame of every file. `target_levels` selects a subset of those levels; no
interpolation is offered, and `vertical_coord` other than `'pressure'` is refused. Grid mode only:
`extend`, `time_label='start'`, `squeeze_height` and forecast mode are refused.

WRF writes the levels in namelist order, normally **descending**, and cfdb-ingest reorders them onto the
ascending coordinate. WRF's own search only works for a descending `press_levels`: with any other order
it leaves later levels entirely missing, and `convert` warns about a level that is missing everywhere.

## Missing values

WRF writes `p_lev_missing` (default -999, the `missing_value` argument) where a level cannot be
interpolated. With `extrap_below_grnd = 1` that is below the lowest model half-level for T/U/V/Q/RH/TD/S,
and below ground for GHT, which WRF interpolates on full levels — so **the masks differ by variable**:
GHT can be valid over high terrain where temperature is missing. A level above the model top is missing
in every mode. All become NaN before any transform. The files do not record `extrap_below_grnd`; the
`masked_cells` counts in the result tell a caller which mode produced them.

## Wind rotation

`U_PL`/`V_PL` are **grid-relative** in the file and the auxiliary stream carries no
`COSALPHA`/`SINALPHA`. They are rotated with an analytic angle from the projection attributes
(`grid_rotation`), which for the tangent-cone Lambert grid is validated against WRF's own `COSALPHA`
(max 2.7e-7 rad on real output). For the secant-cone Lambert, polar stereographic and Mercator grids the
formula is not yet validated: pass `static_path=` a wrfout of the same domain, whose
`COSALPHA`/`SINALPHA` must agree with the formula to 1e-5 rad (else refused) and is then used; without it
the winds are unavailable and the scalars still convert.

## Variables

| Key | cfdb Name | Source Vars | Transform |
|-----|-----------|-------------|-----------|
| `GHT_PL` | `geopotential_height` | GHT_PL | direct |
| `T_PL` | `air_temp` | T_PL | direct |
| `Q_PL` | `mixing_ratio` | Q_PL | direct (WRF's Q_PL is a mixing ratio) |
| `U_PL` | `u_wind` | U_PL, V_PL | grid -> earth rotation |
| `V_PL` | `v_wind` | U_PL, V_PL | grid -> earth rotation |
| `RH_PL` | `relative_humidity` | RH_PL | percent -> fraction, clipped to [0, 1] |
| `TD_PL` | `dew_temp` | TD_PL | direct |
| `S_PL` | `wind_speed` | S_PL | direct |
