# SAIL water vapor isotope dataset — metadata

## Source and provenance

- Built from `atmospheric_isotopes_vsmow_final_cleaned.csv` (SHA-256 `fb0a87851e90ff6fa6ceea2b6ee92c80f6d290bd6458e4b5805caf7ad2585aee`).
- 1-minute time-rolling VSMOW-SLAP corrected Los Gatos analyzer record; humidity correction then 15-day sliding-window VSMOW-SLAP over 17 calibration runs.
- d-excess is the VSMOW-derived form (`d_excess_vsmow_derived`), matching the transfer-entropy analysis.
- Full measurement span, not winter-truncated.

## Columns

| Column | Units | Description |
| --- | --- | --- |
| time | UTC | Measurement minute (1-minute grid) |
| H2O_ppm | ppm | Water vapor mixing ratio |
| dD_vsmow_permil | permil vs VSMOW | delta D (deuterium) |
| d18O_vsmow_permil | permil vs VSMOW | delta 18-O |
| d_excess_vsmow_permil | permil vs VSMOW | Deuterium excess (dD - 8 x d18O) |
| is_interpolated | bool | True in the retained, in-filled malfunction window; False for measured or missing |
| dD_uncertainty_permil | permil | 1-sigma dD uncertainty (humidity model, averaging-reduced) |
| d18O_uncertainty_permil | permil | 1-sigma d18O uncertainty |
| d_excess_uncertainty_permil | permil | 1-sigma d-excess, propagated from dD/d18O (rho=0) |
| d_excess_uncertainty_direct_permil | permil | 1-sigma d-excess from the standalone d-excess fit |
| unc_extrapolated | bool | True where H2O is outside the 1350-8050 ppm fit range |

## Measurement uncertainty

Per-measurement 1-sigma follows a humidity-dependent fit calibrated on 13 standards runs (28,456 points): sigma(H2O) = a*exp(-b*H2O/1000) + c [permil].

| Quantity | a | b | c |
| --- | --- | --- | --- |
| dD | 20.0679 | 0.7753 | 1.9891 |
| d18O | 4.4482 | 0.6409 | 0.5109 |
| d-excess (direct) | 37.7864 | 0.6227 | 4.3991 |

Each published point is a uniform mean of its 5-minute window, so sigma is reduced to sigma/sqrt(N_eff), N_eff = N_native * 0.7719 (Bartlett factor for lag-1 autocorrelation rho1 = 0.1287); N_native is counted from the native stream per window (two-stage uniform averaging approximated as one window mean).

Median over measured points: N_native 30, N_eff 23, sigma_dD 0.641, sigma_d18O 0.190, sigma_d-excess 1.646 permil. 28,043 points are outside the fit range (flagged; below it the fit climbs steeply).

## Cleaning

- Calibration-run periods are absent (instrument on standards); reported as gaps in the coverage report.
- Instrument malfunction 2022-12-12 18:55-19:22 (28 min) linearly interpolated in place and flagged; every other outage left as an explicit NaN gap (measured-only).

## Verification against the analyzed data

Resampling this record with the analysis kernel (Gaussian sigma = 0.25 x interval, +/-4 sigma; linear fill of gaps <= 3 h) reproduces the frozen `final_1hr.csv` / `final_6hr.csv` isotope columns.

**Result: PASS** (tolerance 1e-06 permil/ppm).

| Interval | Column | N | max abs diff | Pass |
| --- | --- | --- | --- | --- |
| 1h | H2O_ppm | 4449 | 5.46e-12 | yes |
| 1h | dD_vsmow_permil | 4449 | 2.27e-13 | yes |
| 1h | d18O_vsmow_permil | 4449 | 2.13e-14 | yes |
| 1h | d_excess_vsmow_permil | 4449 | 2.13e-14 | yes |
| 6h | H2O_ppm | 752 | 3.64e-12 | yes |
| 6h | dD_vsmow_permil | 752 | 1.14e-13 | yes |
| 6h | d18O_vsmow_permil | 752 | 2.13e-14 | yes |
| 6h | d_excess_vsmow_permil | 752 | 1.42e-14 | yes |
