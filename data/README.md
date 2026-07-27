# Manuscript Data — finalized calibrated products + provenance

**Purpose:** self-contained copies of the finalized data this manuscript
reports on, plus the single calibration line actually used and its math,
so the repository reproduces every number without reaching outside itself.
**Status:** active
**Created:** 2026-07 — **Updated:** 2026-07

This records ONLY the pipeline actually used. Multi-GB intermediates are
referenced, not copied. Source: the SAIL isotope-analysis pipeline
(`PROJECTS/SAIL/isotope-analysis-pipeline/`); narrative:
`WRITING/Final Draft/appendix_a_calibration.tex`.

## Contents

### `analysis/`
- `final_1hr.csv` — the analyzed base stream (real-unit, spike-removed,
  1-hour), robust-z normalized into `final_1hr_beta.csv` for transfer
  entropy. Byte-identical to `MODELS/TE_V1.0.0/data/final_1hr.csv` and the
  pipeline's `data/processed/extended_june/unnormalized/final_1hr.csv`.
  Isotope columns `dD, d18O, d_excess, H2O_ppm` are the calibrated Los Gatos
  analyzer record; the rest are ARM/SAIL met + radiation + flux and ERA5.

### `calibration/`
- `standards_calibrated.csv` — the DATA the calibration was fit from:
  17,080 standard-water measurements with raw (`D_del`, `O18_del`) and
  humidity-corrected (`D_del_corrected`, `O18_del_corrected`) isotopes vs
  `H2O_ppm`, plus Los Gatos analyzer diagnostics and per-run identifiers.
- `fit_parameters_summary.csv` — the per-run 4th-order humidity-correction
  polynomial coefficients and quality scores.
- `sliding_window_standards_data.csv` — per-run standard measured-vs-known
  (the rolling VSMOW-SLAP input).
- `sliding_window_vsmow_corrections.csv` — the per-run rolling VSMOW-SLAP
  offsets actually applied.
- `vsmow_correction_validation.csv` — before/after RMSE and residual
  offsets for the rolling normalization.
- `calibration_points.csv` — the five standards' known VSMOW compositions.
- `uncertainty_summary.json` — the humidity-dependent measurement-precision
  fit of record (13 standards runs, 28,456 points; `sigma(H2O) =
  a*exp(-b*H2O/1000)+c` per isotope). Applied per data point in `published/`.
- `autocorr_summary.json` — instrument lag-1 noise autocorrelation
  (rho1 = 0.129) used to reduce per-point sigma by the averaging factor.

### `published/`
- The archival, DOI-ready 1-minute isotope dataset with per-point
  uncertainties and full coverage accounting, plus its metadata and
  coverage report. See `published/README.md`. This IS the analyzed stream
  (verified: it aggregates to `analysis/final_1hr.csv` to ~1e-12).

## Calibration line (exactly as used; math per step)

1. **Raw Los Gatos analyzer.** Standards + atmospheric vapor. REFERENCED, not copied:
   pipeline `outputs/complete_calibration/data/raw_calibrated.csv` (~2 GB).

2. **Humidity (concentration-dependence) correction.** A fourth-order
   polynomial in H2O removes the low-humidity bias of each isotope
   [`fit_parameters_summary.csv`]:
   `delta_corrected = delta_measured - P4(H2O)`,
   `P4(H2O) = c0 + c1*H2O + c2*H2O^2 + c3*H2O^3 + c4*H2O^4`.

3. **Rolling VSMOW-SLAP normalization.** A sliding window over the
   campaign builds a time-varying offset from the per-run standards
   (measured vs known, `sliding_window_standards_data.csv` ->
   `sliding_window_vsmow_corrections.csv`), placing every measurement on
   the VSMOW-SLAP scale while tracking instrument drift. Validation
   [`vsmow_correction_validation.csv`]: RMSE 6.87 -> 1.39 permil dD
   (~80%), 3.97 -> 1.07 permil d18O (~73%); final residual offsets
   -0.5 +/- 1.3 permil dD, +0.02 +/- 1.1 permil d18O.

4. **Humidity-dependent measurement precision.** Derived from the
   rolling-normalized standards' scatter binned in H2O and fit with
   `sigma(H2O) = a * exp(-b * H2O/1000) + c` per isotope. The actual fit
   (the "updated uncertainty metrics") is `calibration/uncertainty_summary.json`
   (dD a=20.07, b=0.775, c=1.99; d18O a=4.45, b=0.641, c=0.511); it
   supersedes any earlier characterization. It is applied per data point in
   the published 1-minute product (`published/`, via
   `../code/pub_isotope_uncertainty.py`), reduced by the 5-minute-window
   sample count with the `calibration/autocorr_summary.json` autocorrelation.

5. **Cleaning + resample + merge.** A several-point ~200 permil positive
   dD instrument-malfunction spike is removed (the spike-removed record is
   the accurate one). The calibrated 1-minute record is resampled to
   hourly (isotopes = mean) and merged with met/radiation/flux/reanalysis
   to give `analysis/final_1hr.csv`. The 1-minute calibrated product
   `outputs/time_rolling_vsmow_analysis/atmospheric_isotopes_vsmow_final_cleaned.csv`
   is bundled (cleaned, with per-point uncertainties) as the published
   dataset in `published/`. **CORRECTION (2026-07-16):** it IS the analyzed
   stream — aggregating it with the analysis kernel reproduces
   `final_1hr`/`final_6hr` to ~1e-12 (verified; the 2022-12-12 malfunction
   window is already interpolated in it). The earlier note that it "predates
   spike removal / r = 0.997 / never report from it" was mistaken; that r
   compared the 1-minute series against the 1-hour series without
   aggregating first.

## Provenance / Notes

- Authoritative analyzed stream and the vintage trap: memory
  `manuscript-isotope-base-stream`. The pipeline's `complete_calibration/`
  outputs use a static linear VSMOW fit and are a DIFFERENT branch that is
  NOT used here; they are deliberately excluded.
- Generating code lives in `../code/` and `MODELS/TE_V1.0.0/`
  (`pub_style.py`, the calibration figure script, `pub_data_coverage.py`).
