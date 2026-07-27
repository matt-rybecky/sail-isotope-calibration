# SAIL water vapor stable isotope dataset (recalibrated) + calibration code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21630381.svg)](https://doi.org/10.5281/zenodo.21630381)

The archival, DOI-ready 1-minute water vapor stable isotope record (dD, d18O,
d-excess) from the SAIL campaign (East River watershed, Colorado, winter
2022-23), with per-point uncertainties and full coverage accounting, together
with the exact calibration code that produced it. Released to accompany a
research manuscript on phase-change signatures in water vapor stable isotopes.

This dataset is a recalibration of the ARM SAIL water vapor isotope
measurements (Galewsky WVISO, https://doi.org/10.5439/2280801): a
humidity (concentration-dependence) correction followed by a 15-day
sliding-window VSMOW-SLAP normalization over 17 standards runs, with a
humidity-dependent per-point uncertainty model. It is the exact stream the
transfer-entropy analysis was run on.

## What is included

- **The dataset** (`data/published/`) — `sail_water_vapor_isotopes_1min.csv`,
  a complete 276,154-minute grid over the full measurement span
  (2022-11-21 20:12 to 2023-06-01 14:45), 261,757 measured minutes, with
  VSMOW dD/d18O/d-excess, per-point 1-sigma uncertainties, and interpolation
  flags. See `data/published/dataset_metadata.md` for the column dictionary,
  source SHA-256, and the uncertainty model.
- **The calibration code** (`code/`) — the humidity correction and rolling
  VSMOW-SLAP corrector, the standards fitting, the per-point uncertainty
  model of record (`pub_isotope_uncertainty.py`), the dataset builder
  (`pub_published_dataset.py`), and the aggregation check that verifies the
  1-minute product reproduces the analyzed hourly stream (`pub_te_match.py`).
- **The calibration inputs** (`data/calibration/`) — the standards
  measurements, per-run humidity-correction polynomials, sliding-window
  VSMOW-SLAP offsets, the five standards' known VSMOW compositions, and the
  uncertainty/autocorrelation summaries the model is built from.
- **An intercomparison against ARM** (`comparison/`) — this recalibration
  against the ARM-archived predecessor (DOI 10.5439/2280801), an independent
  earlier calibration of the same Los Gatos analyzer measurements.

## The dataset

Columns: `time` (UTC minute), `H2O_ppm`, `dD_vsmow_permil`,
`d18O_vsmow_permil`, `d_excess_vsmow_permil`, `is_interpolated`,
`dD_uncertainty_permil`, `d18O_uncertainty_permil`,
`d_excess_uncertainty_permil` (propagated), `d_excess_uncertainty_direct_permil`,
`unc_extrapolated`. Calibration-run periods (instrument on standards) appear
as explicit NaN gaps; one 28-minute instrument-malfunction window is
interpolated in place and flagged. Full provenance, the humidity-dependent
uncertainty fit, and the aggregation-verification table are in
`data/published/dataset_metadata.md`; the gap-by-gap coverage report is in
`data/published/coverage_report.{md,csv}`.

## Calibration method

Documented step by step, with the math, in `data/README.md`:

1. Raw Los Gatos analyzer (standards + atmospheric vapor).
2. Humidity correction: a fourth-order polynomial in H2O removes the
   low-humidity concentration bias of each isotope, per run.
3. Rolling VSMOW-SLAP normalization: 15-day sliding-window linear correction
   to the VSMOW-SLAP scale from 17 standards runs.
4. Per-point uncertainty: a humidity-dependent 1-sigma model
   (`sigma(H2O) = a*exp(-b*H2O/1000) + c`, 13 standards runs, 28,456 points),
   reduced by the within-window averaging factor.

## Comparison to the ARM archived record

`comparison/` intercompares this recalibration against the ARM-archived SAIL
WVISO datastream (DOI 10.5439/2280801) on a common hourly grid (n = 4,404
overlapping hours). The two are independent calibrations of the same analyzer
measurements, so the tools are Deming (errors-in-both) regression and
Bland-Altman agreement, plus difference-vs-time (drift) and difference-vs-H2O.

| variable | bias (this study − ARM) | Deming slope | Pearson r | drift (permil / 30 d) |
| --- | --- | --- | --- | --- |
| dD       | +9.33 permil | 1.022 | 0.995 | −0.78 (flat) |
| d18O     | +1.71 permil | 1.306 | 0.964 | +1.18 |
| d-excess | −4.33 permil | 1.345 | 0.528 | −10.18 |

dD differs by a near-constant offset (flat drift: the two calibrations agree
in shape). d18O carries a seasonal drift with a clear humidity dependence, the
expected signature of this study's rolling VSMOW-SLAP plus humidity correction
against ARM's earlier static calibration. Because d-excess = dD − 8·d18O, that
d18O drift amplifies into the large seasonal d-excess divergence. Figures:
`comparison/arm_overlay_timeseries` (both calibrations overlaid),
`comparison/arm_scatter` (this study vs ARM with 1:1 and Deming fits),
`comparison/arm_difference_timeseries` (the differences), plus per-isotope
Bland-Altman diagnostics; statistics in `comparison/arm_comparison_stats.{md,csv}`.

## Install

From the repository root, in a fresh virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.9 or newer.

## Reproducing

The dataset ships built; the code regenerates it and the comparison.

- **Rebuild the published dataset** (needs the multi-GB pipeline VSMOW output,
  referenced in `data/README.md`, not shipped):

  ```bash
  cd code
  python3 pub_published_dataset.py
  ```

- **Rerun the ARM intercomparison.** The ARM WVISO record is external; order
  it from the ARM archive (DOI 10.5439/2280801) and point the script at the
  delivered file (a plain-CSV `.dat`, or an ARM netCDF):

  ```bash
  cd code
  python3 pub_arm_comparison.py --arm-path /path/to/ARM_wviso.dat \
      --out-dir arm_comparison_output
  # discover ARM variable names first, if needed:
  python3 pub_arm_comparison.py --arm-path /path/to/file --list-vars
  # validate the harness against our own file:
  python3 pub_arm_comparison.py --self-test
  ```

  The committed figures under `comparison/` were produced this way from the
  ARM-archived `.dat` (columns `Time, Mixing Ratio (g/kg), dD (permil),
  d18O (permil)`), mapped by the defaults in `pub_arm_comparison.py`.

## Repository map

- `data/published/` — the DOI dataset, its metadata, and coverage report.
- `data/calibration/` — the calibration inputs (standards, polynomials,
  VSMOW-SLAP offsets, uncertainty/autocorrelation summaries).
- `data/analysis/final_1hr.csv` — the analyzed hourly stream the 1-minute
  dataset aggregates to (isotope columns plus the met/radiation/flux/ERA5
  variables used downstream); provided for the aggregation check.
- `code/` — calibration, uncertainty, dataset-builder, ARM-comparison, and
  standards-fitting scripts, plus the figure style module.
- `comparison/` — the ARM intercomparison figures and statistics of record.

## Data provenance

Source measurements: the ARM SAIL water vapor isotope datastream (Galewsky
WVISO, https://doi.org/10.5439/2280801), a Los Gatos off-axis
integrated-cavity-output analyzer at the SAIL/SPLASH Kettle Ponds site. This
release is a recalibration of that record; the raw ARM datastream is cited by
DOI, not redistributed here.

## Citation

See `CITATION.cff`. Please cite the accompanying manuscript (DOI pending) and
this dataset/code release:

> Rybecky, M., & Galewsky, J. (2026). *SAIL water vapor stable isotope dataset
> (recalibrated, 1-minute) and calibration code*. Zenodo.
> https://doi.org/10.5281/zenodo.21630381

The DOI above is the concept DOI and always resolves to the latest version;
`10.5281/zenodo.21630382` is the frozen v1.0.0 snapshot.

## Releasing to Zenodo

This repository is set up for the GitHub-to-Zenodo archival workflow (the same
one used for the accompanying software release):

1. Create the GitHub repository `matt-rybecky/sail-isotope-calibration` and
   push this repository to it.
2. In Zenodo (logged in with GitHub), toggle the repository ON under
   Account → GitHub, so Zenodo watches it for releases.
3. Create a GitHub Release tagged `v1.0.0`. Zenodo archives that release and
   mints a DOI (a version DOI plus a concept DOI that always points to the
   latest).
4. Set the Zenodo record's upload type to "Dataset", license to CC-BY-4.0, and
   the creators to Rybecky and Galewsky (Zenodo pulls most of this from
   `CITATION.cff`).
5. Add the minted concept DOI to `CITATION.cff` and the README badge, commit,
   and (optionally) cut `v1.0.1` so the DOI is embedded in the archived record.

## License

CC BY 4.0 (see `LICENSE`). Underlying third-party components retain their own
upstream licenses where noted in their source files.
