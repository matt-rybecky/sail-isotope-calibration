#!/usr/bin/env python3
"""pub_published_dataset.py — the published 1-minute water vapor isotope dataset.

Builds the archival, DOI-ready isotope data product from the 1-minute
time-rolling VSMOW-SLAP output of the SAIL isotope pipeline, over the FULL
measurement span (not winter-truncated). The published record is the exact
1-minute stream the transfer-entropy analysis consumes, so aggregating it
reproduces the analyzed `final_1hr.csv` / `final_6hr.csv` isotope columns.

Provenance chain (verified on disk):
  raw Los Gatos analyzer -> humidity correction -> 15-day sliding-window VSMOW-SLAP
  (17 calibration runs) -> 5-minute rolling on a 1-minute grid ->
  `atmospheric_isotopes_vsmow_final_cleaned.csv` (the tidy publish form,
  float-identical to the `..._corrected_5min_rolling.csv` columns
  `D_del_vsmow` / `O18_del_vsmow` / `d_excess_vsmow_derived` / `H2O_ppm`
  that `generate_extended_datasets.py` reads to build the TE datasets).

Cleaning already baked into the source and re-asserted here declaratively:
calibration-run periods are absent (instrument on standards) and appear as
gaps, labeled by date match to the standards file; the one malfunction
window (2022-12-12 18:55-19:22, 28 min) is linearly interpolated in place,
retained as the analysis consumed it and flagged `is_interpolated`; every
other outage is an explicit NaN gap (measured-only policy).

The TE builder `generate_extended_datasets.py` has a stale hardcoded root
and does not run as-is, so its Gaussian-resample + conservative-interpolation
math is REPLICATED here (sigma = 0.25 x interval, +/-4 sigma, linear fill of
gaps <= 3 h) to verify the match rather than imported. Per-point isotope
uncertainties come from `pub_isotope_uncertainty.py`.

Run:
    python3 pub_published_dataset.py

Outputs (WRITING/Manuscript/data/published/):
    sail_water_vapor_isotopes_1min.csv   published dataset (schema A)
    coverage_report.{csv,md}             per-gap timing/length/type + summary
    dataset_metadata.md                  provenance, columns, verification

Author: Matthew Rybecky
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pub_isotope_uncertainty import (UNC_COLS, add_uncertainty_columns,
                                     FIT_DD, FIT_D18O, FIT_DEXCESS_DIRECT)
from pub_te_match import verify

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Paths (resolved from this file: Manuscript/code -> UNM_MASTERS_RESEARCH)
# --------------------------------------------------------------------------
UNM = Path(__file__).resolve().parents[3]
PIPE = UNM / 'PROJECTS' / 'SAIL' / 'isotope-analysis-pipeline'
ROLL = PIPE / 'outputs' / 'time_rolling_vsmow_analysis'
SRC_1MIN = ROLL / 'atmospheric_isotopes_vsmow_final_cleaned.csv'
SRC_TE_CONSUMED = ROLL / 'atmospheric_isotopes_vsmow_corrected_5min_rolling.csv'
SRC_NATIVE = ROLL / 'atmospheric_isotopes_vsmow_corrected_full.csv'
STANDARDS = ROLL / 'sliding_window_standards_data.csv'
TE_FILES = {1: UNM / 'MODELS' / 'TE_V1.0.0' / 'data' / 'final_1hr.csv',
            6: UNM / 'MODELS' / 'TE_V1.0.0' / 'data' / 'final_6hr.csv'}
OUT_DIR = UNM / 'WRITING' / 'Manuscript' / 'data' / 'published'

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
GRID_FREQ = '1min'
DATE_START, DATE_END = '2022-11-21', '2023-06-01'
OUTLIER_PERIODS: List[Tuple[str, str]] = [
    ('2022-12-12 18:55:00', '2022-12-12 19:22:00'),   # instrument malfunction
]
VERIFY_TOL = 1e-6               # max abs isotope difference vs the TE datasets
PUBLISHED_CSV = 'sail_water_vapor_isotopes_1min.csv'

# Isotope columns: published (schema A) name in `final_cleaned` order.
ISO_COLS = ['H2O_ppm', 'dD_vsmow_permil', 'd18O_vsmow_permil',
            'd_excess_vsmow_permil']
# Published column -> analyzed TE column, for the aggregation cross-check.
PUB_TO_TE = {'H2O_ppm': 'H2O_ppm', 'dD_vsmow_permil': 'dD',
             'd18O_vsmow_permil': 'd18O', 'd_excess_vsmow_permil': 'd_excess'}
# Published column -> raw TE-consumed column in the 5-min-rolling file.
PUB_TO_CONSUMED = {'H2O_ppm': 'H2O_ppm', 'dD_vsmow_permil': 'D_del_vsmow',
                   'd18O_vsmow_permil': 'O18_del_vsmow',
                   'd_excess_vsmow_permil': 'd_excess_vsmow_derived'}
UNITS = {'H2O_ppm': 'ppm', 'dD_vsmow_permil': 'permil vs VSMOW',
         'd18O_vsmow_permil': 'permil vs VSMOW',
         'd_excess_vsmow_permil': 'permil vs VSMOW'}


# --------------------------------------------------------------------------
# Load and provenance
# --------------------------------------------------------------------------
def sha256(path: Path) -> str:
    """SHA-256 of a file, for the metadata provenance record."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_source() -> pd.DataFrame:
    """Load the tidy 1-minute VSMOW output (measured minutes only).

    Returns
    -------
    pandas.DataFrame
        Columns ``time`` + the four :data:`ISO_COLS`, sorted, de-duplicated.
    """
    df = pd.read_csv(SRC_1MIN, parse_dates=['Time']).rename(columns={'Time': 'time'})
    df = df[['time'] + ISO_COLS].sort_values('time').reset_index(drop=True)
    df = df.drop_duplicates('time')
    for c in ISO_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    logger.info(f"source: {len(df):,} measured minutes "
                f"{df['time'].min()} -> {df['time'].max()}")
    return df


def assert_te_source_identity(src: pd.DataFrame) -> None:
    """Confirm the published columns equal the exact columns TE consumed.

    The TE builder reads `..._corrected_5min_rolling.csv`; the published
    file is the tidy `..._final_cleaned.csv`. This asserts they are the
    same numbers so the published product is provably the analyzed stream.
    """
    te = pd.read_csv(SRC_TE_CONSUMED, parse_dates=['Time']).rename(
        columns={'Time': 'time'})
    te = te.sort_values('time').drop_duplicates('time').reset_index(drop=True)
    # Prefix the compared columns so H2O_ppm (identically named) never collides.
    te_sub = te[['time'] + list(PUB_TO_CONSUMED.values())].rename(
        columns={raw: f'cmp__{raw}' for raw in PUB_TO_CONSUMED.values()})
    merged = src.merge(te_sub, on='time', how='inner')
    if len(merged) != len(src):
        raise SystemExit(f"time index mismatch: {len(merged)} of {len(src)} "
                         "rows align with the TE-consumed file")
    for pub, raw in PUB_TO_CONSUMED.items():
        diff = (merged[pub] - merged[f'cmp__{raw}']).abs().max()
        if not (diff <= VERIFY_TOL or np.isnan(diff)):
            raise SystemExit(f"{pub} differs from TE-consumed {raw}: "
                             f"max |diff| = {diff:g}")
    logger.info("identity check: published columns == TE-consumed columns "
                "(max |diff| within tol)")


# --------------------------------------------------------------------------
# Full grid, interpolation flag, gaps
# --------------------------------------------------------------------------
def to_full_grid(src: pd.DataFrame) -> pd.DataFrame:
    """Reindex the measured stream onto a complete 1-minute grid.

    Missing minutes (calibration runs and outages) become explicit NaN
    rows so the coverage report describes the whole span.
    """
    grid = pd.date_range(src['time'].min(), src['time'].max(), freq=GRID_FREQ)
    out = src.set_index('time').reindex(grid)
    out.index.name = 'time'
    n_missing = int(out[ISO_COLS[1]].isna().sum())
    logger.info(f"grid: {len(out):,} minutes, {n_missing:,} missing "
                f"({100 * n_missing / len(out):.1f}%)")
    return out.reset_index()


def mark_interpolated(grid: pd.DataFrame) -> pd.DataFrame:
    """Flag the retained, in-filled malfunction window(s).

    ``is_interpolated`` is True only where a present value falls inside a
    declared outlier period (kept to match the analyzed stream); every
    genuine outage stays NaN and unflagged.
    """
    flag = pd.Series(False, index=grid.index)
    present = grid[ISO_COLS[1]].notna()
    for start, end in OUTLIER_PERIODS:
        within = (grid['time'] >= pd.Timestamp(start)) & \
                 (grid['time'] <= pd.Timestamp(end))
        flag |= within & present
    grid['is_interpolated'] = flag
    logger.info(f"is_interpolated: {int(flag.sum())} minutes flagged "
                f"(retained malfunction window)")
    return grid


def find_gaps(grid: pd.DataFrame) -> pd.DataFrame:
    """Enumerate contiguous missing runs on the 1-minute grid.

    Returns
    -------
    pandas.DataFrame
        One row per gap: ``start`` and ``end`` (first/last missing minute),
        ``duration_min``, ``duration_h``.
    """
    missing = grid[ISO_COLS[1]].isna().values
    times = grid['time'].values
    rows: List[Dict] = []
    i, n = 0, len(missing)
    while i < n:
        if missing[i]:
            j = i
            while j + 1 < n and missing[j + 1]:
                j += 1
            start, end = pd.Timestamp(times[i]), pd.Timestamp(times[j])
            dur = int((end - start).total_seconds() // 60) + 1
            rows.append({'start': start, 'end': end, 'duration_min': dur,
                         'duration_h': round(dur / 60, 2)})
            i = j + 1
        else:
            i += 1
    logger.info(f"gaps: {len(rows)} outage/calibration blocks")
    return pd.DataFrame(rows)


def calibration_dates() -> set:
    """Set of calendar dates on which a calibration run was measured."""
    std = pd.read_csv(STANDARDS, parse_dates=['datetime'])
    return set(std['datetime'].dt.normalize())


def classify_gaps(gaps: pd.DataFrame, cal_dates: set) -> pd.DataFrame:
    """Label each gap calibration vs outage by date match (date-level).

    The standards file timestamps calibration runs at date granularity
    only, so a gap is tagged ``calibration`` when any date it spans matches
    a calibration-run date, else ``outage``.
    """
    def label(row: pd.Series) -> str:
        days = pd.date_range(row['start'].normalize(),
                             row['end'].normalize(), freq='D')
        return 'calibration' if any(d in cal_dates for d in days) else 'outage'

    gaps = gaps.copy()
    gaps['type'] = gaps.apply(label, axis=1)
    return gaps


def summarize(grid: pd.DataFrame, gaps: pd.DataFrame) -> Dict:
    """Coverage summary over the full 1-minute span."""
    n_grid = len(grid)
    n_valid = int(grid[ISO_COLS[1]].notna().sum())
    by_type = gaps.groupby('type')['duration_min'].agg(['count', 'sum']) \
        if len(gaps) else pd.DataFrame()
    return {
        'span_start': grid['time'].min(), 'span_end': grid['time'].max(),
        'grid_minutes': n_grid, 'valid_minutes': n_valid,
        'uptime_pct': round(100 * n_valid / n_grid, 2),
        'n_gaps': len(gaps),
        'longest_gap_min': int(gaps['duration_min'].max()) if len(gaps) else 0,
        'calibration_gaps': int(by_type.loc['calibration', 'count'])
        if 'calibration' in by_type.index else 0,
        'calibration_minutes': int(by_type.loc['calibration', 'sum'])
        if 'calibration' in by_type.index else 0,
        'outage_gaps': int(by_type.loc['outage', 'count'])
        if 'outage' in by_type.index else 0,
        'outage_minutes': int(by_type.loc['outage', 'sum'])
        if 'outage' in by_type.index else 0,
        'interpolated_minutes': int(grid['is_interpolated'].sum()),
    }


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------
def write_dataset(grid: pd.DataFrame) -> Path:
    """Write the published schema-A CSV."""
    cols = ['time'] + ISO_COLS + ['is_interpolated'] + UNC_COLS
    path = OUT_DIR / PUBLISHED_CSV
    grid[cols].to_csv(path, index=False)
    logger.info(f"wrote {path}")
    return path


def write_coverage(gaps: pd.DataFrame, summary: Dict) -> None:
    """Write the per-gap table (csv) and a human-readable report (md)."""
    gaps.to_csv(OUT_DIR / 'coverage_report.csv', index=False)
    lines = ['# Published isotope dataset — coverage report', '',
             f"Span: {summary['span_start']} to {summary['span_end']} "
             f"({summary['grid_minutes']:,} one-minute grid points).", '',
             f"Measured minutes: {summary['valid_minutes']:,} "
             f"({summary['uptime_pct']}% uptime). "
             f"Interpolated (retained malfunction window): "
             f"{summary['interpolated_minutes']}.", '',
             f"Gaps: {summary['n_gaps']} total "
             f"({summary['calibration_gaps']} calibration runs = "
             f"{summary['calibration_minutes']:,} min; "
             f"{summary['outage_gaps']} outages = "
             f"{summary['outage_minutes']:,} min). Longest gap "
             f"{summary['longest_gap_min']:,} min.", '',
             'Calibration vs outage is assigned by matching a gap date to a '
             'calibration-run date (date granularity; the standards file '
             'carries no time of day).', '',
             '| Start | End | Duration (min) | Duration (h) | Type |',
             '| --- | --- | --- | --- | --- |']
    for _, r in gaps.iterrows():
        lines.append(f"| {r['start']} | {r['end']} | {r['duration_min']} | "
                     f"{r['duration_h']} | {r['type']} |")
    (OUT_DIR / 'coverage_report.md').write_text('\n'.join(lines) + '\n')
    logger.info(f"wrote {OUT_DIR / 'coverage_report.md'}")


def write_metadata(src_hash: str, summary: Dict, verify_rows: List[Dict],
                   unc: Dict) -> None:
    """Write the provenance + column dictionary + verification metadata."""
    ok = all(r['pass'] for r in verify_rows)
    lines = ['# SAIL water vapor isotope dataset — metadata', '',
             '## Source and provenance', '',
             f'- Built from `{SRC_1MIN.name}` (SHA-256 `{src_hash}`).',
             '- 1-minute time-rolling VSMOW-SLAP corrected Los Gatos '
             'analyzer record; '
             'humidity correction then 15-day sliding-window VSMOW-SLAP over '
             '17 calibration runs.',
             '- d-excess is the VSMOW-derived form (`d_excess_vsmow_derived`), '
             'matching the transfer-entropy analysis.',
             '- Full measurement span, not winter-truncated.', '',
             '## Columns', '',
             '| Column | Units | Description |',
             '| --- | --- | --- |',
             '| time | UTC | Measurement minute (1-minute grid) |']
    desc = {'H2O_ppm': 'Water vapor mixing ratio',
            'dD_vsmow_permil': 'delta D (deuterium)',
            'd18O_vsmow_permil': 'delta 18-O',
            'd_excess_vsmow_permil': 'Deuterium excess (dD - 8 x d18O)'}
    for c in ISO_COLS:
        lines.append(f'| {c} | {UNITS[c]} | {desc[c]} |')
    lines += ['| is_interpolated | bool | True in the retained, in-filled '
              'malfunction window; False for measured or missing |',
              '| dD_uncertainty_permil | permil | 1-sigma dD uncertainty '
              '(humidity model, averaging-reduced) |',
              '| d18O_uncertainty_permil | permil | 1-sigma d18O uncertainty |',
              '| d_excess_uncertainty_permil | permil | 1-sigma d-excess, '
              'propagated from dD/d18O (rho=0) |',
              '| d_excess_uncertainty_direct_permil | permil | 1-sigma '
              'd-excess from the standalone d-excess fit |',
              '| unc_extrapolated | bool | True where H2O is outside the '
              f'{unc["fit_h2o_range"][0]:.0f}-{unc["fit_h2o_range"][1]:.0f} '
              'ppm fit range |', '',
              '## Measurement uncertainty', '',
              'Per-measurement 1-sigma follows a humidity-dependent fit '
              'calibrated on 13 standards runs (28,456 points): '
              'sigma(H2O) = a*exp(-b*H2O/1000) + c [permil].', '',
              '| Quantity | a | b | c |',
              '| --- | --- | --- | --- |',
              f'| dD | {FIT_DD["a"]:.4f} | {FIT_DD["b"]:.4f} | '
              f'{FIT_DD["c"]:.4f} |',
              f'| d18O | {FIT_D18O["a"]:.4f} | {FIT_D18O["b"]:.4f} | '
              f'{FIT_D18O["c"]:.4f} |',
              f'| d-excess (direct) | {FIT_DEXCESS_DIRECT["a"]:.4f} | '
              f'{FIT_DEXCESS_DIRECT["b"]:.4f} | {FIT_DEXCESS_DIRECT["c"]:.4f} |',
              '',
              'Each published point is a uniform mean of its 5-minute '
              'window, so sigma is reduced to sigma/sqrt(N_eff), '
              f'N_eff = N_native * {unc["n_eff_factor"]} (Bartlett factor for '
              f'lag-1 autocorrelation rho1 = {unc["rho1"]:.4f}); N_native is '
              'counted from the native stream per window (two-stage uniform '
              'averaging approximated as one window mean).',
              '',
              f'Median over measured points: N_native '
              f'{unc["median_n_native"]:.0f}, N_eff {unc["median_n_eff"]:.0f}, '
              f'sigma_dD {unc["median_sigma_dD"]:.3f}, sigma_d18O '
              f'{unc["median_sigma_d18O"]:.3f}, sigma_d-excess '
              f'{unc["median_sigma_dexcess"]:.3f} permil. '
              f'{unc["n_extrapolated"]:,} points are outside the fit range '
              '(flagged; below it the fit climbs steeply).', '',
              '## Cleaning', '',
              '- Calibration-run periods are absent (instrument on '
              'standards); reported as gaps in the coverage report.',
              '- Instrument malfunction 2022-12-12 18:55-19:22 (28 min) '
              'linearly interpolated in place and flagged; every other '
              'outage left as an explicit NaN gap (measured-only).', '',
              '## Verification against the analyzed data', '',
              'Resampling this record with the analysis kernel (Gaussian '
              'sigma = 0.25 x interval, +/-4 sigma; linear fill of gaps '
              '<= 3 h) reproduces the frozen `final_1hr.csv` / '
              '`final_6hr.csv` isotope columns.', '',
              f'**Result: {"PASS" if ok else "FAIL"}** '
              f'(tolerance {VERIFY_TOL:g} permil/ppm).', '',
              '| Interval | Column | N | max abs diff | Pass |',
              '| --- | --- | --- | --- | --- |']
    for r in verify_rows:
        lines.append(f"| {r['interval_h']}h | {r['column']} | "
                     f"{r['n_compared']} | {r['max_abs_diff']:.2e} | "
                     f"{'yes' if r['pass'] else 'NO'} |")
    (OUT_DIR / 'dataset_metadata.md').write_text('\n'.join(lines) + '\n')
    logger.info(f"wrote {OUT_DIR / 'dataset_metadata.md'}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = load_source()
    assert_te_source_identity(src)
    grid = mark_interpolated(to_full_grid(src))
    grid, unc_prov = add_uncertainty_columns(grid, SRC_NATIVE)
    logger.info(f"uncertainty: median sigma_dD={unc_prov['median_sigma_dD']:.3f}, "
                f"sigma_d18O={unc_prov['median_sigma_d18O']:.3f}, "
                f"sigma_dex={unc_prov['median_sigma_dexcess']:.3f} permil; "
                f"median N_eff={unc_prov['median_n_eff']:.1f}; "
                f"{unc_prov['n_extrapolated']} extrapolated")
    gaps = classify_gaps(find_gaps(grid), calibration_dates())
    summary = summarize(grid, gaps)
    verify_rows = verify(src, TE_FILES, DATE_START, DATE_END, PUB_TO_TE,
                         VERIFY_TOL)
    write_dataset(grid)
    write_coverage(gaps, summary)
    write_metadata(sha256(SRC_1MIN), summary, verify_rows, unc_prov)
    ok = all(r['pass'] for r in verify_rows)
    logger.info(f"DONE — verification {'PASS' if ok else 'FAIL'}; "
                f"{summary['valid_minutes']:,}/{summary['grid_minutes']:,} "
                f"measured minutes, {summary['n_gaps']} gaps")


if __name__ == '__main__':
    main()
