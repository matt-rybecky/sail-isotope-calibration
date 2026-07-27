#!/usr/bin/env python3
"""pub_te_match.py — verify the published stream reproduces the TE datasets.

Replicates the Gaussian resample + conservative interpolation of
`generate_extended_datasets.py` (sigma = 0.25 x interval, +/-4 sigma window;
linear fill of gaps <= 3 h), aggregates the published 1-minute isotopes to
1 h and 6 h, and diffs against the frozen `final_{1,6}hr.csv` isotope
columns. The builder has a stale hardcoded root and cannot be imported, so
its math is replicated here.

Imported by `pub_published_dataset.py`; not a script.

Author: Matthew Rybecky
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SIGMA_FRACTION = 0.25       # sigma = fraction * interval (TE builder)
MAX_GAP_HOURS = 3           # conservative interpolation ceiling (TE builder)


def _gaussian_resample(t_ns: np.ndarray, values: np.ndarray,
                       targets: pd.DatetimeIndex, sigma_s: float) -> np.ndarray:
    """Weighted Gaussian resample (TE-builder kernel, +/-4 sigma window)."""
    t_sec = t_ns / 1e9
    window = sigma_s * 4
    out = np.full(len(targets), np.nan)
    for i, tt in enumerate(targets):
        tgt = tt.value / 1e9
        m = np.abs(t_sec - tgt) <= window
        if not m.any():
            continue
        tw, vw = t_sec[m], values[m]
        v = ~np.isnan(vw)
        if not v.any():
            continue
        tw, vw = tw[v], vw[v]
        w = np.exp(-0.5 * ((tw - tgt) / sigma_s) ** 2)
        out[i] = np.average(vw, weights=w)
    return out


def target_grid(interval_h: int, date_start: str, date_end: str) -> pd.DatetimeIndex:
    """Hour-anchored target grid over the span (TE-builder convention)."""
    start = pd.Timestamp(date_start)
    grid = pd.date_range(start.replace(minute=0, second=0, microsecond=0),
                         pd.Timestamp(date_end), freq=f'{interval_h}h')
    return grid[(grid >= start) & (grid <= pd.Timestamp(date_end))]


def _interp_conservative(series: pd.Series, max_gap_h: float) -> pd.Series:
    """Linear-fill resampled gaps up to ``max_gap_h`` (TE-builder rule)."""
    result = series.copy()
    idx, vals = series.index, series.values
    missing = np.isnan(vals)
    i, n = 0, len(vals)
    while i < n:
        if missing[i] and i > 0:
            j = i
            while j + 1 < n and missing[j + 1]:
                j += 1
            if j + 1 < n:
                gap_h = (idx[j + 1] - idx[i]).total_seconds() / 3600
                a, b = vals[i - 1], vals[j + 1]
                if gap_h <= max_gap_h and not np.isnan(a) and not np.isnan(b):
                    result.iloc[i:j + 1] = np.linspace(a, b, j - i + 3)[1:-1]
            i = j + 1
        else:
            i += 1
    return result


def verify(src: pd.DataFrame, te_files: Dict[int, Path], date_start: str,
           date_end: str, pub_to_te: Dict[str, str], tol: float) -> List[Dict]:
    """Aggregate the published stream and diff against the TE datasets.

    The builder filtered its isotope input to ``time <= date_end`` before
    resampling, so the same cutoff is applied here; the published file keeps
    the full span, only this reproduction check mirrors the one-sided final
    window.

    Parameters
    ----------
    src : pandas.DataFrame
        Published measured stream: ``time`` + the published isotope columns.
    te_files : dict
        Resolution (hours) -> frozen ``final_<r>hr.csv`` path.
    date_start, date_end : str
        Span bounds matching the TE builder.
    pub_to_te : dict
        Published column -> analyzed TE column name.
    tol : float
        Pass threshold on the max absolute isotope difference.

    Returns
    -------
    list of dict
        One record per (resolution, column): n compared, max abs diff, pass.
    """
    src = src[src['time'] <= pd.Timestamp(date_end)]
    t_ns = src['time'].values.astype('int64')
    rows: List[Dict] = []
    for interval_h, te_path in te_files.items():
        sigma_s = interval_h * 3600 * SIGMA_FRACTION
        targets = target_grid(interval_h, date_start, date_end)
        te = pd.read_csv(te_path, parse_dates=['time']).set_index('time')
        te = te.reindex(targets)
        for pub, te_col in pub_to_te.items():
            res = pd.Series(_gaussian_resample(t_ns, src[pub].values, targets,
                                               sigma_s), index=targets)
            res = _interp_conservative(res, MAX_GAP_HOURS)
            both = (~res.isna()) & (~te[te_col].isna())
            diff = float((res[both] - te[te_col][both]).abs().max()) \
                if both.any() else float('nan')
            rows.append({'interval_h': interval_h, 'column': pub,
                         'n_compared': int(both.sum()), 'max_abs_diff': diff,
                         'pass': bool(diff <= tol)})
            logger.info(f"verify {interval_h}h {pub:22s} "
                        f"n={int(both.sum()):4d} max|diff|={diff:.2e} "
                        f"{'PASS' if diff <= tol else 'FAIL'}")
    return rows
