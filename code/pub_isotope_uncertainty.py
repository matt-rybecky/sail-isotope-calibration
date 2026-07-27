#!/usr/bin/env python3
"""pub_isotope_uncertainty.py — humidity-dependent per-point isotope uncertainty.

Attaches 1-sigma measurement uncertainties (dD, d18O, d-excess) to each
point of the published 1-minute isotope dataset, from the humidity-dependent
instrument-precision model calibrated on the SAIL standards runs.

Model of record (`isotope-analysis-pipeline/outputs/complete_calibration/
uncertainty_summary.json`; 13 standards runs, 28,456 points per isotope,
binned std-dev fit over H2O, R^2 ~ 0.91), identical to the frozen
`TE_V1.0.0/pub_uncertainty_propagation.py`:

    sigma_measurement(H2O) = a * exp(-b * H2O_ppm / 1000) + c   [permil, 1-sigma]

That sigma is the precision of a SINGLE native (~10 s) measurement. Each
published point is a uniform mean of the native samples in its 5-minute
window (native -> 1-minute bins -> centered 5-point mean), so the per-point
uncertainty is reduced by the effective sample count:

    sigma_point = sigma_measurement(H2O) / sqrt(N_eff)
    N_eff       = N_native_in_window * (1 - rho1) / (1 + rho1)

with rho1 = 0.1287 the instrument lag-1 noise autocorrelation from the same
standards runs (`uncertainty_noise_floor/autocorr_summary.json`; Bartlett
factor 0.7719). N_native is counted directly from the native stream, so it
tracks real sampling density and gaps. Two-stage uniform averaging is
treated as one uniform mean over the window (minute counts are near-equal at
~6 native samples per minute), stated as an approximation.

d-excess (= dD - 8*d18O) is reported two ways: propagated from the dD and
d18O fits assuming independent errors (rho = 0, empirically supported), and
from the standalone d-excess calibration fit.

This module is imported by `pub_published_dataset.py`; it is not a script.

Author: Matthew Rybecky
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Exponential-decay fit parameters, verbatim from uncertainty_summary.json
# (VSMOW-corrected; matches pub_uncertainty_propagation.py).
FIT_DD: Dict[str, float] = {'a': 20.067927926153214, 'b': 0.7753385147586956,
                            'c': 1.9891044311787667}
FIT_D18O: Dict[str, float] = {'a': 4.448187915400027, 'b': 0.6409368410401182,
                              'c': 0.5108651683264613}
FIT_DEXCESS_DIRECT: Dict[str, float] = {'a': 37.786399605823036,
                                        'b': 0.6227048707197911,
                                        'c': 4.399145783929147}

# Instrument lag-1 noise autocorrelation (autocorr_summary.json).
RHO1 = 0.12871332722287127
N_EFF_FACTOR = (1.0 - RHO1) / (1.0 + RHO1)     # Bartlett factor ~ 0.7719

# H2O range over which the fit was calibrated (ppm); outside is extrapolated.
FIT_H2O_RANGE: Tuple[float, float] = (1350.0, 8050.0)
ROLLING_WINDOW_MIN = 5                          # centered window of the product

# Columns this module appends, in output order.
UNC_COLS = ['dD_uncertainty_permil', 'd18O_uncertainty_permil',
            'd_excess_uncertainty_permil', 'd_excess_uncertainty_direct_permil',
            'unc_extrapolated']


def sigma_from_fit(h2o_ppm: np.ndarray, fit: Dict[str, float]) -> np.ndarray:
    """Per-measurement 1-sigma from the exponential-decay fit.

    Parameters
    ----------
    h2o_ppm : numpy.ndarray
        Water vapor concentration (ppm); NaN propagates to NaN.
    fit : dict
        Keys ``a``, ``b``, ``c`` of ``a*exp(-b*H2O/1000)+c``.

    Returns
    -------
    numpy.ndarray
        Single-measurement 1-sigma uncertainty (permil).
    """
    return fit['a'] * np.exp(-fit['b'] * h2o_ppm / 1000.0) + fit['c']


def native_counts_per_window(grid_time: pd.Series, native_file: Path) -> np.ndarray:
    """Native (~10 s) sample count in each point's 5-minute window.

    Reconstructs the averaging denominator of the published product: bin the
    native timestamps to 1-minute counts, then a centered rolling sum over
    the window reproduces the samples that fed each output minute.

    Parameters
    ----------
    grid_time : pandas.Series
        The published 1-minute grid timestamps (datetime).
    native_file : pathlib.Path
        `atmospheric_isotopes_vsmow_corrected_full.csv` (native cadence).

    Returns
    -------
    numpy.ndarray
        Native sample count per window, aligned to ``grid_time``.
    """
    nat = pd.read_csv(native_file, usecols=['Time'], parse_dates=['Time'])
    per_min = nat['Time'].dt.floor('1min').value_counts().sort_index()
    full = pd.date_range(grid_time.min(), grid_time.max(), freq='1min')
    counts = per_min.reindex(full, fill_value=0)
    window = counts.rolling(window=ROLLING_WINDOW_MIN, center=True,
                            min_periods=1).sum()
    return window.reindex(pd.DatetimeIndex(grid_time)).values


def add_uncertainty_columns(grid: pd.DataFrame,
                            native_file: Path) -> Tuple[pd.DataFrame, Dict]:
    """Append per-point isotope uncertainty columns to the published grid.

    Parameters
    ----------
    grid : pandas.DataFrame
        Published grid with ``time`` and ``H2O_ppm``; gap rows are NaN.
    native_file : pathlib.Path
        Native-cadence file for the per-window sample count.

    Returns
    -------
    (pandas.DataFrame, dict)
        The grid with :data:`UNC_COLS` added, and a provenance summary
        (median effective N and median uncertainties over measured points).
    """
    h2o = grid['H2O_ppm'].values
    n_native = native_counts_per_window(grid['time'], native_file)
    n_eff = np.maximum(n_native * N_EFF_FACTOR, 1.0)   # a mean never raises sigma
    reduce = 1.0 / np.sqrt(n_eff)

    s_dd = sigma_from_fit(h2o, FIT_DD) * reduce
    s_d18o = sigma_from_fit(h2o, FIT_D18O) * reduce
    s_dex_prop = np.sqrt(s_dd ** 2 + 64.0 * s_d18o ** 2)          # rho = 0
    s_dex_direct = sigma_from_fit(h2o, FIT_DEXCESS_DIRECT) * reduce

    grid['dD_uncertainty_permil'] = s_dd
    grid['d18O_uncertainty_permil'] = s_d18o
    grid['d_excess_uncertainty_permil'] = s_dex_prop
    grid['d_excess_uncertainty_direct_permil'] = s_dex_direct
    lo, hi = FIT_H2O_RANGE
    grid['unc_extrapolated'] = ((h2o < lo) | (h2o > hi)) & ~np.isnan(h2o)

    m = grid['H2O_ppm'].notna().values
    prov = {
        'rho1': RHO1, 'n_eff_factor': round(N_EFF_FACTOR, 4),
        'fit_h2o_range': list(FIT_H2O_RANGE),
        'median_n_native': float(np.median(n_native[m])),
        'median_n_eff': float(np.median(n_eff[m])),
        'median_sigma_dD': float(np.nanmedian(s_dd[m])),
        'median_sigma_d18O': float(np.nanmedian(s_d18o[m])),
        'median_sigma_dexcess': float(np.nanmedian(s_dex_prop[m])),
        'n_extrapolated': int(grid['unc_extrapolated'].sum()),
    }
    return grid, prov
