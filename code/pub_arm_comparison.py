#!/usr/bin/env python3
"""pub_arm_comparison.py — compare our calibrated isotope stream to ARM's.

Intercompares the packaged 1-minute VSMOW-SLAP product
(``data/published/sail_water_vapor_isotopes_1min.csv``) against the
ARM-archived SAIL water-vapor-isotope datastream
(Galewsky WVISO, DOI 10.5439/2280801), which is an *independent, earlier*
calibration of the same Los Gatos analyzer measurements. Both series are put
on a common time grid (hourly means by default), then compared per isotope
with method-agreement statistics and diagnostic figures.

This is an instrument-intercomparison, not a raw-vs-calibrated check: both are
calibrated products, so the right tools are Deming (errors-in-both) regression
and Bland-Altman agreement, plus difference-vs-time (drift, the expected
signature of static vs. rolling VSMOW normalization) and difference-vs-H2O
(humidity-correction differences).

Because the ARM variable names are not known until the files are in hand, the
ARM side is mapped explicitly. Run ``--list-vars`` first to dump the ARM
file's variables, then pass ``--arm-dD/-d18O/-h2o/-time`` (or edit ARM_MAP).

Usage
-----
    # 1. discover the ARM variable names
    python3 pub_arm_comparison.py --arm-path /path/to/arm/*.nc --list-vars

    # 2. run the comparison (example var names; set to the real ones)
    python3 pub_arm_comparison.py --arm-path /path/to/arm/*.nc \
        --arm-time time --arm-dD dD --arm-d18O d18O --arm-h2o wv_mixing_ratio

    # validate the harness against our own file (expect perfect agreement)
    python3 pub_arm_comparison.py --self-test

Author: Matthew Rybecky
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DEFAULT_OURS = (HERE.parent / 'data' / 'published'
                / 'sail_water_vapor_isotopes_1min.csv')

# Our published column names (fixed; see data/published/dataset_metadata.md).
OURS_MAP = {
    'time': 'time',
    'dD': 'dD_vsmow_permil',
    'd18O': 'd18O_vsmow_permil',
    'd_excess': 'd_excess_vsmow_permil',
    'H2O': 'H2O_ppm',
}

# ARM column/variable names. Filled for the ARM-archived SAIL WVISO delivery
# (galewsky-wviso, DOI 10.5439/2280801): a single .dat with columns
# ``Time, Mixing Ratio (g/kg), dD (permil), d18O (permil)`` (leading spaces
# stripped on load). Override on the CLI for a different ARM file, or run
# --list-vars first. d_excess is absent in the ARM stream and derived as
# dD - 8*d18O; H2O is left unmapped because the ARM unit (g/kg) differs from
# our ppmv, and the H2O covariate is taken from our stream.
ARM_MAP: Dict[str, Optional[str]] = {
    'time': 'Time',
    'dD': 'dD (permil)',
    'd18O': 'd18O (permil)',
    'd_excess': None,
    'H2O': None,
}

ISOTOPES = ['dD', 'd18O', 'd_excess']   # variables compared (H2O is a covariate)

# Axis labels for the added calibration-comparison figures (matplotlib text,
# so "permil" is spelled out rather than using the LaTeX \permil macro).
PANEL_TAGS = ('(a)', '(b)', '(c)')
VAR_LABELS = {
    'dD': r'$\delta$D (permil)',
    'd18O': r'$\delta^{18}$O (permil)',
    'd_excess': 'd-excess (permil)',
}
DIFF_LABELS = {
    'dD': r'$\Delta\,\delta$D (permil)',
    'd18O': r'$\Delta\,\delta^{18}$O (permil)',
    'd_excess': r'$\Delta$ d-excess (permil)',
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_ours(path: Path, measured_only: bool = True) -> pd.DataFrame:
    """Load our published 1-minute stream as time-indexed dD/d18O/d_excess/H2O.

    Parameters
    ----------
    path : Path
        Path to ``sail_water_vapor_isotopes_1min.csv``.
    measured_only : bool
        Drop the in-filled malfunction window (``is_interpolated``) so the
        comparison uses measured points only.
    """
    df = pd.read_csv(path, parse_dates=[OURS_MAP['time']])
    if measured_only and 'is_interpolated' in df.columns:
        df = df[~df['is_interpolated'].astype(bool)]
    out = pd.DataFrame({'time': df[OURS_MAP['time']]})
    for key in ('dD', 'd18O', 'd_excess', 'H2O'):
        out[key] = pd.to_numeric(df[OURS_MAP[key]], errors='coerce')
    return out.set_index('time').sort_index()


def _open_arm(arm_path: str):
    """Open one/many ARM netCDF files or a single CSV; return an xarray or df."""
    paths = sorted(glob.glob(arm_path)) if any(c in arm_path for c in '*?[') \
        else [arm_path]
    if not paths:
        raise FileNotFoundError(f"no ARM files matched: {arm_path}")
    if paths[0].lower().endswith(('.csv', '.dat', '.txt')):
        return ('csv', paths)
    import xarray as xr  # netCDF path (ARM standard); imported lazily
    ds = xr.open_mfdataset(paths, combine='by_coords') if len(paths) > 1 \
        else xr.open_dataset(paths[0])
    return ('nc', ds)


def list_arm_vars(arm_path: str) -> None:
    """Print the ARM file's variables (+ units) so ARM_MAP can be filled."""
    kind, obj = _open_arm(arm_path)
    if kind == 'csv':
        df = pd.read_csv(obj[0], nrows=5)
        logger.info("ARM CSV columns: %s", list(df.columns))
        return
    logger.info("ARM netCDF variables (name: units — long_name):")
    for name, var in obj.variables.items():
        units = var.attrs.get('units', '')
        long = var.attrs.get('long_name', '')
        logger.info("  %-28s %-14s %s", name, units, long)


def load_arm(arm_path: str, amap: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Load ARM data as time-indexed dD/d18O/d_excess/H2O (UTC).

    Missing d_excess is derived as ``dD - 8*d18O``. Requires at least
    ``time``, ``dD``, ``d18O`` to be mapped.
    """
    for req in ('time', 'dD', 'd18O'):
        if not amap.get(req):
            raise ValueError(f"ARM_MAP['{req}'] is unset — run --list-vars and "
                             f"map it (or pass --arm-{req}).")
    kind, obj = _open_arm(arm_path)
    if kind == 'csv':
        raw = pd.read_csv(obj[0])
        raw.columns = [c.strip() for c in raw.columns]   # ARM .dat has leading spaces
        t = pd.to_datetime(raw[amap['time']])
        cols = {'time': t}
        for key in ('dD', 'd18O', 'H2O', 'd_excess'):
            if amap.get(key):
                cols[key] = pd.to_numeric(raw[amap[key]], errors='coerce')
        df = pd.DataFrame(cols)
    else:
        take = {k: v for k, v in amap.items() if v and v in obj.variables}
        sub = obj[list(take.values())].to_dataframe().reset_index()
        rename = {v: k for k, v in take.items()}
        df = sub.rename(columns=rename)
        df['time'] = pd.to_datetime(df[amap['time']]) if amap['time'] in df \
            else pd.to_datetime(df['time'])
    if 'd_excess' not in df.columns or df['d_excess'].isna().all():
        df['d_excess'] = df['dD'] - 8.0 * df['d18O']
    if 'H2O' not in df.columns:
        df['H2O'] = np.nan
    keep = ['time', 'dD', 'd18O', 'd_excess', 'H2O']
    return df[keep].dropna(subset=['time']).set_index('time').sort_index()


def align(ours: pd.DataFrame, arm: pd.DataFrame, cadence: str = '1h'
          ) -> pd.DataFrame:
    """Resample both to a common cadence (mean) and inner-join overlaps.

    Returns a frame with ``<var>_ours``/``<var>_arm`` columns plus ``H2O_ours``
    for the humidity covariate, over timestamps where both are present.
    """
    o = ours.resample(cadence).mean()
    a = arm.resample(cadence).mean()
    j = o.join(a, lsuffix='_ours', rsuffix='_arm', how='inner')
    j = j.rename(columns={'H2O_ours': 'H2O'})  # H2O covariate from our stream
    return j


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def deming(x: np.ndarray, y: np.ndarray, delta: float = 1.0) -> tuple:
    """Deming (errors-in-both) regression slope/intercept.

    ``delta`` is var(err_y)/var(err_x); 1.0 is orthogonal regression.
    """
    xb, yb = x.mean(), y.mean()
    sxx = np.sum((x - xb) ** 2) / (len(x) - 1)
    syy = np.sum((y - yb) ** 2) / (len(y) - 1)
    sxy = np.sum((x - xb) * (y - yb)) / (len(x) - 1)
    if abs(sxy) < 1e-15:
        return float('nan'), float('nan')
    slope = ((syy - delta * sxx)
             + np.sqrt((syy - delta * sxx) ** 2 + 4 * delta * sxy ** 2)) \
        / (2 * sxy)
    return float(slope), float(yb - slope * xb)


def compare_variable(df: pd.DataFrame, var: str) -> Dict:
    """Agreement statistics for one isotope (ours vs ARM) over the overlap."""
    o = df[f'{var}_ours'].values
    a = df[f'{var}_arm'].values
    m = ~np.isnan(o) & ~np.isnan(a)
    o, a = o[m], a[m]
    diff = o - a
    n = len(o)
    if n < 3:
        return {'variable': var, 'n': n}
    slope, intercept = deming(a, o)          # x=ARM, y=ours
    r = float(np.corrcoef(a, o)[0, 1])
    # drift of the difference in time (permil per 30 days)
    t = df.index[m].astype('int64').to_numpy() / 1e9
    tt = (t - t.mean()) / (30 * 86400)
    drift = float(np.polyfit(tt, diff, 1)[0]) if np.ptp(tt) > 0 else float('nan')
    # humidity dependence of the difference
    h = df['H2O'].values[m] if 'H2O' in df else np.full(n, np.nan)
    hm = ~np.isnan(h)
    r_h2o = float(np.corrcoef(h[hm], diff[hm])[0, 1]) if hm.sum() > 3 \
        else float('nan')
    sd = float(diff.std(ddof=1))
    return {
        'variable': var, 'n': n,
        'mean_ours': float(o.mean()), 'mean_arm': float(a.mean()),
        'bias_ours_minus_arm': float(diff.mean()), 'sd_diff': sd,
        'rmse': float(np.sqrt((diff ** 2).mean())),
        'median_abs_diff': float(np.median(np.abs(diff))),
        'pearson_r': r, 'deming_slope': slope, 'deming_intercept': intercept,
        'ba_loa_low': float(diff.mean() - 1.96 * sd),
        'ba_loa_high': float(diff.mean() + 1.96 * sd),
        'drift_permil_per_30d': drift, 'r_diff_vs_H2O': r_h2o,
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Per-isotope 2x2 diagnostic panel: scatter, Bland-Altman, drift, H2O."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        sys.path.insert(0, str(HERE))
        from pub_style import setup_style, FULL_WIDTH   # noqa
        try:
            setup_style()
        except Exception:
            pass
        width = FULL_WIDTH
    except Exception:
        width = 7.5

    written: List[Path] = []
    for var in ISOTOPES:
        sub = df[[f'{var}_ours', f'{var}_arm', 'H2O']].dropna(
            subset=[f'{var}_ours', f'{var}_arm'])
        if len(sub) < 3:
            continue
        o = sub[f'{var}_ours'].values
        a = sub[f'{var}_arm'].values
        diff = o - a
        mean = (o + a) / 2
        bias, sd = diff.mean(), diff.std(ddof=1)
        slope, intercept = deming(a, o)

        fig, ax = plt.subplots(2, 2, figsize=(width, width * 0.8))
        # (0,0) scatter ours vs ARM with 1:1 and Deming line
        ax[0, 0].scatter(a, o, s=3, c='k', alpha=0.4, linewidths=0)
        lo, hi = np.nanmin([a, o]), np.nanmax([a, o])
        ax[0, 0].plot([lo, hi], [lo, hi], 'k--', lw=0.8, label='1:1')
        ax[0, 0].plot([lo, hi], [intercept + slope * lo, intercept + slope * hi],
                      color='0.45', lw=1.2, label='Deming')
        ax[0, 0].set_xlabel(f'ARM {var}'); ax[0, 0].set_ylabel(f'ours {var}')
        ax[0, 0].legend(fontsize=6, frameon=False)
        ax[0, 0].set_title(f'slope={slope:.3f}  int={intercept:.2f}', fontsize=7)
        # (0,1) Bland-Altman
        ax[0, 1].scatter(mean, diff, s=3, c='k', alpha=0.4, linewidths=0)
        for y, ls in ((bias, '-'), (bias + 1.96 * sd, ':'),
                      (bias - 1.96 * sd, ':')):
            ax[0, 1].axhline(y, color='0.45', ls=ls, lw=0.9)
        ax[0, 1].set_xlabel(f'mean {var}')
        ax[0, 1].set_ylabel('ours − ARM')
        ax[0, 1].set_title(f'bias={bias:.2f}  LoA±{1.96*sd:.2f}', fontsize=7)
        # (1,0) difference vs time (drift)
        ax[1, 0].scatter(sub.index, diff, s=3, c='k', alpha=0.4, linewidths=0)
        ax[1, 0].axhline(0, color='0.45', lw=0.8)
        ax[1, 0].set_xlabel('time'); ax[1, 0].set_ylabel('ours − ARM')
        for lab in ax[1, 0].get_xticklabels():
            lab.set_rotation(30); lab.set_ha('right'); lab.set_fontsize(6)
        # (1,1) difference vs H2O
        if sub['H2O'].notna().any():
            ax[1, 1].scatter(sub['H2O'], diff, s=3, c='k', alpha=0.4,
                             linewidths=0)
            ax[1, 1].axhline(0, color='0.45', lw=0.8)
            ax[1, 1].set_xlabel('H2O (ppm)'); ax[1, 1].set_ylabel('ours − ARM')
        else:
            ax[1, 1].axis('off')
        fig.suptitle(f'{var}: ours vs ARM (n={len(sub)})', fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        stem = out_dir / f'arm_comparison_{var}'
        fig.savefig(f'{stem}.png', dpi=200)
        fig.savefig(f'{stem}.pdf')
        plt.close(fig)
        written += [Path(f'{stem}.png'), Path(f'{stem}.pdf')]
    return written


# --------------------------------------------------------------------------- #
# Calibration-comparison figures (pub_style: pure B&W, print-true, PNG + PDF)
# --------------------------------------------------------------------------- #
def _mpl_and_style(width: str = 'full'):
    """Return (pyplot, pub_style) with the repo B&W style applied."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE))
    import pub_style as ps
    ps.setup_style(width=width)
    return plt, ps


def make_overlay_figure(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Overlaid time series of the two calibrations (this study vs ARM),
    one stacked panel per isotope. ARM is the sanctioned context gray; this
    study is black."""
    plt, ps = _mpl_and_style()
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(ps.FULL_WIDTH, 7.2))
    for ax, var, tag in zip(axes, ISOTOPES, PANEL_TAGS):
        sub = df[[f'{var}_ours', f'{var}_arm']].dropna()
        ax.plot(sub.index, sub[f'{var}_arm'], color=ps.CONTEXT_GRAY, lw=0.8,
                zorder=2, label='ARM (DOI 10.5439/2280801)')
        ax.plot(sub.index, sub[f'{var}_ours'], color='black', lw=0.8,
                zorder=3, label='This study')
        ax.set_ylabel(VAR_LABELS[var])
        ps.panel_label(ax, tag)
        ps.date_axis(ax)
    axes[0].legend(loc='upper right', fontsize=7, frameon=False)
    axes[-1].set_xlabel('2022–2023')
    return ps.save_figure(fig, out_dir / 'arm_overlay_timeseries')


def make_scatter_figure(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Scatter of this study vs ARM per isotope, with the 1:1 line (dashed
    black) and the Deming errors-in-both fit (context gray)."""
    plt, ps = _mpl_and_style()
    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL_WIDTH, 2.9))
    for ax, var, tag in zip(axes, ISOTOPES, PANEL_TAGS):
        sub = df[[f'{var}_ours', f'{var}_arm']].dropna()
        a, o = sub[f'{var}_arm'].to_numpy(), sub[f'{var}_ours'].to_numpy()
        ax.scatter(a, o, s=2, c='black', alpha=0.3, linewidths=0, zorder=2)
        lo, hi = float(min(a.min(), o.min())), float(max(a.max(), o.max()))
        ax.plot([lo, hi], [lo, hi], ls='--', color='black', lw=0.8, zorder=3)
        slope, intercept = deming(a, o)
        ax.plot([lo, hi], [intercept + slope * lo, intercept + slope * hi],
                color=ps.CONTEXT_GRAY, lw=1.1, zorder=4)
        r = float(np.corrcoef(a, o)[0, 1])
        ax.set_xlabel(f'ARM {VAR_LABELS[var]}')
        ax.set_ylabel(f'This study {VAR_LABELS[var]}')
        ax.text(0.04, 0.96, f'{tag}\nDeming slope {slope:.3f}\nr {r:.3f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=7)
    return ps.save_figure(fig, out_dir / 'arm_scatter')


def make_difference_figure(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Difference time series (this study minus ARM), one stacked panel per
    isotope. The seasonal trend is the signature of the rolling VSMOW-SLAP
    normalization against ARM's static calibration."""
    plt, ps = _mpl_and_style()
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(ps.FULL_WIDTH, 7.2))
    for ax, var, tag in zip(axes, ISOTOPES, PANEL_TAGS):
        sub = df[[f'{var}_ours', f'{var}_arm']].dropna()
        diff = sub[f'{var}_ours'] - sub[f'{var}_arm']
        ax.axhline(0.0, color=ps.CONTEXT_GRAY, lw=0.8, zorder=1)
        ax.plot(sub.index, diff, color='black', lw=0.7, zorder=2)
        ax.set_ylabel(DIFF_LABELS[var])
        ps.panel_label(ax, tag)
        ps.date_axis(ax)
    axes[-1].set_xlabel('2022–2023')
    return ps.save_figure(fig, out_dir / 'arm_difference_timeseries')


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _to_markdown(table: pd.DataFrame) -> str:
    """GitHub-flavored markdown table (no tabulate dependency)."""
    def fmt(v):
        return f'{v:.4g}' if isinstance(v, float) else str(v)
    cols = list(table.columns)
    head = '| ' + ' | '.join(cols) + ' |'
    rule = '| ' + ' | '.join('---' for _ in cols) + ' |'
    body = ['| ' + ' | '.join(fmt(v) for v in row) + ' |'
            for row in table.itertuples(index=False)]
    return '\n'.join([head, rule, *body]) + '\n'



def run(ours_path: Path, arm_path: Optional[str], amap: Dict,
        cadence: str, measured_only: bool, out_dir: Path,
        self_test: bool) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    ours = load_ours(ours_path, measured_only=measured_only)
    logger.info("ours: %d points, %s to %s", len(ours),
                ours.index.min(), ours.index.max())
    if self_test:
        arm = ours.copy()                     # compare to self -> perfect
        logger.info("SELF-TEST: comparing the dataset against itself")
    else:
        arm = load_arm(arm_path, amap)
        logger.info("ARM: %d points, %s to %s", len(arm),
                    arm.index.min(), arm.index.max())
    joined = align(ours, arm, cadence=cadence)
    logger.info("overlap on %s grid: %d bins", cadence, len(joined))
    rows = [compare_variable(joined, v) for v in ISOTOPES]
    table = pd.DataFrame(rows)
    with pd.option_context('display.float_format', '{:.4f}'.format):
        logger.info("\n%s", table.to_string(index=False))
    table.to_csv(out_dir / 'arm_comparison_stats.csv', index=False)
    (out_dir / 'arm_comparison_stats.md').write_text(_to_markdown(table))
    figs = make_figures(joined, out_dir)
    figs += make_overlay_figure(joined, out_dir)
    figs += make_scatter_figure(joined, out_dir)
    figs += make_difference_figure(joined, out_dir)
    logger.info("wrote %d figures + stats table to %s", len(figs), out_dir)
    return table


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ours-path', type=Path, default=DEFAULT_OURS)
    p.add_argument('--arm-path', default=None,
                   help='ARM netCDF file/dir/glob, or a CSV')
    p.add_argument('--arm-time', default=ARM_MAP['time'])
    p.add_argument('--arm-dD', default=ARM_MAP['dD'])
    p.add_argument('--arm-d18O', default=ARM_MAP['d18O'])
    p.add_argument('--arm-d_excess', default=ARM_MAP['d_excess'])
    p.add_argument('--arm-h2o', default=ARM_MAP['H2O'])
    p.add_argument('--cadence', default='1h', help="pandas offset, e.g. 1h, 6h")
    p.add_argument('--include-interpolated', action='store_true',
                   help='keep the in-filled malfunction window')
    p.add_argument('--out-dir', type=Path, default=Path('arm_comparison_output'))
    p.add_argument('--list-vars', action='store_true',
                   help='print ARM file variables and exit')
    p.add_argument('--self-test', action='store_true',
                   help='compare the dataset to itself (harness validation)')
    args = p.parse_args()

    if args.list_vars:
        if not args.arm_path:
            p.error('--list-vars needs --arm-path')
        list_arm_vars(args.arm_path)
        return
    if not args.self_test and not args.arm_path:
        p.error('provide --arm-path (or use --self-test)')

    amap = {'time': args.arm_time, 'dD': args.arm_dD, 'd18O': args.arm_d18O,
            'd_excess': args.arm_d_excess, 'H2O': args.arm_h2o}
    run(args.ours_path, args.arm_path, amap, args.cadence,
        measured_only=not args.include_interpolated, out_dir=args.out_dir,
        self_test=args.self_test)


if __name__ == '__main__':
    main()
