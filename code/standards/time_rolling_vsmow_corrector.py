#!/usr/bin/env python3
"""
Time-Rolling VSMOW-SLAP Correction for Isotope Data

This script implements a sophisticated time-interpolated VSMOW-SLAP correction
that accounts for instrument drift over time using selected high-quality standards runs.

Key Features:
1. Uses specific high-quality runs for calibration
2. Calculates time-varying offsets through interpolation
3. Applies correction to both standards and atmospheric data
4. Compares derived vs direct d-excess calculations
5. Generates comprehensive validation and publication plots
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Import required modules
from data_processing.humidity_calibration import HumidityBiasCorrector

class TimeRollingVSMOWCorrector:
    """
    Implements time-rolling VSMOW-SLAP correction with sophisticated interpolation.
    """
    
    def __init__(self, 
                 standards_data_path: Path,
                 humidity_calibration_path: Path,
                 known_standards_path: Path,
                 atmospheric_data_path: Path,
                 output_dir: Path,
                 calibration_runs: List[int]):
        """
        Initialize the time-rolling VSMOW corrector.
        """
        self.standards_data_path = Path(standards_data_path)
        self.humidity_calibration_path = Path(humidity_calibration_path)
        self.known_standards_path = Path(known_standards_path)
        self.atmospheric_data_path = Path(atmospheric_data_path)
        self.output_dir = Path(output_dir)
        self.calibration_runs = calibration_runs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        self._setup_plotting_style()
        
        # Load data and initialize components
        print("🔧 Initializing Time-Rolling VSMOW Corrector...")
        self._load_data()
        self._extract_calibration_runs()
        
    def _setup_plotting_style(self):
        """Set up professional black and white plotting style."""
        plt.style.use('classic')
        plt.rcParams.update({
            # Font settings
            'font.size': 11,
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans', 'sans-serif'],
            
            # Figure settings
            'figure.figsize': (12, 8),
            'figure.dpi': 300,
            'figure.facecolor': 'white',
            
            # Axes settings
            'axes.titlesize': 13,
            'axes.titleweight': 'bold',
            'axes.titlepad': 12,
            'axes.labelsize': 11,
            'axes.labelweight': 'bold',
            'axes.labelpad': 8,
            'axes.linewidth': 1.2,
            'axes.edgecolor': 'black',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'axes.axisbelow': True,
            
            # Grid settings
            'grid.color': 'gray',
            'grid.linestyle': ':',
            'grid.linewidth': 0.8,
            'grid.alpha': 0.7,
            
            # Tick settings
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'xtick.major.size': 6,
            'ytick.major.size': 6,
            'xtick.minor.size': 3,
            'ytick.minor.size': 3,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'xtick.color': 'black',
            'ytick.color': 'black',
            
            # Legend settings
            'legend.fontsize': 10,
            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.edgecolor': 'black',
            'legend.facecolor': 'white',
            'legend.framealpha': 1.0,
            'legend.borderpad': 0.5,
            
            # Line and marker settings
            'lines.linewidth': 1.5,
            'lines.markersize': 6,
            'lines.markeredgewidth': 0.8,
            
            # Save settings
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.15,
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'none'
        })
    
    def _load_data(self):
        """Load all required datasets."""
        print("  📂 Loading datasets...")
        
        # Load standards data
        self.standards_data = pd.read_csv(self.standards_data_path)
        if 'Time' in self.standards_data.columns:
            self.standards_data['Time'] = pd.to_datetime(self.standards_data['Time'])
            self.standards_data['date'] = self.standards_data['Time'].dt.date
        
        # Load known standards
        self.known_standards = pd.read_csv(self.known_standards_path)
        if 'Time' in self.known_standards.columns:
            self.known_standards['Time'] = pd.to_datetime(self.known_standards['Time'])
            self.known_standards['date'] = self.known_standards['Time'].dt.date
        
        # Load atmospheric data
        self.atmospheric_data = pd.read_csv(self.atmospheric_data_path)
        if 'Time' in self.atmospheric_data.columns:
            self.atmospheric_data['Time'] = pd.to_datetime(self.atmospheric_data['Time'])
        
        # Initialize humidity corrector
        self.humidity_corrector = HumidityBiasCorrector(self.humidity_calibration_path)
        
        print(f"    ✅ Standards data: {len(self.standards_data):,} points")
        print(f"    ✅ Known standards: {len(self.known_standards)} reference values")
        print(f"    ✅ Atmospheric data: {len(self.atmospheric_data):,} points")
    
    def _extract_calibration_runs(self):
        """Extract and match the specified calibration runs."""
        print(f"  🎯 Extracting calibration runs: {self.calibration_runs}")
        
        self.calibration_data = {}
        self.matched_standards = {}
        
        for run_id in self.calibration_runs:
            # Extract run data
            run_data = self.standards_data[
                self.standards_data['standards_run_id'] == run_id
            ].copy()
            
            if len(run_data) > 10:  # Only include runs with sufficient data
                self.calibration_data[run_id] = run_data
                
                # Match with known standards by date
                run_date = run_data['Time'].dt.date.iloc[0]
                
                known_match = self.known_standards[
                    self.known_standards['date'] == run_date
                ]
                
                if len(known_match) > 0:
                    standard_info = known_match.iloc[0]
                    self.matched_standards[run_id] = {
                        'date': run_date,
                        'datetime': pd.to_datetime(run_date),
                        'name': standard_info['Name'],
                        'dD_known': standard_info['dD_known'],
                        'd18O_known': standard_info['d18O_known'],
                        'n_points': len(run_data)
                    }
                    print(f"    ✅ Run {run_id} ({run_date}) -> {standard_info['Name']}")
                else:
                    print(f"    ❌ Run {run_id} ({run_date}) -> No known standard match")
            else:
                print(f"    ❌ Run {run_id} -> Insufficient data ({len(run_data)} points)")
        
        print(f"  📊 Successfully matched {len(self.matched_standards)} calibration runs")
    
    def calculate_time_varying_offsets(self):
        """Calculate time-varying VSMOW corrections using sliding window linear regressions."""
        print("🧮 Calculating sliding window VSMOW-SLAP corrections...")
        
        # First, prepare all standards data with measurements
        all_standards_data = []
        
        for run_id in self.calibration_runs:
            if run_id in self.matched_standards:
                run_data = self.calibration_data[run_id]
                standard_info = self.matched_standards[run_id]
                
                # Apply humidity correction
                corrected_data = self.humidity_corrector.apply_correction(run_data)
                
                # Calculate robust statistics for this run
                dd_stats = self._calculate_windowed_statistics(corrected_data['D_del_corrected'])
                d18o_stats = self._calculate_windowed_statistics(corrected_data['O18_del_corrected'])
                
                all_standards_data.append({
                    'run_id': run_id,
                    'datetime': standard_info['datetime'],
                    'standard_name': standard_info['name'],
                    'dD_measured': dd_stats['median'],
                    'dD_known': standard_info['dD_known'],
                    'd18O_measured': d18o_stats['median'],
                    'd18O_known': standard_info['d18O_known'],
                    'dD_mad': dd_stats['mad'],
                    'd18O_mad': d18o_stats['mad'],
                    'n_points': len(corrected_data)
                })
                
                print(f"    Run {run_id} ({standard_info['name']}): δD={dd_stats['median']:.1f}‰, δ18O={d18o_stats['median']:.1f}‰")
        
        self.standards_df = pd.DataFrame(all_standards_data)
        
        # Calculate sliding window linear corrections
        self._calculate_sliding_window_corrections()
        
        # Save standards data
        standards_file = self.output_dir / "sliding_window_standards_data.csv"
        self.standards_df.to_csv(standards_file, index=False)
        
        # Save corrections data
        corrections_file = self.output_dir / "sliding_window_vsmow_corrections.csv"
        self.corrections_df.to_csv(corrections_file, index=False)
        
        print(f"  💾 Saved standards data: {standards_file.name}")
        print(f"  💾 Saved corrections data: {corrections_file.name}")
        print(f"  📊 Generated {len(self.corrections_df)} sliding window corrections")
    
    def _calculate_sliding_window_corrections(self):
        """Calculate slope and intercept corrections for 15-day intervals using expanding windows."""
        print("  📊 Calculating sliding window linear corrections...")
        
        # Determine study period
        start_date = self.standards_df['datetime'].min()
        end_date = self.standards_df['datetime'].max()
        study_days = (end_date - start_date).days
        
        print(f"    📅 Study period: {start_date.date()} to {end_date.date()} ({study_days} days)")
        
        # Create 15-day intervals
        interval_days = 15
        correction_points = []
        
        # Start from the first standard and go to the last
        current_date = start_date
        point_number = 0
        
        while current_date <= end_date:
            point_number += 1
            
            # Try expanding windows: 15, 20, 25, 30 days
            window_standards = None
            window_half_size = None
            
            for half_window_days in [15, 20, 25, 30]:
                window_start = current_date - pd.Timedelta(days=half_window_days)
                window_end = current_date + pd.Timedelta(days=half_window_days)
                
                # Find standards within this window
                candidate_standards = self.standards_df[
                    (self.standards_df['datetime'] >= window_start) & 
                    (self.standards_df['datetime'] <= window_end)
                ].copy()
                
                # Check if we have enough standards and they're not all identical
                if len(candidate_standards) >= 3:
                    # Check if we have diversity in standards for regression
                    dd_unique = len(np.unique(candidate_standards['dD_known'].values))
                    d18o_unique = len(np.unique(candidate_standards['d18O_known'].values))
                    
                    if dd_unique >= 2 and d18o_unique >= 2:
                        # Found suitable window
                        window_standards = candidate_standards
                        window_half_size = half_window_days
                        break
                    elif half_window_days == 30:
                        # Last attempt - accept even if all identical (will skip later)
                        window_standards = candidate_standards
                        window_half_size = half_window_days
                
            if window_standards is not None and len(window_standards) >= 3:
                # Check again for diversity before attempting regression
                dd_unique = len(np.unique(window_standards['dD_known'].values))
                d18o_unique = len(np.unique(window_standards['d18O_known'].values))
                
                if dd_unique >= 2 and d18o_unique >= 2:
                    # Calculate linear regression for δD: measured = slope * known + intercept
                    dd_slope, dd_intercept, dd_r2, dd_n = self._fit_measured_vs_known_regression(
                        window_standards['dD_known'].values,
                        window_standards['dD_measured'].values,
                        'δD'
                    )
                    
                    # Calculate linear regression for δ18O
                    d18o_slope, d18o_intercept, d18o_r2, d18o_n = self._fit_measured_vs_known_regression(
                        window_standards['d18O_known'].values,
                        window_standards['d18O_measured'].values,
                        'δ18O'
                    )
                    
                    correction_points.append({
                        'point_number': point_number,
                        'center_date': current_date,
                        'days_from_start': (current_date - start_date).days,
                        'window_start': current_date - pd.Timedelta(days=window_half_size),
                        'window_end': current_date + pd.Timedelta(days=window_half_size),
                        'window_size': window_half_size * 2,
                        'n_standards': len(window_standards),
                        'dD_slope': dd_slope,
                        'dD_intercept': dd_intercept,
                        'dD_r_squared': dd_r2,
                        'd18O_slope': d18o_slope,
                        'd18O_intercept': d18o_intercept,
                        'd18O_r_squared': d18o_r2,
                        'standards_used': ', '.join(window_standards['standard_name'].unique())
                    })
                    
                    window_info = f"±{window_half_size}d" if window_half_size > 15 else "std"
                    print(f"      Point {point_number} ({current_date.date()}): {len(window_standards)} standards ({window_info}), "
                          f"δD R²={dd_r2:.3f}, δ18O R²={d18o_r2:.3f}")
                else:
                    print(f"      Point {point_number} ({current_date.date()}): {len(window_standards)} standards but insufficient diversity - SKIPPED")
            else:
                total_standards = len(window_standards) if window_standards is not None else 0
                print(f"      Point {point_number} ({current_date.date()}): Only {total_standards} standards even with ±30d - SKIPPED")
            
            # Move to next interval
            current_date += pd.Timedelta(days=interval_days)
        
        self.corrections_df = pd.DataFrame(correction_points)
        print(f"    ✅ Generated {len(self.corrections_df)} correction points from {point_number} intervals")
        
        # Create interpolation functions for time-varying corrections
        self._create_sliding_window_interpolation_functions()
    
    def _fit_measured_vs_known_regression(self, known_values, measured_values, isotope_name):
        """Fit linear regression: measured = slope * known + intercept"""
        from scipy import stats
        
        # Perform linear regression (diversity check done before calling this function)
        slope, intercept, r_value, p_value, std_err = stats.linregress(known_values, measured_values)
        r_squared = r_value ** 2
        n_points = len(known_values)
        
        return slope, intercept, r_squared, n_points
    
    def _create_sliding_window_interpolation_functions(self):
        """Create interpolation functions for time-varying slope and intercept corrections."""
        print("  📈 Creating sliding window interpolation functions...")
        
        # Extract time points and correction parameters
        time_points = self.corrections_df['days_from_start'].values
        dd_slopes = self.corrections_df['dD_slope'].values
        dd_intercepts = self.corrections_df['dD_intercept'].values
        d18o_slopes = self.corrections_df['d18O_slope'].values
        d18o_intercepts = self.corrections_df['d18O_intercept'].values
        
        # Store calibration period bounds
        self.min_calibration_day = time_points.min()
        self.max_calibration_day = time_points.max()
        
        # Create interpolation functions for slopes and intercepts
        self.dd_slope_interp = interpolate.interp1d(
            time_points, dd_slopes, kind='linear', 
            bounds_error=False, fill_value=(dd_slopes[0], dd_slopes[-1])
        )
        self.dd_intercept_interp = interpolate.interp1d(
            time_points, dd_intercepts, kind='linear',
            bounds_error=False, fill_value=(dd_intercepts[0], dd_intercepts[-1])
        )
        self.d18o_slope_interp = interpolate.interp1d(
            time_points, d18o_slopes, kind='linear',
            bounds_error=False, fill_value=(d18o_slopes[0], d18o_slopes[-1])
        )
        self.d18o_intercept_interp = interpolate.interp1d(
            time_points, d18o_intercepts, kind='linear',
            bounds_error=False, fill_value=(d18o_intercepts[0], d18o_intercepts[-1])
        )
        
        # Store reference datetime for conversion
        self.reference_datetime = self.standards_df['datetime'].min()
        
        print(f"    ✅ Sliding window interpolation functions created")
        print(f"    📅 Reference date: {self.reference_datetime.date()}")
        print(f"    📏 Correction span: {time_points.min():.0f} to {time_points.max():.0f} days")
        print(f"    📈 δD slope range: {dd_slopes.min():.4f} to {dd_slopes.max():.4f}")
        print(f"    📈 δ18O slope range: {d18o_slopes.min():.4f} to {d18o_slopes.max():.4f}")
        print(f"    ⚠️  Data outside correction period will use boundary values")

    def _calculate_windowed_statistics(self, data_series: pd.Series) -> dict:
        """Calculate robust windowed statistics for calibration run data."""
        # Remove outliers using IQR method for robustness
        Q1 = data_series.quantile(0.25)
        Q3 = data_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out outliers
        filtered_data = data_series[(data_series >= lower_bound) & (data_series <= upper_bound)]
        
        # Calculate robust statistics
        median_val = filtered_data.median()
        mad_val = np.median(np.abs(filtered_data - median_val))  # Median absolute deviation
        std_val = filtered_data.std()
        
        return {
            'median': median_val,
            'std': std_val,
            'mad': mad_val,
            'n_filtered': len(filtered_data),
            'n_removed': len(data_series) - len(filtered_data)
        }
    
    def _create_linear_calibration_functions(self):
        """Create linear calibration using slope and intercept from deviations."""
        print("  📊 Creating linear calibration functions...")
        
        from scipy import stats
        
        # Extract time points and deviations
        time_points = self.offset_df['days_from_start'].values
        dd_deviations = self.offset_df['dD_deviation'].values
        d18o_deviations = self.offset_df['d18O_deviation'].values
        
        # Store calibration period bounds
        self.min_calibration_day = time_points.min()
        self.max_calibration_day = time_points.max()
        
        # Fit linear regression to deviations over time
        dd_slope, dd_intercept, dd_r_value, dd_p_value, dd_stderr = stats.linregress(time_points, dd_deviations)
        d18o_slope, d18o_intercept, d18o_r_value, d18o_p_value, d18o_stderr = stats.linregress(time_points, d18o_deviations)
        
        # Store calibration parameters
        self.dd_calibration = {
            'slope': dd_slope,
            'intercept': dd_intercept,
            'r_squared': dd_r_value**2,
            'p_value': dd_p_value,
            'std_error': dd_stderr
        }
        
        self.d18o_calibration = {
            'slope': d18o_slope,
            'intercept': d18o_intercept,
            'r_squared': d18o_r_value**2,
            'p_value': d18o_p_value,
            'std_error': d18o_stderr
        }
        
        # Create linear interpolation functions for time-varying corrections
        self.dd_interp = interpolate.interp1d(
            time_points, dd_deviations, kind='linear', 
            bounds_error=False, fill_value=(dd_deviations[0], dd_deviations[-1])
        )
        self.d18o_interp = interpolate.interp1d(
            time_points, d18o_deviations, kind='linear',
            bounds_error=False, fill_value=(d18o_deviations[0], d18o_deviations[-1])
        )
        
        # Store reference datetime for conversion
        self.reference_datetime = min([s['datetime'] for s in self.matched_standards.values()])
        
        print(f"    ✅ Linear calibration functions created")
        print(f"    📅 Reference date: {self.reference_datetime.date()}")
        print(f"    📏 Calibration span: {time_points.min():.0f} to {time_points.max():.0f} days")
        print(f"    📈 δD linear fit: slope = {dd_slope:.4f}‰/day, R² = {dd_r_value**2:.3f}")
        print(f"    📈 δ18O linear fit: slope = {d18o_slope:.4f}‰/day, R² = {d18o_r_value**2:.3f}")
        print(f"    ⚠️  Data outside calibration period will use boundary values")
    
    def apply_time_varying_vsmow_correction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply time-varying sliding window VSMOW correction to data."""
        print(f"  🔧 Applying sliding window VSMOW correction to {len(data):,} points...")
        
        # Calculate days from reference
        days_from_ref = (data['Time'] - self.reference_datetime).dt.total_seconds() / (24 * 3600)
        
        # Check how much data falls outside calibration period
        before_calib = np.sum(days_from_ref < self.min_calibration_day)
        after_calib = np.sum(days_from_ref > self.max_calibration_day)
        within_calib = len(data) - before_calib - after_calib
        
        print(f"    📊 Data distribution:")
        print(f"        Before correction period: {before_calib:,} points ({100*before_calib/len(data):.1f}%)")
        print(f"        Within correction period: {within_calib:,} points ({100*within_calib/len(data):.1f}%)")
        print(f"        After correction period: {after_calib:,} points ({100*after_calib/len(data):.1f}%)")
        
        # Get interpolated slope and intercept values for these times
        dd_slopes = self.dd_slope_interp(days_from_ref)
        dd_intercepts = self.dd_intercept_interp(days_from_ref)
        d18o_slopes = self.d18o_slope_interp(days_from_ref)
        d18o_intercepts = self.d18o_intercept_interp(days_from_ref)
        
        # Apply sliding window VSMOW correction using slope and intercept
        # Corrected = (Measured - intercept) / slope
        # This inverts the relationship: measured = slope * true + intercept
        corrected_data = data.copy()
        corrected_data['D_del_vsmow'] = (data['D_del_corrected'] - dd_intercepts) / dd_slopes
        corrected_data['O18_del_vsmow'] = (data['O18_del_corrected'] - d18o_intercepts) / d18o_slopes
        
        # Calculate d-excess from VSMOW-corrected values
        corrected_data['d_excess_vsmow_derived'] = (
            corrected_data['D_del_vsmow'] - 8 * corrected_data['O18_del_vsmow']
        )
        
        # Also calculate d-excess correction using direct slope/intercept approach
        d_excess_original = data['D_del_corrected'] - 8 * data['O18_del_corrected']
        d_excess_slope = dd_slopes - 8 * d18o_slopes
        d_excess_intercept = dd_intercepts - 8 * d18o_intercepts
        corrected_data['d_excess_vsmow_direct'] = (d_excess_original - d_excess_intercept) / d_excess_slope
        
        # Store correction parameters used for each point
        corrected_data['dD_slope_applied'] = dd_slopes
        corrected_data['dD_intercept_applied'] = dd_intercepts
        corrected_data['d18O_slope_applied'] = d18o_slopes
        corrected_data['d18O_intercept_applied'] = d18o_intercepts
        
        return corrected_data
    
    def validate_vsmow_correction(self):
        """Validate VSMOW correction by applying to calibration standards."""
        print("✅ Validating VSMOW correction on calibration standards...")
        
        validation_results = []
        
        for run_id in self.calibration_runs:
            if run_id in self.matched_standards:
                run_data = self.calibration_data[run_id]
                standard_info = self.matched_standards[run_id]
                
                # Apply humidity correction first
                humidity_corrected = self.humidity_corrector.apply_correction(run_data)
                
                # Apply VSMOW correction
                vsmow_corrected = self.apply_time_varying_vsmow_correction(humidity_corrected)
                
                # Calculate statistics
                dd_vsmow_mean = vsmow_corrected['D_del_vsmow'].mean()
                d18o_vsmow_mean = vsmow_corrected['O18_del_vsmow'].mean()
                dd_vsmow_std = vsmow_corrected['D_del_vsmow'].std()
                d18o_vsmow_std = vsmow_corrected['O18_del_vsmow'].std()
                
                # Calculate final offsets (should be ~0 for calibration standards)
                final_dd_offset = dd_vsmow_mean - standard_info['dD_known']
                final_d18o_offset = d18o_vsmow_mean - standard_info['d18O_known']
                
                validation_results.append({
                    'run_id': run_id,
                    'standard_name': standard_info['name'],
                    'date': standard_info['date'],
                    'dD_vsmow_mean': dd_vsmow_mean,
                    'dD_vsmow_std': dd_vsmow_std,
                    'dD_known': standard_info['dD_known'],
                    'dD_final_offset': final_dd_offset,
                    'd18O_vsmow_mean': d18o_vsmow_mean,
                    'd18O_vsmow_std': d18o_vsmow_std,
                    'd18O_known': standard_info['d18O_known'],
                    'd18O_final_offset': final_d18o_offset
                })
                
                print(f"    Run {run_id}: Final δD offset = {final_dd_offset:+.3f}‰, δ18O offset = {final_d18o_offset:+.3f}‰")
        
        self.validation_df = pd.DataFrame(validation_results)
        
        # Save validation results
        validation_file = self.output_dir / "vsmow_correction_validation.csv"
        self.validation_df.to_csv(validation_file, index=False)
        
        print(f"  💾 Saved validation results: {validation_file.name}")
        
        # Calculate validation statistics
        print(f"  📊 Validation statistics:")
        print(f"      δD final offset: {self.validation_df['dD_final_offset'].mean():+.3f} ± {self.validation_df['dD_final_offset'].std():.3f}‰")
        print(f"      δ18O final offset: {self.validation_df['d18O_final_offset'].mean():+.3f} ± {self.validation_df['d18O_final_offset'].std():.3f}‰")
    
    def process_atmospheric_data_with_vsmow(self):
        """Process atmospheric data with complete VSMOW correction."""
        print("🌍 Processing atmospheric data with time-rolling VSMOW correction...")
        
        # Apply humidity correction first
        print("  🔧 Applying humidity correction...")
        humidity_corrected = self.humidity_corrector.apply_correction(self.atmospheric_data)
        
        # Calculate original d-excess
        humidity_corrected['d_excess_humidity_corrected'] = (
            humidity_corrected['D_del_corrected'] - 8 * humidity_corrected['O18_del_corrected']
        )
        
        # Apply VSMOW correction
        print("  🌐 Applying time-rolling VSMOW correction...")
        vsmow_corrected = self.apply_time_varying_vsmow_correction(humidity_corrected)
        
        # Create atmospheric isotope plots matching humidity correction style
        print("  🎨 Creating atmospheric isotope plots...")
        rolling_data = self._create_atmospheric_time_series_plots(vsmow_corrected)
        self._create_atmospheric_histogram_plots(vsmow_corrected)
        
        self.vsmow_atmospheric_data = vsmow_corrected
        self.vsmow_rolling_data = rolling_data
        
        # Save processed data
        print("  💾 Saving VSMOW-corrected datasets...")
        self._save_vsmow_atmospheric_data()
        
        print("✅ Atmospheric VSMOW correction complete!")
    
    
    def _save_vsmow_atmospheric_data(self):
        """Save VSMOW-corrected atmospheric datasets."""
        # Full dataset
        output_columns = [
            'Time', 'H2O_ppm', 'D_del', 'O18_del', 
            'D_del_corrected', 'O18_del_corrected',
            'D_del_vsmow', 'O18_del_vsmow',
            'd_excess_humidity_corrected', 'd_excess_vsmow_derived', 'd_excess_vsmow_direct',
            'dD_slope_applied', 'dD_intercept_applied', 'd18O_slope_applied', 'd18O_intercept_applied'
        ]
        
        full_output = self.vsmow_atmospheric_data[output_columns].copy()
        
        # Save full dataset
        full_path = self.output_dir / "atmospheric_isotopes_vsmow_corrected_full.csv"
        full_output.to_csv(full_path, index=False)
        
        # Save rolling averages
        rolling_path = self.output_dir / "atmospheric_isotopes_vsmow_corrected_5min_rolling.csv"
        self.vsmow_rolling_data.to_csv(rolling_path, index=False)
        
        # Create clean VSMOW-corrected dataset
        clean_vsmow = self.vsmow_atmospheric_data[
            ['Time', 'H2O_ppm', 'D_del_vsmow', 'O18_del_vsmow', 'd_excess_vsmow_derived']
        ].copy()
        clean_vsmow.columns = ['Time', 'H2O_ppm', 'dD_vsmow_permil', 'd18O_vsmow_permil', 'd_excess_vsmow_permil']
        
        clean_path = self.output_dir / "atmospheric_isotopes_vsmow_final_cleaned.csv"
        clean_vsmow.to_csv(clean_path, index=False)
        
        # Save metadata
        metadata = {
            'processing_info': {
                'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'correction_sequence': ['humidity_correction', 'time_rolling_vsmow_correction'],
                'calibration_runs_used': self.calibration_runs,
                'reference_datetime': self.reference_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'interpolation_method': 'cubic_spline',
                'total_records': len(self.vsmow_atmospheric_data)
            },
            'column_descriptions': {
                'Time': 'Measurement timestamp (UTC)',
                'H2O_ppm': 'Water vapor concentration (parts per million)',
                'dD_vsmow_permil': 'Deuterium content (per mil vs VSMOW)',
                'd18O_vsmow_permil': 'Oxygen-18 content (per mil vs VSMOW)', 
                'd_excess_vsmow_permil': 'Deuterium excess (per mil vs VSMOW)'
            }
        }
        
        metadata_path = self.output_dir / "vsmow_correction_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"    📁 Full dataset: {full_path.name} ({len(full_output):,} records)")
        print(f"    📁 Rolling averages: {rolling_path.name} ({len(self.vsmow_rolling_data):,} records)")
        print(f"    📁 Final cleaned: {clean_path.name}")
        print(f"    📁 Metadata: {metadata_path.name}")
    
    def create_comprehensive_plots(self):
        """Create comprehensive publication-quality plots."""
        print("🎨 Creating comprehensive VSMOW correction plots...")
        
        # 1. Time-varying offset plots
        self._plot_time_varying_offsets()
        
        # 2. Validation plots
        self._plot_validation_results()
        
        # 3. Atmospheric time series comparison
        self._plot_atmospheric_comparison()
        
        # 4. Distribution analysis
        self._plot_distribution_analysis()
        
        print("✅ All VSMOW correction plots created!")
    
    def _plot_time_varying_offsets(self):
        """Plot time-varying slope and intercept corrections with sliding windows."""
        print("    📈 Creating sliding window correction plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Sliding Window VSMOW-SLAP Calibration\n'
                    '15-Day Intervals with Linear Regression Corrections', 
                    fontsize=15, fontweight='bold', y=0.98)
        
        # Create time range for interpolation plotting
        time_range = np.linspace(
            self.corrections_df['days_from_start'].min() - 10,
            self.corrections_df['days_from_start'].max() + 10,
            1000
        )
        dd_slope_vals = self.dd_slope_interp(time_range)
        dd_intercept_vals = self.dd_intercept_interp(time_range)
        d18o_slope_vals = self.d18o_slope_interp(time_range)
        d18o_intercept_vals = self.d18o_intercept_interp(time_range)
        
        # Convert to actual dates for x-axis
        date_range = [self.reference_datetime + pd.Timedelta(days=d) for d in time_range]
        correction_dates = [self.reference_datetime + pd.Timedelta(days=d) for d in self.corrections_df['days_from_start']]
        
        # Plot 1: δD slopes over time
        axes[0,0].scatter(correction_dates, self.corrections_df['dD_slope'], s=80, c='blue', 
                         alpha=0.8, edgecolors='black', linewidth=1, zorder=3,
                         label='Sliding window slopes')
        axes[0,0].plot(date_range, dd_slope_vals, 'b-', linewidth=2, alpha=0.8,
                      label='Linear interpolation')
        axes[0,0].axhline(1.0, color='gray', linestyle='--', alpha=0.8, linewidth=1, label='Perfect slope = 1.0')
        axes[0,0].set_ylabel('δD Slope (measured/known)', fontweight='bold')
        axes[0,0].set_title('δD Slope Evolution', fontweight='bold')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[0,0].tick_params(axis='x', rotation=45, labelsize=9)
        
        # Plot 2: δD intercepts over time
        axes[0,1].scatter(correction_dates, self.corrections_df['dD_intercept'], s=80, c='red', 
                         alpha=0.8, edgecolors='black', linewidth=1, zorder=3,
                         label='Sliding window intercepts')
        axes[0,1].plot(date_range, dd_intercept_vals, 'r-', linewidth=2, alpha=0.8,
                      label='Linear interpolation')
        axes[0,1].axhline(0.0, color='gray', linestyle='--', alpha=0.8, linewidth=1, label='Perfect intercept = 0')
        axes[0,1].set_ylabel('δD Intercept (‰)', fontweight='bold')
        axes[0,1].set_title('δD Intercept Evolution', fontweight='bold')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[0,1].tick_params(axis='x', rotation=45, labelsize=9)
        
        # Plot 3: δ18O slopes over time  
        axes[1,0].scatter(correction_dates, self.corrections_df['d18O_slope'], s=80, c='blue',
                         alpha=0.8, edgecolors='black', linewidth=1, zorder=3,
                         label='Sliding window slopes')
        axes[1,0].plot(date_range, d18o_slope_vals, 'b-', linewidth=2, alpha=0.8,
                      label='Linear interpolation')
        axes[1,0].axhline(1.0, color='gray', linestyle='--', alpha=0.8, linewidth=1, label='Perfect slope = 1.0')
        axes[1,0].set_ylabel('δ18O Slope (measured/known)', fontweight='bold')
        axes[1,0].set_title('δ18O Slope Evolution', fontweight='bold')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[1,0].tick_params(axis='x', rotation=45, labelsize=9)
        
        # Plot 4: δ18O intercepts over time
        axes[1,1].scatter(correction_dates, self.corrections_df['d18O_intercept'], s=80, c='red',
                         alpha=0.8, edgecolors='black', linewidth=1, zorder=3,
                         label='Sliding window intercepts')
        axes[1,1].plot(date_range, d18o_intercept_vals, 'r-', linewidth=2, alpha=0.8,
                      label='Linear interpolation')
        axes[1,1].axhline(0.0, color='gray', linestyle='--', alpha=0.8, linewidth=1, label='Perfect intercept = 0')
        axes[1,1].set_ylabel('δ18O Intercept (‰)', fontweight='bold')
        axes[1,1].set_title('δ18O Intercept Evolution', fontweight='bold')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[1,1].tick_params(axis='x', rotation=45, labelsize=9)
        
        # Plot 5: Standards used with color coding (use different subplot)
        standards_colors = {'UNM Greenland': 'black', 'UNM South Pole': 'gray', 
                           'USGS46': 'dimgray', 'USGS47': 'darkgray', 'W-64444': 'lightgray'}
        
        for i, std_name in enumerate(standards_colors.keys()):
            std_data = self.standards_df[self.standards_df['standard_name'] == std_name]
            if len(std_data) > 0:
                std_dates = std_data['datetime'].tolist()
                axes[0,2].scatter(std_dates, [i] * len(std_dates), 
                                 c=standards_colors[std_name], s=100, alpha=0.8,
                                 edgecolors='black', linewidth=1, label=std_name)
        
        axes[0,2].set_ylabel('Standard Type', fontweight='bold')
        axes[0,2].set_title('Calibration Standards Timeline', fontweight='bold')
        axes[0,2].set_yticks(range(len(standards_colors)))
        axes[0,2].set_yticklabels(list(standards_colors.keys()))
        axes[0,2].legend(loc='upper right')
        axes[0,2].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[0,2].tick_params(axis='x', rotation=45, labelsize=9)
        
        # Plot 6: Sliding window calibration statistics
        offset_stats_text = f"""Sliding Window Calibration Summary:
        
Total runs used: {len(self.calibration_runs)}
Standards data points: {len(self.standards_df)}
Correction windows: {len(self.corrections_df)}
Time span: {self.corrections_df['days_from_start'].max():.0f} days

δD Slope Statistics:
Mean: {self.corrections_df['dD_slope'].mean():.4f}
Range: {self.corrections_df['dD_slope'].min():.4f} to {self.corrections_df['dD_slope'].max():.4f}
Avg R²: {self.corrections_df['dD_r_squared'].mean():.3f}

δ18O Slope Statistics:
Mean: {self.corrections_df['d18O_slope'].mean():.4f}
Range: {self.corrections_df['d18O_slope'].min():.4f} to {self.corrections_df['d18O_slope'].max():.4f}
Avg R²: {self.corrections_df['d18O_r_squared'].mean():.3f}"""
        
        axes[1,1].text(0.05, 0.95, offset_stats_text, transform=axes[1,1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
        axes[1,1].set_title('Calibration Statistics', fontweight='bold')
        axes[1,1].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])  # Add padding for suptitle
        
        # Save plot
        offset_plot_path = self.output_dir / "linear_vsmow_calibration_plots.png"
        plt.savefig(offset_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      📊 Saved offset plot: {offset_plot_path.name}")
    
    def _plot_validation_results(self):
        """Plot validation results showing correction effectiveness."""
        print("    ✅ Creating validation plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('VSMOW Correction Validation\n'
                    'Before and After Time-Rolling Correction',
                    fontsize=15, fontweight='bold', y=0.95)
        
        # Plot validation for each isotope
        for i, (isotope, known_col, offset_col) in enumerate([
            ('δD', 'dD_known', 'dD_final_offset'),
            ('δ18O', 'd18O_known', 'd18O_final_offset')
        ]):
            
            # Before correction (from standards_df)
            ax_before = axes[0, i]
            ax_before.scatter(self.standards_df[known_col.replace('known', 'known')], 
                             self.standards_df[known_col.replace('known', 'measured')],
                             s=80, c='gray', alpha=0.7, edgecolors='black', linewidth=1)
            
            # Add 1:1 line
            min_val = min(self.standards_df[known_col].min(), self.standards_df[known_col.replace('known', 'measured')].min())
            max_val = max(self.standards_df[known_col].max(), self.standards_df[known_col.replace('known', 'measured')].max())
            ax_before.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='1:1 line')
            
            ax_before.set_xlabel(f'{isotope} Known (‰ VSMOW)', fontweight='bold')
            ax_before.set_ylabel(f'{isotope} Measured (‰ after humidity corr)', fontweight='bold')
            ax_before.set_title(f'{isotope}: Before VSMOW Correction', fontweight='bold')
            ax_before.legend()
            ax_before.grid(True, alpha=0.7, linestyle=':', color='gray')
            
            # After correction
            ax_after = axes[1, i]
            if isotope == 'δD':
                measured_after = self.validation_df['dD_vsmow_mean']
                std_col = 'dD_vsmow_std'
            else:  # δ18O
                measured_after = self.validation_df['d18O_vsmow_mean']
                std_col = 'd18O_vsmow_std'
            ax_after.scatter(self.validation_df[known_col], measured_after,
                           s=80, c='black', alpha=0.8, edgecolors='black', linewidth=1)
            
            # Error bars
            ax_after.errorbar(self.validation_df[known_col], measured_after,
                            yerr=self.validation_df[std_col], fmt='none',
                            alpha=0.6, color='black', capsize=3, capthick=1.5)
            
            # Add 1:1 line
            min_val = min(self.validation_df[known_col].min(), measured_after.min())
            max_val = max(self.validation_df[known_col].max(), measured_after.max())
            ax_after.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='1:1 line')
            
            ax_after.set_xlabel(f'{isotope} Known (‰ VSMOW)', fontweight='bold')
            ax_after.set_ylabel(f'{isotope} Measured (‰ VSMOW corrected)', fontweight='bold')
            ax_after.set_title(f'{isotope}: After VSMOW Correction', fontweight='bold')
            ax_after.legend()
            ax_after.grid(True, alpha=0.7, linestyle=':', color='gray')
        
        # Summary statistics plot
        ax_summary = axes[0, 2]
        
        # Calculate improvement metrics
        # Calculate before-correction deviations from standards data
        dd_deviations_before = self.standards_df['dD_measured'] - self.standards_df['dD_known']
        d18o_deviations_before = self.standards_df['d18O_measured'] - self.standards_df['d18O_known']
        
        dd_rmse_before = np.sqrt(np.mean(dd_deviations_before**2))
        d18o_rmse_before = np.sqrt(np.mean(d18o_deviations_before**2))
        dd_rmse_after = np.sqrt(np.mean(self.validation_df['dD_final_offset']**2))
        d18o_rmse_after = np.sqrt(np.mean(self.validation_df['d18O_final_offset']**2))
        
        summary_text = f"""Correction Performance:

δD Improvement:
RMSE before: {dd_rmse_before:.3f}‰
RMSE after: {dd_rmse_after:.3f}‰
Improvement: {((dd_rmse_before - dd_rmse_after)/dd_rmse_before)*100:.1f}%

δ18O Improvement:
RMSE before: {d18o_rmse_before:.3f}‰
RMSE after: {d18o_rmse_after:.3f}‰
Improvement: {((d18o_rmse_before - d18o_rmse_after)/d18o_rmse_before)*100:.1f}%

Final Offsets:
δD: {self.validation_df['dD_final_offset'].mean():+.3f} ± {self.validation_df['dD_final_offset'].std():.3f}‰
δ18O: {self.validation_df['d18O_final_offset'].mean():+.3f} ± {self.validation_df['d18O_final_offset'].std():.3f}‰"""
        
        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
        ax_summary.set_title('Validation Summary', fontweight='bold')
        ax_summary.axis('off')
        
        # Final offset distribution
        ax_dist = axes[1, 2]
        x_pos = [0, 1]
        dd_offsets = [self.validation_df['dD_final_offset'].mean()]
        d18o_offsets = [self.validation_df['d18O_final_offset'].mean()]
        dd_errs = [self.validation_df['dD_final_offset'].std()]
        d18o_errs = [self.validation_df['d18O_final_offset'].std()]
        
        ax_dist.bar([0], dd_offsets, yerr=dd_errs, capsize=5, color='black', alpha=0.7,
                   edgecolor='black', linewidth=1, label='δD')
        ax_dist.bar([1], d18o_offsets, yerr=d18o_errs, capsize=5, color='gray', alpha=0.7,
                   edgecolor='black', linewidth=1, label='δ18O')
        
        ax_dist.axhline(0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Perfect correction')
        ax_dist.set_xticks([0, 1])
        ax_dist.set_xticklabels(['δD', 'δ18O'])
        ax_dist.set_ylabel('Final Offset (‰)', fontweight='bold')
        ax_dist.set_title('Final Correction Performance', fontweight='bold')
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.7, linestyle=':', color='gray')
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])  # Add padding for suptitle
        
        # Save plot
        validation_plot_path = self.output_dir / "vsmow_correction_validation.png"
        plt.savefig(validation_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      📊 Saved validation plot: {validation_plot_path.name}")
    
    def _plot_atmospheric_comparison(self):
        """Plot atmospheric data comparison through the correction process."""
        print("    🌍 Creating atmospheric comparison plots...")
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 20))
        fig.suptitle('Atmospheric Water Vapor Isotopes: Complete Correction Process\n'
                    '5-Minute Rolling Averages', fontsize=16, fontweight='bold', y=0.98)
        
        time_data = self.vsmow_rolling_data['Time']
        
        # Plot 1: H2O concentration
        axes[0].plot(time_data, self.vsmow_rolling_data['H2O_ppm'], 'k-', linewidth=1.5, alpha=0.8)
        axes[0].set_ylabel('H₂O Concentration (ppm)', fontweight='bold', fontsize=12)
        axes[0].set_title('Water Vapor Concentration', fontweight='bold', fontsize=13)
        axes[0].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[0].tick_params(labelsize=10)
        
        # Plot 2: δD through correction process
        axes[1].plot(time_data, self.vsmow_rolling_data['D_del'], 'lightgray', linewidth=1.2, alpha=0.7,
                    label='Original')
        axes[1].plot(time_data, self.vsmow_rolling_data['D_del_corrected'], 'gray', linewidth=1.5, alpha=0.8,
                    label='Humidity corrected')
        axes[1].plot(time_data, self.vsmow_rolling_data['D_del_vsmow'], 'black', linewidth=1.8,
                    label='VSMOW corrected')
        axes[1].set_ylabel('δD (‰)', fontweight='bold', fontsize=12)
        axes[1].set_title('Deuterium Content: Complete Correction Process', fontweight='bold', fontsize=13)
        axes[1].legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
        axes[1].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[1].tick_params(labelsize=10)
        
        # Plot 3: δ18O through correction process
        axes[2].plot(time_data, self.vsmow_rolling_data['O18_del'], 'lightgray', linewidth=1.2, alpha=0.7,
                    label='Original')
        axes[2].plot(time_data, self.vsmow_rolling_data['O18_del_corrected'], 'gray', linewidth=1.5, alpha=0.8,
                    label='Humidity corrected')
        axes[2].plot(time_data, self.vsmow_rolling_data['O18_del_vsmow'], 'black', linewidth=1.8,
                    label='VSMOW corrected')
        axes[2].set_ylabel('δ18O (‰)', fontweight='bold', fontsize=12)
        axes[2].set_title('Oxygen-18 Content: Complete Correction Process', fontweight='bold', fontsize=13)
        axes[2].legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
        axes[2].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[2].tick_params(labelsize=10)
        
        # Plot 4: d-excess comparison
        axes[3].plot(time_data, self.vsmow_rolling_data['d_excess_humidity_corrected'], 'gray', 
                    linewidth=1.5, alpha=0.8, label='d-excess (humidity corrected)')
        axes[3].plot(time_data, self.vsmow_rolling_data['d_excess_vsmow_derived'], 'black', 
                    linewidth=1.8, label='d-excess (VSMOW derived)')
        axes[3].plot(time_data, self.vsmow_rolling_data['d_excess_vsmow_direct'], 'k--', 
                    linewidth=1.5, alpha=0.7, label='d-excess (VSMOW direct)')
        axes[3].set_ylabel('d-excess (‰)', fontweight='bold', fontsize=12)
        axes[3].set_xlabel('Time', fontweight='bold', fontsize=12)
        axes[3].set_title('Deuterium Excess: Correction Methods Comparison', fontweight='bold', fontsize=13)
        axes[3].legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
        axes[3].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[3].tick_params(labelsize=10, axis='x', rotation=45)
        
        # Format x-axis for all subplots
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Add padding for suptitle
        
        # Save plot
        atmospheric_plot_path = self.output_dir / "atmospheric_vsmow_correction_complete.png"
        plt.savefig(atmospheric_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      📊 Saved atmospheric plot: {atmospheric_plot_path.name}")
    
    def _plot_distribution_analysis(self):
        """Plot distribution analysis of VSMOW-corrected data."""
        print("    📊 Creating distribution analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('VSMOW-Corrected Data: Distribution Analysis\n'
                    'Statistical Properties and Data Quality Assessment', 
                    fontsize=15, fontweight='bold', y=0.96)
        
        # Use rolling data for cleaner plots
        rolling_data = self.vsmow_rolling_data
        
        # Plot 1: δD distribution
        axes[0,0].hist(rolling_data['D_del_vsmow'], bins=60, alpha=0.7, color='black', 
                      edgecolor='black', density=True)
        axes[0,0].set_xlabel('δD (‰ vs VSMOW)', fontweight='bold')
        axes[0,0].set_ylabel('Probability Density', fontweight='bold')
        axes[0,0].set_title('δD Distribution (VSMOW-Corrected)', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: δ18O distribution
        axes[0,1].hist(rolling_data['O18_del_vsmow'], bins=60, alpha=0.7, color='black', 
                      edgecolor='black', density=True)
        axes[0,1].set_xlabel('δ18O (‰ vs VSMOW)', fontweight='bold')
        axes[0,1].set_ylabel('Probability Density', fontweight='bold')
        axes[0,1].set_title('δ18O Distribution (VSMOW-Corrected)', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: d-excess distribution
        axes[1,0].hist(rolling_data['d_excess_vsmow_derived'], bins=60, alpha=0.7, color='black', 
                      edgecolor='black', density=True)
        axes[1,0].set_xlabel('d-excess (‰)', fontweight='bold')
        axes[1,0].set_ylabel('Probability Density', fontweight='bold')
        axes[1,0].set_title('d-excess Distribution (VSMOW-Corrected)', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Statistical summary
        stats_text = f"""VSMOW-Corrected Data Statistics:
        
δD (‰ vs VSMOW):
  Mean: {rolling_data['D_del_vsmow'].mean():.2f}
  Std: ±{rolling_data['D_del_vsmow'].std():.2f}
  Range: {rolling_data['D_del_vsmow'].min():.1f} to {rolling_data['D_del_vsmow'].max():.1f}

δ18O (‰ vs VSMOW):
  Mean: {rolling_data['O18_del_vsmow'].mean():.2f}
  Std: ±{rolling_data['O18_del_vsmow'].std():.2f}
  Range: {rolling_data['O18_del_vsmow'].min():.1f} to {rolling_data['O18_del_vsmow'].max():.1f}

d-excess (‰):
  Mean: {rolling_data['d_excess_vsmow_derived'].mean():.2f}
  Std: ±{rolling_data['d_excess_vsmow_derived'].std():.2f}
  Range: {rolling_data['d_excess_vsmow_derived'].min():.1f} to {rolling_data['d_excess_vsmow_derived'].max():.1f}

Data Points: {len(rolling_data):,}"""
        
        axes[1,1].text(0.05, 0.95, stats_text, transform=axes[1,1].transAxes,
                      fontsize=11, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
        axes[1,1].set_title('Summary Statistics', fontweight='bold')
        axes[1,1].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])  # Add padding for suptitle
        
        # Save plot
        distribution_plot_path = self.output_dir / "vsmow_corrected_distribution_analysis.png"
        plt.savefig(distribution_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      📊 Saved distribution analysis: {distribution_plot_path.name}")
        
        # Plot 1: Time series comparison
        axes[0,0].plot(rolling_data['Time'], rolling_data['d_excess_humidity_corrected'], 
                      color='gray', linewidth=1.5, alpha=0.7, label='Humidity corrected')
        axes[0,0].plot(rolling_data['Time'], rolling_data['d_excess_vsmow_derived'], 
                      color='black', linewidth=1.8, label='VSMOW derived')
        axes[0,0].plot(rolling_data['Time'], rolling_data['d_excess_vsmow_direct'], 
                      'k--', linewidth=1.5, alpha=0.8, label='VSMOW direct')
        
        axes[0,0].set_ylabel('d-excess (‰)', fontweight='bold')
        axes[0,0].set_title('d-excess Time Series Comparison', fontweight='bold')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[0,0].tick_params(axis='x', rotation=45, labelsize=9)
        
        # Plot 2: Direct comparison scatter plot
        axes[0,1].scatter(rolling_data['d_excess_vsmow_direct'], rolling_data['d_excess_vsmow_derived'],
                         alpha=0.6, s=8, c='black', edgecolors='none')
        
        # Add 1:1 line
        min_val = min(rolling_data['d_excess_vsmow_direct'].min(), rolling_data['d_excess_vsmow_derived'].min())
        max_val = max(rolling_data['d_excess_vsmow_direct'].max(), rolling_data['d_excess_vsmow_derived'].max())
        axes[0,1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='1:1 line')
        
        axes[0,1].set_xlabel('d-excess VSMOW Direct (‰)', fontweight='bold')
        axes[0,1].set_ylabel('d-excess VSMOW Derived (‰)', fontweight='bold')
        axes[0,1].set_title('Direct vs Derived d-excess', fontweight='bold')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.7, linestyle=':', color='gray')
        
        # Plot 3: Difference distribution
        axes[0,2].hist(dexcess_diff, bins=50, alpha=0.7, color='black', edgecolor='black', 
                      linewidth=0.5, density=True)
        axes[0,2].axvline(dexcess_diff.mean(), color='red', linestyle='-', linewidth=2, alpha=0.8,
                         label=f'Mean: {dexcess_diff.mean():.4f}‰')
        axes[0,2].axvline(0, color='gray', linestyle='--', alpha=0.8, linewidth=2, label='Perfect match')
        
        axes[0,2].set_xlabel('Difference (Derived - Direct) (‰)', fontweight='bold')
        axes[0,2].set_ylabel('Probability Density', fontweight='bold')
        axes[0,2].set_title('Method Difference Distribution', fontweight='bold')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.7, linestyle=':', color='gray')
        
        # Plot 4: Difference over time
        axes[1,0].plot(rolling_data['Time'], dexcess_diff, 'k-', linewidth=1.5, alpha=0.8)
        axes[1,0].axhline(dexcess_diff.mean(), color='red', linestyle='-', linewidth=2, alpha=0.8,
                         label=f'Mean difference: {dexcess_diff.mean():.4f}‰')
        axes[1,0].axhline(0, color='gray', linestyle='--', alpha=0.8, linewidth=2, label='Perfect match')
        
        axes[1,0].set_ylabel('Difference (Derived - Direct) (‰)', fontweight='bold')
        axes[1,0].set_xlabel('Time', fontweight='bold')
        axes[1,0].set_title('Method Difference Over Time', fontweight='bold')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[1,0].tick_params(axis='x', rotation=45, labelsize=9)
        
        # Plot 5: Statistics comparison
        methods_data = [
            rolling_data['d_excess_humidity_corrected'],
            rolling_data['d_excess_vsmow_derived'],
            rolling_data['d_excess_vsmow_direct']
        ]
        methods_labels = ['Humidity\nCorrected', 'VSMOW\nDerived', 'VSMOW\nDirect']
        
        box_plot = axes[1,1].boxplot(methods_data, labels=methods_labels, patch_artist=True)
        
        # Color the boxes
        colors = ['gray', 'black', 'darkgray']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1,1].set_ylabel('d-excess (‰)', fontweight='bold')
        axes[1,1].set_title('Statistical Comparison', fontweight='bold')
        axes[1,1].grid(True, alpha=0.7, linestyle=':', color='gray')
        
        # Plot 6: Summary statistics
        stats_text = f"""Method Comparison Statistics:

Humidity Corrected:
Mean: {rolling_data['d_excess_humidity_corrected'].mean():.2f}‰
Std: {rolling_data['d_excess_humidity_corrected'].std():.2f}‰

VSMOW Derived (dD - 8×d18O):
Mean: {rolling_data['d_excess_vsmow_derived'].mean():.2f}‰
Std: {rolling_data['d_excess_vsmow_derived'].std():.2f}‰

VSMOW Direct (offset corrected):
Mean: {rolling_data['d_excess_vsmow_direct'].mean():.2f}‰
Std: {rolling_data['d_excess_vsmow_direct'].std():.2f}‰

Method Agreement:
Mean difference: {dexcess_diff.mean():.4f}‰
RMS difference: {np.sqrt(np.mean(dexcess_diff**2)):.4f}‰
Correlation: {np.corrcoef(rolling_data['d_excess_vsmow_derived'], rolling_data['d_excess_vsmow_direct'])[0,1]:.6f}"""
        
        axes[1,2].text(0.05, 0.95, stats_text, transform=axes[1,2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
        axes[1,2].set_title('Method Comparison Statistics', fontweight='bold')
        axes[1,2].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])  # Add padding for suptitle
        
        # Save plot
        dexcess_plot_path = self.output_dir / "dexcess_methods_comparison.png"
        plt.savefig(dexcess_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      📊 Saved d-excess comparison: {dexcess_plot_path.name}")
    
    def _plot_distribution_analysis(self):
        """Plot distribution analysis comparing all correction stages."""
        print("    📊 Creating distribution analysis plots...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Atmospheric Isotope Distributions: Complete Correction Process\n'
                    'Original → Humidity Corrected → VSMOW Corrected', 
                    fontsize=15, fontweight='bold', y=0.95)
        
        # Use full data for distributions (more statistical power)
        data = self.vsmow_atmospheric_data
        
        # Define isotope data for plotting
        isotope_data = [
            ('D_del', 'D_del_corrected', 'D_del_vsmow', 'δD', '‰'),
            ('O18_del', 'O18_del_corrected', 'O18_del_vsmow', 'δ18O', '‰'),
            ('d_excess_humidity_corrected', 'd_excess_vsmow_derived', 'd_excess_vsmow_direct', 'd-excess', '‰')
        ]
        
        for i, (orig_col, humid_col, vsmow_col, name, units) in enumerate(isotope_data):
            # Handle d-excess special case
            if name == 'd-excess':
                orig_data = data['D_del'] - 8 * data['O18_del']
                humid_data = data[humid_col]
                vsmow_data = data[vsmow_col]  # Use derived method
            else:
                orig_data = data[orig_col]
                humid_data = data[humid_col]
                vsmow_data = data[vsmow_col]
            
            # Top row: Histogram comparison
            ax_hist = axes[0, i]
            
            # Calculate bins for all data
            all_data = np.concatenate([orig_data.dropna(), humid_data.dropna(), vsmow_data.dropna()])
            bins = np.linspace(np.percentile(all_data, 1), np.percentile(all_data, 99), 50)
            
            # Plot histograms
            ax_hist.hist(orig_data.dropna(), bins=bins, alpha=0.5, color='lightgray', 
                        density=True, label='Original', edgecolor='black', linewidth=0.5)
            ax_hist.hist(humid_data.dropna(), bins=bins, alpha=0.7, color='gray',
                        density=True, label='Humidity corrected', edgecolor='black', 
                        linewidth=0.5, histtype='step')
            ax_hist.hist(vsmow_data.dropna(), bins=bins, alpha=1.0,
                        density=True, label='VSMOW corrected', edgecolor='black', 
                        linewidth=1.5, histtype='step', facecolor='none')
            
            # Add mean lines
            ax_hist.axvline(orig_data.mean(), color='lightgray', linestyle='--', linewidth=1.5, alpha=0.8)
            ax_hist.axvline(humid_data.mean(), color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
            ax_hist.axvline(vsmow_data.mean(), color='black', linestyle='-', linewidth=2, alpha=0.8)
            
            ax_hist.set_ylabel('Probability Density', fontweight='bold')
            ax_hist.set_xlabel(f'{name} ({units})', fontweight='bold')
            ax_hist.set_title(f'{name} Distribution Evolution', fontweight='bold')
            ax_hist.legend(fontsize=9)
            ax_hist.grid(True, alpha=0.7, linestyle=':', color='gray')
            
            # Middle row: Q-Q plot (Original vs VSMOW)
            ax_qq = axes[1, i]
            
            # Create Q-Q plot
            orig_sorted = np.sort(orig_data.dropna())
            vsmow_sorted = np.sort(vsmow_data.dropna())
            
            # Interpolate to same length
            n_points = min(len(orig_sorted), len(vsmow_sorted))
            orig_interp = np.interp(np.linspace(0, 1, n_points), 
                                   np.linspace(0, 1, len(orig_sorted)), orig_sorted)
            vsmow_interp = np.interp(np.linspace(0, 1, n_points),
                                    np.linspace(0, 1, len(vsmow_sorted)), vsmow_sorted)
            
            ax_qq.scatter(orig_interp, vsmow_interp, alpha=0.6, s=4, color='black', edgecolors='none')
            
            # Add 1:1 line
            min_val = min(orig_interp.min(), vsmow_interp.min())
            max_val = max(orig_interp.max(), vsmow_interp.max())
            ax_qq.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8,
                      linewidth=2, label='1:1 Reference')
            
            ax_qq.set_xlabel(f'Original {name} ({units})', fontweight='bold')
            ax_qq.set_ylabel(f'VSMOW Corrected {name} ({units})', fontweight='bold')
            ax_qq.set_title(f'{name} Q-Q Plot', fontweight='bold')
            ax_qq.legend(fontsize=9)
            ax_qq.grid(True, alpha=0.7, linestyle=':', color='gray')
            
            # Bottom row: Statistical summary
            ax_stats = axes[2, i]
            
            # Calculate statistics
            orig_mean = orig_data.mean()
            humid_mean = humid_data.mean()
            vsmow_mean = vsmow_data.mean()
            orig_std = orig_data.std()
            humid_std = humid_data.std()
            vsmow_std = vsmow_data.std()
            
            stats_text = f"""{name} Statistics:

Original:
Mean: {orig_mean:.2f}{units}
Std: {orig_std:.2f}{units}

Humidity Corrected:
Mean: {humid_mean:.2f}{units}
Std: {humid_std:.2f}{units}
Δ from original: {humid_mean - orig_mean:+.2f}{units}

VSMOW Corrected:
Mean: {vsmow_mean:.2f}{units}
Std: {vsmow_std:.2f}{units}
Δ from original: {vsmow_mean - orig_mean:+.2f}{units}
Δ from humidity: {vsmow_mean - humid_mean:+.2f}{units}"""
            
            ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                         fontsize=9, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
            ax_stats.set_title(f'{name} Summary Statistics', fontweight='bold')
            ax_stats.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])  # Add padding for suptitle
        
        # Save plot
        distribution_plot_path = self.output_dir / "complete_correction_distributions.png"
        plt.savefig(distribution_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      📊 Saved distribution analysis: {distribution_plot_path.name}")
    
    def run_complete_vsmow_analysis(self):
        """Run the complete time-rolling VSMOW analysis."""
        print("🚀 Starting complete time-rolling VSMOW-SLAP analysis...\n")
        
        # Calculate time-varying offsets
        self.calculate_time_varying_offsets()
        print()
        
        # Validate correction
        self.validate_vsmow_correction()
        print()
        
        # Process atmospheric data
        self.process_atmospheric_data_with_vsmow()
        print()
        
        # Create comprehensive plots
        self.create_comprehensive_plots()
        print()
        
        # Create summary report
        self._create_summary_report()
        
        print("🎉 Complete time-rolling VSMOW-SLAP analysis finished!")
        print(f"📁 All outputs saved to: {self.output_dir}")
        
        return self.vsmow_atmospheric_data, self.validation_df
    
    def _create_summary_report(self):
        """Create comprehensive summary report."""
        report_path = self.output_dir / "time_rolling_vsmow_analysis_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("TIME-ROLLING VSMOW-SLAP CORRECTION ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("SLIDING WINDOW CALIBRATION SETUP:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Calibration runs used: {self.calibration_runs}\n")
            f.write(f"Successfully matched: {len(self.matched_standards)} runs\n")
            f.write(f"Time span: {self.corrections_df['days_from_start'].max():.0f} days\n")
            f.write(f"Reference date: {self.reference_datetime.date()}\n")
            f.write(f"Correction method: 15-day sliding windows with linear regression\n")
            f.write(f"Correction windows: {len(self.corrections_df)}\n\n")
            
            f.write("SLIDING WINDOW STATISTICS:\n")
            f.write("-" * 20 + "\n")
            # Calculate deviations from standards data
            dd_deviations = self.standards_df['dD_measured'] - self.standards_df['dD_known']
            d18o_deviations = self.standards_df['d18O_measured'] - self.standards_df['d18O_known']
            
            f.write(f"δD deviation range: {dd_deviations.min():+.2f} to {dd_deviations.max():+.2f}‰\n")
            f.write(f"δD deviation mean: {dd_deviations.mean():+.2f} ± {self.standards_df['dD_mad'].mean():.2f}‰ (MAD)\n")
            f.write(f"δD slope range: {self.corrections_df['dD_slope'].min():.4f} to {self.corrections_df['dD_slope'].max():.4f}\n")
            f.write(f"δD avg R²: {self.corrections_df['dD_r_squared'].mean():.3f}\n")
            f.write(f"δ18O deviation range: {d18o_deviations.min():+.2f} to {d18o_deviations.max():+.2f}‰\n")
            f.write(f"δ18O deviation mean: {d18o_deviations.mean():+.2f} ± {self.standards_df['d18O_mad'].mean():.2f}‰ (MAD)\n")
            f.write(f"δ18O slope range: {self.corrections_df['d18O_slope'].min():.4f} to {self.corrections_df['d18O_slope'].max():.4f}\n")
            f.write(f"δ18O avg R²: {self.corrections_df['d18O_r_squared'].mean():.3f}\n\n")
            
            f.write("VALIDATION RESULTS:\n")
            f.write("-" * 19 + "\n")
            f.write(f"Final δD offset: {self.validation_df['dD_final_offset'].mean():+.3f} ± {self.validation_df['dD_final_offset'].std():.3f}‰\n")
            f.write(f"Final δ18O offset: {self.validation_df['d18O_final_offset'].mean():+.3f} ± {self.validation_df['d18O_final_offset'].std():.3f}‰\n\n")
            
            f.write("ATMOSPHERIC DATA PROCESSING:\n")
            f.write("-" * 29 + "\n")
            f.write(f"Total atmospheric records: {len(self.vsmow_atmospheric_data):,}\n")
            f.write(f"Rolling average records: {len(self.vsmow_rolling_data):,}\n")
            f.write(f"Time range: {self.atmospheric_data['Time'].min()} to {self.atmospheric_data['Time'].max()}\n\n")
            
            # d-excess comparison
            if hasattr(self, 'vsmow_rolling_data'):
                dexcess_diff = (self.vsmow_rolling_data['d_excess_vsmow_derived'] - 
                               self.vsmow_rolling_data['d_excess_vsmow_direct'])
                f.write("D-EXCESS METHOD COMPARISON:\n")
                f.write("-" * 28 + "\n")
                f.write(f"Mean difference (derived - direct): {dexcess_diff.mean():.4f}‰\n")
                f.write(f"RMS difference: {np.sqrt(np.mean(dexcess_diff**2)):.4f}‰\n")
                correlation = np.corrcoef(self.vsmow_rolling_data['d_excess_vsmow_derived'], 
                                        self.vsmow_rolling_data['d_excess_vsmow_direct'])[0,1]
                f.write(f"Correlation coefficient: {correlation:.6f}\n\n")
            
            f.write("OUTPUT FILES GENERATED:\n")
            f.write("-" * 23 + "\n")
            output_files = list(self.output_dir.glob("*.png")) + list(self.output_dir.glob("*.csv")) + list(self.output_dir.glob("*.json"))
            for file_path in sorted(output_files):
                f.write(f"  {file_path.name}\n")
        
        print(f"📄 Summary report saved: {report_path.name}")

    def _create_atmospheric_time_series_plots(self, corrected_data: pd.DataFrame):
        """Create atmospheric time series plots matching humidity correction style."""
        print("    🎨 Creating VSMOW-corrected atmospheric time series plots...")
        
        # Create 5-minute rolling averages
        rolling_data = self._create_rolling_averages(corrected_data)
        
        # Create comprehensive time series figure - matching humidity correction style
        fig, axes = plt.subplots(4, 1, figsize=(16, 20))
        fig.suptitle('Atmospheric Water Vapor Isotopes: Humidity and VSMOW-SLAP Corrections\n'
                    '5-Minute Rolling Averages', fontsize=16, fontweight='bold', y=0.98)
        
        time_data = rolling_data['Time']
        
        # Plot 1: H2O concentration
        axes[0].plot(time_data, rolling_data['H2O_ppm'], 'k-', linewidth=1.5, alpha=0.8)
        axes[0].set_ylabel('H₂O Concentration (ppm)', fontweight='bold', fontsize=12)
        axes[0].set_title('Water Vapor Concentration', fontweight='bold', fontsize=13)
        axes[0].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[0].tick_params(labelsize=10)
        
        # Plot 2: δD progression through corrections
        axes[1].plot(time_data, rolling_data['D_del'], 'lightgray', linewidth=1.2, alpha=0.7,
                    label='Original δD')
        axes[1].plot(time_data, rolling_data['D_del_corrected'], 'gray', linewidth=1.5, alpha=0.8,
                    label='Humidity Corrected δD')
        axes[1].plot(time_data, rolling_data['D_del_vsmow'], 'black', linewidth=1.8,
                    label='VSMOW-SLAP Corrected δD')
        axes[1].set_ylabel('δD (‰)', fontweight='bold', fontsize=12)
        axes[1].set_title('Deuterium Content: Original → Humidity Corrected → VSMOW-SLAP Scale', 
                         fontweight='bold', fontsize=13)
        axes[1].legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
        axes[1].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[1].tick_params(labelsize=10)
        
        # Plot 3: δ18O progression through corrections
        axes[2].plot(time_data, rolling_data['O18_del'], 'lightgray', linewidth=1.2, alpha=0.7,
                    label='Original δ18O')
        axes[2].plot(time_data, rolling_data['O18_del_corrected'], 'gray', linewidth=1.5, alpha=0.8,
                    label='Humidity Corrected δ18O')
        axes[2].plot(time_data, rolling_data['O18_del_vsmow'], 'black', linewidth=1.8,
                    label='VSMOW-SLAP Corrected δ18O')
        axes[2].set_ylabel('δ18O (‰)', fontweight='bold', fontsize=12)
        axes[2].set_title('Oxygen-18 Content: Original → Humidity Corrected → VSMOW-SLAP Scale', 
                         fontweight='bold', fontsize=13)
        axes[2].legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
        axes[2].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[2].tick_params(labelsize=10)
        
        # Plot 4: d-excess derived comparison
        axes[3].plot(time_data, rolling_data['d_excess_original'], 'lightgray', linewidth=1.2, alpha=0.7,
                    label='Original d-excess')
        axes[3].plot(time_data, rolling_data['d_excess_corrected'], 'gray', linewidth=1.5, alpha=0.8,
                    label='Humidity Corrected d-excess')
        axes[3].plot(time_data, rolling_data['d_excess_vsmow_derived'], 'black', linewidth=1.8,
                    label='VSMOW-SLAP Corrected d-excess')
        axes[3].set_ylabel('d-excess (‰)', fontweight='bold', fontsize=12)
        axes[3].set_xlabel('Time', fontweight='bold', fontsize=12)
        axes[3].set_title('Deuterium Excess: Original → Humidity → VSMOW-SLAP', 
                         fontweight='bold', fontsize=13)
        axes[3].legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
        axes[3].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[3].tick_params(labelsize=10, axis='x', rotation=45)
        
        # Format x-axis for all subplots
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Add padding for suptitle
        
        # Save plot
        plot_path = self.output_dir / "atmospheric_isotopes_vsmow_corrected_time_series.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      📊 Saved VSMOW time series: {plot_path.name}")
        return rolling_data
        
    def _create_rolling_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create 5-minute rolling averages for atmospheric data."""
        print("    📈 Creating 5-minute rolling averages...")
        
        # Set time as index for resampling
        data_indexed = data.set_index('Time')
        
        # Select relevant columns for rolling average - matching existing method
        isotope_columns = [
            'H2O_ppm', 'D_del', 'O18_del', 'D_del_corrected', 'O18_del_corrected',
            'D_del_vsmow', 'O18_del_vsmow', 'd_excess_humidity_corrected',
            'd_excess_vsmow_derived', 'd_excess_vsmow_direct'
        ]
        
        # Also include original d-excess calculations for plotting
        if 'd_excess_original' not in data.columns:
            data['d_excess_original'] = data['D_del'] - 8 * data['O18_del']
        if 'd_excess_corrected' not in data.columns:
            data['d_excess_corrected'] = data['D_del_corrected'] - 8 * data['O18_del_corrected']
        
        data_indexed = data.set_index('Time')
        isotope_columns.extend(['d_excess_original', 'd_excess_corrected'])
        
        # Create rolling averages using time-based window
        # First resample to regular intervals, then apply rolling average
        resampled_data = data_indexed[isotope_columns].resample('1min').mean()
        
        # Apply 5-minute rolling average
        rolling_data = resampled_data.rolling(
            window=5, center=True, min_periods=1
        ).mean()
        
        # Reset index to get Time back as column
        rolling_data = rolling_data.reset_index()
        
        # Remove rows with insufficient data (at edges)
        rolling_data = rolling_data.dropna()
        
        print(f"      Created {len(rolling_data):,} rolling average points from {len(data):,} original points")
        print(f"      Rolling average time range: {rolling_data['Time'].min()} to {rolling_data['Time'].max()}")
        
        return rolling_data
        
    def _create_atmospheric_histogram_plots(self, corrected_data: pd.DataFrame):
        """Create histogram plots for VSMOW-corrected atmospheric data."""
        print("    📊 Creating VSMOW-corrected atmospheric histogram plots...")
        
        # Create comprehensive histogram figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Atmospheric Water Vapor Isotopes: Distribution Analysis\n'
                    'Original → Humidity Corrected → VSMOW-SLAP Corrected', 
                    fontsize=15, fontweight='bold', y=0.95)
        
        # δD histograms
        axes[0,0].hist(corrected_data['D_del'], bins=60, alpha=0.5, color='lightgray', 
                      density=True, edgecolor='gray', linewidth=0.5, label='Original')
        axes[0,0].hist(corrected_data['D_del_corrected'], bins=60, alpha=0.7, color='gray', 
                      density=True, edgecolor='black', linewidth=0.5, label='Humidity Corrected')
        axes[0,0].hist(corrected_data['D_del_vsmow'], bins=60, alpha=0.8, 
                      facecolor='none', edgecolor='black', linewidth=2, density=True,
                      histtype='step', label='VSMOW-SLAP Corrected')
        axes[0,0].set_xlabel('δD (‰)', fontweight='bold')
        axes[0,0].set_ylabel('Probability Density', fontweight='bold')
        axes[0,0].set_title('δD Distribution', fontweight='bold')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # δ18O histograms  
        axes[0,1].hist(corrected_data['O18_del'], bins=60, alpha=0.5, color='lightgray', 
                      density=True, edgecolor='gray', linewidth=0.5, label='Original')
        axes[0,1].hist(corrected_data['O18_del_corrected'], bins=60, alpha=0.7, color='gray', 
                      density=True, edgecolor='black', linewidth=0.5, label='Humidity Corrected')
        axes[0,1].hist(corrected_data['O18_del_vsmow'], bins=60, alpha=0.8, 
                      facecolor='none', edgecolor='black', linewidth=2, density=True,
                      histtype='step', label='VSMOW-SLAP Corrected')
        axes[0,1].set_xlabel('δ18O (‰)', fontweight='bold')
        axes[0,1].set_ylabel('Probability Density', fontweight='bold')
        axes[0,1].set_title('δ18O Distribution', fontweight='bold')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # d-excess histograms
        d_excess_orig = corrected_data['D_del'] - 8 * corrected_data['O18_del']
        d_excess_humid = corrected_data['D_del_corrected'] - 8 * corrected_data['O18_del_corrected']
        
        axes[0,2].hist(d_excess_orig, bins=60, alpha=0.5, color='lightgray', 
                      density=True, edgecolor='gray', linewidth=0.5, label='Original')
        axes[0,2].hist(d_excess_humid, bins=60, alpha=0.7, color='gray', 
                      density=True, edgecolor='black', linewidth=0.5, label='Humidity Corrected')
        axes[0,2].hist(corrected_data['d_excess_vsmow_derived'], bins=60, alpha=0.8, 
                      facecolor='none', edgecolor='black', linewidth=2, density=True,
                      histtype='step', label='VSMOW-SLAP Corrected')
        axes[0,2].set_xlabel('d-excess (‰)', fontweight='bold')
        axes[0,2].set_ylabel('Probability Density', fontweight='bold')
        axes[0,2].set_title('d-excess Distribution', fontweight='bold')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # Statistical summary plots - before and after correction comparisons
        corrections = ['Original', 'Humidity Corrected', 'VSMOW-SLAP Corrected']
        dd_data = [corrected_data['D_del'], corrected_data['D_del_corrected'], corrected_data['D_del_vsmow']]
        d18o_data = [corrected_data['O18_del'], corrected_data['O18_del_corrected'], corrected_data['O18_del_vsmow']]
        dexcess_data = [d_excess_orig, d_excess_humid, corrected_data['d_excess_vsmow_derived']]
        
        # Box plots for each isotope
        axes[1,0].boxplot(dd_data, labels=corrections, patch_artist=True)
        axes[1,0].set_ylabel('δD (‰)', fontweight='bold')
        axes[1,0].set_title('δD Statistical Summary', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].tick_params(axis='x', rotation=45)
        
        axes[1,1].boxplot(d18o_data, labels=corrections, patch_artist=True)
        axes[1,1].set_ylabel('δ18O (‰)', fontweight='bold')
        axes[1,1].set_title('δ18O Statistical Summary', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].tick_params(axis='x', rotation=45)
        
        axes[1,2].boxplot(dexcess_data, labels=corrections, patch_artist=True)
        axes[1,2].set_ylabel('d-excess (‰)', fontweight='bold')
        axes[1,2].set_title('d-excess Statistical Summary', fontweight='bold')
        axes[1,2].grid(True, alpha=0.3)
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])  # Add padding for suptitle
        
        # Save plot
        plot_path = self.output_dir / "atmospheric_isotopes_vsmow_corrected_histograms.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      📊 Saved VSMOW histograms: {plot_path.name}")
    
    def process_atmospheric_data_with_vsmow_plots(self, atmospheric_data_path: Path):
        """Process atmospheric data and create comprehensive VSMOW-corrected plots."""
        print("🌍 Processing atmospheric data with VSMOW correction and creating plots...")
        
        # Load and apply corrections
        atm_data = pd.read_csv(atmospheric_data_path)
        if 'Time' in atm_data.columns:
            atm_data['Time'] = pd.to_datetime(atm_data['Time'])
        
        print(f"  📊 Loaded {len(atm_data):,} atmospheric measurements")
        
        # Apply humidity correction first
        print("  🔧 Applying humidity correction...")
        humidity_corrected = self.humidity_corrector.apply_correction(atm_data)
        
        # Apply VSMOW correction
        print("  🎯 Applying VSMOW-SLAP correction...")
        vsmow_corrected = self.apply_time_varying_vsmow_correction(humidity_corrected)
        
        # Create comprehensive atmospheric plots
        print("  🎨 Creating atmospheric plots...")
        rolling_data = self._create_atmospheric_time_series_plots(vsmow_corrected)
        self._create_atmospheric_histogram_plots(vsmow_corrected)
        
        # Save corrected atmospheric data
        output_file = self.output_dir / "atmospheric_data_vsmow_corrected_5min_rolling.csv"
        rolling_data.to_csv(output_file, index=False)
        
        print(f"  💾 Saved VSMOW-corrected atmospheric data: {output_file.name}")
        print(f"✅ Atmospheric VSMOW processing and plotting complete!")
        
        return vsmow_corrected, rolling_data

def main():
    """Main execution function."""
    project_root = Path(__file__).parent.parent
    
    # Define paths
    standards_data_path = project_root / "data" / "raw" / "separated" / "standards_only_raw.csv"
    humidity_calib_path = project_root / "outputs" / "selective_calibration" / "selective_humidity_calibration.json"  
    known_standards_path = project_root / "data" / "raw" / "known_isotope_standards_runs.csv"
    atmospheric_data_path = project_root / "outputs" / "improved_humidity_analysis" / "atmospheric_isotopes_humidity_corrected_full.csv"
    output_dir = project_root / "outputs" / "time_rolling_vsmow_analysis"
    
    # Define calibration runs (specified by user)
    calibration_runs = [1, 2, 4, 6, 7, 8, 10, 13, 15, 16, 17, 18, 19, 23, 24, 33, 36]
    
    # Verify input files exist
    required_files = [
        (standards_data_path, "Standards data"),
        (humidity_calib_path, "Humidity calibration"),
        (known_standards_path, "Known standards"),
        (atmospheric_data_path, "Atmospheric data (humidity corrected)")
    ]
    
    for file_path, name in required_files:
        if not file_path.exists():
            print(f"❌ {name} file not found: {file_path}")
            return
    
    # Run time-rolling VSMOW analysis
    corrector = TimeRollingVSMOWCorrector(
        standards_data_path=standards_data_path,
        humidity_calibration_path=humidity_calib_path,
        known_standards_path=known_standards_path,
        atmospheric_data_path=atmospheric_data_path,
        output_dir=output_dir,
        calibration_runs=calibration_runs
    )
    
    results = corrector.run_complete_vsmow_analysis()
    
    return corrector, results

if __name__ == "__main__":
    corrector, results = main()