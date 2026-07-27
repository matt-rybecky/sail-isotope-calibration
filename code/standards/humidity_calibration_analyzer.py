#!/usr/bin/env python3
"""
Humidity Calibration Analysis and Visualization

This script extracts key functions from the calibration process to create:
1. Uncalibrated standard run visualizations with humidity correction curve overlays
2. Corrected/flattened data after humidity calibration application
3. Standard vs known value comparisons for VSMOW/SLAP calibration preparation

Created for comprehensive analysis of the humidity calibration pipeline.
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
import warnings
warnings.filterwarnings('ignore')

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Import required modules
from data_processing.humidity_calibration import HumidityBiasCorrector
from data_processing.vsmow_calibration import VSMOWCalibrator

class HumidityCalibrationAnalyzer:
    """
    Comprehensive analyzer for humidity calibration visualization and validation.
    """
    
    def __init__(self, 
                 standards_data_path: Path,
                 humidity_calibration_path: Path,
                 output_dir: Path):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        standards_data_path : Path
            Path to standards data CSV
        humidity_calibration_path : Path  
            Path to humidity calibration JSON file
        output_dir : Path
            Directory to save output plots and data
        """
        self.standards_data_path = Path(standards_data_path)
        self.humidity_calibration_path = Path(humidity_calibration_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print("🔬 Loading standards data...")
        self.standards_data = self._load_standards_data()
        
        print("🔧 Loading humidity calibration...")
        self.humidity_corrector = HumidityBiasCorrector(self.humidity_calibration_path)
        
        # Extract standards runs
        print("📊 Identifying standards runs...")
        self.standards_runs = self._extract_standards_runs()
        
        print(f"✅ Initialized analyzer with {len(self.standards_runs)} standards runs")
        
        # Set up plotting style
        self._setup_plotting_style()
    
    def _load_standards_data(self) -> pd.DataFrame:
        """Load and prepare standards data."""
        df = pd.read_csv(self.standards_data_path)
        
        # Ensure time column is datetime
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
        
        return df
    
    def _extract_standards_runs(self) -> Dict[int, pd.DataFrame]:
        """Extract individual standards runs based on standards_run_id."""
        runs = {}
        
        if 'standards_run_id' in self.standards_data.columns:
            run_ids = self.standards_data['standards_run_id'].unique()
            
            for run_id in run_ids:
                if pd.notna(run_id):  # Skip NaN run IDs
                    run_data = self.standards_data[
                        self.standards_data['standards_run_id'] == run_id
                    ].copy()
                    
                    if len(run_data) > 10:  # Only include runs with sufficient data
                        runs[int(run_id)] = run_data
        
        return runs
    
    def _setup_plotting_style(self):
        """Set up professional black and white plotting style."""
        plt.style.use('classic')
        plt.rcParams.update({
            # Font settings
            'font.size': 11,
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
            
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
    
    def _evaluate_humidity_correction_polynomial(self, 
                                               h2o_values: np.ndarray,
                                               isotope: str) -> np.ndarray:
        """
        Evaluate the humidity correction polynomial for visualization.
        
        Parameters:
        -----------
        h2o_values : np.ndarray
            H2O concentration values
        isotope : str
            Isotope name ('dD' or 'd18O')
            
        Returns:
        --------
        np.ndarray
            Correction polynomial values
        """
        calib_info = self.humidity_corrector.calibration_functions[isotope]
        calib_poly = calib_info['calibration_polynomial']
        
        return self.humidity_corrector._evaluate_polynomial(h2o_values, calib_poly)
    
    def create_uncalibrated_standards_plots(self):
        """
        Create plots showing uncalibrated standards data with humidity correction overlays.
        Plot for every utilized standards run.
        """
        print("📈 Creating uncalibrated standards visualizations...")
        
        # Get calibration info for titles
        with open(self.humidity_calibration_path) as f:
            calib_data = json.load(f)
        
        # Track which runs were used in calibration
        used_dates_dd = set(calib_data['calibrations']['dD']['source_dates'])
        used_dates_d18o = set(calib_data['calibrations']['d18O']['source_dates'])
        
        for run_id, run_data in self.standards_runs.items():
            # Skip runs with insufficient data
            if len(run_data) < 50:
                continue
                
            # Extract date for comparison
            run_date = run_data['Time'].dt.date.iloc[0].strftime('%Y-%m-%d')
            
            # Determine if this run was used in calibration
            used_in_dd = run_date in used_dates_dd
            used_in_d18o = run_date in used_dates_d18o
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Standards Run {run_id} - Uncalibrated Data with Humidity Correction\n{run_date}', 
                        fontsize=14, fontweight='bold', y=0.98)
            
            # Extract data
            time_data = run_data['Time']
            h2o_data = run_data['H2O_ppm']
            dd_data = run_data['D_del']
            d18o_data = run_data['O18_del']
            
            # Plot 1: Time series of H2O concentration
            axes[0].plot(time_data, h2o_data, 'k-', linewidth=1.2, alpha=0.8)
            axes[0].set_xlabel('Time', fontweight='bold')
            axes[0].set_ylabel('H₂O Concentration (ppm)', fontweight='bold')
            axes[0].set_title('H₂O Time Series', fontweight='bold')
            axes[0].tick_params(axis='x', rotation=45, labelsize=9)
            axes[0].tick_params(axis='y', labelsize=10)
            axes[0].grid(True, alpha=0.7, linestyle=':', color='gray')
            
            # Plot 2: δD vs H2O with correction curve
            h2o_range = np.linspace(h2o_data.min() * 0.8, h2o_data.max() * 1.2, 500)
            
            # Plot uncalibrated data
            axes[1].scatter(h2o_data, dd_data, alpha=0.7, s=15, c='black', 
                           marker='o', edgecolors='none', label='Uncalibrated data')
            
            # Plot correction polynomial if this isotope has calibration
            if 'dD' in self.humidity_corrector.calibration_functions:
                try:
                    correction_values = self._evaluate_humidity_correction_polynomial(h2o_range, 'dD')
                    axes[1].plot(h2o_range, correction_values, 'k-', linewidth=2.5,
                               linestyle='--', label='Humidity correction curve')
                    
                    # Mark reference H2O point
                    ref_h2o = self.humidity_corrector.reference_h2o
                    ref_correction = self._evaluate_humidity_correction_polynomial(
                        np.array([ref_h2o]), 'dD')[0]
                    axes[1].axvline(ref_h2o, color='gray', linestyle='-', alpha=0.8, linewidth=1.5,
                                  label=f'Reference H₂O ({ref_h2o:.0f} ppm)')
                    axes[1].plot(ref_h2o, ref_correction, 's', color='black', markersize=8, 
                               markerfacecolor='white', markeredgewidth=2,
                               label='Reference point')
                except:
                    print(f"  Warning: Could not plot δD correction curve for run {run_id}")
            
            axes[1].set_xlabel('H₂O Concentration (ppm)', fontweight='bold')
            axes[1].set_ylabel('δD (‰ vs. working standard)', fontweight='bold')
            title_suffix = " ✓" if used_in_dd else ""
            axes[1].set_title(f'δD vs H₂O{title_suffix}', fontweight='bold')
            axes[1].legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
            axes[1].grid(True, alpha=0.7, linestyle=':', color='gray')
            axes[1].tick_params(labelsize=10)
            
            # Plot 3: δ18O vs H2O with correction curve  
            axes[2].scatter(h2o_data, d18o_data, alpha=0.7, s=15, c='black',
                           marker='^', edgecolors='none', label='Uncalibrated data')
            
            if 'd18O' in self.humidity_corrector.calibration_functions:
                try:
                    correction_values = self._evaluate_humidity_correction_polynomial(h2o_range, 'd18O')
                    axes[2].plot(h2o_range, correction_values, 'k-', linewidth=2.5,
                               linestyle='-.', label='Humidity correction curve')
                    
                    # Mark reference H2O point
                    ref_h2o = self.humidity_corrector.reference_h2o
                    ref_correction = self._evaluate_humidity_correction_polynomial(
                        np.array([ref_h2o]), 'd18O')[0]
                    axes[2].axvline(ref_h2o, color='gray', linestyle='-', alpha=0.8, linewidth=1.5,
                                  label=f'Reference H₂O ({ref_h2o:.0f} ppm)')
                    axes[2].plot(ref_h2o, ref_correction, '^', color='black', markersize=8,
                               markerfacecolor='white', markeredgewidth=2,
                               label='Reference point')
                except:
                    print(f"  Warning: Could not plot δ18O correction curve for run {run_id}")
            
            axes[2].set_xlabel('H₂O Concentration (ppm)', fontweight='bold')
            axes[2].set_ylabel('δ18O (‰ vs. working standard)', fontweight='bold')
            title_suffix = " ✓" if used_in_d18o else ""
            axes[2].set_title(f'δ18O vs H₂O{title_suffix}', fontweight='bold')
            axes[2].legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
            axes[2].grid(True, alpha=0.7, linestyle=':', color='gray')
            axes[2].tick_params(labelsize=10)
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"uncalibrated_standards_run_{run_id}_{run_date}.png"
            plot_path = self.output_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ Saved uncalibrated plot: {plot_filename}")
        
        print("📈 Completed uncalibrated standards visualizations")
    
    def apply_humidity_correction_to_standards(self):
        """Apply humidity correction to all standards runs."""
        print("🔧 Applying humidity correction to standards runs...")
        
        corrected_runs = {}
        
        for run_id, run_data in self.standards_runs.items():
            # Apply humidity correction
            corrected_data = self.humidity_corrector.apply_correction(
                run_data,
                h2o_column='H2O_ppm',
                dd_column='D_del',
                d18o_column='O18_del'
            )
            
            corrected_runs[run_id] = corrected_data
            
        self.corrected_standards_runs = corrected_runs
        print(f"✅ Applied humidity correction to {len(corrected_runs)} standards runs")
    
    def create_corrected_standards_plots(self):
        """
        Create plots showing corrected/flattened standards data.
        Data should be flattened across the H2O range.
        """
        print("📊 Creating corrected standards visualizations...")
        
        for run_id, corrected_data in self.corrected_standards_runs.items():
            # Skip runs with insufficient data
            if len(corrected_data) < 50:
                continue
            
            # Extract date for filename
            run_date = corrected_data['Time'].dt.date.iloc[0].strftime('%Y-%m-%d')
            
            # Create figure with 4 subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Standards Run {run_id} - Humidity Corrected Data\n{run_date}', 
                        fontsize=15, fontweight='bold', y=0.96)
            
            # Extract data
            time_data = corrected_data['Time']
            h2o_data = corrected_data['H2O_ppm']
            
            # Original and corrected isotope data
            dd_original = corrected_data['D_del']
            d18o_original = corrected_data['O18_del']
            
            dd_corrected = corrected_data.get('D_del_corrected', dd_original)
            d18o_corrected = corrected_data.get('O18_del_corrected', d18o_original)
            
            # Plot 1: δD before and after correction
            axes[0,0].scatter(h2o_data, dd_original, alpha=0.5, s=12, c='gray',
                             marker='o', edgecolors='none', label='Original')
            axes[0,0].scatter(h2o_data, dd_corrected, alpha=0.8, s=15, c='black',
                             marker='o', edgecolors='none', label='Humidity corrected')
            
            # Calculate and show mean values
            dd_orig_mean = dd_original.mean()
            dd_corr_mean = dd_corrected.mean()
            dd_orig_std = dd_original.std()
            dd_corr_std = dd_corrected.std()
            
            axes[0,0].axhline(dd_orig_mean, color='gray', linestyle=':', alpha=0.8, linewidth=1.5,
                             label=f'Original mean: {dd_orig_mean:.1f} ± {dd_orig_std:.1f}‰')
            axes[0,0].axhline(dd_corr_mean, color='black', linestyle='--', alpha=0.8, linewidth=2,
                             label=f'Corrected mean: {dd_corr_mean:.1f} ± {dd_corr_std:.1f}‰')
            
            axes[0,0].set_xlabel('H₂O Concentration (ppm)', fontweight='bold')
            axes[0,0].set_ylabel('δD (‰ vs. working standard)', fontweight='bold')
            axes[0,0].set_title('δD: Before vs After Humidity Correction', fontweight='bold')
            axes[0,0].legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
            axes[0,0].grid(True, alpha=0.7, linestyle=':', color='gray')
            axes[0,0].tick_params(labelsize=10)
            
            # Plot 2: δ18O before and after correction
            axes[0,1].scatter(h2o_data, d18o_original, alpha=0.5, s=12, c='gray',
                             marker='^', edgecolors='none', label='Original')
            axes[0,1].scatter(h2o_data, d18o_corrected, alpha=0.8, s=15, c='black',
                             marker='^', edgecolors='none', label='Humidity corrected')
            
            # Calculate and show mean values
            d18o_orig_mean = d18o_original.mean()
            d18o_corr_mean = d18o_corrected.mean()
            d18o_orig_std = d18o_original.std()
            d18o_corr_std = d18o_corrected.std()
            
            axes[0,1].axhline(d18o_orig_mean, color='gray', linestyle=':', alpha=0.8, linewidth=1.5,
                             label=f'Original mean: {d18o_orig_mean:.1f} ± {d18o_orig_std:.1f}‰')
            axes[0,1].axhline(d18o_corr_mean, color='black', linestyle='--', alpha=0.8, linewidth=2,
                             label=f'Corrected mean: {d18o_corr_mean:.1f} ± {d18o_corr_std:.1f}‰')
            
            axes[0,1].set_xlabel('H₂O Concentration (ppm)', fontweight='bold')
            axes[0,1].set_ylabel('δ18O (‰ vs. working standard)', fontweight='bold')
            axes[0,1].set_title('δ18O: Before vs After Humidity Correction', fontweight='bold')
            axes[0,1].legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
            axes[0,1].grid(True, alpha=0.7, linestyle=':', color='gray')
            axes[0,1].tick_params(labelsize=10)
            
            # Plot 3: Time series of corrected δD
            axes[1,0].plot(time_data, dd_corrected, 'k-', alpha=0.8, linewidth=1.2)
            axes[1,0].axhline(dd_corr_mean, color='black', linestyle='--', alpha=0.8, linewidth=2,
                             label=f'Mean: {dd_corr_mean:.1f} ± {dd_corr_std:.1f}‰')
            # Add ±1σ bands
            axes[1,0].axhline(dd_corr_mean + dd_corr_std, color='gray', linestyle=':', alpha=0.6,
                             label='±1σ')
            axes[1,0].axhline(dd_corr_mean - dd_corr_std, color='gray', linestyle=':', alpha=0.6)
            
            axes[1,0].set_xlabel('Time', fontweight='bold')
            axes[1,0].set_ylabel('δD (‰, humidity corrected)', fontweight='bold')
            axes[1,0].set_title('Corrected δD Time Series', fontweight='bold')
            axes[1,0].legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
            axes[1,0].grid(True, alpha=0.7, linestyle=':', color='gray')
            axes[1,0].tick_params(axis='x', rotation=45, labelsize=9)
            axes[1,0].tick_params(axis='y', labelsize=10)
            
            # Plot 4: Time series of corrected δ18O
            axes[1,1].plot(time_data, d18o_corrected, 'k-', alpha=0.8, linewidth=1.2)
            axes[1,1].axhline(d18o_corr_mean, color='black', linestyle='--', alpha=0.8, linewidth=2,
                             label=f'Mean: {d18o_corr_mean:.1f} ± {d18o_corr_std:.1f}‰')
            # Add ±1σ bands
            axes[1,1].axhline(d18o_corr_mean + d18o_corr_std, color='gray', linestyle=':', alpha=0.6,
                             label='±1σ')
            axes[1,1].axhline(d18o_corr_mean - d18o_corr_std, color='gray', linestyle=':', alpha=0.6)
            
            axes[1,1].set_xlabel('Time', fontweight='bold')
            axes[1,1].set_ylabel('δ18O (‰, humidity corrected)', fontweight='bold')
            axes[1,1].set_title('Corrected δ18O Time Series', fontweight='bold')
            axes[1,1].legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
            axes[1,1].grid(True, alpha=0.7, linestyle=':', color='gray')
            axes[1,1].tick_params(axis='x', rotation=45, labelsize=9)
            axes[1,1].tick_params(axis='y', labelsize=10)
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"corrected_standards_run_{run_id}_{run_date}.png"
            plot_path = self.output_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ Saved corrected plot: {plot_filename}")
        
        print("📊 Completed corrected standards visualizations")
    
    def calculate_standard_offsets(self, known_values_file: Optional[Path] = None):
        """
        Calculate offsets between measured corrected values and known standard values.
        This prepares data for VSMOW/SLAP calibration.
        
        Parameters:
        -----------
        known_values_file : Path, optional
            CSV file with known standard values (standard_name, dD_known, d18O_known)
        """
        print("🧮 Calculating standard offsets for VSMOW/SLAP calibration...")
        
        # Default known values for common standards (VSMOW relative)
        known_standards = {
            'USGS46': {'dD_VSMOW': -235.8, 'd18O_VSMOW': -29.80},
            'UNM Greenland': {'dD_VSMOW': -189.5, 'd18O_VSMOW': -24.76},
            'UNM South Pole': {'dD_VSMOW': -428.0, 'd18O_VSMOW': -54.11},
            'W-64444': {'dD_VSMOW': -108.7, 'd18O_VSMOW': -14.20},
            'USGS45': {'dD_VSMOW': -10.3, 'd18O_VSMOW': -2.24},
        }
        
        # Load known values from file if provided
        if known_values_file and known_values_file.exists():
            known_df = pd.read_csv(known_values_file)
            for _, row in known_df.iterrows():
                known_standards[row['standard_name']] = {
                    'dD_VSMOW': row['dD_known'],
                    'd18O_VSMOW': row['d18O_known']
                }
        
        offsets_data = []
        
        for run_id, corrected_data in self.corrected_standards_runs.items():
            # Get run info
            run_date = corrected_data['Time'].dt.date.iloc[0].strftime('%Y-%m-%d')
            
            # Try to identify standard from MIU_DESC or other indicators
            if 'MIU_DESC' in corrected_data.columns:
                miu_desc = corrected_data['MIU_DESC'].iloc[0]
                # Extract standard name (customize based on your naming convention)
                standard_name = str(miu_desc).strip()
            else:
                standard_name = f"Unknown_Run_{run_id}"
            
            # Calculate mean corrected values
            dd_corrected_mean = corrected_data['D_del_corrected'].mean()
            d18o_corrected_mean = corrected_data['O18_del_corrected'].mean()
            dd_corrected_std = corrected_data['D_del_corrected'].std()
            d18o_corrected_std = corrected_data['O18_del_corrected'].std()
            
            # Find matching known values
            known_match = None
            for known_std, known_vals in known_standards.items():
                if known_std.upper() in standard_name.upper() or \
                   any(part in standard_name.upper() for part in known_std.upper().split()):
                    known_match = known_vals
                    matched_name = known_std
                    break
            
            # Calculate offsets if we have a match
            if known_match:
                dd_offset = dd_corrected_mean - known_match['dD_VSMOW']
                d18o_offset = d18o_corrected_mean - known_match['d18O_VSMOW']
                
                offsets_data.append({
                    'run_id': run_id,
                    'run_date': run_date,
                    'standard_name': matched_name,
                    'miu_desc': standard_name,
                    'dD_measured_mean': dd_corrected_mean,
                    'dD_measured_std': dd_corrected_std,
                    'dD_known': known_match['dD_VSMOW'],
                    'dD_offset': dd_offset,
                    'd18O_measured_mean': d18o_corrected_mean,
                    'd18O_measured_std': d18o_corrected_std,
                    'd18O_known': known_match['d18O_VSMOW'],
                    'd18O_offset': d18o_offset,
                    'n_points': len(corrected_data)
                })
            else:
                print(f"  ⚠️  No known values found for {standard_name} (Run {run_id})")
        
        # Save offsets data
        if offsets_data:
            offsets_df = pd.DataFrame(offsets_data)
            offsets_file = self.output_dir / "standard_offsets_for_vsmow_calibration.csv"
            offsets_df.to_csv(offsets_file, index=False)
            
            # Create offsets visualization
            self._plot_offsets_analysis(offsets_df)
            
            print(f"✅ Calculated offsets for {len(offsets_data)} standards runs")
            print(f"📁 Saved offsets data: {offsets_file.name}")
            
            return offsets_df
        else:
            print("❌ No matching standards found for offset calculation")
            return None
    
    def _plot_offsets_analysis(self, offsets_df: pd.DataFrame):
        """Create comprehensive offsets analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Standard Offsets Analysis for VSMOW/SLAP Calibration', 
                    fontsize=15, fontweight='bold', y=0.96)
        
        # Plot 1: δD measured vs known
        axes[0,0].scatter(offsets_df['dD_known'], offsets_df['dD_measured_mean'], 
                         alpha=0.8, s=80, c='black', marker='o', edgecolors='none')
        axes[0,0].errorbar(offsets_df['dD_known'], offsets_df['dD_measured_mean'],
                          yerr=offsets_df['dD_measured_std'], fmt='none', alpha=0.6, 
                          color='black', capsize=3, capthick=1.5)
        
        # Plot 1:1 line
        min_val = min(offsets_df['dD_known'].min(), offsets_df['dD_measured_mean'].min())
        max_val = max(offsets_df['dD_known'].max(), offsets_df['dD_measured_mean'].max())
        axes[0,0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2,
                      label='1:1 Reference Line')
        
        axes[0,0].set_xlabel('δD Known (‰ VSMOW)', fontweight='bold')
        axes[0,0].set_ylabel('δD Measured (‰, humidity corrected)', fontweight='bold')
        axes[0,0].set_title('δD: Measured vs Known Values', fontweight='bold')
        axes[0,0].legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
        axes[0,0].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[0,0].tick_params(labelsize=10)
        
        # Plot 2: δ18O measured vs known
        axes[0,1].scatter(offsets_df['d18O_known'], offsets_df['d18O_measured_mean'], 
                         alpha=0.8, s=80, c='black', marker='^', edgecolors='none')
        axes[0,1].errorbar(offsets_df['d18O_known'], offsets_df['d18O_measured_mean'],
                          yerr=offsets_df['d18O_measured_std'], fmt='none', alpha=0.6,
                          color='black', capsize=3, capthick=1.5)
        
        # Plot 1:1 line
        min_val = min(offsets_df['d18O_known'].min(), offsets_df['d18O_measured_mean'].min())
        max_val = max(offsets_df['d18O_known'].max(), offsets_df['d18O_measured_mean'].max())
        axes[0,1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2,
                      label='1:1 Reference Line')
        
        axes[0,1].set_xlabel('δ18O Known (‰ VSMOW)', fontweight='bold')
        axes[0,1].set_ylabel('δ18O Measured (‰, humidity corrected)', fontweight='bold')
        axes[0,1].set_title('δ18O: Measured vs Known Values', fontweight='bold')
        axes[0,1].legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
        axes[0,1].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[0,1].tick_params(labelsize=10)
        
        # Plot 3: δD offsets by standard
        unique_standards = offsets_df['standard_name'].unique()
        # Use grayscale colors for different standards
        grays = np.linspace(0.2, 0.8, len(unique_standards))
        markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', 'h', '*']
        
        for i, standard in enumerate(unique_standards):
            std_data = offsets_df[offsets_df['standard_name'] == standard]
            marker = markers[i % len(markers)]
            color = str(grays[i])
            
            axes[1,0].scatter([i] * len(std_data), std_data['dD_offset'], 
                             c=color, s=80, alpha=0.8, marker=marker, 
                             edgecolors='black', linewidth=0.5, label=standard)
            axes[1,0].errorbar([i] * len(std_data), std_data['dD_offset'],
                              yerr=std_data['dD_measured_std'], fmt='none', alpha=0.6,
                              color='black', capsize=3, capthick=1.5)
        
        axes[1,0].axhline(0, color='black', linestyle='--', alpha=0.8, linewidth=2,
                         label='Zero Offset Reference')
        axes[1,0].set_xticks(range(len(unique_standards)))
        axes[1,0].set_xticklabels(unique_standards, rotation=45, ha='right')
        axes[1,0].set_ylabel('δD Offset (‰)', fontweight='bold')
        axes[1,0].set_title('δD Offsets by Standard', fontweight='bold')
        axes[1,0].legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
        axes[1,0].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[1,0].tick_params(labelsize=10)
        
        # Plot 4: δ18O offsets by standard
        for i, standard in enumerate(unique_standards):
            std_data = offsets_df[offsets_df['standard_name'] == standard]
            marker = markers[i % len(markers)]
            color = str(grays[i])
            
            axes[1,1].scatter([i] * len(std_data), std_data['d18O_offset'], 
                             c=color, s=80, alpha=0.8, marker=marker,
                             edgecolors='black', linewidth=0.5, label=standard)
            axes[1,1].errorbar([i] * len(std_data), std_data['d18O_offset'],
                              yerr=std_data['d18O_measured_std'], fmt='none', alpha=0.6,
                              color='black', capsize=3, capthick=1.5)
        
        axes[1,1].axhline(0, color='black', linestyle='--', alpha=0.8, linewidth=2,
                         label='Zero Offset Reference')
        axes[1,1].set_xticks(range(len(unique_standards)))
        axes[1,1].set_xticklabels(unique_standards, rotation=45, ha='right')
        axes[1,1].set_ylabel('δ18O Offset (‰)', fontweight='bold')
        axes[1,1].set_title('δ18O Offsets by Standard', fontweight='bold')
        axes[1,1].legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
        axes[1,1].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[1,1].tick_params(labelsize=10)
        
        plt.tight_layout()
        
        # Save plot
        offsets_plot_path = self.output_dir / "standard_offsets_analysis.png"
        plt.savefig(offsets_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Saved offsets analysis: {offsets_plot_path.name}")
    
    def run_complete_analysis(self, known_values_file: Optional[Path] = None):
        """Run the complete humidity calibration analysis."""
        print("🚀 Starting complete humidity calibration analysis...\n")
        
        # Step 1: Create uncalibrated visualizations
        self.create_uncalibrated_standards_plots()
        print()
        
        # Step 2: Apply humidity correction
        self.apply_humidity_correction_to_standards()
        print()
        
        # Step 3: Create corrected visualizations
        self.create_corrected_standards_plots()
        print()
        
        # Step 4: Calculate offsets for VSMOW/SLAP calibration
        offsets_df = self.calculate_standard_offsets(known_values_file)
        print()
        
        # Create summary report
        self._create_summary_report(offsets_df)
        
        print("🎉 Complete humidity calibration analysis finished!")
        print(f"📁 All outputs saved to: {self.output_dir}")
        
        return offsets_df
    
    def _create_summary_report(self, offsets_df: Optional[pd.DataFrame]):
        """Create a summary report of the analysis."""
        report_path = self.output_dir / "humidity_calibration_analysis_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("HUMIDITY CALIBRATION ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Standards Data: {self.standards_data_path.name}\n")
            f.write(f"Calibration File: {self.humidity_calibration_path.name}\n\n")
            
            f.write(f"Total Standards Runs Analyzed: {len(self.standards_runs)}\n")
            f.write(f"Reference H2O: {self.humidity_corrector.reference_h2o} ppm\n\n")
            
            # Calibration info
            with open(self.humidity_calibration_path) as calib_file:
                calib_data = json.load(calib_file)
            
            f.write("HUMIDITY CALIBRATION FUNCTIONS:\n")
            f.write("-" * 30 + "\n")
            for isotope, calib_info in calib_data['calibrations'].items():
                f.write(f"{isotope}:\n")
                f.write(f"  Coefficients: {calib_info['coefficients']}\n")
                f.write(f"  Source dates: {', '.join(calib_info['source_dates'])}\n")
                f.write(f"  Source standards: {', '.join(calib_info['source_standards'])}\n\n")
            
            # Offsets summary if available
            if offsets_df is not None and len(offsets_df) > 0:
                f.write("STANDARD OFFSETS SUMMARY:\n")
                f.write("-" * 25 + "\n")
                f.write(f"Standards with known values: {len(offsets_df)}\n\n")
                
                for _, row in offsets_df.iterrows():
                    f.write(f"{row['standard_name']} (Run {row['run_id']}, {row['run_date']}):\n")
                    f.write(f"  δD: {row['dD_measured_mean']:.1f}‰ (known: {row['dD_known']:.1f}‰, offset: {row['dD_offset']:+.1f}‰)\n")
                    f.write(f"  δ18O: {row['d18O_measured_mean']:.1f}‰ (known: {row['d18O_known']:.1f}‰, offset: {row['d18O_offset']:+.1f}‰)\n")
                    f.write(f"  N points: {row['n_points']}\n\n")
                
                # Overall statistics
                f.write("OFFSET STATISTICS:\n")
                f.write("-" * 18 + "\n")
                f.write(f"δD mean offset: {offsets_df['dD_offset'].mean():+.2f} ± {offsets_df['dD_offset'].std():.2f}‰\n")
                f.write(f"δ18O mean offset: {offsets_df['d18O_offset'].mean():+.2f} ± {offsets_df['d18O_offset'].std():.2f}‰\n\n")
            
            f.write("FILES GENERATED:\n")
            f.write("-" * 15 + "\n")
            output_files = list(self.output_dir.glob("*.png")) + list(self.output_dir.glob("*.csv"))
            for file_path in sorted(output_files):
                f.write(f"  {file_path.name}\n")
        
        print(f"📄 Summary report saved: {report_path.name}")

def main():
    """Main execution function."""
    # Define paths
    project_root = Path(__file__).parent.parent
    
    standards_data_path = project_root / "data" / "raw" / "separated" / "standards_only_raw.csv"
    humidity_calib_path = project_root / "outputs" / "selective_calibration" / "selective_humidity_calibration.json"
    output_dir = project_root / "outputs" / "humidity_calibration_analysis"
    
    # Optional: known values file (create if you have specific known values)
    known_values_file = None  # Set to path if you have a CSV with known standard values
    
    # Verify input files exist
    if not standards_data_path.exists():
        print(f"❌ Standards data file not found: {standards_data_path}")
        return
    
    if not humidity_calib_path.exists():
        print(f"❌ Humidity calibration file not found: {humidity_calib_path}")
        return
    
    # Run analysis
    analyzer = HumidityCalibrationAnalyzer(
        standards_data_path=standards_data_path,
        humidity_calibration_path=humidity_calib_path,
        output_dir=output_dir
    )
    
    offsets_df = analyzer.run_complete_analysis(known_values_file)
    
    return analyzer, offsets_df

if __name__ == "__main__":
    analyzer, offsets_df = main()