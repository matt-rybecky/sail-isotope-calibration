#!/usr/bin/env python3
"""
Improved Humidity Calibration Analysis and Visualization

This script fixes the issues with standard identification and provides detailed
analysis of the humidity correction function application.

Key improvements:
1. Date-based standard identification using known_isotope_standards_runs.csv
2. Investigation of correction function application
3. Comparison with existing calibration results
4. Professional black and white plots with proper fonts
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

class ImprovedHumidityCalibrationAnalyzer:
    """
    Improved analyzer for humidity calibration with proper standard identification.
    """
    
    def __init__(self, 
                 standards_data_path: Path,
                 humidity_calibration_path: Path,
                 known_standards_path: Path,
                 output_dir: Path):
        """
        Initialize the improved analyzer.
        """
        self.standards_data_path = Path(standards_data_path)
        self.humidity_calibration_path = Path(humidity_calibration_path)
        self.known_standards_path = Path(known_standards_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        self._setup_plotting_style()
        
        # Load data
        print("🔬 Loading standards data...")
        self.standards_data = self._load_standards_data()
        
        print("📋 Loading known standards reference...")
        self.known_standards = self._load_known_standards()
        
        print("🔧 Loading humidity calibration...")
        self.humidity_corrector = HumidityBiasCorrector(self.humidity_calibration_path)
        
        # Extract and match standards runs
        print("🔍 Identifying and matching standards runs...")
        self.standards_runs = self._extract_and_match_standards_runs()
        
        print(f"✅ Successfully matched {len(self.matched_standards)} out of {len(self.standards_runs)} standards runs")
    
    def _setup_plotting_style(self):
        """Set up professional black and white plotting style."""
        plt.style.use('classic')
        plt.rcParams.update({
            # Font settings - use system available fonts
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
    
    def _load_standards_data(self) -> pd.DataFrame:
        """Load and prepare standards data."""
        df = pd.read_csv(self.standards_data_path)
        
        # Ensure time column is datetime
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df['date'] = df['Time'].dt.date
        
        return df
    
    def _load_known_standards(self) -> pd.DataFrame:
        """Load known standards reference values."""
        df = pd.read_csv(self.known_standards_path)
        
        # Ensure date column is datetime
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df['date'] = df['Time'].dt.date
        
        return df
    
    def _extract_and_match_standards_runs(self) -> Dict[int, pd.DataFrame]:
        """Extract standards runs and match them with known standards."""
        runs = {}
        self.matched_standards = {}
        
        if 'standards_run_id' in self.standards_data.columns:
            run_ids = self.standards_data['standards_run_id'].unique()
            
            for run_id in run_ids:
                if pd.notna(run_id):  # Skip NaN run IDs
                    run_data = self.standards_data[
                        self.standards_data['standards_run_id'] == run_id
                    ].copy()
                    
                    if len(run_data) > 10:  # Only include runs with sufficient data
                        runs[int(run_id)] = run_data
                        
                        # Try to match with known standards by date
                        run_date = run_data['Time'].dt.date.iloc[0]
                        
                        # Find matching known standard
                        known_match = self.known_standards[
                            self.known_standards['date'] == run_date
                        ]
                        
                        if len(known_match) > 0:
                            standard_info = known_match.iloc[0]
                            self.matched_standards[int(run_id)] = {
                                'date': run_date,
                                'name': standard_info['Name'],
                                'dD_known': standard_info['dD_known'],
                                'd18O_known': standard_info['d18O_known'],
                                'n_points': len(run_data)
                            }
                            print(f"  ✅ Matched Run {run_id} ({run_date}) -> {standard_info['Name']}")
                        else:
                            print(f"  ❓ Run {run_id} ({run_date}) - no matching known standard")
        
        return runs
    
    def _evaluate_humidity_correction_polynomial(self, 
                                               h2o_values: np.ndarray,
                                               isotope: str) -> np.ndarray:
        """Evaluate the humidity correction polynomial."""
        if isotope not in self.humidity_corrector.calibration_functions:
            return np.zeros_like(h2o_values)
            
        calib_info = self.humidity_corrector.calibration_functions[isotope]
        calib_poly = calib_info['calibration_polynomial']
        
        return self.humidity_corrector._evaluate_polynomial(h2o_values, calib_poly)
    
    def _investigate_correction_function(self):
        """Investigate the correction function in detail."""
        print("🔬 Investigating humidity correction function...")
        
        # Load calibration details
        with open(self.humidity_calibration_path, 'r') as f:
            calib_data = json.load(f)
        
        print(f"\n📊 Calibration Details:")
        print(f"   Reference H2O: {self.humidity_corrector.reference_h2o} ppm")
        print(f"   Method: {calib_data['metadata']['method']}")
        
        for isotope in ['dD', 'd18O']:
            if isotope in calib_data['calibrations']:
                calib_info = calib_data['calibrations'][isotope]
                print(f"\n   {isotope} Calibration:")
                print(f"     Coefficients: {calib_info['coefficients']}")
                print(f"     Source dates: {calib_info['source_dates']}")
                print(f"     Source standards: {calib_info['source_standards']}")
                
                # Check if these dates match our known standards
                for date_str in calib_info['source_dates']:
                    date_obj = pd.to_datetime(date_str).date()
                    known_match = self.known_standards[
                        self.known_standards['date'] == date_obj
                    ]
                    if len(known_match) > 0:
                        standard_name = known_match.iloc[0]['Name']
                        print(f"       {date_str} -> {standard_name} ✓")
                    else:
                        print(f"       {date_str} -> No match ❓")
    
    def create_detailed_correction_analysis(self):
        """Create detailed analysis of correction application."""
        print("📈 Creating detailed correction function analysis...")
        
        # Create a comprehensive analysis figure
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Humidity Correction Function Analysis', fontsize=16, fontweight='bold', y=0.95)
        
        # H2O range for function plotting
        h2o_range = np.linspace(500, 8000, 1000)
        
        # Plot 1: dD correction function
        if 'dD' in self.humidity_corrector.calibration_functions:
            dd_correction = self._evaluate_humidity_correction_polynomial(h2o_range, 'dD')
            dd_ref_correction = self._evaluate_humidity_correction_polynomial(
                np.array([self.humidity_corrector.reference_h2o]), 'dD')[0]
            
            axes[0,0].plot(h2o_range, dd_correction, 'k-', linewidth=2.5, label='Correction polynomial')
            axes[0,0].axhline(dd_ref_correction, color='gray', linestyle='--', alpha=0.8,
                             label=f'Reference value ({self.humidity_corrector.reference_h2o} ppm)')
            axes[0,0].axvline(self.humidity_corrector.reference_h2o, color='gray', linestyle=':', alpha=0.8)
            
            # Show correction = polynomial - reference
            correction_applied = dd_correction - dd_ref_correction
            axes[0,1].plot(h2o_range, correction_applied, 'k-', linewidth=2.5)
            axes[0,1].axhline(0, color='gray', linestyle='--', alpha=0.8, label='Zero correction')
            axes[0,1].axvline(self.humidity_corrector.reference_h2o, color='gray', linestyle=':', alpha=0.8,
                             label='Reference H₂O')
        
        # Plot similar for d18O
        if 'd18O' in self.humidity_corrector.calibration_functions:
            d18o_correction = self._evaluate_humidity_correction_polynomial(h2o_range, 'd18O')
            d18o_ref_correction = self._evaluate_humidity_correction_polynomial(
                np.array([self.humidity_corrector.reference_h2o]), 'd18O')[0]
            
            axes[1,0].plot(h2o_range, d18o_correction, 'k-', linewidth=2.5, label='Correction polynomial')
            axes[1,0].axhline(d18o_ref_correction, color='gray', linestyle='--', alpha=0.8,
                             label=f'Reference value ({self.humidity_corrector.reference_h2o} ppm)')
            axes[1,0].axvline(self.humidity_corrector.reference_h2o, color='gray', linestyle=':', alpha=0.8)
            
            correction_applied = d18o_correction - d18o_ref_correction
            axes[1,1].plot(h2o_range, correction_applied, 'k-', linewidth=2.5)
            axes[1,1].axhline(0, color='gray', linestyle='--', alpha=0.8, label='Zero correction')
            axes[1,1].axvline(self.humidity_corrector.reference_h2o, color='gray', linestyle=':', alpha=0.8,
                             label='Reference H₂O')
        
        # Plot examples from actual data
        example_runs = list(self.matched_standards.keys())[:2]  # Take first 2 matched runs
        
        for i, run_id in enumerate(example_runs):
            if i >= 2:
                break
                
            run_data = self.standards_runs[run_id]
            standard_info = self.matched_standards[run_id]
            
            h2o_data = run_data['H2O_ppm']
            dd_data = run_data['D_del']
            d18o_data = run_data['O18_del']
            
            # Apply correction manually to show the process
            corrected_data = self.humidity_corrector.apply_correction(run_data)
            dd_corrected = corrected_data['D_del_corrected']
            d18o_corrected = corrected_data['O18_del_corrected']
            
            # Plot in the third column
            ax = axes[i, 2]
            ax.scatter(h2o_data, dd_data, alpha=0.6, s=12, color='gray', 
                      label=f'Original δD ({standard_info["name"]})')
            ax.scatter(h2o_data, dd_corrected, alpha=0.8, s=15, color='black',
                      label=f'Corrected δD')
            
            # Show known value
            ax.axhline(standard_info['dD_known'], color='black', linestyle='--', linewidth=2,
                      label=f'Known value: {standard_info["dD_known"]:.1f}‰')
            
            ax.set_xlabel('H₂O Concentration (ppm)', fontweight='bold')
            ax.set_ylabel('δD (‰)', fontweight='bold')
            ax.set_title(f'Example: {standard_info["name"]} ({standard_info["date"]})', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.7, linestyle=':', color='gray')
        
        # Label axes
        for ax, title in zip(axes.flat[:4], 
                           ['δD Correction Polynomial', 'δD Applied Correction (poly - ref)',
                            'δ18O Correction Polynomial', 'δ18O Applied Correction (poly - ref)']):
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('H₂O Concentration (ppm)', fontweight='bold')
            ax.set_ylabel('Value (‰)', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.7, linestyle=':', color='gray')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "detailed_correction_function_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Saved detailed correction analysis: {plot_path.name}")
    
    def create_matched_standards_analysis(self):
        """Create comprehensive analysis of all isotopes for properly matched standards."""
        print("🎯 Creating comprehensive matched standards analysis...")
        
        if not self.matched_standards:
            print("❌ No matched standards found for analysis")
            return
        
        # Create comprehensive plots for each matched standard showing all isotopes
        for run_id, standard_info in self.matched_standards.items():
            self._create_individual_standard_analysis(run_id, standard_info)
        
        # Create summary comparison plot
        self._create_all_standards_summary()
    
    def _create_individual_standard_analysis(self, run_id: int, standard_info: Dict):
        """Create detailed analysis for individual standard showing all isotopes."""
        run_data = self.standards_runs[run_id]
        corrected_data = self.humidity_corrector.apply_correction(run_data)
        
        # Create figure with subplots for each isotope
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{standard_info["name"]} (Run {run_id}, {standard_info["date"]})\n'
                    f'Humidity Correction Analysis - All Isotopes', 
                    fontsize=15, fontweight='bold', y=0.95)
        
        h2o_data = run_data['H2O_ppm']
        
        # δD analysis
        dd_original = run_data['D_del']
        dd_corrected = corrected_data['D_del_corrected']
        known_dd = standard_info['dD_known']
        
        self._plot_isotope_correction(axes[0,0], h2o_data, dd_original, dd_corrected, 
                                     known_dd, 'δD', '‰')
        
        # δ18O analysis  
        d18o_original = run_data['O18_del']
        d18o_corrected = corrected_data['O18_del_corrected']
        known_d18o = standard_info['d18O_known']
        
        self._plot_isotope_correction(axes[0,1], h2o_data, d18o_original, d18o_corrected,
                                     known_d18o, 'δ18O', '‰')
        
        # d-excess analysis (calculate from δD and δ18O)
        dxs_original = dd_original - 8 * d18o_original
        dxs_corrected = dd_corrected - 8 * d18o_corrected
        known_dxs = known_dd - 8 * known_d18o
        
        self._plot_isotope_correction(axes[0,2], h2o_data, dxs_original, dxs_corrected,
                                     known_dxs, 'd-excess', '‰')
        
        # Time series plots for each isotope
        time_data = run_data['Time']
        
        # δD time series
        axes[1,0].plot(time_data, dd_original, 'o-', color='gray', alpha=0.7, markersize=3,
                      linewidth=1, label='Original')
        axes[1,0].plot(time_data, dd_corrected, 'o-', color='black', alpha=0.8, markersize=3,
                      linewidth=1.5, label='Corrected')
        axes[1,0].axhline(known_dd, color='black', linestyle='--', linewidth=2, alpha=0.8,
                         label=f'Known: {known_dd:.1f}‰')
        axes[1,0].set_ylabel('δD (‰)', fontweight='bold')
        axes[1,0].set_title('δD Time Series', fontweight='bold')
        axes[1,0].legend(fontsize=9)
        axes[1,0].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[1,0].tick_params(axis='x', rotation=45, labelsize=8)
        
        # δ18O time series
        axes[1,1].plot(time_data, d18o_original, '^-', color='gray', alpha=0.7, markersize=3,
                      linewidth=1, label='Original')
        axes[1,1].plot(time_data, d18o_corrected, '^-', color='black', alpha=0.8, markersize=3,
                      linewidth=1.5, label='Corrected')
        axes[1,1].axhline(known_d18o, color='black', linestyle='--', linewidth=2, alpha=0.8,
                         label=f'Known: {known_d18o:.1f}‰')
        axes[1,1].set_ylabel('δ18O (‰)', fontweight='bold')
        axes[1,1].set_title('δ18O Time Series', fontweight='bold')
        axes[1,1].legend(fontsize=9)
        axes[1,1].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[1,1].tick_params(axis='x', rotation=45, labelsize=8)
        
        # d-excess time series
        axes[1,2].plot(time_data, dxs_original, 's-', color='gray', alpha=0.7, markersize=3,
                      linewidth=1, label='Original')
        axes[1,2].plot(time_data, dxs_corrected, 's-', color='black', alpha=0.8, markersize=3,
                      linewidth=1.5, label='Corrected')
        axes[1,2].axhline(known_dxs, color='black', linestyle='--', linewidth=2, alpha=0.8,
                         label=f'Known: {known_dxs:.1f}‰')
        axes[1,2].set_ylabel('d-excess (‰)', fontweight='bold')
        axes[1,2].set_title('d-excess Time Series', fontweight='bold')
        axes[1,2].legend(fontsize=9)
        axes[1,2].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[1,2].tick_params(axis='x', rotation=45, labelsize=8)
        
        plt.tight_layout()
        
        # Save individual standard plot
        plot_path = self.output_dir / f"standard_{standard_info['name'].replace(' ', '_')}_run_{run_id}_complete_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Saved {standard_info['name']} complete analysis: {plot_path.name}")
    
    def _plot_isotope_correction(self, ax, h2o_data, original, corrected, known_value, 
                                isotope_name, units):
        """Helper function to plot isotope correction analysis."""
        # Scatter plot
        ax.scatter(h2o_data, original, alpha=0.5, s=12, color='gray',
                  marker='o', label='Original')
        ax.scatter(h2o_data, corrected, alpha=0.8, s=15, color='black',
                  marker='o', label='Corrected')
        
        # Statistics
        corr_mean = corrected.mean()
        corr_std = corrected.std()
        offset = corr_mean - known_value
        
        # Reference lines
        ax.axhline(known_value, color='black', linestyle='--', linewidth=2, alpha=0.8,
                  label=f'Known: {known_value:.1f}{units}')
        ax.axhline(corr_mean, color='gray', linestyle=':', linewidth=1.5, alpha=0.8,
                  label=f'Corrected mean: {corr_mean:.1f}{units}')
        
        # Formatting
        ax.set_xlabel('H₂O Concentration (ppm)', fontweight='bold')
        ax.set_ylabel(f'{isotope_name} ({units})', fontweight='bold')
        ax.set_title(f'{isotope_name} Correction\nOffset: {offset:+.1f}{units}', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.7, linestyle=':', color='gray')
        ax.tick_params(labelsize=9)
    
    def _create_all_standards_summary(self):
        """Create summary plot of all matched standards."""
        if len(self.matched_standards) < 2:
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('All Matched Standards: Humidity Correction Summary', 
                    fontsize=15, fontweight='bold', y=0.95)
        
        isotope_info = [
            ('δD', 'dD_known', '‰'),
            ('δ18O', 'd18O_known', '‰'),
            ('d-excess', None, '‰')  # Will calculate from δD and δ18O
        ]
        
        for i, (isotope_name, known_key, units) in enumerate(isotope_info):
            ax = axes[i]
            
            for run_id, standard_info in self.matched_standards.items():
                run_data = self.standards_runs[run_id]
                corrected_data = self.humidity_corrector.apply_correction(run_data)
                
                h2o_data = run_data['H2O_ppm']
                
                if isotope_name == 'δD':
                    original = run_data['D_del']
                    corrected = corrected_data['D_del_corrected']
                    known_value = standard_info[known_key]
                elif isotope_name == 'δ18O':
                    original = run_data['O18_del']
                    corrected = corrected_data['O18_del_corrected']
                    known_value = standard_info[known_key]
                else:  # d-excess
                    dd_corrected = corrected_data['D_del_corrected']
                    d18o_corrected = corrected_data['O18_del_corrected']
                    original = run_data['D_del'] - 8 * run_data['O18_del']
                    corrected = dd_corrected - 8 * d18o_corrected
                    known_value = standard_info['dD_known'] - 8 * standard_info['d18O_known']
                
                # Plot with different markers for different standards
                label = f"{standard_info['name']} (R{run_id})"
                alpha = 0.6
                size = 20
                
                ax.scatter(h2o_data, corrected, alpha=alpha, s=size, 
                          label=label, edgecolors='black', linewidth=0.5)
                ax.axhline(known_value, linestyle='--', alpha=0.7, linewidth=1,
                          label=f'{standard_info["name"]} known: {known_value:.1f}{units}')
            
            ax.set_xlabel('H₂O Concentration (ppm)', fontweight='bold')
            ax.set_ylabel(f'{isotope_name} ({units})', fontweight='bold')
            ax.set_title(f'{isotope_name} - All Standards (Corrected)', fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.7, linestyle=':', color='gray')
        
        plt.tight_layout()
        
        # Save summary plot
        plot_path = self.output_dir / "all_standards_summary_corrected.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Saved all standards summary: {plot_path.name}")
    
    def create_offsets_summary(self):
        """Create summary of offsets for properly matched standards."""
        print("📋 Creating offsets summary...")
        
        if not self.matched_standards:
            print("❌ No matched standards for offset calculation")
            return None
        
        # Calculate offsets for matched standards
        offsets_data = []
        
        for run_id, standard_info in self.matched_standards.items():
            run_data = self.standards_runs[run_id]
            corrected_data = self.humidity_corrector.apply_correction(run_data)
            
            # Calculate means and standard deviations
            dd_corr_mean = corrected_data['D_del_corrected'].mean()
            d18o_corr_mean = corrected_data['O18_del_corrected'].mean()
            dd_corr_std = corrected_data['D_del_corrected'].std()
            d18o_corr_std = corrected_data['O18_del_corrected'].std()
            
            # Calculate offsets
            dd_offset = dd_corr_mean - standard_info['dD_known']
            d18o_offset = d18o_corr_mean - standard_info['d18O_known']
            
            offsets_data.append({
                'run_id': run_id,
                'date': standard_info['date'],
                'standard_name': standard_info['name'],
                'dD_measured_mean': dd_corr_mean,
                'dD_measured_std': dd_corr_std,
                'dD_known': standard_info['dD_known'],
                'dD_offset': dd_offset,
                'd18O_measured_mean': d18o_corr_mean,
                'd18O_measured_std': d18o_corr_std,
                'd18O_known': standard_info['d18O_known'],
                'd18O_offset': d18o_offset,
                'n_points': standard_info['n_points']
            })
        
        offsets_df = pd.DataFrame(offsets_data)
        
        # Save offsets data
        offsets_file = self.output_dir / "corrected_standard_offsets.csv"
        offsets_df.to_csv(offsets_file, index=False)
        
        # Create offsets visualization
        self._plot_corrected_offsets(offsets_df)
        
        print(f"✅ Calculated offsets for {len(offsets_df)} matched standards")
        print(f"📁 Saved offsets: {offsets_file.name}")
        
        return offsets_df
    
    def process_atmospheric_data(self, atmospheric_data_path: Path):
        """
        Process atmospheric data with humidity correction and create publication plots.
        """
        print("🌍 Processing atmospheric data with humidity correction...")
        
        # Load atmospheric data
        print(f"  📂 Loading atmospheric data: {atmospheric_data_path.name}")
        atm_data = pd.read_csv(atmospheric_data_path)
        
        # Ensure time column is datetime
        if 'Time' in atm_data.columns:
            atm_data['Time'] = pd.to_datetime(atm_data['Time'])
        
        print(f"  📊 Loaded {len(atm_data):,} atmospheric measurements")
        print(f"  📅 Time range: {atm_data['Time'].min()} to {atm_data['Time'].max()}")
        
        # Apply humidity correction
        print("  🔧 Applying humidity correction...")
        corrected_atm_data = self.humidity_corrector.apply_correction(atm_data)
        
        # Calculate d-excess for original and corrected data
        corrected_atm_data['d_excess_original'] = (
            corrected_atm_data['D_del'] - 8 * corrected_atm_data['O18_del']
        )
        corrected_atm_data['d_excess_corrected'] = (
            corrected_atm_data['D_del_corrected'] - 8 * corrected_atm_data['O18_del_corrected']
        )
        
        # Create 5-minute rolling averages
        print("  📈 Creating 5-minute rolling averages...")
        rolling_data = self._create_rolling_averages(corrected_atm_data)
        
        # Create publication-style plots
        print("  🎨 Creating publication-style plots...")
        self._create_atmospheric_time_series_plots(rolling_data)
        self._create_atmospheric_histograms(corrected_atm_data)
        
        # Save processed datasets
        print("  💾 Saving processed datasets...")
        self._save_processed_atmospheric_data(corrected_atm_data, rolling_data)
        
        print("✅ Atmospheric data processing complete!")
        
        return corrected_atm_data, rolling_data
    
    def _create_rolling_averages(self, data: pd.DataFrame, window: str = '5min') -> pd.DataFrame:
        """Create rolling averages for time series data."""
        # Set time as index for resampling
        data_indexed = data.set_index('Time')
        
        # Define columns to average
        isotope_columns = [
            'H2O_ppm', 'D_del', 'O18_del', 'D_del_corrected', 'O18_del_corrected',
            'd_excess_original', 'd_excess_corrected'
        ]
        
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
        
        print(f"    Created {len(rolling_data):,} rolling average points from {len(data):,} original points")
        print(f"    Rolling average time range: {rolling_data['Time'].min()} to {rolling_data['Time'].max()}")
        
        return rolling_data
    
    def _create_atmospheric_time_series_plots(self, rolling_data: pd.DataFrame):
        """Create publication-style time series plots."""
        print("    🎨 Creating time series plots...")
        
        # Create comprehensive time series figure
        fig, axes = plt.subplots(4, 1, figsize=(16, 20))
        fig.suptitle('Atmospheric Water Vapor Isotopes: Before and After Humidity Correction\n'
                    '5-Minute Rolling Averages', fontsize=16, fontweight='bold', y=0.98)
        
        time_data = rolling_data['Time']
        
        # Plot 1: H2O concentration
        axes[0].plot(time_data, rolling_data['H2O_ppm'], 'k-', linewidth=1.5, alpha=0.8)
        axes[0].set_ylabel('H₂O Concentration (ppm)', fontweight='bold', fontsize=12)
        axes[0].set_title('Water Vapor Concentration', fontweight='bold', fontsize=13)
        axes[0].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[0].tick_params(labelsize=10)
        
        # Plot 2: δD before and after correction
        axes[1].plot(time_data, rolling_data['D_del'], 'gray', linewidth=1.5, alpha=0.7,
                    label='Original δD')
        axes[1].plot(time_data, rolling_data['D_del_corrected'], 'black', linewidth=1.8,
                    label='Humidity Corrected δD')
        axes[1].set_ylabel('δD (‰ vs. working standard)', fontweight='bold', fontsize=12)
        axes[1].set_title('Deuterium Content: Original vs Humidity Corrected', fontweight='bold', fontsize=13)
        axes[1].legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
        axes[1].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[1].tick_params(labelsize=10)
        
        # Plot 3: δ18O before and after correction
        axes[2].plot(time_data, rolling_data['O18_del'], 'gray', linewidth=1.5, alpha=0.7,
                    label='Original δ18O')
        axes[2].plot(time_data, rolling_data['O18_del_corrected'], 'black', linewidth=1.8,
                    label='Humidity Corrected δ18O')
        axes[2].set_ylabel('δ18O (‰ vs. working standard)', fontweight='bold', fontsize=12)
        axes[2].set_title('Oxygen-18 Content: Original vs Humidity Corrected', fontweight='bold', fontsize=13)
        axes[2].legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
        axes[2].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[2].tick_params(labelsize=10)
        
        # Plot 4: d-excess before and after correction
        axes[3].plot(time_data, rolling_data['d_excess_original'], 'gray', linewidth=1.5, alpha=0.7,
                    label='Original d-excess')
        axes[3].plot(time_data, rolling_data['d_excess_corrected'], 'black', linewidth=1.8,
                    label='Humidity Corrected d-excess')
        axes[3].set_ylabel('d-excess (‰)', fontweight='bold', fontsize=12)
        axes[3].set_xlabel('Time', fontweight='bold', fontsize=12)
        axes[3].set_title('Deuterium Excess: Original vs Humidity Corrected', fontweight='bold', fontsize=13)
        axes[3].legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
        axes[3].grid(True, alpha=0.7, linestyle=':', color='gray')
        axes[3].tick_params(labelsize=10, axis='x', rotation=45)
        
        # Format x-axis for all subplots
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save time series plot
        ts_plot_path = self.output_dir / "atmospheric_time_series_humidity_correction.png"
        plt.savefig(ts_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      📊 Saved time series plot: {ts_plot_path.name}")
        
        # Create focused comparison plots
        self._create_focused_comparison_plots(rolling_data)
    
    def _create_focused_comparison_plots(self, rolling_data: pd.DataFrame):
        """Create focused before/after comparison plots."""
        print("    🎯 Creating focused comparison plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Humidity Correction Impact on Atmospheric Water Vapor Isotopes\n'
                    '5-Minute Rolling Averages', fontsize=15, fontweight='bold', y=0.95)
        
        time_data = rolling_data['Time']
        
        # Define isotope pairs and properties
        isotope_pairs = [
            ('D_del', 'D_del_corrected', 'δD (‰)', 'Deuterium'),
            ('O18_del', 'O18_del_corrected', 'δ18O (‰)', 'Oxygen-18'),
            ('d_excess_original', 'd_excess_corrected', 'd-excess (‰)', 'Deuterium Excess')
        ]
        
        for i, (orig_col, corr_col, ylabel, title) in enumerate(isotope_pairs):
            ax = axes[i]
            
            # Plot original and corrected
            ax.plot(time_data, rolling_data[orig_col], color='gray', linewidth=1.8, alpha=0.7,
                   label='Original')
            ax.plot(time_data, rolling_data[corr_col], color='black', linewidth=2,
                   label='Humidity Corrected')
            
            # Calculate and show statistics
            orig_mean = rolling_data[orig_col].mean()
            corr_mean = rolling_data[corr_col].mean()
            orig_std = rolling_data[orig_col].std()
            corr_std = rolling_data[corr_col].std()
            
            # Add statistics text box
            stats_text = f'Original: {orig_mean:.1f} ± {orig_std:.1f}‰\n'
            stats_text += f'Corrected: {corr_mean:.1f} ± {corr_std:.1f}‰\n'
            stats_text += f'Δ Mean: {corr_mean - orig_mean:+.1f}‰'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', 
                   alpha=0.9, edgecolor='black'))
            
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(f'{title} Correction Impact', fontweight='bold')
            ax.legend(fontsize=10, frameon=True, fancybox=False, edgecolor='black')
            ax.grid(True, alpha=0.7, linestyle=':', color='gray')
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.tick_params(axis='y', labelsize=10)
        
        axes[1].set_xlabel('Time', fontweight='bold')
        
        plt.tight_layout()
        
        # Save focused comparison plot
        comp_plot_path = self.output_dir / "atmospheric_humidity_correction_comparison.png"
        plt.savefig(comp_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      📊 Saved comparison plot: {comp_plot_path.name}")
    
    def _create_atmospheric_histograms(self, data: pd.DataFrame):
        """Create publication-style histograms showing before/after distributions."""
        print("    📊 Creating histogram distributions...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Atmospheric Water Vapor Isotope Distributions\n'
                    'Before and After Humidity Correction', fontsize=16, fontweight='bold', y=0.95)
        
        # Define isotope pairs for histograms
        histogram_data = [
            ('D_del', 'D_del_corrected', 'δD (‰)', 'Deuterium'),
            ('O18_del', 'O18_del_corrected', 'δ18O (‰)', 'Oxygen-18'),
            ('d_excess_original', 'd_excess_corrected', 'd-excess (‰)', 'Deuterium Excess')
        ]
        
        for i, (orig_col, corr_col, xlabel, title) in enumerate(histogram_data):
            # Top row: overlaid histograms
            ax_top = axes[0, i]
            
            # Calculate histogram bins
            all_data = np.concatenate([data[orig_col].dropna(), data[corr_col].dropna()])
            bins = np.linspace(np.percentile(all_data, 1), np.percentile(all_data, 99), 50)
            
            # Plot histograms
            ax_top.hist(data[orig_col].dropna(), bins=bins, alpha=0.6, color='gray', 
                       density=True, label='Original', edgecolor='black', linewidth=0.5)
            ax_top.hist(data[corr_col].dropna(), bins=bins, alpha=1.0, 
                       density=True, label='Humidity Corrected', edgecolor='black', 
                       linewidth=1.5, histtype='step', facecolor='none')
            
            # Add statistical information
            orig_mean = data[orig_col].mean()
            corr_mean = data[corr_col].mean()
            orig_std = data[orig_col].std()
            corr_std = data[corr_col].std()
            
            ax_top.axvline(orig_mean, color='gray', linestyle='--', linewidth=2, alpha=0.8,
                          label=f'Original mean: {orig_mean:.1f}‰')
            ax_top.axvline(corr_mean, color='black', linestyle='-', linewidth=2, alpha=0.8,
                          label=f'Corrected mean: {corr_mean:.1f}‰')
            
            ax_top.set_ylabel('Probability Density', fontweight='bold')
            ax_top.set_title(f'{title} Distribution', fontweight='bold')
            ax_top.legend(fontsize=9)
            ax_top.grid(True, alpha=0.7, linestyle=':', color='gray')
            
            # Bottom row: Q-Q plots
            ax_bottom = axes[1, i]
            
            # Create Q-Q plot
            orig_sorted = np.sort(data[orig_col].dropna())
            corr_sorted = np.sort(data[corr_col].dropna())
            
            # Interpolate to same length for comparison
            n_points = min(len(orig_sorted), len(corr_sorted))
            orig_interp = np.interp(np.linspace(0, 1, n_points), 
                                   np.linspace(0, 1, len(orig_sorted)), orig_sorted)
            corr_interp = np.interp(np.linspace(0, 1, n_points),
                                   np.linspace(0, 1, len(corr_sorted)), corr_sorted)
            
            ax_bottom.scatter(orig_interp, corr_interp, alpha=0.6, s=8, color='black',
                             edgecolors='none')
            
            # Add 1:1 line
            min_val = min(orig_interp.min(), corr_interp.min())
            max_val = max(orig_interp.max(), corr_interp.max())
            ax_bottom.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8,
                          linewidth=2, label='1:1 Reference')
            
            ax_bottom.set_xlabel(f'Original {xlabel}', fontweight='bold')
            ax_bottom.set_ylabel(f'Corrected {xlabel}', fontweight='bold')
            ax_bottom.set_title(f'{title} Q-Q Plot', fontweight='bold')
            ax_bottom.legend(fontsize=9)
            ax_bottom.grid(True, alpha=0.7, linestyle=':', color='gray')
        
        # Set xlabel for top row
        for i in range(3):
            axes[0, i].set_xlabel(histogram_data[i][2], fontweight='bold')
        
        plt.tight_layout()
        
        # Save histogram plot
        hist_plot_path = self.output_dir / "atmospheric_isotope_histograms.png"
        plt.savefig(hist_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      📊 Saved histogram plot: {hist_plot_path.name}")
    
    def _save_processed_atmospheric_data(self, corrected_data: pd.DataFrame, rolling_data: pd.DataFrame):
        """Save processed atmospheric datasets with descriptive labels."""
        print("    💾 Saving processed datasets...")
        
        # Prepare full corrected dataset
        output_columns = [
            'Time', 'H2O_ppm', 'D_del', 'O18_del', 
            'D_del_corrected', 'O18_del_corrected',
            'd_excess_original', 'd_excess_corrected'
        ]
        
        full_output = corrected_data[output_columns].copy()
        
        # Add descriptive metadata
        full_output.attrs = {
            'description': 'Atmospheric water vapor isotope measurements with humidity bias correction',
            'correction_method': 'Polynomial humidity correction based on standards analysis',
            'reference_h2o_ppm': self.humidity_corrector.reference_h2o,
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'correction_applied_below_ppm': self.humidity_corrector.reference_h2o
        }
        
        # Save full corrected dataset
        full_output_path = self.output_dir / "atmospheric_isotopes_humidity_corrected_full.csv"
        full_output.to_csv(full_output_path, index=False)
        
        # Save 5-minute rolling averaged dataset
        rolling_output_path = self.output_dir / "atmospheric_isotopes_humidity_corrected_5min_rolling.csv"
        rolling_data.to_csv(rolling_output_path, index=False)
        
        # Create isotope-only datasets for each type
        isotope_only_original = corrected_data[['Time', 'H2O_ppm', 'D_del', 'O18_del', 'd_excess_original']].copy()
        isotope_only_original.columns = ['Time', 'H2O_ppm', 'dD_permil', 'd18O_permil', 'd_excess_permil']
        
        isotope_only_corrected = corrected_data[['Time', 'H2O_ppm', 'D_del_corrected', 'O18_del_corrected', 'd_excess_corrected']].copy()
        isotope_only_corrected.columns = ['Time', 'H2O_ppm', 'dD_permil', 'd18O_permil', 'd_excess_permil']
        
        # Save isotope-only datasets
        isotope_orig_path = self.output_dir / "atmospheric_isotopes_original_cleaned.csv"
        isotope_corr_path = self.output_dir / "atmospheric_isotopes_humidity_corrected_cleaned.csv"
        
        isotope_only_original.to_csv(isotope_orig_path, index=False)
        isotope_only_corrected.to_csv(isotope_corr_path, index=False)
        
        # Create metadata file
        metadata = {
            'processing_info': {
                'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'humidity_correction_method': 'Polynomial humidity bias correction',
                'reference_h2o_ppm': float(self.humidity_corrector.reference_h2o),
                'correction_applied_below_ppm': float(self.humidity_corrector.reference_h2o),
                'original_records': len(corrected_data),
                'rolling_average_window': '5 minutes'
            },
            'column_descriptions': {
                'Time': 'Measurement timestamp (UTC)',
                'H2O_ppm': 'Water vapor concentration (parts per million)',
                'dD_permil': 'Deuterium content (per mil vs working standard)',
                'd18O_permil': 'Oxygen-18 content (per mil vs working standard)',
                'd_excess_permil': 'Deuterium excess = dD - 8*d18O (per mil)'
            },
            'files_generated': {
                'atmospheric_isotopes_humidity_corrected_full.csv': 'Complete dataset with original and corrected values',
                'atmospheric_isotopes_humidity_corrected_5min_rolling.csv': '5-minute rolling averages',
                'atmospheric_isotopes_original_cleaned.csv': 'Original isotope values (cleaned format)',
                'atmospheric_isotopes_humidity_corrected_cleaned.csv': 'Humidity corrected isotope values (cleaned format)'
            }
        }
        
        metadata_path = self.output_dir / "atmospheric_data_processing_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Print summary
        print(f"      📁 Full dataset: {full_output_path.name} ({len(full_output):,} records)")
        print(f"      📁 5-min rolling: {rolling_output_path.name} ({len(rolling_data):,} records)")
        print(f"      📁 Original cleaned: {isotope_orig_path.name}")
        print(f"      📁 Corrected cleaned: {isotope_corr_path.name}")
        print(f"      📁 Metadata: {metadata_path.name}")

    def _plot_corrected_offsets(self, offsets_df: pd.DataFrame):
        """Plot the corrected offsets analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Humidity Corrected Standards: Offset Analysis', 
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
        
        axes[0,0].set_xlabel('δD Known (‰)', fontweight='bold')
        axes[0,0].set_ylabel('δD Measured (‰, humidity corrected)', fontweight='bold')
        axes[0,0].set_title('δD: Measured vs Known Values', fontweight='bold')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.7, linestyle=':', color='gray')
        
        # Similar for δ18O...
        axes[0,1].scatter(offsets_df['d18O_known'], offsets_df['d18O_measured_mean'], 
                         alpha=0.8, s=80, c='black', marker='^', edgecolors='none')
        axes[0,1].errorbar(offsets_df['d18O_known'], offsets_df['d18O_measured_mean'],
                          yerr=offsets_df['d18O_measured_std'], fmt='none', alpha=0.6,
                          color='black', capsize=3, capthick=1.5)
        
        min_val = min(offsets_df['d18O_known'].min(), offsets_df['d18O_measured_mean'].min())
        max_val = max(offsets_df['d18O_known'].max(), offsets_df['d18O_measured_mean'].max())
        axes[0,1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2,
                      label='1:1 Reference Line')
        
        axes[0,1].set_xlabel('δ18O Known (‰)', fontweight='bold')
        axes[0,1].set_ylabel('δ18O Measured (‰, humidity corrected)', fontweight='bold')
        axes[0,1].set_title('δ18O: Measured vs Known Values', fontweight='bold')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.7, linestyle=':', color='gray')
        
        # Offset plots by standard
        unique_standards = offsets_df['standard_name'].unique()
        x_pos = np.arange(len(unique_standards))
        
        # δD offsets
        dd_offsets = [offsets_df[offsets_df['standard_name'] == std]['dD_offset'].mean() 
                     for std in unique_standards]
        dd_errors = [offsets_df[offsets_df['standard_name'] == std]['dD_measured_std'].mean() 
                    for std in unique_standards]
        
        axes[1,0].bar(x_pos, dd_offsets, yerr=dd_errors, capsize=5, color='black', alpha=0.7,
                     edgecolor='black', linewidth=1)
        axes[1,0].axhline(0, color='gray', linestyle='--', alpha=0.8, linewidth=2)
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels(unique_standards, rotation=45, ha='right')
        axes[1,0].set_ylabel('δD Offset (‰)', fontweight='bold')
        axes[1,0].set_title('δD Offsets by Standard', fontweight='bold')
        axes[1,0].grid(True, alpha=0.7, linestyle=':', color='gray')
        
        # δ18O offsets
        d18o_offsets = [offsets_df[offsets_df['standard_name'] == std]['d18O_offset'].mean() 
                       for std in unique_standards]
        d18o_errors = [offsets_df[offsets_df['standard_name'] == std]['d18O_measured_std'].mean() 
                      for std in unique_standards]
        
        axes[1,1].bar(x_pos, d18o_offsets, yerr=d18o_errors, capsize=5, color='black', alpha=0.7,
                     edgecolor='black', linewidth=1)
        axes[1,1].axhline(0, color='gray', linestyle='--', alpha=0.8, linewidth=2)
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(unique_standards, rotation=45, ha='right')
        axes[1,1].set_ylabel('δ18O Offset (‰)', fontweight='bold')
        axes[1,1].set_title('δ18O Offsets by Standard', fontweight='bold')
        axes[1,1].grid(True, alpha=0.7, linestyle=':', color='gray')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "corrected_offsets_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Saved corrected offsets analysis: {plot_path.name}")
    
    def run_complete_analysis(self, atmospheric_data_path: Optional[Path] = None):
        """Run the complete improved analysis including atmospheric data processing."""
        print("🚀 Starting complete humidity calibration and atmospheric analysis...\n")
        
        # Investigate correction function
        self._investigate_correction_function()
        print()
        
        # Create detailed correction analysis
        self.create_detailed_correction_analysis()
        print()
        
        # Create matched standards analysis
        self.create_matched_standards_analysis()
        print()
        
        # Create offsets summary
        offsets_df = self.create_offsets_summary()
        print()
        
        # Process atmospheric data if provided
        atmospheric_results = None
        if atmospheric_data_path and atmospheric_data_path.exists():
            atmospheric_results = self.process_atmospheric_data(atmospheric_data_path)
            print()
        
        # Create summary report
        self._create_improved_summary_report(offsets_df, atmospheric_results)
        
        print("🎉 Complete humidity calibration and atmospheric analysis finished!")
        print(f"📁 All outputs saved to: {self.output_dir}")
        
        return offsets_df, atmospheric_results
    
    def _create_improved_summary_report(self, offsets_df, atmospheric_results=None):
        """Create an improved summary report."""
        report_path = self.output_dir / "improved_analysis_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("IMPROVED HUMIDITY CALIBRATION ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("INPUT FILES:\n")
            f.write("-" * 12 + "\n")
            f.write(f"Standards Data: {self.standards_data_path.name}\n")
            f.write(f"Known Standards: {self.known_standards_path.name}\n")
            f.write(f"Calibration File: {self.humidity_calibration_path.name}\n\n")
            
            f.write("MATCHING RESULTS:\n")
            f.write("-" * 16 + "\n")
            f.write(f"Total standards runs: {len(self.standards_runs)}\n")
            f.write(f"Successfully matched: {len(self.matched_standards)}\n")
            f.write(f"Reference H₂O: {self.humidity_corrector.reference_h2o} ppm\n\n")
            
            if self.matched_standards:
                f.write("MATCHED STANDARDS:\n")
                f.write("-" * 18 + "\n")
                for run_id, info in self.matched_standards.items():
                    f.write(f"Run {run_id} ({info['date']}): {info['name']} "
                           f"(δD: {info['dD_known']:.1f}‰, δ18O: {info['d18O_known']:.1f}‰)\n")
                f.write("\n")
            
            if offsets_df is not None and len(offsets_df) > 0:
                f.write("OFFSET STATISTICS:\n")
                f.write("-" * 18 + "\n")
                f.write(f"δD mean offset: {offsets_df['dD_offset'].mean():+.2f} ± {offsets_df['dD_offset'].std():.2f}‰\n")
                f.write(f"δ18O mean offset: {offsets_df['d18O_offset'].mean():+.2f} ± {offsets_df['d18O_offset'].std():.2f}‰\n\n")
                
                f.write("INDIVIDUAL OFFSETS:\n")
                f.write("-" * 19 + "\n")
                for _, row in offsets_df.iterrows():
                    f.write(f"{row['standard_name']} (Run {row['run_id']}):\n")
                    f.write(f"  δD: {row['dD_offset']:+.1f}‰, δ18O: {row['d18O_offset']:+.1f}‰\n")
        
        print(f"📄 Improved summary saved: {report_path.name}")

def main():
    """Main execution function."""
    project_root = Path(__file__).parent.parent
    
    standards_data_path = project_root / "data" / "raw" / "separated" / "standards_only_raw.csv"
    humidity_calib_path = project_root / "outputs" / "selective_calibration" / "selective_humidity_calibration.json"
    known_standards_path = project_root / "data" / "raw" / "known_isotope_standards_runs.csv"
    atmospheric_data_path = project_root / "data" / "raw" / "separated" / "atmospheric_only_raw.csv"
    output_dir = project_root / "outputs" / "improved_humidity_analysis"
    
    # Verify input files exist
    required_files = [
        (standards_data_path, "Standards data"), 
        (humidity_calib_path, "Humidity calibration"),
        (known_standards_path, "Known standards")
    ]
    
    for file_path, name in required_files:
        if not file_path.exists():
            print(f"❌ {name} file not found: {file_path}")
            return
    
    # Check if atmospheric data exists
    if not atmospheric_data_path.exists():
        print(f"⚠️  Atmospheric data file not found: {atmospheric_data_path}")
        print("   Standards analysis will proceed without atmospheric processing")
        atmospheric_data_path = None
    
    # Run improved analysis
    analyzer = ImprovedHumidityCalibrationAnalyzer(
        standards_data_path=standards_data_path,
        humidity_calibration_path=humidity_calib_path,
        known_standards_path=known_standards_path,
        output_dir=output_dir
    )
    
    results = analyzer.run_complete_analysis(atmospheric_data_path)
    
    return analyzer, results

if __name__ == "__main__":
    analyzer, offsets_df = main()