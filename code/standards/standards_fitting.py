#!/usr/bin/env python3
"""
Standards fitting script for isotope analysis.

Creates B&W publication-ready plots for each high-quality standards run:
1. Time series of H2O concentration
2. δD vs H2O scatter plot with power law fit
3. δ18O vs H2O scatter plot with power law fit

Saves fit parameters for both log and polynomial fits.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Add src to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root))

# Import functions from simple_test.py
sys.path.insert(0, str(current_dir))
from simple_test import (
    load_isotope_data_simple, 
    load_single_csv_data,
    detect_standards_periods_simple, 
    extract_standards_runs_simple,
    assess_run_quality_simple,
    setup_logging_simple,
    find_data_directories
)

def setup_plotting_style():
    """Set up B&W publication-ready plotting style."""
    plt.style.use('classic')
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'figure.figsize': (15, 5),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    # B&W color scheme
    colors = {
        'H2O': 'black',
        'dD': 'black', 
        'd18O': 'black'
    }
    return colors

def load_known_standards(data_dir):
    """Load known isotope standards from CSV file."""
    
    standards_file = Path(data_dir).parent / "raw" / "known_isotope_standards_runs.csv"
    
    if not standards_file.exists():
        print(f"⚠️  Known standards file not found: {standards_file}")
        return None
    
    try:
        standards_df = pd.read_csv(standards_file)
        standards_df['Time'] = pd.to_datetime(standards_df['Time'])
        print(f"✅ Loaded {len(standards_df)} known standards")
        return standards_df
    except Exception as e:
        print(f"❌ Error loading known standards: {e}")
        return None

def match_run_to_known_standard(run_data, known_standards):
    """
    Match a standards run to known isotope values based on date/time.
    
    Parameters:
    -----------
    run_data : pd.DataFrame
        Standards run data
    known_standards : pd.DataFrame
        Known standard values with columns: Time, d18O_known, dD_known
        
    Returns:
    --------
    dict or None
        Matched standard info with known values, or None if no match
    """
    
    if known_standards is None:
        return None
    
    # Get the run start time
    if 'start_time' not in run_data.columns:
        return None
    
    run_start = run_data['start_time'].iloc[0]
    if pd.isna(run_start):
        return None
    
    # Convert to datetime if needed
    if not isinstance(run_start, pd.Timestamp):
        try:
            run_start = pd.to_datetime(run_start)
        except:
            return None
    
    # Find exact date match in known standards
    run_date = run_start.date()
    
    for idx, standard in known_standards.iterrows():
        # Handle case where Time column might be just date (no time component)
        if isinstance(standard['Time'], str):
            try:
                standard_datetime = pd.to_datetime(standard['Time'])
                standard_date = standard_datetime.date()
            except:
                continue
        else:
            standard_date = standard['Time'].date()
        
        if run_date == standard_date:
            return {
                'timestamp': standard['Time'],
                'd18O_known': standard['d18O_known'],
                'dD_known': standard['dD_known'],
                'name': standard.get('Name', 'Unknown'),
                'run_date': run_date,
                'standard_date': standard_date
            }
    
    return None

def remove_last_4_minutes(run_data):
    """
    Remove the last 4 minutes of data from a standards run.
    
    Parameters:
    -----------
    run_data : pd.DataFrame
        Standards run data with Time column
        
    Returns:
    --------
    pd.DataFrame
        Run data with last 4 minutes removed
    """
    
    if 'Time' not in run_data.columns:
        return run_data
    
    # Convert Time to datetime if it's not already
    time_col = run_data['Time']
    if not pd.api.types.is_datetime64_any_dtype(time_col):
        try:
            time_col = pd.to_datetime(time_col)
        except:
            return run_data
    
    # Find the maximum time and subtract 4 minutes
    max_time = time_col.max()
    cutoff_time = max_time - pd.Timedelta(minutes=4)
    
    # Filter out the last 4 minutes
    mask = time_col <= cutoff_time
    filtered_data = run_data[mask].copy()
    
    if len(filtered_data) < len(run_data):
        print(f"   ✂️  Removed last 4 minutes: {len(run_data)} → {len(filtered_data)} points")
    
    return filtered_data

def combine_runs_by_standard(quality_results, runs):
    """
    Combine standards runs that have the same known standard.
    
    Parameters:
    -----------
    quality_results : list
        List of quality results with known standards
    runs : list
        List of run data DataFrames
        
    Returns:
    --------
    dict
        Dictionary with standard names as keys and combined run info as values
    """
    
    combined_standards = {}
    
    for quality_info in quality_results:
        known_standard = quality_info.get('known_standard')
        if not known_standard:
            continue
            
        standard_name = known_standard['name']
        run_id = quality_info['run_id']
        run_data = runs[run_id]
        
        if standard_name not in combined_standards:
            combined_standards[standard_name] = {
                'name': standard_name,
                'known_values': {
                    'dD_known': known_standard['dD_known'],
                    'd18O_known': known_standard['d18O_known']
                },
                'runs': [],
                'dates': [],
                'quality_scores': []
            }
        
        combined_standards[standard_name]['runs'].append(run_data)
        combined_standards[standard_name]['dates'].append(known_standard['run_date'])
        combined_standards[standard_name]['quality_scores'].append(quality_info['quality_score'])
    
    return combined_standards

def create_combined_standard_plots(combined_standards, colors, output_dir, fit_params):
    """
    Create combined plots for each standard showing all runs together.
    
    Parameters:
    -----------
    combined_standards : dict
        Combined standards data
    colors : dict
        Color scheme
    output_dir : Path
        Output directory
    fit_params : dict
        Fit parameters storage
    """
    
    print(f"\n📊 Creating combined plots by standard...")
    
    for standard_name, standard_data in combined_standards.items():
        print(f"   Creating combined plot for {standard_name} ({len(standard_data['runs'])} runs)")
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot title with standard info
        dates_str = ", ".join([str(d) for d in sorted(standard_data['dates'])])
        known_dD = standard_data['known_values']['dD_known']
        known_d18O = standard_data['known_values']['d18O_known']
        
        fig.suptitle(f"{standard_name} Combined Runs ({dates_str})\nδD = {known_dD:.1f}‰, δ18O = {known_d18O:.1f}‰", 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Color palette for different runs
        run_colors = plt.cm.tab10(np.linspace(0, 1, len(standard_data['runs'])))
        
        all_h2o_data = []
        all_dd_data = []
        all_d18o_data = []
        
        # Plot 1: H2O time series for all runs
        for i, run_data in enumerate(standard_data['runs']):
            # Remove last 4 minutes
            run_data = remove_last_4_minutes(run_data)
            
            if 'Time' not in run_data.columns:
                continue
                
            time = run_data['Time']
            h2o = run_data.get('H2O_ppm', None)
            
            if h2o is None:
                continue
            
            # Convert time to minutes from start of each run
            if pd.api.types.is_datetime64_any_dtype(time):
                time_start = time.min()
                minutes_from_start = (time - time_start).dt.total_seconds() / 60
            else:
                try:
                    time_dt = pd.to_datetime(time)
                    time_start = time_dt.min()
                    minutes_from_start = (time_dt - time_start).dt.total_seconds() / 60
                except:
                    minutes_from_start = np.arange(len(time))
            
            # Add run offset for visual separation
            minutes_offset = minutes_from_start + (i * 200)  # 200 minute separation
            
            ax1.plot(minutes_offset, h2o, color=run_colors[i], linewidth=1.5, alpha=0.8,
                    label=f'Run {i+1} ({standard_data["dates"][i]})')
            ax1.scatter(minutes_offset, h2o, color=run_colors[i], s=6, alpha=0.6)
            
            # Store data for combined isotope plots
            dd = run_data.get('D_del', None)
            d18o = run_data.get('O18_del', None)
            
            if h2o is not None and dd is not None:
                # Apply 400 ppm filter
                valid_mask = (h2o >= 400) & h2o.notna() & dd.notna()
                if valid_mask.sum() > 0:
                    all_h2o_data.extend(h2o[valid_mask])
                    all_dd_data.extend(dd[valid_mask])
            
            if h2o is not None and d18o is not None:
                # Apply 400 ppm filter
                valid_mask = (h2o >= 400) & h2o.notna() & d18o.notna()
                if valid_mask.sum() > 0:
                    if len(all_d18o_data) == 0:  # First time adding d18O data
                        all_h2o_data = []  # Reset for d18O (in case h2o was added for dD)
                        all_h2o_data.extend(h2o[valid_mask])
                    all_d18o_data.extend(d18o[valid_mask])
        
        ax1.set_xlabel('Time (minutes, offset per run)', fontweight='bold')
        ax1.set_ylabel('H2O (ppm)', fontweight='bold')
        ax1.set_title('H2O Time Series (All Runs)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc='best')
        
        # Set x-axis ticks every 30 minutes
        from matplotlib.ticker import MultipleLocator
        ax1.xaxis.set_major_locator(MultipleLocator(60))  # Wider spacing for combined plot
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
        
        # Plot 2: Combined δD vs H2O
        if all_dd_data and all_h2o_data:
            h2o_array = np.array(all_h2o_data)
            dd_array = np.array(all_dd_data)
            
            ax2.scatter(h2o_array, dd_array, color='black', s=8, alpha=0.6, marker='.')
            
            # Add known standard line
            ax2.axhline(y=known_dD, color='gray', linestyle='--', linewidth=1,
                       label=f'{standard_name}: {known_dD:.1f}‰')
            
            # Fit combined data
            if len(h2o_array) > 10:
                try:
                    h2o_pos = h2o_array[h2o_array >= 400]
                    dd_pos = dd_array[h2o_array >= 400]
                    
                    if len(h2o_pos) > 3:
                        dd_shifted = dd_pos - dd_pos.min() + 1
                        log_h2o = np.log(h2o_pos)
                        log_dd = np.log(dd_shifted)
                        
                        coeffs = np.polyfit(log_h2o, log_dd, 1)
                        b, log_a = coeffs[0], coeffs[1]
                        
                        dd_pred = np.exp(log_a) * (h2o_pos ** b) + dd_pos.min() - 1
                        r_squared = 1 - np.sum((dd_pos - dd_pred)**2) / np.sum((dd_pos - np.mean(dd_pos))**2)
                        
                        h2o_smooth = np.linspace(h2o_pos.min(), h2o_pos.max(), 100)
                        dd_fit = np.exp(log_a) * (h2o_smooth ** b) + dd_pos.min() - 1
                        ax2.plot(h2o_smooth, dd_fit, '-', color='black', alpha=0.8, linewidth=1,
                                label=f'Combined fit: δD ∝ H2O^{b:.3f}, R²={r_squared:.3f}')
                        
                        # Store combined fit parameters
                        fit_params[f'combined_{standard_name}_dD'] = {
                            'standard_name': standard_name,
                            'n_runs': len(standard_data['runs']),
                            'dates': dates_str,
                            'exponent': float(b),
                            'r_squared': float(r_squared),
                            'n_points': len(h2o_pos)
                        }
                except:
                    pass
            
            ax2.set_xlabel('H2O (ppm)', fontweight='bold')
            ax2.set_ylabel('δD (‰)', fontweight='bold')
            ax2.set_title('δD vs H2O (Combined)', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            if ax2.get_legend_handles_labels()[0]:
                ax2.legend(loc='best', fontsize=8)
        
        # Plot 3: Combined δ18O vs H2O (similar structure as δD)
        if all_d18o_data and len(all_h2o_data) >= len(all_d18o_data):
            h2o_array = np.array(all_h2o_data[-len(all_d18o_data):])  # Match lengths
            d18o_array = np.array(all_d18o_data)
            
            ax3.scatter(h2o_array, d18o_array, color='black', s=8, alpha=0.6, marker='.')
            
            # Add known standard line
            ax3.axhline(y=known_d18O, color='gray', linestyle='--', linewidth=1,
                       label=f'{standard_name}: {known_d18O:.1f}‰')
            
            # Fit combined data
            if len(h2o_array) > 10:
                try:
                    h2o_pos = h2o_array[h2o_array >= 400]
                    d18o_pos = d18o_array[h2o_array >= 400]
                    
                    if len(h2o_pos) > 3:
                        d18o_shifted = d18o_pos - d18o_pos.min() + 1
                        log_h2o = np.log(h2o_pos)
                        log_d18o = np.log(d18o_shifted)
                        
                        coeffs = np.polyfit(log_h2o, log_d18o, 1)
                        b, log_a = coeffs[0], coeffs[1]
                        
                        d18o_pred = np.exp(log_a) * (h2o_pos ** b) + d18o_pos.min() - 1
                        r_squared = 1 - np.sum((d18o_pos - d18o_pred)**2) / np.sum((d18o_pos - np.mean(d18o_pos))**2)
                        
                        h2o_smooth = np.linspace(h2o_pos.min(), h2o_pos.max(), 100)
                        d18o_fit = np.exp(log_a) * (h2o_smooth ** b) + d18o_pos.min() - 1
                        ax3.plot(h2o_smooth, d18o_fit, '-', color='black', alpha=0.8, linewidth=1,
                                label=f'Combined fit: δ18O ∝ H2O^{b:.3f}, R²={r_squared:.3f}')
                        
                        # Store combined fit parameters
                        fit_params[f'combined_{standard_name}_d18O'] = {
                            'standard_name': standard_name,
                            'n_runs': len(standard_data['runs']),
                            'dates': dates_str,
                            'exponent': float(b),
                            'r_squared': float(r_squared),
                            'n_points': len(h2o_pos)
                        }
                except:
                    pass
            
            ax3.set_xlabel('H2O (ppm)', fontweight='bold')
            ax3.set_ylabel('δ18O (‰)', fontweight='bold')
            ax3.set_title('δ18O vs H2O (Combined)', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            if ax3.get_legend_handles_labels()[0]:
                ax3.legend(loc='best', fontsize=8)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save combined plot
        safe_name = standard_name.replace(' ', '_').replace('/', '_')
        plot_filename = f"combined_{safe_name}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Saved combined plot: {plot_filename}")

def predict_unknown_standard_values(run_data, known_standards, run_id):
    """
    Make a best guess at isotopic values for unknown standards runs based on 
    patterns in known standards and high H2O measurements.
    
    Parameters:
    -----------
    run_data : pd.DataFrame
        Standards run data
    known_standards : pd.DataFrame
        Known standard values
    run_id : int
        Run identifier for logging
        
    Returns:
    --------
    dict or None
        Predicted standard values, or None if prediction not possible
    """
    
    # Get high H2O values (> 7000 ppm) from the run
    h2o = run_data.get('H2O_ppm', None)
    dd = run_data.get('D_del', None)
    d18o = run_data.get('O18_del', None)
    
    if h2o is None or (dd is None and d18o is None):
        return None
    
    # Filter for high H2O values where humidity bias is minimal
    high_h2o_mask = (h2o > 7000) & h2o.notna()
    
    if high_h2o_mask.sum() < 3:  # Need at least 3 points for reliable estimate
        return None
    
    # Calculate mean isotope values at high H2O
    predicted_values = {}
    
    if dd is not None and dd[high_h2o_mask].notna().sum() > 0:
        predicted_values['dD_predicted'] = dd[high_h2o_mask].mean()
    
    if d18o is not None and d18o[high_h2o_mask].notna().sum() > 0:
        predicted_values['d18O_predicted'] = d18o[high_h2o_mask].mean()
    
    if not predicted_values:
        return None
    
    # Find the closest known standard based on isotopic similarity
    best_match_distance = float('inf')
    best_match_standard = None
    
    for idx, standard in known_standards.iterrows():
        distance = 0
        valid_comparisons = 0
        
        if 'dD_predicted' in predicted_values and not pd.isna(standard['dD_known']):
            distance += abs(predicted_values['dD_predicted'] - standard['dD_known'])
            valid_comparisons += 1
            
        if 'd18O_predicted' in predicted_values and not pd.isna(standard['d18O_known']):
            distance += abs(predicted_values['d18O_predicted'] - standard['d18O_known']) * 8  # Weight d18O more heavily
            valid_comparisons += 1
        
        if valid_comparisons > 0:
            normalized_distance = distance / valid_comparisons
            if normalized_distance < best_match_distance:
                best_match_distance = normalized_distance
                best_match_standard = standard
    
    if best_match_standard is None:
        return None
    
    # Create prediction result
    prediction = {
        'dD_predicted': predicted_values.get('dD_predicted'),
        'd18O_predicted': predicted_values.get('d18O_predicted'),
        'closest_known_standard': {
            'dD_known': best_match_standard['dD_known'],
            'd18O_known': best_match_standard['d18O_known'],
            'name': best_match_standard.get('Name', 'Unknown'),
            'date': best_match_standard['Time']
        },
        'match_distance': best_match_distance,
        'n_high_h2o_points': high_h2o_mask.sum(),
        'confidence': 'high' if best_match_distance < 10 else 'medium' if best_match_distance < 25 else 'low'
    }
    
    print(f"   🔮 Run {run_id} predicted values: δD={prediction['dD_predicted']:.1f}‰, δ18O={prediction['d18O_predicted']:.1f}‰ "
          f"(closest to {best_match_standard.get('Name', 'Unknown')}, distance={best_match_distance:.1f}, confidence={prediction['confidence']})")
    
    return prediction

def create_run_plots(run_data, run_id, quality_info, colors, output_dir, fit_params):
    """
    Create 3 B&W publication plots for a single standards run.
    
    Parameters:
    -----------
    run_data : pd.DataFrame
        Data from standards run
    run_id : int
        Run identifier
    quality_info : dict
        Quality assessment information
    colors : dict
        Color scheme for plots (B&W)
    output_dir : Path
        Output directory for plots
    fit_params : dict
        Dictionary to store fit parameters
    """
    
    # Extract data
    if 'Time' not in run_data.columns:
        print(f"⚠️  No Time column in run {run_id}, skipping")
        return
    
    # Remove last 4 minutes from the run
    run_data = remove_last_4_minutes(run_data)
    
    time = run_data['Time']
    h2o = run_data.get('H2O_ppm', None)
    dd = run_data.get('D_del', None)  
    d18o = run_data.get('O18_del', None)
    
    # Check required columns
    if h2o is None:
        print(f"⚠️  No H2O_ppm column in run {run_id}, skipping")
        return
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Run info for title
    date_str = quality_info.get('date_str', 'unknown')
    score = quality_info.get('quality_score', 0)
    duration = quality_info.get('duration_min', 0)
    n_points = quality_info.get('n_points', 0)
    h2o_range = quality_info.get('h2o_range', 0)
    
    run_title = f"Run {run_id} ({date_str}) - Score: {score:.1f}, Duration: {duration:.1f}min, Points: {n_points}, Range: {h2o_range:.0f}ppm"
    fig.suptitle(run_title, fontsize=13, y=0.98, color='black')
    
    # Initialize fit parameters for this run
    fit_params[f'run_{run_id}'] = {
        'metadata': {
            'run_id': run_id,
            'date': date_str,
            'quality_score': score,
            'n_points': n_points,
            'h2o_range': h2o_range,
            'duration_min': duration
        }
    }
    
    # Plot 1: H2O time series with time in minutes from start
    # Convert time to minutes from start
    if pd.api.types.is_datetime64_any_dtype(time):
        time_start = time.min()
        minutes_from_start = (time - time_start).dt.total_seconds() / 60
    else:
        # If time is not datetime, try to convert
        try:
            time_dt = pd.to_datetime(time)
            time_start = time_dt.min()
            minutes_from_start = (time_dt - time_start).dt.total_seconds() / 60
        except:
            # Fallback: use index as time
            minutes_from_start = np.arange(len(time))
    
    ax1.plot(minutes_from_start, h2o, color=colors['H2O'], linewidth=1.5, alpha=0.8)
    ax1.scatter(minutes_from_start, h2o, color=colors['H2O'], s=8, alpha=0.6)
    ax1.set_xlabel('Time (minutes from start)', fontweight='bold')
    ax1.set_ylabel('H2O (ppm)', fontweight='bold')
    ax1.set_title('H2O Time Series', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Set x-axis ticks every 30 minutes
    from matplotlib.ticker import MultipleLocator
    ax1.xaxis.set_major_locator(MultipleLocator(30))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
    
    # Plot 2: δD vs H2O
    if dd is not None:
        # Apply 400 ppm filter for plotting
        plot_mask = (h2o >= 400) & h2o.notna() & dd.notna()
        if plot_mask.sum() > 0:
            ax2.scatter(h2o[plot_mask], dd[plot_mask], color=colors['dD'], s=12, alpha=0.7, marker='.')
        
        # Add horizontal lines for known or predicted δD values
        known_standard = quality_info.get('known_standard')
        predicted_standard = quality_info.get('predicted_standard')
        
        if known_standard and not pd.isna(known_standard['dD_known']):
            known_dD = known_standard['dD_known']
            standard_name = known_standard.get('name', 'Unknown')
            ax2.axhline(y=known_dD, color='gray', linestyle='--', linewidth=1, 
                       label=f'{standard_name}: {known_dD:.1f}‰')
        elif predicted_standard and predicted_standard['dD_predicted']:
            pred_dD = predicted_standard['dD_predicted']
            confidence = predicted_standard['confidence']
            closest_name = predicted_standard['closest_known_standard']['name']
            line_style = ':' if confidence == 'medium' else '-.' if confidence == 'low' else '--'
            ax2.axhline(y=pred_dD, color='orange', linestyle=line_style, linewidth=1, 
                       label=f'Predicted ({closest_name}): {pred_dD:.1f}‰')
        
        ax2.set_xlabel('H2O (ppm)', fontweight='bold')
        ax2.set_ylabel('δD (‰)', fontweight='bold')
        ax2.set_title('δD vs H2O', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line and save parameters
        if len(h2o.dropna()) > 5:
            try:
                # Fit power law trend: y = a * x^b + c
                valid_mask = ~(h2o.isna() | dd.isna())
                if valid_mask.sum() > 5:
                    h2o_clean = h2o[valid_mask]
                    dd_clean = dd[valid_mask]
                    
                    # Power law fit using log-log transformation
                    # Apply 400 ppm filter following standards requirements
                    h2o_pos = h2o_clean[h2o_clean >= 400]
                    dd_pos = dd_clean[h2o_clean >= 400]
                    
                    if len(h2o_pos) > 3:
                        # Handle negative values for log transformation
                        dd_shifted = dd_pos - dd_pos.min() + 1
                        log_h2o = np.log(h2o_pos)
                        log_dd = np.log(dd_shifted)
                        
                        # Linear fit in log space
                        coeffs = np.polyfit(log_h2o, log_dd, 1)
                        b, log_a = coeffs[0], coeffs[1]
                        
                        # Calculate R²
                        dd_pred = np.exp(log_a) * (h2o_pos ** b) + dd_pos.min() - 1
                        r_squared = 1 - np.sum((dd_pos - dd_pred)**2) / np.sum((dd_pos - np.mean(dd_pos))**2)
                        
                        # Generate smooth curve
                        h2o_smooth = np.linspace(h2o_pos.min(), h2o_pos.max(), 100)
                        dd_fit = np.exp(log_a) * (h2o_smooth ** b) + dd_pos.min() - 1
                        ax2.plot(h2o_smooth, dd_fit, '-', color='black', alpha=0.8, linewidth=1, 
                                label=f'Power law: δD ∝ H2O^{b:.2f}, R²={r_squared:.3f}')
                        
                        # Save fit parameters
                        fit_params[f'run_{run_id}']['dD_power_law'] = {
                            'exponent': float(b),
                            'log_a': float(log_a),
                            'a_effective': float(np.exp(log_a)),
                            'offset': float(dd_pos.min() - 1),
                            'r_squared': float(r_squared),
                            'n_points': len(h2o_pos)
                        }
                        
                        # Also try polynomial fit
                        poly_coeffs = np.polyfit(h2o_pos, dd_pos, 2)
                        poly_fit = np.polyval(poly_coeffs, h2o_smooth)
                        dd_poly_pred = np.polyval(poly_coeffs, h2o_pos)
                        poly_r2 = 1 - np.sum((dd_pos - dd_poly_pred)**2) / np.sum((dd_pos - np.mean(dd_pos))**2)
                        
                        fit_params[f'run_{run_id}']['dD_polynomial'] = {
                            'coefficients': [float(c) for c in poly_coeffs],
                            'r_squared': float(poly_r2),
                            'n_points': len(h2o_pos)
                        }
                        
            except Exception as e:
                print(f"   ⚠️  Fit failed for δD in run {run_id}: {e}")
        
        # Add legend if there are any labeled items
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend(loc='best', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No δD data', transform=ax2.transAxes, 
                ha='center', va='center', fontsize=12, color='black')
        ax2.set_xlabel('H2O (ppm)', fontweight='bold')
        ax2.set_ylabel('δD (‰)', fontweight='bold')
        ax2.set_title('δD vs H2O', fontweight='bold')
    
    # Plot 3: δ18O vs H2O  
    if d18o is not None:
        # Apply 400 ppm filter for plotting
        plot_mask = (h2o >= 400) & h2o.notna() & d18o.notna()
        if plot_mask.sum() > 0:
            ax3.scatter(h2o[plot_mask], d18o[plot_mask], color=colors['d18O'], s=12, alpha=0.7, marker='.')
        
        # Add horizontal lines for known or predicted δ18O values
        known_standard = quality_info.get('known_standard')
        predicted_standard = quality_info.get('predicted_standard')
        
        if known_standard and not pd.isna(known_standard['d18O_known']):
            known_d18O = known_standard['d18O_known']
            standard_name = known_standard.get('name', 'Unknown')
            ax3.axhline(y=known_d18O, color='gray', linestyle='--', linewidth=1, 
                       label=f'{standard_name}: {known_d18O:.1f}‰')
        elif predicted_standard and predicted_standard['d18O_predicted']:
            pred_d18O = predicted_standard['d18O_predicted']
            confidence = predicted_standard['confidence']
            closest_name = predicted_standard['closest_known_standard']['name']
            line_style = ':' if confidence == 'medium' else '-.' if confidence == 'low' else '--'
            ax3.axhline(y=pred_d18O, color='orange', linestyle=line_style, linewidth=1, 
                       label=f'Predicted ({closest_name}): {pred_d18O:.1f}‰')
        
        ax3.set_xlabel('H2O (ppm)', fontweight='bold')
        ax3.set_ylabel('δ18O (‰)', fontweight='bold')
        ax3.set_title('δ18O vs H2O', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add trend line and save parameters
        if len(h2o.dropna()) > 5:
            try:
                # Fit power law trend: y = a * x^b + c
                valid_mask = ~(h2o.isna() | d18o.isna())
                if valid_mask.sum() > 5:
                    h2o_clean = h2o[valid_mask]
                    d18o_clean = d18o[valid_mask]
                    
                    # Power law fit using log-log transformation
                    # Apply 400 ppm filter following standards requirements
                    h2o_pos = h2o_clean[h2o_clean >= 400]
                    d18o_pos = d18o_clean[h2o_clean >= 400]
                    
                    if len(h2o_pos) > 3:
                        # Handle negative values for log transformation
                        d18o_shifted = d18o_pos - d18o_pos.min() + 1
                        log_h2o = np.log(h2o_pos)
                        log_d18o = np.log(d18o_shifted)
                        
                        # Linear fit in log space
                        coeffs = np.polyfit(log_h2o, log_d18o, 1)
                        b, log_a = coeffs[0], coeffs[1]
                        
                        # Calculate R²
                        d18o_pred = np.exp(log_a) * (h2o_pos ** b) + d18o_pos.min() - 1
                        r_squared = 1 - np.sum((d18o_pos - d18o_pred)**2) / np.sum((d18o_pos - np.mean(d18o_pos))**2)
                        
                        # Generate smooth curve
                        h2o_smooth = np.linspace(h2o_pos.min(), h2o_pos.max(), 100)
                        d18o_fit = np.exp(log_a) * (h2o_smooth ** b) + d18o_pos.min() - 1
                        ax3.plot(h2o_smooth, d18o_fit, '-', color='black', alpha=0.8, linewidth=1,
                                label=f'Power law: δ18O ∝ H2O^{b:.2f}, R²={r_squared:.3f}')
                        
                        # Save fit parameters
                        fit_params[f'run_{run_id}']['d18O_power_law'] = {
                            'exponent': float(b),
                            'log_a': float(log_a),
                            'a_effective': float(np.exp(log_a)),
                            'offset': float(d18o_pos.min() - 1),
                            'r_squared': float(r_squared),
                            'n_points': len(h2o_pos)
                        }
                        
                        # Also try polynomial fit
                        poly_coeffs = np.polyfit(h2o_pos, d18o_pos, 2)
                        poly_fit = np.polyval(poly_coeffs, h2o_smooth)
                        d18o_poly_pred = np.polyval(poly_coeffs, h2o_pos)
                        poly_r2 = 1 - np.sum((d18o_pos - d18o_poly_pred)**2) / np.sum((d18o_pos - np.mean(d18o_pos))**2)
                        
                        fit_params[f'run_{run_id}']['d18O_polynomial'] = {
                            'coefficients': [float(c) for c in poly_coeffs],
                            'r_squared': float(poly_r2),
                            'n_points': len(h2o_pos)
                        }
                        
            except Exception as e:
                print(f"   ⚠️  Fit failed for δ18O in run {run_id}: {e}")
        
        # Add legend if there are any labeled items
        if ax3.get_legend_handles_labels()[0]:
            ax3.legend(loc='best', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No δ18O data', transform=ax3.transAxes,
                ha='center', va='center', fontsize=12, color='black')
        ax3.set_xlabel('H2O (ppm)', fontweight='bold')
        ax3.set_ylabel('δ18O (‰)', fontweight='bold')
        ax3.set_title('δ18O vs H2O', fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save plot
    plot_filename = f"run_{run_id:02d}_{date_str.replace('/', '')}_score{score:.0f}.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Saved: {plot_filename}")

def create_summary_plot(quality_results, output_dir):
    """Create B&W summary plot showing all run quality scores."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract data
    run_ids = [r['run_id'] for r in quality_results]
    scores = [r['quality_score'] for r in quality_results]
    dates = [r['date_str'] for r in quality_results]
    
    # Plot 1: Quality scores bar chart (B&W)
    bars = ax1.bar(run_ids, scores, color='lightgray', edgecolor='black', alpha=0.7)
    ax1.axhline(y=100, color='black', linestyle='--', alpha=0.8, linewidth=2, label='Score = 100 threshold')
    ax1.set_xlabel('Run ID', fontweight='bold')
    ax1.set_ylabel('Quality Score', fontweight='bold')
    ax1.set_title('Standards Run Quality Scores', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add score labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score:.0f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Score distribution (B&W)
    ax2.hist(scores, bins=15, alpha=0.7, color='lightgray', edgecolor='black')
    ax2.axvline(x=100, color='black', linestyle='--', alpha=0.8, linewidth=2, label='Score = 100 threshold')
    ax2.set_xlabel('Quality Score', fontweight='bold')
    ax2.set_ylabel('Number of Runs', fontweight='bold')
    ax2.set_title('Quality Score Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save summary plot
    summary_path = output_dir / "quality_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Summary plot saved: {summary_path}")

def standards_fitting():
    """Main function to create standards fitting analysis."""
    
    print("📊 STANDARDS FITTING ANALYSIS")
    print("=" * 50)
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "outputs" / "standards_fitting"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / "standards_fitting.log"
    logger = setup_logging_simple(log_file)
    
    print(f"📁 Output directory: {output_dir}")
    print(f"📝 Log file: {log_file}")
    
    # Setup plotting
    colors = setup_plotting_style()
    
    # Initialize fit parameters storage
    fit_params = {}
    
    # Find available data directories
    data_directories = find_data_directories()
    
    if not data_directories:
        print("❌ No isotope data directories found!")
        print("   Please ensure data is in 'data/raw/' or set ISOTOPE_DATA_ROOT environment variable")
        return
    
    print(f"📊 Found {len(data_directories)} data directories:")
    for i, dir_path in enumerate(data_directories):
        print(f"   {i+1}. {dir_path}")
    
    # Load known standards
    print(f"\n🎯 Loading known isotope standards...")
    known_standards = None
    for data_dir in data_directories:
        known_standards = load_known_standards(data_dir)
        if known_standards is not None:
            break
    
    if known_standards is None:
        print("❌ Cannot proceed without known standards file!")
        print("   Please ensure 'known_isotope_standards_runs.csv' exists in data/raw/")
        return
    
    combined_data = None
    
    print(f"\n1️⃣ Loading isotope data...")
    
    for data_dir in data_directories:
        print(f"\n🔍 Trying directory: {data_dir}")
        
        # Try CSV first
        combined_data = load_single_csv_data(data_dir)
        if combined_data is not None:
            break
            
        # Try original txt files as fallback
        combined_data = load_isotope_data_simple(data_dir)
        if combined_data is not None:
            break
    
    if combined_data is None:
        print("❌ Cannot proceed without data")
        return
    
    print(f"\n2️⃣ Detecting standards periods...")
    periods = detect_standards_periods_simple(combined_data)
    
    if not periods:
        print("❌ No standards periods found")
        return
    
    print(f"\n3️⃣ Extracting standards runs...")
    runs = extract_standards_runs_simple(combined_data, periods)
    
    if not runs:
        print("❌ No standards runs extracted")
        return
    
    # Show summary of all extracted runs with their dates
    print(f"📋 Found {len(runs)} standards runs total:")
    for i, run_data in enumerate(runs):
        if 'start_time' in run_data.columns and len(run_data) > 0:
            start_time = run_data['start_time'].iloc[0]
            if pd.notna(start_time):
                date_str = pd.to_datetime(start_time).strftime('%Y-%m-%d')
                print(f"   Run {i}: {date_str}")
            else:
                print(f"   Run {i}: No valid start time")
        else:
            print(f"   Run {i}: No start_time column")
    
    print(f"\n4️⃣ Assessing run quality and matching to known standards...")
    quality_results = []
    
    for i, run_data in enumerate(runs):
        quality = assess_run_quality_simple(run_data)
        quality['run_id'] = i
        
        if 'start_time' in run_data.columns:
            quality['date_str'] = run_data['start_time'].iloc[0].strftime('%m/%d')
        else:
            quality['date_str'] = 'unknown'
        
        # Try to match this run to a known standard
        known_match = match_run_to_known_standard(run_data, known_standards)
        quality['known_standard'] = known_match
        
        if known_match is not None:
            quality_results.append(quality)
            print(f"   ✅ Run {i} matched to known standard {known_match['name']}: δD={known_match['dD_known']:.1f}‰, δ18O={known_match['d18O_known']:.1f}‰")
        else:
            print(f"   ❌ Run {i} excluded: no match to known standards")
    
    # Count only known standards (since we're only including those now)
    known_count = len(quality_results)  # All results now have known standards
    
    print(f"\n📊 {known_count} runs with known standards will be plotted")
    
    if not quality_results:
        print("❌ No runs matched to known standards!")
        return
    
    # Sort by quality score for organization (but plot all)
    quality_results.sort(key=lambda x: x['quality_score'], reverse=True)
    
    # Plot ALL standards runs regardless of score
    high_quality_runs = quality_results
    
    print(f"\n5️⃣ Creating B&W plots and fitting parameters for {len(high_quality_runs)} runs...")
    
    # Create plots for each high-quality run
    for i, quality_info in enumerate(high_quality_runs):
        run_id = quality_info['run_id']
        run_data = runs[run_id]
        
        print(f"   Creating plots for Run {run_id} (Score: {quality_info['quality_score']:.1f})")
        create_run_plots(run_data, run_id, quality_info, colors, output_dir, fit_params)
    
    print(f"\n6️⃣ Creating summary plot...")
    create_summary_plot(quality_results, output_dir)
    
    # Create combined plots by standard
    print(f"\n7️⃣ Creating combined plots by standard...")
    combined_standards = combine_runs_by_standard(quality_results, runs)
    if combined_standards:
        create_combined_standard_plots(combined_standards, colors, output_dir, fit_params)
        print(f"   Created combined plots for {len(combined_standards)} unique standards")
    else:
        print("   No standards to combine")
    
    # Save fit parameters
    print(f"\n8️⃣ Saving fit parameters...")
    params_file = output_dir / "fit_parameters.json"
    with open(params_file, 'w') as f:
        json.dump(fit_params, f, indent=2)
    print(f"💾 Fit parameters saved: {params_file}")
    
    # Create parameter summary CSV
    print(f"\n9️⃣ Creating parameter summary...")
    create_parameter_summary(fit_params, output_dir)
    
    print(f"\n✅ STANDARDS FITTING COMPLETED!")
    print(f"📁 All plots saved to: {output_dir}")
    print(f"📊 Fit parameters: {params_file}")

def create_parameter_summary(fit_params, output_dir):
    """Create summary CSV of all fit parameters."""
    
    summary_data = []
    
    for run_key, run_data in fit_params.items():
        base_info = {
            'run_id': run_data['metadata']['run_id'],
            'date': run_data['metadata']['date'],
            'quality_score': run_data['metadata']['quality_score'],
            'n_points': run_data['metadata']['n_points'],
            'h2o_range': run_data['metadata']['h2o_range']
        }
        
        # δD parameters
        if 'dD_power_law' in run_data:
            dD_power = run_data['dD_power_law']
            row = {**base_info,
                   'isotope': 'dD',
                   'fit_type': 'power_law',
                   'exponent': dD_power['exponent'],
                   'r_squared': dD_power['r_squared'],
                   'n_fit_points': dD_power['n_points']}
            summary_data.append(row)
        
        if 'dD_polynomial' in run_data:
            dD_poly = run_data['dD_polynomial']
            row = {**base_info,
                   'isotope': 'dD',
                   'fit_type': 'polynomial',
                   'coeff_0': dD_poly['coefficients'][2],
                   'coeff_1': dD_poly['coefficients'][1],
                   'coeff_2': dD_poly['coefficients'][0],
                   'r_squared': dD_poly['r_squared'],
                   'n_fit_points': dD_poly['n_points']}
            summary_data.append(row)
        
        # δ18O parameters
        if 'd18O_power_law' in run_data:
            d18O_power = run_data['d18O_power_law']
            row = {**base_info,
                   'isotope': 'd18O',
                   'fit_type': 'power_law',
                   'exponent': d18O_power['exponent'],
                   'r_squared': d18O_power['r_squared'],
                   'n_fit_points': d18O_power['n_points']}
            summary_data.append(row)
        
        if 'd18O_polynomial' in run_data:
            d18O_poly = run_data['d18O_polynomial']
            row = {**base_info,
                   'isotope': 'd18O',
                   'fit_type': 'polynomial',
                   'coeff_0': d18O_poly['coefficients'][2],
                   'coeff_1': d18O_poly['coefficients'][1],
                   'coeff_2': d18O_poly['coefficients'][0],
                   'r_squared': d18O_poly['r_squared'],
                   'n_fit_points': d18O_poly['n_points']}
            summary_data.append(row)
    
    # Save as CSV
    if summary_data:
        df = pd.DataFrame(summary_data)
        summary_path = output_dir / "fit_parameters_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"📊 Parameter summary saved: {summary_path}")

if __name__ == "__main__":
    standards_fitting()