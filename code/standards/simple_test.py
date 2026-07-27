#!/usr/bin/env python3
"""
Simple test script for the isotope calibration system.
This version uses direct imports to avoid relative import issues.
"""

import sys
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.metrics import r2_score
from datetime import datetime, timedelta

# Add the src directory to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"

# Add to path
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root))

def setup_logging_simple(log_file=None):
    """Simple logging setup."""
    logger = logging.getLogger('isotope_test')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    return logger

def load_isotope_data_simple(data_dir):
    """Simple isotope data loader."""
    
    print(f"🔍 Looking for isotope data files in: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return None
    
    # Find all .txt files
    txt_files = list(Path(data_dir).glob("*.txt"))
    
    if not txt_files:
        print(f"❌ No .txt files found in {data_dir}")
        return None
    
    print(f"📁 Found {len(txt_files)} text files")
    
    # Load and combine files
    dataframes = []
    
    for file_path in sorted(txt_files):
        try:
            print(f"   Loading: {file_path.name}")
            
            # Read CSV with header in row 1 (0-indexed)
            df = pd.read_csv(file_path, delimiter=',', header=1)
            
            # Clean column names
            df.columns = df.columns.str.replace(' ', '')
            
            # Remove NaN rows
            initial_len = len(df)
            df = df.dropna()
            
            if len(df) < initial_len:
                print(f"     Removed {initial_len - len(df)} NaN rows")
            
            # Convert time and voltage columns
            if 'Time' in df.columns:
                df['Time'] = pd.to_datetime(df['Time'])
            
            if 'WVISS_VOLTAGE' in df.columns:
                df['WVISS_VOLTAGE'] = pd.to_numeric(df['WVISS_VOLTAGE'], errors='coerce')
            
            # Add source file info
            df['source_file'] = file_path.name
            
            dataframes.append(df)
            print(f"     ✅ Loaded {len(df)} rows")
            
        except Exception as e:
            print(f"     ❌ Error loading {file_path.name}: {e}")
    
    if not dataframes:
        print("❌ No files loaded successfully")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Sort by time if available
    if 'Time' in combined_df.columns:
        combined_df = combined_df.sort_values('Time').reset_index(drop=True)
    
    print(f"✅ Combined dataset: {len(combined_df)} rows, {len(combined_df.columns)} columns")
    print(f"   Columns: {list(combined_df.columns)}")
    
    # Show voltage distribution
    if 'WVISS_VOLTAGE' in combined_df.columns:
        n_standards = (combined_df['WVISS_VOLTAGE'] > 0).sum()
        n_measurements = (combined_df['WVISS_VOLTAGE'] == 0).sum()
        print(f"   Standards points: {n_standards}")
        print(f"   Measurement points: {n_measurements}")
    
    return combined_df

def detect_standards_periods_simple(df):
    """Simple standards period detection."""
    
    if 'WVISS_VOLTAGE' not in df.columns or 'Time' not in df.columns:
        print("❌ Missing required columns for standards detection")
        return []
    
    print("🔍 Detecting standards periods...")
    
    # Sort by time
    df = df.sort_values('Time').reset_index(drop=True)
    
    voltage = df['WVISS_VOLTAGE'].values
    time = df['Time'].values
    
    periods = []
    in_standards = False
    start_time = None
    
    for i in range(len(voltage) - 1):
        current_v = voltage[i]
        next_v = voltage[i + 1]
        
        # Start of standards
        if current_v <= 0 and next_v > 0 and not in_standards:
            start_time = time[i + 1]
            in_standards = True
            
        # End of standards
        elif current_v > 0 and next_v <= 0 and in_standards:
            end_time = time[i]
            if start_time is not None:
                periods.append((pd.Timestamp(start_time), pd.Timestamp(end_time)))
            in_standards = False
            start_time = None
    
    # Handle case where data ends during standards
    if in_standards and start_time is not None:
        periods.append((pd.Timestamp(start_time), pd.Timestamp(time[-1])))
    
    print(f"✅ Found {len(periods)} standards periods")
    
    return periods

def extract_standards_runs_simple(df, periods, memory_buffer_min=10):
    """Simple standards run extraction."""
    
    print(f"📊 Extracting standards runs (with {memory_buffer_min}min memory buffer)...")
    
    runs = []
    
    for i, (start_time, end_time) in enumerate(periods):
        # Apply memory buffer
        buffered_start = start_time + timedelta(minutes=memory_buffer_min)
        
        if buffered_start >= end_time:
            print(f"   Run {i}: Too short after buffer, skipping")
            continue
        
        # Extract data
        mask = (df['Time'] >= buffered_start) & (df['Time'] <= end_time)
        run_data = df[mask].copy()
        
        # Filter to standards only
        if 'WVISS_VOLTAGE' in run_data.columns:
            run_data = run_data[run_data['WVISS_VOLTAGE'] > 0].copy()
        
        if len(run_data) > 0:
            run_data['run_id'] = i
            run_data['start_time'] = buffered_start
            run_data['end_time'] = end_time
            run_data['duration_min'] = (end_time - buffered_start).total_seconds() / 60
            
            runs.append(run_data)
            
            print(f"   Run {i}: {len(run_data)} points, "
                  f"{buffered_start.strftime('%m/%d %H:%M')} to "
                  f"{end_time.strftime('%m/%d %H:%M')} "
                  f"({run_data['duration_min'].iloc[0]:.1f} min)")
    
    print(f"✅ Extracted {len(runs)} standards runs")
    return runs

def assess_run_quality_simple(run_data, h2o_col='H2O_ppm'):
    """Simple quality assessment with low mixing ratio weighting. Filters H2O < 400 ppm."""
    
    if h2o_col not in run_data.columns:
        return {'quality_score': 0, 'n_points': 0, 'h2o_range': 0}
    
    # Filter out H2O < 400 ppm following standards requirements
    h2o_all_values = run_data[h2o_col].dropna()
    h2o_values = h2o_all_values[h2o_all_values >= 400]
    
    if len(h2o_values) == 0:
        return {'quality_score': 0, 'n_points': 0, 'h2o_range': 0, 'note': 'No data ≥ 400 ppm'}
    
    # Calculate metrics
    n_points = len(h2o_values)
    h2o_min = h2o_values.min()
    h2o_max = h2o_values.max()
    h2o_range = h2o_max - h2o_min
    duration = run_data['duration_min'].iloc[0] if 'duration_min' in run_data.columns else 0
    
    # Calculate low mixing ratio coverage
    low_h2o_points = (h2o_values < 2000).sum()  # Points below 2000 ppm
    very_low_h2o_points = (h2o_values < 1000).sum()  # Points below 1000 ppm
    low_coverage_fraction = low_h2o_points / n_points if n_points > 0 else 0
    
    # Simple scoring
    score = 0
    score += min(duration / 60, 2) * 15  # Duration score (reduced weight)
    score += min(h2o_range / 5000, 1) * 20  # Range score (reduced weight)
    score += min(h2o_min / 1000, 1) * 15  # Min humidity score (reduced weight)
    score += min(n_points / 1000, 1) * 15  # Data points score (reduced weight)
    
    # NEW: Low mixing ratio bonus (major emphasis)
    score += low_coverage_fraction * 25  # Bonus for fraction of low H2O points
    score += min(very_low_h2o_points / 100, 1) * 20  # Extra bonus for very low H2O points
    
    # NEW: Minimum humidity bonus (inverse relationship - lower is better)
    if h2o_min < 1500:
        min_bonus = max(0, (1500 - h2o_min) / 1000 * 15)  # Up to 15 point bonus
        score += min_bonus
    
    # Penalties
    if h2o_min < 300:  # Very dry air penalty (too extreme)
        score -= 20
    elif h2o_min < 500:  # Slightly dry air penalty (reduced)
        score -= 5
    
    # Check for decreasing trend (stepping pattern bonus)
    if len(h2o_values) > 10:
        x = np.arange(len(h2o_values))
        slope = np.polyfit(x, h2o_values, 1)[0]
        if slope <= 0:  # Non-increasing (good stepping pattern)
            score += 10
    
    return {
        'quality_score': score,
        'n_points': n_points,
        'h2o_range': h2o_range,
        'h2o_min': h2o_min,
        'h2o_max': h2o_max,
        'duration_min': duration,
        'low_h2o_points': low_h2o_points,
        'very_low_h2o_points': very_low_h2o_points,
        'low_coverage_fraction': low_coverage_fraction
    }

def load_single_csv_data(data_dir):
    """Load data from a single CSV file in the raw data directory."""
    
    print(f"🔍 Looking for CSV files in: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return None
    
    # Look for CSV files
    csv_files = list(Path(data_dir).glob("*.csv"))
    
    if not csv_files:
        print(f"❌ No .csv files found in {data_dir}")
        return None
    
    if len(csv_files) == 1:
        csv_file = csv_files[0]
        print(f"📁 Found single CSV file: {csv_file.name}")
    else:
        print(f"📁 Found {len(csv_files)} CSV files:")
        for f in csv_files:
            print(f"   - {f.name}")
        
        # Use the first one or let user choose
        csv_file = csv_files[0]
        print(f"   Using: {csv_file.name}")
    
    try:
        print(f"📊 Loading: {csv_file.name}")
        
        # Read CSV 
        df = pd.read_csv(csv_file)
        
        print(f"   Initial shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Convert time column if it exists
        time_columns = ['Time', 'time', 'DateTime', 'datetime']
        time_col = None
        for col in time_columns:
            if col in df.columns:
                time_col = col
                break
        
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])
            print(f"   ✅ Converted {time_col} to datetime")
        
        # Convert WVISS_VOLTAGE if it exists
        if 'WVISS_VOLTAGE' in df.columns:
            df['WVISS_VOLTAGE'] = pd.to_numeric(df['WVISS_VOLTAGE'], errors='coerce')
            print(f"   ✅ Converted WVISS_VOLTAGE to numeric")
        
        # Remove NaN rows
        initial_len = len(df)
        df = df.dropna()
        
        if len(df) < initial_len:
            print(f"   Removed {initial_len - len(df)} NaN rows")
        
        print(f"✅ Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
        
        # Show voltage distribution if available
        if 'WVISS_VOLTAGE' in df.columns:
            n_standards = (df['WVISS_VOLTAGE'] > 0).sum()
            n_measurements = (df['WVISS_VOLTAGE'] == 0).sum()
            print(f"   Standards points: {n_standards}")
            print(f"   Measurement points: {n_measurements}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading {csv_file.name}: {e}")
        return None

def find_data_directories():
    """Find available data directories with isotope data."""
    project_root = Path(__file__).parent.parent
    
    # Possible data directories to check
    candidate_dirs = [
        project_root / "data" / "raw",
        Path(os.getenv('ISOTOPE_DATA_ROOT', project_root / "data" / "external")),
        Path.home() / "EPS Masters" / "Research" / "Data" / "isodata",
        Path.home() / "EPS Masters" / "Research" / "Data" / "cleaned data"
    ]
    
    available_dirs = []
    
    for data_dir in candidate_dirs:
        if data_dir.exists():
            # Check for isotope data files
            txt_files = list(data_dir.glob("*.txt"))
            csv_files = list(data_dir.glob("*.csv"))
            if txt_files or csv_files:
                available_dirs.append(data_dir)
    
    return available_dirs

def test_calibration_system():
    """Main test function."""
    
    print("🧪 SIMPLE ISOTOPE CALIBRATION TEST")
    print("=" * 50)
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "outputs" / "simple_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / "simple_test.log"
    logger = setup_logging_simple(log_file)
    
    print(f"📁 Output directory: {output_dir}")
    print(f"📝 Log file: {log_file}")
    
    # Find available data directories
    data_directories = find_data_directories()
    
    if not data_directories:
        print("❌ No isotope data directories found!")
        print("   Please ensure data is in 'data/raw/' or set ISOTOPE_DATA_ROOT environment variable")
        return
    
    print(f"📊 Found {len(data_directories)} data directories:")
    for i, dir_path in enumerate(data_directories):
        print(f"   {i+1}. {dir_path}")
    
    combined_data = None
    
    print(f"\n1️⃣ Loading isotope data...")
    
    for data_dir in data_directories:
        print(f"\n🔍 Trying directory: {data_dir}")
        
        # Try CSV first (your new single file)
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
    
    # Save combined data
    combined_file = output_dir / "combined_data.csv"
    combined_data.to_csv(combined_file, index=False)
    print(f"💾 Combined data saved to: {combined_file}")
    
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
    
    print(f"\n4️⃣ Assessing run quality...")
    quality_results = []
    
    for i, run_data in enumerate(runs):
        quality = assess_run_quality_simple(run_data)
        quality['run_id'] = i
        
        if 'start_time' in run_data.columns:
            quality['date_str'] = run_data['start_time'].iloc[0].strftime('%m/%d')
        else:
            quality['date_str'] = 'unknown'
        
        quality_results.append(quality)
        
        print(f"   Run {i} ({quality['date_str']}): "
              f"Score={quality['quality_score']:.1f}, "
              f"Points={quality['n_points']}, "
              f"Range={quality['h2o_range']:.0f}ppm")
    
    # Sort by quality
    quality_results.sort(key=lambda x: x['quality_score'], reverse=True)
    
    # Filter runs with score > 100
    high_quality_runs = [result for result in quality_results if result['quality_score'] > 100]
    
    print(f"\n📊 HIGH QUALITY STANDARDS RUNS (Score > 100):")
    if high_quality_runs:
        for i, result in enumerate(high_quality_runs):
            print(f"   {i+1}. Run {result['run_id']} ({result['date_str']}): "
                  f"Score={result['quality_score']:.1f}")
        print(f"\n✅ Found {len(high_quality_runs)} runs with score > 100")
    else:
        print("   ⚠️  No runs found with score > 100")
        print("   📊 Top 5 runs by score:")
        for i, result in enumerate(quality_results[:5]):
            print(f"      {i+1}. Run {result['run_id']} ({result['date_str']}): "
                  f"Score={result['quality_score']:.1f}")
    
    # Save results
    results_df = pd.DataFrame(quality_results)
    results_file = output_dir / "quality_assessment.csv"
    results_df.to_csv(results_file, index=False)
    
    # Save individual run data files for inspection
    for i, run_data in enumerate(runs):
        run_file = output_dir / f"run_{i:02d}_{run_data['start_time'].iloc[0].strftime('%m%d_%H%M')}.csv"
        run_data.to_csv(run_file, index=False)
    
    print(f"\n✅ TEST COMPLETED SUCCESSFULLY!")
    print(f"📁 All outputs saved to: {output_dir}")
    print(f"📊 Quality assessment: {results_file}")
    print(f"🔍 Individual run files saved for inspection")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"1. Review the quality assessment results")
    print(f"2. Inspect individual run files to verify data quality")
    print(f"3. Create known_values.csv with your reference isotope values")
    print(f"4. Run full calibration workflow")

if __name__ == "__main__":
    test_calibration_system()