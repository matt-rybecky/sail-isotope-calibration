#!/usr/bin/env python3
"""
VSMOW-SLAP Scale Calibration Module

This module contains functions for loading and applying VSMOW-SLAP scale calibrations
based on linear fits between humidity-corrected measurements and known standard values.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Callable, Tuple

class VSMOWCalibrator:
    """
    Apply VSMOW-SLAP scale calibration using saved linear calibrations.
    """
    
    def __init__(self, calibration_file: Path):
        """
        Initialize calibrator with calibration file.
        
        Parameters:
        -----------
        calibration_file : Path
            Path to VSMOW-SLAP calibration JSON file
        """
        self.calibration_file = Path(calibration_file)
        self.calibration_functions = {}
        self.calibration_metadata = {}
        self.load_calibration()
    
    def load_calibration(self) -> None:
        """Load calibration functions from file."""
        
        with open(self.calibration_file, 'r') as f:
            calib_data = json.load(f)
        
        self.calibration_metadata = calib_data['metadata']
        
        # Create calibration functions for each isotope
        for isotope, calib_info in calib_data['calibrations'].items():
            slope = calib_info['slope']
            intercept = calib_info['intercept']
            
            # Create calibration function
            def make_calibration_function(s: float, i: float) -> Callable:
                def calibrate_to_vsmow(measured_values: np.ndarray) -> np.ndarray:
                    """Apply VSMOW-SLAP calibration to measured values."""
                    return s * np.asarray(measured_values) + i
                return calibrate_to_vsmow
            
            self.calibration_functions[isotope] = {
                'calibration_function': make_calibration_function(slope, intercept),
                'slope': slope,
                'intercept': intercept,
                'r_squared': calib_info['r_squared'],
                'rmse': calib_info['rmse'],
                'mae': calib_info['mae'],
                'n_points': calib_info['n_points']
            }
    
    def apply_calibration(self, data: pd.DataFrame,
                         dd_column: str = 'D_del_corrected',
                         d18o_column: str = 'O18_del_corrected') -> pd.DataFrame:
        """
        Apply VSMOW-SLAP calibration to humidity-corrected data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with humidity-corrected isotope values
        dd_column : str
            δD column name (humidity-corrected)
        d18o_column : str
            δ18O column name (humidity-corrected)
            
        Returns:
        --------
        pd.DataFrame
            Data with VSMOW-calibrated isotope columns added
        """
        
        calibrated_data = data.copy()
        
        # Apply dD VSMOW calibration
        if 'dD' in self.calibration_functions and dd_column in data.columns:
            calibration_function = self.calibration_functions['dD']['calibration_function']
            
            dd_values = data[dd_column].values
            dd_vsmow = calibration_function(dd_values)
            calibrated_data[f'dD_VSMOW'] = dd_vsmow
        
        # Apply d18O VSMOW calibration
        if 'd18O' in self.calibration_functions and d18o_column in data.columns:
            calibration_function = self.calibration_functions['d18O']['calibration_function']
            
            d18o_values = data[d18o_column].values
            d18o_vsmow = calibration_function(d18o_values)
            calibrated_data[f'd18O_VSMOW'] = d18o_vsmow
        
        # Calculate d-excess if both isotopes are available
        if 'dD_VSMOW' in calibrated_data.columns and 'd18O_VSMOW' in calibrated_data.columns:
            calibrated_data['d_excess_VSMOW'] = calibrated_data['dD_VSMOW'] - 8 * calibrated_data['d18O_VSMOW']
        
        return calibrated_data
    
    def get_calibration_info(self) -> Dict:
        """
        Get calibration information and statistics.
        
        Returns:
        --------
        dict
            Calibration metadata and fit statistics
        """
        info = {
            'metadata': self.calibration_metadata,
            'isotopes': {}
        }
        
        for isotope, calib_data in self.calibration_functions.items():
            info['isotopes'][isotope] = {
                'slope': calib_data['slope'],
                'intercept': calib_data['intercept'],
                'r_squared': calib_data['r_squared'],
                'rmse': calib_data['rmse'],
                'mae': calib_data['mae'],
                'n_points': calib_data['n_points'],
                'equation': f'VSMOW = {calib_data["slope"]:.6f} * measured + {calib_data["intercept"]:.3f}'
            }
        
        return info

def load_vsmow_calibration(calibration_file: Path) -> VSMOWCalibrator:
    """
    Load VSMOW-SLAP calibration from file.
    
    Parameters:
    -----------
    calibration_file : Path
        Path to VSMOW-SLAP calibration JSON file
        
    Returns:
    --------
    VSMOWCalibrator
        Loaded VSMOW calibrator
    """
    return VSMOWCalibrator(calibration_file)