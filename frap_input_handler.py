import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

@dataclass
class FRAPCurveData:
    """
    Holds data for a single FRAP curve.
    """
    time: np.ndarray
    roi_intensity: np.ndarray
    ref_intensity: np.ndarray
    background_intensity: np.ndarray
    metadata: dict
    
    # Processed data
    normalized_intensity: Optional[np.ndarray] = None
    time_post_bleach: Optional[np.ndarray] = None
    intensity_post_bleach: Optional[np.ndarray] = None
    
    # Time Zero Correction
    f_zero_comp: Optional[float] = None
    
    def __post_init__(self):
        # Ensure arrays are numpy arrays
        self.time = np.array(self.time)
        self.roi_intensity = np.array(self.roi_intensity)
        self.ref_intensity = np.array(self.ref_intensity)
        self.background_intensity = np.array(self.background_intensity)

class FRAPInputHandler:
    """
    Handles data input, standardization, and initial preprocessing.
    """
    
    @staticmethod
    def load_csv(filepath: str, metadata: Optional[dict] = None) -> FRAPCurveData:
        """
        Loads FRAP data from a CSV file.
        Expected columns: 'Time', 'ROI', 'Reference', 'Background'
        """
        try:
            df = pd.read_csv(filepath)
            
            # Map columns (flexible mapping can be added later)
            # For now, assume standard names or map common variations
            col_map = {
                'Time': ['Time', 'time', 't'],
                'ROI': ['ROI', 'roi', 'Bleach', 'bleach'],
                'Reference': ['Reference', 'ref', 'Ref', 'Control'],
                'Background': ['Background', 'bg', 'BG', 'background']
            }
            
            mapped_cols = {}
            for key, candidates in col_map.items():
                found = False
                for cand in candidates:
                    if cand in df.columns:
                        mapped_cols[key] = cand
                        found = True
                        break
                if not found:
                    raise ValueError(f"Column for '{key}' not found in CSV. Expected one of: {candidates}")
            
            return FRAPCurveData(
                time=df[mapped_cols['Time']].values,
                roi_intensity=df[mapped_cols['ROI']].values,
                ref_intensity=df[mapped_cols['Reference']].values,
                background_intensity=df[mapped_cols['Background']].values,
                metadata=metadata or {}
            )
            
        except Exception as e:
            logger.error(f"Error loading CSV {filepath}: {e}")
            raise

    @staticmethod
    def double_normalization(data: FRAPCurveData, bleach_frame_idx: int, pre_bleach_frames: int = 5) -> FRAPCurveData:
        """
        Implements double normalization:
        F_norm(t) = (F_ROI(t) / F_prebleach_ROI) * (F_ref(prebleach) / F_ref(t))
        
        Also performs background subtraction first.
        """
        # Background subtraction
        roi_bg_corr = data.roi_intensity - data.background_intensity
        ref_bg_corr = data.ref_intensity - data.background_intensity
        
        # Avoid division by zero or negative values
        roi_bg_corr = np.maximum(roi_bg_corr, 1e-6)
        ref_bg_corr = np.maximum(ref_bg_corr, 1e-6)
        
        # Calculate pre-bleach averages
        # Assuming bleach_frame_idx is the index of the first frame AFTER bleach or the bleach frame itself.
        # Usually pre-bleach is 0 to bleach_frame_idx (exclusive)
        
        if bleach_frame_idx < 1:
             raise ValueError("Bleach frame index must be > 0 to have pre-bleach data.")
             
        # Use the specified number of pre-bleach frames, or all available if fewer
        start_frame = max(0, bleach_frame_idx - pre_bleach_frames)
        
        f_pre_roi = np.mean(roi_bg_corr[start_frame:bleach_frame_idx])
        f_pre_ref = np.mean(ref_bg_corr[start_frame:bleach_frame_idx])
        
        # Double Normalization
        # F_norm(t) = (F_ROI(t) / F_pre_roi) * (F_pre_ref / F_ref(t))
        
        norm_intensity = (roi_bg_corr / f_pre_roi) * (f_pre_ref / ref_bg_corr)
        
        data.normalized_intensity = norm_intensity
        return data

    @staticmethod
    def time_zero_correction(data: FRAPCurveData, bleach_frame_idx: int, initial_points: int = 5) -> FRAPCurveData:
        """
        Corrects for initial rapid recovery.
        1. Identify initial phase of recovery.
        2. Back-extrapolate to t=0 (bleach time).
        3. Determine F(0)_comp.
        """
        if data.normalized_intensity is None:
            raise ValueError("Data must be normalized before Time Zero Correction.")
            
        time = data.time
        intensity = data.normalized_intensity
        
        # Post-bleach data
        # Assuming bleach_frame_idx is the frame where intensity is minimum (bleach event)
        # Recovery starts immediately after
        
        t_bleach = time[bleach_frame_idx]
        
        # Select initial recovery points for extrapolation
        # Start from the point AFTER the minimum (bleach) to avoid the bleach artifact itself if it's deep
        start_fit = bleach_frame_idx + 1
        end_fit = min(len(time), start_fit + initial_points)
        
        if end_fit <= start_fit:
             raise ValueError("Not enough points for Time Zero extrapolation.")

        t_fit = time[start_fit:end_fit]
        y_fit = intensity[start_fit:end_fit]
        
        # Simple linear extrapolation for the very initial phase is often robust enough for F(0)
        # Or use a simple exponential if curvature is high. 
        # The prompt suggests "simple exponential, polynomial, or initial phase of theoretical diffusion model"
        # Let's use a simple linear fit for the very first few points as it's most robust for "t=0" intercept 
        # if the points are close to t=0. If they are further, exponential is better.
        # Let's try linear first as it's standard for "back-extrapolation" of the immediate recovery.
        
        # Using linear regression: y = mx + c
        slope, intercept = np.polyfit(t_fit, y_fit, 1)
        
        # Extrapolate to t_bleach
        f_zero_comp = slope * t_bleach + intercept
        
        # Ensure F(0) is not lower than the actual measured minimum (which might be noise) 
        # or higher than the first recovery point.
        # Actually, F(0) CAN be lower than measured if acquisition was slow.
        # But it shouldn't be higher than the first recovery point.
        
        data.f_zero_comp = f_zero_comp
        
        # Set post-bleach data for fitting
        # We start from t=0 (relative to bleach)
        data.time_post_bleach = time[bleach_frame_idx:] - t_bleach
        data.intensity_post_bleach = intensity[bleach_frame_idx:]
        
        # Important: The first point in intensity_post_bleach is the measured bleach depth.
        # For fitting, we might want to replace it or enforce the curve to start at f_zero_comp.
        # The prompt says: "Mandate that F(0)_comp is used as the starting intensity (bleach depth) for all subsequent theoretical model fittings."
        
        return data
