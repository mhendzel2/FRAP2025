"""
FRAP Signal Extraction and Normalization Module
Robust background subtraction and signal processing
"""
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def extract_signal_with_background(
    img: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    bg_inner_offset: float = 5.0,
    bg_outer_offset: float = 10.0,
    method: str = "median"
) -> tuple[float, float, float]:
    """
    Extract signal and background from ROI
    
    Parameters
    ----------
    img : np.ndarray
        2D image
    cx, cy : float
        ROI center
    radius : float
        ROI radius
    bg_inner_offset : float
        Inner radius offset for background annulus
    bg_outer_offset : float
        Outer radius offset for background annulus
    method : str
        'median' or 'mean' for background estimation
        
    Returns
    -------
    tuple[float, float, float]
        (signal_raw, signal_bg, signal_corr)
    """
    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    
    # ROI mask
    roi_mask = dist <= radius
    
    # Background annulus mask
    bg_inner = radius + bg_inner_offset
    bg_outer = radius + bg_outer_offset
    bg_mask = (dist > bg_inner) & (dist <= bg_outer)
    
    # Extract raw signal
    if not roi_mask.any():
        logger.warning(f"Empty ROI at ({cx:.1f}, {cy:.1f})")
        return np.nan, np.nan, np.nan
    
    signal_raw = np.mean(img[roi_mask])
    
    # Extract background with outlier rejection
    if not bg_mask.any():
        logger.warning(f"Empty background annulus at ({cx:.1f}, {cy:.1f})")
        signal_bg = 0.0
    else:
        bg_pixels = img[bg_mask]
        
        # Robust background estimation
        if method == "median":
            signal_bg = np.median(bg_pixels)
        elif method == "mean":
            # Remove outliers using IQR
            q1, q3 = np.percentile(bg_pixels, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            bg_filtered = bg_pixels[(bg_pixels >= lower) & (bg_pixels <= upper)]
            
            if len(bg_filtered) > 0:
                signal_bg = np.mean(bg_filtered)
            else:
                signal_bg = np.median(bg_pixels)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # Corrected signal
    signal_corr = signal_raw - signal_bg
    
    return signal_raw, signal_bg, signal_corr


def normalize_traces(
    traces: np.ndarray,
    I0: float,
    I_inf: float,
    method: str = "standard"
) -> np.ndarray:
    """
    Normalize recovery traces
    
    Parameters
    ----------
    traces : np.ndarray
        Signal traces (can be 1D or 2D)
    I0 : float
        Intensity immediately after bleach
    I_inf : float
        Plateau intensity
    method : str
        'standard': (I - I0) / (I_inf - I0)
        'full': (I - I0) / (pre_bleach - I0)
        
    Returns
    -------
    np.ndarray
        Normalized traces
    """
    if method == "standard":
        denominator = I_inf - I0
    elif method == "full":
        # Would need pre_bleach as parameter
        raise NotImplementedError("Full normalization requires pre_bleach parameter")
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if abs(denominator) < 1e-10:
        logger.warning("Near-zero denominator in normalization")
        return np.full_like(traces, np.nan)
    
    normalized = (traces - I0) / denominator
    return normalized


def compute_pre_bleach_intensity(
    intensities: np.ndarray,
    bleach_frame: int,
    n_frames: int = 5
) -> float:
    """
    Compute average pre-bleach intensity
    
    Parameters
    ----------
    intensities : np.ndarray
        Intensity time series
    bleach_frame : int
        Index of bleach frame
    n_frames : int
        Number of frames to average before bleach
        
    Returns
    -------
    float
        Average pre-bleach intensity
    """
    start_frame = max(0, bleach_frame - n_frames)
    end_frame = bleach_frame
    
    if end_frame <= start_frame:
        logger.warning("Insufficient pre-bleach frames")
        return intensities[0] if len(intensities) > 0 else np.nan
    
    pre_bleach = intensities[start_frame:end_frame]
    return np.mean(pre_bleach)


def find_bleach_frame(intensities: np.ndarray, method: str = "min") -> int:
    """
    Find bleach frame index
    
    Parameters
    ----------
    intensities : np.ndarray
        Intensity time series
    method : str
        'min': frame with minimum intensity
        'derivative': largest negative derivative
        
    Returns
    -------
    int
        Bleach frame index
    """
    if method == "min":
        return int(np.argmin(intensities))
    
    elif method == "derivative":
        # Compute first derivative
        diff = np.diff(intensities)
        # Find largest drop
        return int(np.argmin(diff)) + 1
    
    else:
        raise ValueError(f"Unknown method: {method}")


def extract_roi_circle(
    img: np.ndarray,
    cx: float,
    cy: float,
    radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract pixels within circular ROI
    
    Parameters
    ----------
    img : np.ndarray
        2D image
    cx, cy : float
        Center coordinates
    radius : float
        ROI radius
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (pixel_values, mask)
    """
    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = dist <= radius
    
    return img[mask], mask


def compute_snr(
    signal: float,
    background: float,
    bg_std: Optional[float] = None
) -> float:
    """
    Compute signal-to-noise ratio
    
    Parameters
    ----------
    signal : float
        Signal intensity
    background : float
        Background intensity
    bg_std : float, optional
        Background standard deviation (if known)
        
    Returns
    -------
    float
        SNR
    """
    signal_above_bg = signal - background
    
    if bg_std is None or bg_std == 0:
        # Assume Poisson noise
        bg_std = np.sqrt(max(background, 1))
    
    snr = signal_above_bg / bg_std
    return snr


def rolling_baseline_correction(
    intensities: np.ndarray,
    window: int = 10,
    quantile: float = 0.1
) -> np.ndarray:
    """
    Apply rolling baseline correction
    
    Parameters
    ----------
    intensities : np.ndarray
        Intensity time series
    window : int
        Rolling window size
    quantile : float
        Quantile for baseline estimation
        
    Returns
    -------
    np.ndarray
        Baseline-corrected intensities
    """
    baseline = np.zeros_like(intensities)
    half_window = window // 2
    
    for i in range(len(intensities)):
        start = max(0, i - half_window)
        end = min(len(intensities), i + half_window + 1)
        baseline[i] = np.quantile(intensities[start:end], quantile)
    
    corrected = intensities - baseline
    return corrected


def photobleaching_correction(
    intensities: np.ndarray,
    time: np.ndarray,
    method: str = "exponential"
) -> np.ndarray:
    """
    Correct for imaging photobleaching
    
    Parameters
    ----------
    intensities : np.ndarray
        Intensity time series
    time : np.ndarray
        Time points
    method : str
        'exponential': fit single exponential
        'linear': linear correction
        
    Returns
    -------
    np.ndarray
        Corrected intensities
    """
    if method == "linear":
        # Simple linear correction
        slope, intercept = np.polyfit(time, intensities, 1)
        trend = slope * time + intercept
        corrected = intensities - trend + intercept
        return corrected
    
    elif method == "exponential":
        from scipy.optimize import curve_fit
        
        def exp_decay(t, A, k, offset):
            return A * np.exp(-k * t) + offset
        
        try:
            popt, _ = curve_fit(
                exp_decay,
                time,
                intensities,
                p0=[intensities[0], 0.01, intensities[-1]],
                maxfev=1000
            )
            trend = exp_decay(time, *popt)
            corrected = intensities - trend + popt[2]
            return corrected
        except RuntimeError:
            logger.warning("Exponential fit failed, using linear correction")
            return photobleaching_correction(intensities, time, method="linear")
    
    else:
        raise ValueError(f"Unknown method: {method}")


def detect_motion_artifacts(
    positions: np.ndarray,
    threshold_px: float = 5.0
) -> tuple[np.ndarray, dict]:
    """
    Detect motion artifacts in ROI trajectory
    
    Parameters
    ----------
    positions : np.ndarray
        (N, 2) array of (x, y) positions
    threshold_px : float
        Threshold for flagging sudden jumps
        
    Returns
    -------
    tuple[np.ndarray, dict]
        Boolean array of artifact frames and summary statistics
    """
    # Compute frame-to-frame displacement
    displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    
    # Detect outliers
    median_disp = np.median(displacements)
    mad = np.median(np.abs(displacements - median_disp))
    threshold = median_disp + 5 * mad
    
    # Also use absolute threshold
    artifacts = displacements > max(threshold, threshold_px)
    
    # Pad to match original length
    artifacts = np.concatenate([[False], artifacts])
    
    stats = {
        'n_artifacts': artifacts.sum(),
        'mean_displacement': np.mean(displacements),
        'max_displacement': np.max(displacements),
        'median_displacement': median_disp
    }
    
    return artifacts, stats
