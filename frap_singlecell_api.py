"""
FRAP Single-Cell Analysis API
Clean public API for single-cell FRAP analysis pipeline
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union
import logging

from frap_data_model import DataIO, ROITrace, CellFeatures
from frap_tracking import (
    seed_rois, fit_gaussian_centroid, ROIKalman, 
    track_optical_flow, adapt_radius, hungarian_assignment
)
from frap_signal import (
    extract_signal_with_background, compute_pre_bleach_intensity,
    find_bleach_frame, detect_motion_artifacts
)
from frap_fitting import (
    fit_with_model_selection, compute_mobile_fraction,
    fit_cells_parallel, FitResult
)
from frap_populations import detect_outliers_and_clusters, compute_cluster_statistics
from frap_statistics import multi_parameter_analysis

logger = logging.getLogger(__name__)


def track_movie(
    movie: np.ndarray,
    time_points: np.ndarray,
    exp_id: str,
    movie_id: str,
    initial_rois: Optional[list[tuple[float, float]]] = None,
    initial_radius: float = 10.0,
    use_kalman: bool = True,
    adapt_radius_flag: bool = True,
    dt: float = 1.0
) -> pd.DataFrame:
    """
    Track ROIs across entire movie
    
    Parameters
    ----------
    movie : np.ndarray
        3D array (T, H, W) of movie frames
    time_points : np.ndarray
        Time point for each frame
    exp_id : str
        Experiment ID
    movie_id : str
        Movie ID
    initial_rois : list[tuple[float, float]], optional
        Initial ROI positions, auto-detected if None
    initial_radius : float
        Initial ROI radius
    use_kalman : bool
        Use Kalman filtering
    adapt_radius_flag : bool
        Adapt radius over time
    dt : float
        Time step for Kalman filter
        
    Returns
    -------
    pd.DataFrame
        roi_traces table
    """
    logger.info(f"Tracking movie {movie_id} with {len(movie)} frames")
    
    # Detect initial ROIs if not provided
    if initial_rois is None:
        initial_rois = seed_rois(movie[0], max_rois=50)
    
    n_cells = len(initial_rois)
    logger.info(f"Tracking {n_cells} ROIs")
    
    # Initialize tracking structures
    current_positions = list(initial_rois)
    current_radii = [initial_radius] * n_cells
    kalman_filters = [ROIKalman(dt=dt) if use_kalman else None for _ in range(n_cells)]
    
    # Initialize Kalman filters with first positions
    if use_kalman:
        for i, (x, y) in enumerate(current_positions):
            kalman_filters[i].update(x, y)
    
    traces = []
    prev_frame = movie[0]
    
    # Track through movie
    for frame_idx, (frame, t) in enumerate(zip(movie, time_points)):
        logger.debug(f"Processing frame {frame_idx}/{len(movie)}")
        
        new_positions = []
        new_radii = []
        
        for cell_id, ((x, y), radius) in enumerate(zip(current_positions, current_radii)):
            # Fit Gaussian centroid
            x_gauss, y_gauss, mse_gauss = fit_gaussian_centroid(frame, (int(x), int(y)))
            
            # Try optical flow as fallback
            x_flow, y_flow, err_flow = track_optical_flow(prev_frame, frame, (x, y))
            
            # Choose best method
            if mse_gauss < err_flow:
                x_new, y_new = x_gauss, y_gauss
                roi_method = "gaussian"
            else:
                x_new, y_new = x_flow, y_flow
                roi_method = "optical_flow"
            
            # Apply Kalman smoothing
            if use_kalman and kalman_filters[cell_id] is not None:
                x_smooth, y_smooth = kalman_filters[cell_id].update(x_new, y_new)
                innovation = kalman_filters[cell_id].get_innovation_norm()
            else:
                x_smooth, y_smooth = x_new, y_new
                innovation = 0.0
            
            # Adapt radius
            if adapt_radius_flag:
                radius_new = adapt_radius(frame, x_smooth, y_smooth, radius)
            else:
                radius_new = radius
            
            # Extract signal
            signal_raw, signal_bg, signal_corr = extract_signal_with_background(
                frame, x_smooth, y_smooth, radius_new
            )
            
            # QC flags
            qc_motion = False
            qc_reason = ""
            
            if innovation > 10.0:  # Large innovation
                qc_motion = True
                qc_reason = "large_kalman_innovation"
            elif mse_gauss > 1000:  # Poor fit
                qc_motion = True
                qc_reason = "poor_gaussian_fit"
            
            # Store trace
            trace = ROITrace(
                exp_id=exp_id,
                movie_id=movie_id,
                cell_id=cell_id,
                frame=frame_idx,
                t=t,
                x=x_smooth,
                y=y_smooth,
                radius=radius_new,
                signal_raw=signal_raw,
                signal_bg=signal_bg,
                signal_corr=signal_corr,
                signal_norm=np.nan,  # Filled after fitting
                qc_motion=qc_motion,
                qc_reason=qc_reason
            )
            traces.append(trace)
            
            new_positions.append((x_smooth, y_smooth))
            new_radii.append(radius_new)
        
        current_positions = new_positions
        current_radii = new_radii
        prev_frame = frame
    
    # Convert to DataFrame
    df = DataIO.roi_traces_to_dataframe(traces)
    logger.info(f"Tracked {n_cells} cells across {len(movie)} frames")
    
    return df


def fit_cells(
    roi_traces: pd.DataFrame,
    pre_bleach_window: int = 5,
    robust: bool = True,
    try_2exp: bool = True,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Fit recovery curves for all cells
    
    Parameters
    ----------
    roi_traces : pd.DataFrame
        ROI traces table
    pre_bleach_window : int
        Number of frames for pre-bleach average
    robust : bool
        Use robust fitting
    try_2exp : bool
        Try 2-exponential model
    n_jobs : int
        Parallel jobs
        
    Returns
    -------
    pd.DataFrame
        cell_features table
    """
    logger.info("Fitting recovery curves")
    
    # Group by cell
    cell_features_list = []
    
    for (exp_id, movie_id, cell_id), group in roi_traces.groupby(['exp_id', 'movie_id', 'cell_id']):
        # Sort by time
        group = group.sort_values('t')
        
        # Find bleach frame
        intensities = group['signal_corr'].values
        times = group['t'].values
        
        bleach_frame = find_bleach_frame(intensities)
        
        # Compute pre-bleach
        pre_bleach = compute_pre_bleach_intensity(intensities, bleach_frame, pre_bleach_window)
        
        # Extract post-bleach data
        post_bleach_data = group.iloc[bleach_frame:].copy()
        t_post = post_bleach_data['t'].values - post_bleach_data['t'].values[0]
        y_post = post_bleach_data['signal_corr'].values
        
        if len(t_post) < 4:
            logger.warning(f"Insufficient recovery data for cell {cell_id}")
            continue
        
        # Fit
        fit = fit_with_model_selection(t_post, y_post, try_2exp=try_2exp, robust=robust)
        
        if not fit.success:
            logger.debug(f"Fit failed for cell {cell_id}: {fit.message}")
            continue
        
        # Compute mobile fraction
        mobile_frac = compute_mobile_fraction(fit.I0, fit.I_inf, pre_bleach)
        
        # Compute drift
        positions = group[['x', 'y']].values
        drift_px = np.linalg.norm(positions[-1] - positions[0])
        
        # QC
        bleach_qc = fit.r2 > 0.5 and 0 <= mobile_frac <= 1.0
        
        # Store features
        features = CellFeatures(
            exp_id=exp_id,
            movie_id=movie_id,
            cell_id=cell_id,
            pre_bleach=pre_bleach,
            I0=fit.I0,
            I_inf=fit.I_inf,
            k=fit.k,
            t_half=fit.t_half,
            mobile_frac=mobile_frac,
            r2=fit.r2,
            sse=fit.sse,
            drift_px=drift_px,
            bleach_qc=bleach_qc,
            roi_method="mixed",
            A=fit.A,
            B=fit.B,
            fit_method=fit.fit_method,
            aic=fit.aic,
            bic=fit.bic
        )
        cell_features_list.append(features)
    
    df = DataIO.cell_features_to_dataframe(cell_features_list)
    logger.info(f"Fitted {len(df)} cells")
    
    return df


def flag_motion(
    roi_traces: pd.DataFrame,
    threshold_px: float = 5.0
) -> pd.DataFrame:
    """
    Flag frames with motion artifacts
    
    Parameters
    ----------
    roi_traces : pd.DataFrame
        ROI traces table
    threshold_px : float
        Threshold for flagging motion
        
    Returns
    -------
    pd.DataFrame
        Updated roi_traces with QC flags
    """
    logger.info("Flagging motion artifacts")
    
    df = roi_traces.copy()
    
    for (exp_id, movie_id, cell_id), group in df.groupby(['exp_id', 'movie_id', 'cell_id']):
        positions = group[['x', 'y']].values
        artifacts, stats = detect_motion_artifacts(positions, threshold_px)
        
        # Update QC flags
        indices = group.index
        df.loc[indices, 'qc_motion'] = df.loc[indices, 'qc_motion'] | artifacts
        
        # Update reasons
        for idx, is_artifact in zip(indices, artifacts):
            if is_artifact and not df.loc[idx, 'qc_reason']:
                df.loc[idx, 'qc_reason'] = "motion_artifact"
    
    return df


def analyze(
    cell_features: pd.DataFrame,
    group_col: str = "condition",
    batch_col: str = "exp_id",
    params: Optional[list[str]] = None,
    n_bootstrap: int = 1000,
    random_state: int = 0
) -> dict:
    """
    Complete statistical analysis
    
    Parameters
    ----------
    cell_features : pd.DataFrame
        Cell features table with group and batch columns
    group_col : str
        Group/condition column
    batch_col : str
        Batch/experiment column
    params : list[str], optional
        Parameters to analyze
    n_bootstrap : int
        Bootstrap iterations
    random_state : int
        Random seed
        
    Returns
    -------
    dict
        Analysis results
    """
    if params is None:
        params = ['mobile_frac', 'k', 't_half', 'pre_bleach', 'r2']
    
    logger.info(f"Analyzing {len(params)} parameters across groups")
    
    # Multi-parameter analysis with FDR
    results_df = multi_parameter_analysis(
        cell_features,
        params,
        group_col,
        batch_col,
        n_bootstrap=n_bootstrap,
        random_state=random_state
    )
    
    # Cluster statistics
    if 'cluster' in cell_features.columns:
        cluster_stats = compute_cluster_statistics(cell_features, params)
    else:
        cluster_stats = pd.DataFrame()
    
    return {
        'comparisons': results_df,
        'cluster_stats': cluster_stats,
        'n_cells': len(cell_features),
        'groups': sorted(cell_features[group_col].unique()) if group_col in cell_features.columns else []
    }


# Convenience function for complete pipeline
def analyze_frap_movie(
    movie: np.ndarray,
    time_points: np.ndarray,
    exp_id: str,
    movie_id: str,
    condition: str,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> dict:
    """
    Complete end-to-end analysis of a FRAP movie
    
    Parameters
    ----------
    movie : np.ndarray
        Movie frames (T, H, W)
    time_points : np.ndarray
        Time points
    exp_id : str
        Experiment ID
    movie_id : str
        Movie ID
    condition : str
        Experimental condition
    output_dir : str or Path, optional
        Output directory for saving results
    **kwargs : dict
        Additional parameters for tracking and fitting
        
    Returns
    -------
    dict
        Complete analysis results including roi_traces, cell_features, statistics
    """
    logger.info(f"Starting complete analysis for {movie_id}")
    
    # Track
    roi_traces = track_movie(movie, time_points, exp_id, movie_id, **kwargs)
    
    # Flag motion
    roi_traces = flag_motion(roi_traces)
    
    # Fit
    cell_features = fit_cells(roi_traces, **kwargs)
    
    # Add condition
    cell_features['condition'] = condition
    
    # Population analysis
    cell_features = detect_outliers_and_clusters(cell_features)
    
    # Save if requested
    if output_dir is not None:
        output_dir = Path(output_dir)
        DataIO.save_tables(roi_traces, cell_features, output_dir, format="both")
        logger.info(f"Saved results to {output_dir}")
    
    return {
        'roi_traces': roi_traces,
        'cell_features': cell_features,
        'n_cells': len(cell_features),
        'n_outliers': cell_features['outlier'].sum(),
        'n_clusters': len(cell_features['cluster'].unique()) - (1 if -1 in cell_features['cluster'].values else 0)
    }
