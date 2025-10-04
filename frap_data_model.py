"""
FRAP Single-Cell Data Model
Defines data structures and I/O for single-cell FRAP analysis
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ROITrace:
    """Per-frame ROI measurements"""
    exp_id: str
    movie_id: str
    cell_id: int
    frame: int
    t: float
    x: float
    y: float
    radius: float
    signal_raw: float
    signal_bg: float
    signal_corr: float
    signal_norm: float = np.nan
    qc_motion: bool = False
    qc_reason: str = ""


@dataclass
class CellFeatures:
    """Per-cell derived features and fit parameters"""
    exp_id: str
    movie_id: str
    cell_id: int
    pre_bleach: float
    I0: float
    I_inf: float
    k: float
    t_half: float
    mobile_frac: float
    r2: float
    sse: float
    drift_px: float
    bleach_qc: bool
    roi_method: str
    outlier: bool = False
    cluster: int = -1
    cluster_prob: float = 0.0
    # Additional fit metrics
    A: float = np.nan
    B: float = np.nan
    fit_method: str = "1exp"
    aic: float = np.nan
    bic: float = np.nan


class DataIO:
    """Handles persistence of roi_traces and cell_features tables"""
    
    @staticmethod
    def save_tables(
        roi_traces: pd.DataFrame,
        cell_features: pd.DataFrame,
        output_dir: str | Path,
        format: str = "parquet"
    ) -> tuple[Path, Path]:
        """
        Save roi_traces and cell_features to disk
        
        Parameters
        ----------
        roi_traces : pd.DataFrame
            Long-form per-frame table
        cell_features : pd.DataFrame
            Per-cell features table
        output_dir : str | Path
            Output directory
        format : str
            'parquet', 'csv', or 'both'
            
        Returns
        -------
        tuple[Path, Path]
            Paths to saved roi_traces and cell_features files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        traces_path = None
        features_path = None
        
        if format in ("parquet", "both"):
            traces_path = output_dir / "roi_traces.parquet"
            features_path = output_dir / "cell_features.parquet"
            roi_traces.to_parquet(traces_path, index=False, engine="pyarrow")
            cell_features.to_parquet(features_path, index=False, engine="pyarrow")
            logger.info(f"Saved parquet files to {output_dir}")
            
        if format in ("csv", "both"):
            csv_traces = output_dir / "roi_traces.csv"
            csv_features = output_dir / "cell_features.csv"
            roi_traces.to_csv(csv_traces, index=False)
            cell_features.to_csv(csv_features, index=False)
            logger.info(f"Saved CSV files to {output_dir}")
            if traces_path is None:
                traces_path = csv_traces
                features_path = csv_features
                
        return traces_path, features_path
    
    @staticmethod
    def load_tables(
        input_dir: str | Path,
        format: str = "parquet"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load roi_traces and cell_features from disk
        
        Parameters
        ----------
        input_dir : str | Path
            Input directory
        format : str
            'parquet' or 'csv'
            
        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            roi_traces and cell_features DataFrames
        """
        input_dir = Path(input_dir)
        
        if format == "parquet":
            traces_path = input_dir / "roi_traces.parquet"
            features_path = input_dir / "cell_features.parquet"
            roi_traces = pd.read_parquet(traces_path)
            cell_features = pd.read_parquet(features_path)
        elif format == "csv":
            traces_path = input_dir / "roi_traces.csv"
            features_path = input_dir / "cell_features.csv"
            roi_traces = pd.read_csv(traces_path)
            cell_features = pd.read_csv(features_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Loaded {len(roi_traces)} traces and {len(cell_features)} cells from {input_dir}")
        return roi_traces, cell_features
    
    @staticmethod
    def roi_traces_to_dataframe(traces: list[ROITrace]) -> pd.DataFrame:
        """Convert list of ROITrace objects to DataFrame"""
        if not traces:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=[
                'exp_id', 'movie_id', 'cell_id', 'frame', 't', 'x', 'y', 'radius',
                'signal_raw', 'signal_bg', 'signal_corr', 'signal_norm',
                'qc_motion', 'qc_reason'
            ])
        
        data = []
        for trace in traces:
            data.append({
                'exp_id': trace.exp_id,
                'movie_id': trace.movie_id,
                'cell_id': trace.cell_id,
                'frame': trace.frame,
                't': trace.t,
                'x': trace.x,
                'y': trace.y,
                'radius': trace.radius,
                'signal_raw': trace.signal_raw,
                'signal_bg': trace.signal_bg,
                'signal_corr': trace.signal_corr,
                'signal_norm': trace.signal_norm,
                'qc_motion': trace.qc_motion,
                'qc_reason': trace.qc_reason
            })
        return pd.DataFrame(data)
    
    @staticmethod
    def cell_features_to_dataframe(features: list[CellFeatures]) -> pd.DataFrame:
        """Convert list of CellFeatures objects to DataFrame"""
        if not features:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=[
                'exp_id', 'movie_id', 'cell_id', 'pre_bleach', 'I0', 'I_inf', 'k',
                't_half', 'mobile_frac', 'r2', 'sse', 'drift_px', 'bleach_qc',
                'roi_method', 'outlier', 'cluster', 'cluster_prob', 'A', 'B',
                'fit_method', 'aic', 'bic'
            ])
        
        data = []
        for feat in features:
            data.append({
                'exp_id': feat.exp_id,
                'movie_id': feat.movie_id,
                'cell_id': feat.cell_id,
                'pre_bleach': feat.pre_bleach,
                'I0': feat.I0,
                'I_inf': feat.I_inf,
                'k': feat.k,
                't_half': feat.t_half,
                'mobile_frac': feat.mobile_frac,
                'r2': feat.r2,
                'sse': feat.sse,
                'drift_px': feat.drift_px,
                'bleach_qc': feat.bleach_qc,
                'roi_method': feat.roi_method,
                'outlier': feat.outlier,
                'cluster': feat.cluster,
                'cluster_prob': feat.cluster_prob,
                'A': feat.A,
                'B': feat.B,
                'fit_method': feat.fit_method,
                'aic': feat.aic,
                'bic': feat.bic
            })
        return pd.DataFrame(data)


def validate_roi_traces(df: pd.DataFrame) -> bool:
    """Validate roi_traces DataFrame schema"""
    required_cols = [
        'exp_id', 'movie_id', 'cell_id', 'frame', 't', 'x', 'y', 'radius',
        'signal_raw', 'signal_bg', 'signal_corr', 'signal_norm',
        'qc_motion', 'qc_reason'
    ]
    
    missing = set(required_cols) - set(df.columns)
    if missing:
        logger.error(f"Missing columns in roi_traces: {missing}")
        return False
    return True


def validate_cell_features(df: pd.DataFrame) -> bool:
    """Validate cell_features DataFrame schema"""
    required_cols = [
        'exp_id', 'movie_id', 'cell_id', 'pre_bleach', 'I0', 'I_inf', 'k',
        't_half', 'mobile_frac', 'r2', 'sse', 'drift_px', 'bleach_qc',
        'roi_method', 'outlier', 'cluster', 'cluster_prob'
    ]
    
    missing = set(required_cols) - set(df.columns)
    if missing:
        logger.error(f"Missing columns in cell_features: {missing}")
        return False
    return True
