"""
FRAP Populations Module
Outlier detection and subpopulation clustering
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def prepare_features(
    cell_features: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    exclude_qc_failed: bool = True
) -> tuple[np.ndarray, np.ndarray, RobustScaler]:
    """
    Prepare feature matrix for clustering/outlier detection
    
    Parameters
    ----------
    cell_features : pd.DataFrame
        Cell features table
    feature_cols : list[str], optional
        Columns to use as features
    exclude_qc_failed : bool
        Exclude cells that failed QC
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, RobustScaler]
        (scaled_features, valid_indices, scaler)
    """
    if feature_cols is None:
        feature_cols = ['mobile_frac', 'k', 't_half', 'pre_bleach', 'r2']
    
    # Filter valid cells
    df = cell_features.copy()
    
    if exclude_qc_failed and 'bleach_qc' in df.columns:
        df = df[df['bleach_qc'] == True]
    
    # Drop rows with NaN in feature columns
    valid_mask = df[feature_cols].notna().all(axis=1)
    df = df[valid_mask]
    
    if len(df) == 0:
        logger.warning("No valid cells after filtering")
        return np.array([]), np.array([]), RobustScaler()
    
    # Extract features
    X = df[feature_cols].values
    
    # Handle infinities
    X = np.where(np.isinf(X), np.nan, X)
    
    # Replace remaining NaNs with median
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    valid_indices = df.index.values
    
    logger.info(f"Prepared {X_scaled.shape[0]} cells with {X_scaled.shape[1]} features")
    
    return X_scaled, valid_indices, scaler


def flag_outliers(
    X: np.ndarray, 
    contamination: float = 0.07,
    random_state: int = 0
) -> np.ndarray:
    """
    Flag outliers using ensemble of methods
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (already scaled)
    contamination : float
        Expected proportion of outliers
    random_state : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Boolean array, True for outliers
    """
    if X.shape[0] < 10:
        logger.warning("Too few samples for outlier detection")
        return np.zeros(X.shape[0], dtype=bool)
    
    # Isolation Forest
    try:
        iso = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        outliers_iso = iso.fit_predict(X) == -1
    except Exception as e:
        logger.warning(f"Isolation Forest failed: {e}")
        outliers_iso = np.zeros(X.shape[0], dtype=bool)
    
    # Elliptic Envelope (requires enough samples)
    if X.shape[0] >= X.shape[1] * 2:
        try:
            ee = EllipticEnvelope(
                contamination=contamination,
                random_state=random_state
            )
            outliers_ee = ee.fit_predict(X) == -1
        except Exception as e:
            logger.warning(f"Elliptic Envelope failed: {e}")
            outliers_ee = np.zeros(X.shape[0], dtype=bool)
    else:
        outliers_ee = np.zeros(X.shape[0], dtype=bool)
    
    # Combine: flag as outlier if EITHER method flags it
    outliers = np.logical_or(outliers_iso, outliers_ee)
    
    logger.info(f"Flagged {outliers.sum()} outliers ({outliers.mean()*100:.1f}%)")
    
    return outliers


def gmm_clusters(
    X: np.ndarray, 
    max_k: int = 6,
    random_state: int = 0,
    min_k: int = 1
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Cluster using Gaussian Mixture Model with BIC selection
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (scaled)
    max_k : int
        Maximum number of clusters to try
    random_state : int
        Random seed
    min_k : int
        Minimum number of clusters
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        (labels, probabilities, best_k)
    """
    if X.shape[0] < 4:
        logger.warning("Too few samples for clustering")
        return np.zeros(X.shape[0], dtype=int), np.ones(X.shape[0]), 1
    
    # Limit max_k by sample size
    max_k = min(max_k, X.shape[0] // 2)
    max_k = max(max_k, min_k)
    
    # Try different numbers of components
    bic_scores = []
    models = []
    
    for k in range(min_k, max_k + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='full',
                random_state=random_state,
                n_init=10,
                max_iter=200
            )
            gmm.fit(X)
            bic = gmm.bic(X)
            bic_scores.append(bic)
            models.append(gmm)
        except Exception as e:
            logger.debug(f"GMM with k={k} failed: {e}")
            bic_scores.append(np.inf)
            models.append(None)
    
    # Select best k
    best_idx = np.argmin(bic_scores)
    best_k = min_k + best_idx
    best_model = models[best_idx]
    
    if best_model is None:
        logger.warning("All GMM fits failed")
        return np.zeros(X.shape[0], dtype=int), np.ones(X.shape[0]), 1
    
    logger.info(f"Selected k={best_k} clusters (BIC={bic_scores[best_idx]:.2f})")
    
    # Get labels and probabilities
    labels = best_model.predict(X)
    probs = best_model.predict_proba(X)
    max_probs = np.max(probs, axis=1)
    
    # If only 1 cluster, try DBSCAN as fallback
    if best_k == 1:
        logger.info("Only 1 cluster found, trying DBSCAN")
        return dbscan_fallback(X, random_state)
    
    return labels, max_probs, best_k


def dbscan_fallback(
    X: np.ndarray, 
    random_state: int = 0,
    eps: Optional[float] = None,
    min_samples: int = 5
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Cluster using DBSCAN as fallback
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    random_state : int
        Random seed (for consistency)
    eps : float, optional
        Neighborhood size (auto-determined if None)
    min_samples : int
        Minimum samples per cluster
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        (labels, probabilities, n_clusters)
    """
    # Auto-determine eps if not provided
    if eps is None:
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors.fit(X)
        distances, _ = neighbors.kneighbors(X)
        eps = np.percentile(distances[:, -1], 75)
    
    # Fit DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    logger.info(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")
    
    # Probability = 1 for clustered, 0 for noise
    probs = np.where(labels == -1, 0.0, 1.0)
    
    return labels, probs, n_clusters


def detect_outliers_and_clusters(
    cell_features: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    contamination: float = 0.07,
    max_k: int = 6,
    random_state: int = 0
) -> pd.DataFrame:
    """
    Complete outlier detection and clustering pipeline
    
    Parameters
    ----------
    cell_features : pd.DataFrame
        Cell features table
    feature_cols : list[str], optional
        Features to use
    contamination : float
        Expected outlier proportion
    max_k : int
        Maximum clusters to try
    random_state : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Updated cell_features with outlier, cluster, cluster_prob columns
    """
    df = cell_features.copy()
    
    # Initialize columns
    df['outlier'] = False
    df['cluster'] = -1
    df['cluster_prob'] = 0.0
    
    # Prepare features
    X, valid_indices, scaler = prepare_features(df, feature_cols)
    
    if len(X) == 0:
        logger.warning("No valid features, skipping population analysis")
        return df
    
    # Detect outliers
    outliers = flag_outliers(X, contamination, random_state)
    df.loc[valid_indices, 'outlier'] = outliers
    
    # Cluster on non-outliers
    X_clean = X[~outliers]
    valid_clean_indices = valid_indices[~outliers]
    
    if len(X_clean) >= 4:
        labels, probs, n_clusters = gmm_clusters(X_clean, max_k, random_state)
        df.loc[valid_clean_indices, 'cluster'] = labels
        df.loc[valid_clean_indices, 'cluster_prob'] = probs
        
        logger.info(f"Population analysis complete: {n_clusters} clusters, {outliers.sum()} outliers")
    else:
        logger.warning("Too few non-outlier samples for clustering")
    
    return df


def compute_cluster_statistics(
    cell_features: pd.DataFrame,
    param_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Compute per-cluster statistics
    
    Parameters
    ----------
    cell_features : pd.DataFrame
        Cell features with cluster assignments
    param_cols : list[str], optional
        Parameters to summarize
        
    Returns
    -------
    pd.DataFrame
        Cluster statistics
    """
    if param_cols is None:
        param_cols = ['mobile_frac', 'k', 't_half', 'pre_bleach', 'r2']
    
    # Group by cluster
    stats = []
    
    for cluster_id in sorted(cell_features['cluster'].unique()):
        if cluster_id == -1:
            continue
        
        cluster_data = cell_features[cell_features['cluster'] == cluster_id]
        
        stat_row = {'cluster': cluster_id, 'n_cells': len(cluster_data)}
        
        for param in param_cols:
            if param in cluster_data.columns:
                values = cluster_data[param].dropna()
                if len(values) > 0:
                    stat_row[f'{param}_mean'] = values.mean()
                    stat_row[f'{param}_std'] = values.std()
                    stat_row[f'{param}_median'] = values.median()
                    stat_row[f'{param}_q25'] = values.quantile(0.25)
                    stat_row[f'{param}_q75'] = values.quantile(0.75)
        
        stats.append(stat_row)
    
    return pd.DataFrame(stats)


def compute_separation_metrics(
    X: np.ndarray,
    labels: np.ndarray
) -> dict:
    """
    Compute cluster separation metrics
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    labels : np.ndarray
        Cluster labels
        
    Returns
    -------
    dict
        Separation metrics
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    # Filter out noise points (-1)
    mask = labels != -1
    X_clustered = X[mask]
    labels_clustered = labels[mask]
    
    if len(np.unique(labels_clustered)) < 2:
        return {
            'silhouette': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan
        }
    
    try:
        silhouette = silhouette_score(X_clustered, labels_clustered)
    except:
        silhouette = np.nan
    
    try:
        calinski = calinski_harabasz_score(X_clustered, labels_clustered)
    except:
        calinski = np.nan
    
    try:
        davies = davies_bouldin_score(X_clustered, labels_clustered)
    except:
        davies = np.nan
    
    return {
        'silhouette': silhouette,
        'calinski_harabasz': calinski,
        'davies_bouldin': davies,
        'n_clusters': len(np.unique(labels_clustered)),
        'n_noise': np.sum(labels == -1)
    }
