import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
import logging

logger = logging.getLogger(__name__)

def detect_heterogeneity_v2(df, feature_cols=['k', 'mobile_fraction'], 
                           sensitivity='normal', min_k=1, max_k=3):
    """
    Detects kinetic sub-populations using GMM with adjustable sensitivity.
    
    Args:
        df: DataFrame containing kinetic parameters.
        feature_cols: Columns to use for clustering (e.g., rate constant and mobile fraction).
        sensitivity: 'low', 'normal', 'high'. 
                     'low' requires a massive BIC improvement to add a cluster (less sensitive).
        min_k: Minimum number of populations.
        max_k: Maximum number of populations.
    
    Returns:
        labels: Array of cluster assignments (-1 for outliers/insufficient data).
        n_clusters: Number of clusters found.
        bic_score: The final BIC score.
    """
    # 1. Prepare Data
    data = df[feature_cols].dropna()
    if len(data) < 10:
        return np.full(len(df), -1), 1, np.inf
        
    X = data.values
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Define Sensitivity Thresholds (Delta BIC required to accept a new cluster)
    # Higher threshold = harder to add clusters = less sensitive
    thresholds = {
        'low': 25.0,     # Very strict, requires strong evidence for heterogeneity
        'normal': 10.0,  # Standard statistical convention
        'high': 2.0      # Loose, easily finds sub-populations
    }
    bic_threshold = thresholds.get(sensitivity, 10.0)
    
    # 3. Iterative Model Selection
    best_k = min_k
    best_bic = np.inf
    best_model = None
    
    # Fit the baseline (min_k)
    try:
        base_model = GaussianMixture(n_components=min_k, random_state=42, n_init=10)
        base_model.fit(X_scaled)
        best_bic = base_model.bic(X_scaled)
        best_model = base_model
        best_k = min_k
    except Exception as e:
        logger.error(f"Base clustering failed: {e}")
        return np.full(len(df), -1), 1, np.inf

    # Try adding components
    for k in range(min_k + 1, max_k + 1):
        try:
            model = GaussianMixture(n_components=k, random_state=42, n_init=10)
            model.fit(X_scaled)
            new_bic = model.bic(X_scaled)
            
            # CRITICAL CHECK: Does the new model improve BIC by more than the threshold?
            # Note: Lower BIC is better.
            if best_bic - new_bic > bic_threshold:
                best_bic = new_bic
                best_k = k
                best_model = model
            else:
                # If adding a cluster doesn't significantly help, stop.
                # This prevents "over-splitting" similar populations.
                break
        except Exception:
            continue
            
    # 4. Assign Labels
    labels = best_model.predict(X_scaled)
    
    # Map back to original dataframe indices (handling dropped NaNs)
    full_labels = np.full(len(df), -1)
    # data.index may not be a 0..N integer range; map to positional indices safely
    positions = df.index.get_indexer(data.index)
    full_labels[positions] = labels
    
    return full_labels, best_k, best_bic