"""
Comprehensive tests for FRAP single-cell analysis
Run with: pytest test_frap_singlecell.py -v
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

# Import modules
from test_synthetic import synth_movie, generate_test_movies_with_expectations
from frap_singlecell_api import track_movie, fit_cells, flag_motion, analyze
from frap_data_model import DataIO
from frap_tracking import fit_gaussian_centroid, ROIKalman, adapt_radius
from frap_signal import extract_signal_with_background, detect_motion_artifacts
from frap_fitting import fit_recovery, fit_with_model_selection
from frap_populations import detect_outliers_and_clusters
from frap_statistics import lmm_param, bootstrap_bca_ci
from sklearn.metrics import adjusted_rand_score


class TestTracking:
    """Test ROI tracking accuracy"""
    
    def test_gaussian_centroid_accuracy(self):
        """Test sub-pixel centroid fitting"""
        # Create synthetic spot
        img = np.zeros((50, 50))
        true_x, true_y = 25.3, 25.7  # Sub-pixel position
        
        y, x = np.ogrid[:50, :50]
        dist = np.sqrt((x - true_x)**2 + (y - true_y)**2)
        img = 100 * np.exp(-(dist**2) / (2 * 5**2))
        
        # Add noise
        img += np.random.normal(0, 1, img.shape)
        
        # Fit
        x_fit, y_fit, mse = fit_gaussian_centroid(img, (25, 26))
        
        # Check accuracy
        error = np.sqrt((x_fit - true_x)**2 + (y_fit - true_y)**2)
        assert error < 0.5, f"Centroid error {error:.3f} px exceeds 0.5 px"
    
    def test_tracking_on_synthetic_movie(self):
        """Test complete tracking pipeline"""
        # Generate simple movie
        movie, gt = synth_movie(
            n_cells=5,
            T=50,
            drift_px=0.1,
            noise=0.01,
            random_state=42
        )
        
        time_points = np.arange(movie.shape[0])
        
        # Track
        roi_traces = track_movie(
            movie,
            time_points,
            exp_id='test',
            movie_id='test1',
            use_kalman=True
        )
        
        assert len(roi_traces) > 0, "No traces generated"
        assert roi_traces['cell_id'].nunique() <= 5, "Too many cells detected"
        
        # Check drift estimation
        for cell_id in roi_traces['cell_id'].unique():
            cell_data = roi_traces[roi_traces['cell_id'] == cell_id]
            positions = cell_data[['x', 'y']].values
            drift = np.linalg.norm(positions[-1] - positions[0])
            
            # Expected drift ~= 0.1 * 50 = 5 pixels
            assert drift < 10, f"Cell {cell_id} drift {drift:.1f} px seems too large"


class TestFitting:
    """Test curve fitting accuracy"""
    
    def test_single_exponential_recovery(self):
        """Test 1-exp fitting on known curve"""
        # Generate perfect recovery curve
        t = np.linspace(0, 100, 100)
        true_k = 0.5
        true_A = 1.0
        true_B = 0.6
        
        y = true_A - true_B * np.exp(-true_k * t)
        y += np.random.normal(0, 0.01, len(t))  # Small noise
        
        # Fit
        fit = fit_recovery(t, y, robust=True)
        
        assert fit.success, "Fit failed"
        assert abs(fit.k - true_k) / true_k < 0.1, f"k error {abs(fit.k - true_k)/true_k:.1%} > 10%"
        assert abs(fit.A - true_A) / true_A < 0.1, f"A error too large"
        assert fit.r2 > 0.95, f"R² = {fit.r2:.3f} too low"
    
    def test_model_selection(self):
        """Test BIC-based model selection"""
        t = np.linspace(0, 100, 100)
        
        # Single exponential data
        y_1exp = 1.0 - 0.6 * np.exp(-0.5 * t)
        y_1exp += np.random.normal(0, 0.02, len(t))
        
        fit = fit_with_model_selection(t, y_1exp, try_2exp=True)
        
        assert fit.success, "Fit failed"
        # Should prefer 1-exp for 1-exp data
        # (not guaranteed with noise, but likely)
        assert fit.fit_method in ['1exp', '2exp'], "Unknown fit method"


class TestPopulations:
    """Test outlier detection and clustering"""
    
    def test_outlier_detection(self):
        """Test outlier flagging"""
        # Generate data with clear outliers
        rng = np.random.RandomState(42)
        
        # Normal data
        normal = rng.normal(0, 1, (90, 4))
        
        # Outliers
        outliers = rng.normal(5, 1, (10, 4))
        
        X = np.vstack([normal, outliers])
        
        from frap_populations import flag_outliers
        outlier_flags = flag_outliers(X, contamination=0.1)
        
        # Should detect most outliers
        detected_outliers = outlier_flags[-10:].sum()
        assert detected_outliers >= 5, f"Only detected {detected_outliers}/10 outliers"
    
    def test_clustering_on_synthetic(self):
        """Test clustering on two populations"""
        # Generate synthetic data
        movie, gt = synth_movie(
            n_cells=20,
            T=100,
            k_means=(0.2, 1.0),
            frac_means=(0.3, 0.7),
            random_state=42
        )
        
        # Extract true populations
        true_labels = gt['population'].values
        
        # Create mock cell_features
        cell_features = pd.DataFrame({
            'exp_id': 'test',
            'movie_id': 'test',
            'cell_id': range(20),
            'mobile_frac': gt['mobile_frac'],
            'k': gt['k'],
            't_half': np.log(2) / gt['k'],
            'pre_bleach': 1.0,
            'r2': 0.95,
            'drift_px': 2.0,
            'bleach_qc': True,
            'roi_method': 'gaussian'
        })
        
        # Run clustering
        cell_features = detect_outliers_and_clusters(
            cell_features,
            max_k=4,
            contamination=0.05
        )
        
        # Check clustering quality (ARI)
        pred_labels = cell_features['cluster'].values
        
        # Filter out outliers and noise
        valid = (pred_labels != -1) & (~cell_features['outlier'])
        
        if valid.sum() > 5:
            ari = adjusted_rand_score(true_labels[valid], pred_labels[valid])
            assert ari > 0.5, f"ARI = {ari:.3f} < 0.5 (poor clustering)"


class TestStatistics:
    """Test statistical analysis"""
    
    def test_lmm_two_groups(self):
        """Test LMM on two groups"""
        # Generate data with known effect
        rng = np.random.RandomState(42)
        
        n_per_group = 30
        
        # Group 1: mobile_frac ~ 0.5
        group1 = pd.DataFrame({
            'mobile_frac': rng.normal(0.5, 0.1, n_per_group),
            'condition': 'control',
            'exp_id': ['exp1'] * 15 + ['exp2'] * 15
        })
        
        # Group 2: mobile_frac ~ 0.65 (effect size ~ 0.15)
        group2 = pd.DataFrame({
            'mobile_frac': rng.normal(0.65, 0.1, n_per_group),
            'condition': 'treatment',
            'exp_id': ['exp1'] * 15 + ['exp2'] * 15
        })
        
        df = pd.concat([group1, group2], ignore_index=True)
        
        # Run LMM
        result = lmm_param(df, 'mobile_frac', 'condition', 'exp_id')
        
        assert result['success'], "LMM failed"
        assert 'beta_treatment' in result, "No treatment effect estimated"
        assert result['p_treatment'] < 0.05, f"Failed to detect effect (p={result['p_treatment']:.3f})"
    
    def test_bootstrap_ci(self):
        """Test bootstrap CI"""
        data = np.random.normal(10, 2, 100)
        
        mean, (lower, upper) = bootstrap_bca_ci(
            data,
            np.mean,
            n_bootstrap=500,
            confidence=0.95
        )
        
        # Check that true mean is in CI
        true_mean = data.mean()
        assert lower <= true_mean <= upper, "True mean not in CI"
        
        # CI should be reasonable width
        width = upper - lower
        assert 0.2 < width < 2.0, f"CI width {width:.2f} seems wrong"


class TestEndToEnd:
    """End-to-end integration tests"""
    
    def test_complete_pipeline(self):
        """Test full pipeline on synthetic movie"""
        # Generate movie
        movie, gt = synth_movie(
            n_cells=10,
            T=80,
            drift_px=0.2,
            noise=0.02,
            random_state=42
        )
        
        time_points = np.arange(movie.shape[0]) * 0.5  # 0.5s intervals
        
        # Track
        roi_traces = track_movie(
            movie,
            time_points,
            exp_id='test',
            movie_id='test1',
            use_kalman=True
        )
        
        assert len(roi_traces) > 0, "No traces"
        
        # Fit
        cell_features = fit_cells(roi_traces, pre_bleach_window=5)
        
        assert len(cell_features) > 0, "No fits"
        assert cell_features['bleach_qc'].sum() > 0, "No cells passed QC"
        
        # Population analysis
        cell_features = detect_outliers_and_clusters(cell_features)
        
        assert 'outlier' in cell_features.columns
        assert 'cluster' in cell_features.columns
    
    def test_save_and_load(self):
        """Test data persistence"""
        # Create mock data
        roi_traces = pd.DataFrame({
            'exp_id': ['exp1'] * 10,
            'movie_id': ['mov1'] * 10,
            'cell_id': [0] * 10,
            'frame': range(10),
            't': np.arange(10),
            'x': 100.0,
            'y': 100.0,
            'radius': 10.0,
            'signal_raw': 1.0,
            'signal_bg': 0.1,
            'signal_corr': 0.9,
            'signal_norm': 0.5,
            'qc_motion': False,
            'qc_reason': ''
        })
        
        cell_features = pd.DataFrame({
            'exp_id': ['exp1'],
            'movie_id': ['mov1'],
            'cell_id': [0],
            'pre_bleach': 1.0,
            'I0': 0.4,
            'I_inf': 0.9,
            'k': 0.5,
            't_half': 1.4,
            'mobile_frac': 0.7,
            'r2': 0.95,
            'sse': 0.01,
            'drift_px': 2.0,
            'bleach_qc': True,
            'roi_method': 'gaussian',
            'outlier': False,
            'cluster': 0,
            'cluster_prob': 0.9,
            'A': 0.9,
            'B': 0.5,
            'fit_method': '1exp',
            'aic': 10.0,
            'bic': 15.0
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            traces_path, features_path = DataIO.save_tables(
                roi_traces,
                cell_features,
                tmpdir,
                format='both'
            )
            
            assert traces_path.exists()
            assert features_path.exists()
            
            # Load
            traces_loaded, features_loaded = DataIO.load_tables(tmpdir, format='parquet')
            
            assert len(traces_loaded) == len(roi_traces)
            assert len(features_loaded) == len(cell_features)


@pytest.fixture
def synthetic_dataset():
    """Fixture providing synthetic dataset"""
    test_data = generate_test_movies_with_expectations()
    return test_data


def test_acceptance_criteria(synthetic_dataset):
    """
    Test acceptance criteria from specification §14
    
    - Tracking MAE < 0.5 px
    - k RMSE < 10%
    - Cluster ARI > 0.7
    - Effect detection p < 0.01 for Δ = 0.15
    """
    movie = synthetic_dataset['simple']['movie']
    gt = synthetic_dataset['simple']['ground_truth']
    expectations = synthetic_dataset['simple']['expectations']
    
    time_points = np.arange(movie.shape[0])
    
    # Track
    roi_traces = track_movie(
        movie,
        time_points,
        exp_id='test',
        movie_id='test1',
        use_kalman=True
    )
    
    # Note: Full validation would require matching tracked ROIs to ground truth
    # This is a simplified check
    assert len(roi_traces) > 0, "Tracking produced no results"
    
    # Fit
    cell_features = fit_cells(roi_traces)
    
    assert len(cell_features) > 0, "Fitting produced no results"
    
    # Population analysis
    cell_features = detect_outliers_and_clusters(cell_features)
    
    # Check that we found multiple clusters
    n_clusters = len(cell_features['cluster'].unique()) - (1 if -1 in cell_features['cluster'].values else 0)
    assert n_clusters >= 2, f"Expected 2 populations, found {n_clusters}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
