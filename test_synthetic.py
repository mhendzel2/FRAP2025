"""
Synthetic FRAP Data Generator
Generate controlled synthetic movies for testing
"""
import numpy as np
import pandas as pd
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def synth_movie(
    n_cells: int = 10,
    T: int = 100,
    img_size: tuple[int, int] = (256, 256),
    drift_px: float = 0.2,
    noise: float = 0.02,
    k_means: tuple[float, float] = (0.3, 1.2),
    frac_means: tuple[float, float] = (0.4, 0.7),
    bleach_frame: int = 10,
    pre_bleach_intensity: float = 1.0,
    bleach_depth: float = 0.6,
    roi_radius: int = 10,
    background_level: float = 0.1,
    random_state: int = 0
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Generate synthetic FRAP movie with two subpopulations
    
    Parameters
    ----------
    n_cells : int
        Number of cells
    T : int
        Number of frames
    img_size : tuple[int, int]
        Image dimensions (H, W)
    drift_px : float
        Drift per frame (pixels)
    noise : float
        Noise level (relative to signal)
    k_means : tuple[float, float]
        Recovery rates for two populations
    frac_means : tuple[float, float]
        Mobile fractions for two populations
    bleach_frame : int
        Frame at which bleaching occurs
    pre_bleach_intensity : float
        Pre-bleach intensity
    bleach_depth : float
        Bleach depth (0-1)
    roi_radius : int
        ROI radius
    background_level : float
        Background intensity
    random_state : int
        Random seed
        
    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        (movie, ground_truth_params)
    """
    rng = np.random.RandomState(random_state)
    
    H, W = img_size
    
    # Generate cell positions (avoid edges)
    margin = roi_radius * 3
    positions = []
    for _ in range(n_cells):
        x = rng.randint(margin, W - margin)
        y = rng.randint(margin, H - margin)
        positions.append([x, y])
    positions = np.array(positions)
    
    # Assign cells to populations (50/50 split)
    n_pop1 = n_cells // 2
    populations = np.array([0] * n_pop1 + [1] * (n_cells - n_pop1))
    rng.shuffle(populations)
    
    # Generate parameters for each cell
    k_values = np.where(
        populations == 0,
        rng.normal(k_means[0], k_means[0] * 0.2, n_cells),
        rng.normal(k_means[1], k_means[1] * 0.2, n_cells)
    )
    k_values = np.clip(k_values, 0.1, 3.0)
    
    frac_values = np.where(
        populations == 0,
        rng.normal(frac_means[0], frac_means[0] * 0.15, n_cells),
        rng.normal(frac_means[1], frac_means[1] * 0.15, n_cells)
    )
    frac_values = np.clip(frac_values, 0.1, 0.95)
    
    # Generate drift trajectories
    drift_directions = rng.randn(n_cells, 2)
    drift_directions /= np.linalg.norm(drift_directions, axis=1, keepdims=True)
    
    # Initialize movie
    movie = np.zeros((T, H, W), dtype=np.float32)
    
    # Ground truth data
    ground_truth = []
    
    # Generate frames
    for t in range(T):
        frame = np.ones((H, W)) * background_level
        
        for cell_id in range(n_cells):
            # Current position with drift
            pos = positions[cell_id] + drift_directions[cell_id] * drift_px * t
            cx, cy = pos
            
            # Skip if out of bounds
            if not (margin <= cx < W - margin and margin <= cy < H - margin):
                continue
            
            # Generate intensity for this cell
            if t < bleach_frame:
                # Pre-bleach
                intensity = pre_bleach_intensity
            else:
                # Recovery curve
                t_post = t - bleach_frame
                k = k_values[cell_id]
                mobile_frac = frac_values[cell_id]
                
                # I(t) = I0 + (I_inf - I0) * (1 - exp(-k*t))
                I0 = pre_bleach_intensity * (1 - bleach_depth)
                I_inf = I0 + mobile_frac * bleach_depth * pre_bleach_intensity
                
                intensity = I_inf - (I_inf - I0) * np.exp(-k * t_post)
            
            # Add cell to frame (Gaussian spot)
            y, x = np.ogrid[:H, :W]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            gaussian = np.exp(-(dist ** 2) / (2 * (roi_radius / 2) ** 2))
            frame += intensity * gaussian
            
            # Record ground truth
            ground_truth.append({
                'frame': t,
                'cell_id': cell_id,
                'x': cx,
                'y': cy,
                'intensity': intensity,
                'k': k_values[cell_id],
                'mobile_frac': frac_values[cell_id],
                'population': populations[cell_id]
            })
        
        # Add noise
        noise_map = rng.normal(0, noise, (H, W))
        frame += noise_map
        
        # Clip to valid range
        frame = np.clip(frame, 0, None)
        
        movie[t] = frame
    
    # Convert ground truth to DataFrame
    gt_df = pd.DataFrame(ground_truth)
    
    # Add summary per cell
    summary = gt_df.groupby('cell_id').first()[['k', 'mobile_frac', 'population']]
    summary['true_drift_px'] = drift_px * T
    
    logger.info(f"Generated synthetic movie: {T} frames, {n_cells} cells, 2 populations")
    logger.info(f"Population 0: k={k_means[0]:.2f}, frac={frac_means[0]:.2f}")
    logger.info(f"Population 1: k={k_means[1]:.2f}, frac={frac_means[1]:.2f}")
    
    return movie, summary


def synth_multi_movie_dataset(
    n_movies: int = 5,
    n_cells_per_movie: int = 10,
    conditions: Optional[list[str]] = None,
    **kwargs
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """
    Generate multiple synthetic movies simulating an experiment
    
    Parameters
    ----------
    n_movies : int
        Number of movies
    n_cells_per_movie : int
        Cells per movie
    conditions : list[str], optional
        Condition labels
    **kwargs
        Parameters for synth_movie
        
    Returns
    -------
    tuple[list[np.ndarray], pd.DataFrame]
        (list_of_movies, combined_ground_truth)
    """
    if conditions is None:
        # Split movies between two conditions
        conditions = ['control'] * (n_movies // 2) + ['treatment'] * (n_movies - n_movies // 2)
    
    movies = []
    ground_truths = []
    
    for movie_id, condition in enumerate(conditions):
        # Vary parameters slightly per movie
        random_state = kwargs.get('random_state', 0) + movie_id
        
        # Modify parameters based on condition
        if condition == 'treatment':
            # Treatment increases mobile fraction
            k_means = kwargs.get('k_means', (0.3, 1.2))
            frac_means = (
                kwargs.get('frac_means', (0.4, 0.7))[0] + 0.15,
                kwargs.get('frac_means', (0.4, 0.7))[1] + 0.15
            )
            kwargs_mod = {**kwargs, 'frac_means': frac_means, 'random_state': random_state}
        else:
            kwargs_mod = {**kwargs, 'random_state': random_state}
        
        movie, gt = synth_movie(n_cells=n_cells_per_movie, **kwargs_mod)
        
        gt['movie_id'] = movie_id
        gt['condition'] = condition
        gt['exp_id'] = f"exp_{movie_id // 2}"  # Group movies into experiments
        
        movies.append(movie)
        ground_truths.append(gt)
    
    combined_gt = pd.concat(ground_truths, ignore_index=True)
    
    logger.info(f"Generated dataset: {n_movies} movies across {len(combined_gt['exp_id'].unique())} experiments")
    
    return movies, combined_gt


def add_realistic_artifacts(
    movie: np.ndarray,
    artifact_prob: float = 0.05,
    artifact_magnitude: float = 0.3,
    photobleaching_rate: float = 0.001,
    random_state: int = 0
) -> np.ndarray:
    """
    Add realistic artifacts to synthetic movie
    
    Parameters
    ----------
    movie : np.ndarray
        Clean movie (T, H, W)
    artifact_prob : float
        Probability of artifact per frame
    artifact_magnitude : float
        Magnitude of artifacts
    photobleaching_rate : float
        Imaging photobleaching rate
    random_state : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Movie with artifacts
    """
    rng = np.random.RandomState(random_state)
    movie_corrupted = movie.copy()
    T, H, W = movie.shape
    
    # Add photobleaching
    for t in range(T):
        decay = np.exp(-photobleaching_rate * t)
        movie_corrupted[t] *= decay
    
    # Add frame-specific artifacts
    for t in range(T):
        if rng.rand() < artifact_prob:
            # Random shift or intensity jump
            if rng.rand() < 0.5:
                # Intensity artifact
                movie_corrupted[t] *= (1 + rng.uniform(-artifact_magnitude, artifact_magnitude))
            else:
                # Spatial shift (simulate stage drift spike)
                shift_x = rng.randint(-2, 3)
                shift_y = rng.randint(-2, 3)
                if shift_x != 0 or shift_y != 0:
                    movie_corrupted[t] = np.roll(movie_corrupted[t], shift=(shift_y, shift_x), axis=(0, 1))
    
    return movie_corrupted


def generate_test_movies_with_expectations(
    random_state: int = 42
) -> dict:
    """
    Generate test movies with known expectations for unit tests
    
    Parameters
    ----------
    random_state : int
        Random seed
        
    Returns
    -------
    dict
        Test data and expectations
    """
    # Test case 1: Simple movie with well-separated populations
    movie1, gt1 = synth_movie(
        n_cells=20,
        T=100,
        drift_px=0.1,
        noise=0.01,
        k_means=(0.2, 1.0),
        frac_means=(0.3, 0.7),
        random_state=random_state
    )
    
    # Test case 2: Movie with artifacts
    movie2, gt2 = synth_movie(
        n_cells=15,
        T=80,
        drift_px=0.5,
        noise=0.05,
        k_means=(0.5, 0.8),
        frac_means=(0.5, 0.6),
        random_state=random_state + 1
    )
    movie2_artifacts = add_realistic_artifacts(movie2, artifact_prob=0.1, random_state=random_state)
    
    # Test case 3: Multi-condition dataset
    movies3, gt3 = synth_multi_movie_dataset(
        n_movies=6,
        n_cells_per_movie=12,
        T=100,
        drift_px=0.2,
        noise=0.02,
        random_state=random_state
    )
    
    return {
        'simple': {
            'movie': movie1,
            'ground_truth': gt1,
            'expectations': {
                'tracking_mae': 0.5,  # pixels
                'k_rmse_pct': 10,  # percent
                'cluster_ari': 0.7,  # Adjusted Rand Index
                'effect_p': 0.01  # p-value threshold
            }
        },
        'artifacts': {
            'movie': movie2_artifacts,
            'ground_truth': gt2,
            'clean_movie': movie2
        },
        'multi_condition': {
            'movies': movies3,
            'ground_truth': gt3
        }
    }
