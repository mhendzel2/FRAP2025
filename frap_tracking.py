"""
FRAP ROI Tracking Module
Advanced ROI tracking with sub-pixel accuracy, Kalman filtering, and multi-ROI support
"""
import numpy as np
import cv2
from scipy.optimize import curve_fit
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from skimage.filters import threshold_otsu, sobel
from skimage.morphology import opening, disk, watershed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Try to import filterpy for Kalman filtering
try:
    from filterpy.kalman import KalmanFilter as FilterPyKalman
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False
    logger.warning("filterpy not available, using custom Kalman implementation")


def gaussian_2d(coords: tuple[np.ndarray, np.ndarray], 
                x0: float, y0: float, 
                sigma_x: float, sigma_y: float, 
                amplitude: float, offset: float) -> np.ndarray:
    """2D Gaussian function for curve fitting"""
    x, y = coords
    gauss = offset + amplitude * np.exp(
        -(((x - x0) ** 2) / (2 * sigma_x ** 2) + 
          ((y - y0) ** 2) / (2 * sigma_y ** 2))
    )
    return gauss.ravel()


def fit_gaussian_centroid(
    img: np.ndarray, 
    seed_xy: tuple[int, int], 
    window: int = 15
) -> tuple[float, float, float]:
    """
    Fit 2D Gaussian to determine sub-pixel centroid
    
    Parameters
    ----------
    img : np.ndarray
        2D image array
    seed_xy : tuple[int, int]
        Initial (x, y) position
    window : int
        Size of local window for fitting
        
    Returns
    -------
    tuple[float, float, float]
        (x, y, mse) - refined centroid and mean squared error
    """
    seed_x, seed_y = seed_xy
    h, w = img.shape
    
    # Define window bounds
    half_win = window // 2
    x_min = max(0, seed_x - half_win)
    x_max = min(w, seed_x + half_win + 1)
    y_min = max(0, seed_y - half_win)
    y_max = min(h, seed_y + half_win + 1)
    
    # Extract patch
    patch = img[y_min:y_max, x_min:x_max]
    
    if patch.size == 0 or patch.shape[0] < 5 or patch.shape[1] < 5:
        return float(seed_x), float(seed_y), np.inf
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[y_min:y_max, x_min:x_max]
    
    # Initial parameter guess
    amplitude = patch.max() - patch.min()
    offset = patch.min()
    sigma_init = window / 4.0
    
    p0 = [seed_x, seed_y, sigma_init, sigma_init, amplitude, offset]
    
    # Bounds
    bounds_lower = [x_min, y_min, 1.0, 1.0, 0, 0]
    bounds_upper = [x_max, y_max, window, window, patch.max() * 2, patch.max()]
    
    try:
        popt, pcov = curve_fit(
            gaussian_2d,
            (x_coords, y_coords),
            patch.ravel(),
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            maxfev=1000
        )
        
        x_fit, y_fit = popt[0], popt[1]
        
        # Calculate MSE
        fitted = gaussian_2d((x_coords, y_coords), *popt).reshape(patch.shape)
        mse = np.mean((patch - fitted) ** 2)
        
        return x_fit, y_fit, mse
        
    except (RuntimeError, ValueError) as e:
        logger.debug(f"Gaussian fit failed: {e}")
        return float(seed_x), float(seed_y), np.inf


class ROIKalman:
    """
    2D Kalman filter for ROI tracking with constant velocity model
    """
    
    def __init__(self, dt: float, process_var: float = 1.0, meas_var: float = 2.0):
        """
        Initialize Kalman filter
        
        Parameters
        ----------
        dt : float
            Time step between frames
        process_var : float
            Process noise variance
        meas_var : float
            Measurement noise variance
        """
        self.dt = dt
        self.process_var = process_var
        self.meas_var = meas_var
        
        if FILTERPY_AVAILABLE:
            self._init_filterpy()
        else:
            self._init_custom()
    
    def _init_filterpy(self):
        """Initialize using filterpy library"""
        self.kf = FilterPyKalman(dim_x=4, dim_z=2)
        
        # State: [x, y, vx, vy]
        self.kf.x = np.zeros(4)
        
        # State transition matrix (constant velocity)
        self.kf.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (observe position only)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Covariance matrices
        self.kf.P *= 10.0  # Initial uncertainty
        self.kf.R = np.eye(2) * self.meas_var
        self.kf.Q = np.eye(4) * self.process_var
        
        self.innovation_norm = 0.0
    
    def _init_custom(self):
        """Initialize custom Kalman filter"""
        # State: [x, y, vx, vy]
        self.x = np.zeros(4)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Covariances
        self.P = np.eye(4) * 10.0
        self.R = np.eye(2) * self.meas_var
        self.Q = np.eye(4) * self.process_var
        
        self.innovation_norm = 0.0
    
    def update(self, x_meas: float, y_meas: float) -> tuple[float, float]:
        """
        Update filter with new measurement
        
        Parameters
        ----------
        x_meas, y_meas : float
            Measured position
            
        Returns
        -------
        tuple[float, float]
            Smoothed (x, y) position
        """
        z = np.array([x_meas, y_meas])
        
        if FILTERPY_AVAILABLE:
            # Predict
            self.kf.predict()
            
            # Update
            self.kf.update(z)
            
            # Calculate innovation
            innovation = z - self.kf.H @ self.kf.x_prior
            self.innovation_norm = np.linalg.norm(innovation)
            
            return self.kf.x[0], self.kf.x[1]
        else:
            # Custom implementation
            # Predict
            x_pred = self.F @ self.x
            P_pred = self.F @ self.P @ self.F.T + self.Q
            
            # Innovation
            y_innov = z - self.H @ x_pred
            self.innovation_norm = np.linalg.norm(y_innov)
            
            # Innovation covariance
            S = self.H @ P_pred @ self.H.T + self.R
            
            # Kalman gain
            K = P_pred @ self.H.T @ np.linalg.inv(S)
            
            # Update
            self.x = x_pred + K @ y_innov
            self.P = (np.eye(4) - K @ self.H) @ P_pred
            
            return self.x[0], self.x[1]
    
    def get_innovation_norm(self) -> float:
        """Return the innovation norm from last update"""
        return self.innovation_norm


def adapt_radius(
    img: np.ndarray, 
    cx: float, 
    cy: float, 
    r0: float, 
    r_min: int = 3, 
    r_max: int = 20
) -> float:
    """
    Adapt ROI radius based on gradient energy
    
    Parameters
    ----------
    img : np.ndarray
        2D image
    cx, cy : float
        Center coordinates
    r0 : float
        Initial radius
    r_min, r_max : int
        Radius bounds
        
    Returns
    -------
    float
        Optimal radius
    """
    # Compute gradient magnitude
    grad = sobel(img)
    
    h, w = grad.shape
    cy_int, cx_int = int(round(cy)), int(round(cx))
    
    # Test radii
    radii = np.arange(max(r_min, r0 - 3), min(r_max, r0 + 4), 0.5)
    energies = []
    
    for r in radii:
        # Create ring mask
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        ring = (dist >= r - 1) & (dist <= r + 1)
        
        if not ring.any():
            energies.append(0)
            continue
        
        # Compute energy
        energy = np.sum(grad[ring])
        energies.append(energy)
    
    if not energies or max(energies) == 0:
        return r0
    
    # Find maximum with smoothing
    energies = np.array(energies)
    if len(energies) > 3:
        from scipy.ndimage import gaussian_filter1d
        energies = gaussian_filter1d(energies, sigma=1.0)
    
    best_idx = np.argmax(energies)
    return radii[best_idx]


def track_optical_flow(
    prev_img: np.ndarray, 
    curr_img: np.ndarray, 
    prev_xy: tuple[float, float]
) -> tuple[float, float, float]:
    """
    Track point using Lucas-Kanade optical flow
    
    Parameters
    ----------
    prev_img, curr_img : np.ndarray
        Consecutive frames
    prev_xy : tuple[float, float]
        Previous (x, y) position
        
    Returns
    -------
    tuple[float, float, float]
        (x, y, error)
    """
    # Convert to uint8 if needed
    if prev_img.dtype != np.uint8:
        prev_img = (prev_img / prev_img.max() * 255).astype(np.uint8)
    if curr_img.dtype != np.uint8:
        curr_img = (curr_img / curr_img.max() * 255).astype(np.uint8)
    
    # Prepare point
    p0 = np.array([[prev_xy]], dtype=np.float32)
    
    # LK parameters
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    
    try:
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, p0, None, **lk_params)
        
        if st[0][0] == 1:
            x_new, y_new = p1[0][0]
            return float(x_new), float(y_new), float(err[0][0])
        else:
            return prev_xy[0], prev_xy[1], np.inf
            
    except cv2.error as e:
        logger.debug(f"Optical flow failed: {e}")
        return prev_xy[0], prev_xy[1], np.inf


def seed_rois(
    first_frame: np.ndarray, 
    max_rois: Optional[int] = None,
    min_area: int = 20,
    max_area: int = 500
) -> list[tuple[float, float]]:
    """
    Detect ROIs in first frame using watershed segmentation
    
    Parameters
    ----------
    first_frame : np.ndarray
        First frame of movie
    max_rois : int, optional
        Maximum number of ROIs to return
    min_area, max_area : int
        Area bounds for valid ROIs
        
    Returns
    -------
    list[tuple[float, float]]
        List of (x, y) centroids
    """
    # Normalize
    img = first_frame.astype(float)
    img = (img - img.min()) / (img.max() - img.min())
    
    # Threshold
    try:
        thresh = threshold_otsu(img)
    except ValueError:
        logger.warning("Otsu threshold failed, using mean")
        thresh = img.mean()
    
    binary = img > thresh
    
    # Morphological opening
    binary = opening(binary, disk(2))
    
    # Distance transform
    distance = distance_transform_edt(binary)
    
    # Find peaks
    coords = peak_local_max(distance, min_distance=10, labels=binary)
    
    if len(coords) == 0:
        logger.warning("No ROIs detected")
        return []
    
    # Create markers
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = label(mask)
    
    # Watershed
    labels = watershed(-distance, markers, mask=binary)
    
    # Extract region properties
    regions = regionprops(labels)
    centroids = []
    
    for region in regions:
        if min_area <= region.area <= max_area:
            y, x = region.centroid
            centroids.append((x, y))
    
    # Limit number if requested
    if max_rois is not None and len(centroids) > max_rois:
        # Keep largest regions
        areas = [r.area for r in regions if min_area <= r.area <= max_area]
        indices = np.argsort(areas)[-max_rois:]
        centroids = [centroids[i] for i in indices]
    
    logger.info(f"Detected {len(centroids)} ROIs")
    return centroids


def hungarian_assignment(
    prev_positions: list[tuple[float, float]], 
    curr_positions: list[tuple[float, float]],
    max_distance: float = 50.0,
    intensity_weight: float = 0.0,
    prev_intensities: Optional[list[float]] = None,
    curr_intensities: Optional[list[float]] = None
) -> list[tuple[int, int]]:
    """
    Match ROIs between frames using Hungarian algorithm
    
    Parameters
    ----------
    prev_positions, curr_positions : list[tuple[float, float]]
        ROI positions in previous and current frame
    max_distance : float
        Maximum matching distance
    intensity_weight : float
        Weight for intensity term in cost
    prev_intensities, curr_intensities : list[float], optional
        Intensities for penalty term
        
    Returns
    -------
    list[tuple[int, int]]
        Matched indices (prev_idx, curr_idx)
    """
    if not prev_positions or not curr_positions:
        return []
    
    n_prev = len(prev_positions)
    n_curr = len(curr_positions)
    
    # Build cost matrix
    cost = np.full((n_prev, n_curr), np.inf)
    
    for i, (x1, y1) in enumerate(prev_positions):
        for j, (x2, y2) in enumerate(curr_positions):
            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            if dist <= max_distance:
                cost[i, j] = dist
                
                # Add intensity penalty if available
                if (intensity_weight > 0 and 
                    prev_intensities is not None and 
                    curr_intensities is not None):
                    intensity_diff = abs(prev_intensities[i] - curr_intensities[j])
                    cost[i, j] += intensity_weight * intensity_diff
    
    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Filter out invalid matches
    matches = []
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < np.inf:
            matches.append((i, j))
    
    return matches


try:
    from skimage.segmentation import morphological_geodesic_active_contour
    ACTIVE_CONTOUR_AVAILABLE = True
except ImportError:
    ACTIVE_CONTOUR_AVAILABLE = False


def evolve_mask(
    img: np.ndarray, 
    init_mask: np.ndarray, 
    alpha: float = 0.2, 
    beta: float = 0.1, 
    iters: int = 100
) -> np.ndarray:
    """
    Evolve ROI mask using active contours
    
    Parameters
    ----------
    img : np.ndarray
        Current frame
    init_mask : np.ndarray
        Initial binary mask
    alpha, beta : float
        Smoothness parameters
    iters : int
        Number of iterations
        
    Returns
    -------
    np.ndarray
        Evolved mask
    """
    if not ACTIVE_CONTOUR_AVAILABLE:
        logger.warning("Active contour not available, returning initial mask")
        return init_mask
    
    # Normalize image
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-10)
    
    # Invert for geodesic active contour
    gimage = 1 - img_norm
    
    try:
        evolved = morphological_geodesic_active_contour(
            gimage,
            iterations=iters,
            init_level_set=init_mask,
            smoothing=1,
            threshold='auto',
            balloon=-1
        )
        return evolved.astype(np.uint8)
    except Exception as e:
        logger.debug(f"Active contour failed: {e}")
        return init_mask


def compute_mask_metrics(mask1: np.ndarray, mask2: np.ndarray) -> dict:
    """
    Compute QC metrics between consecutive masks
    
    Parameters
    ----------
    mask1, mask2 : np.ndarray
        Binary masks to compare
        
    Returns
    -------
    dict
        Metrics including IoU and Hausdorff distance
    """
    from scipy.spatial.distance import directed_hausdorff
    
    # IoU
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union > 0 else 0.0
    
    # Hausdorff distance
    try:
        coords1 = np.argwhere(mask1)
        coords2 = np.argwhere(mask2)
        if len(coords1) > 0 and len(coords2) > 0:
            hausdorff = max(
                directed_hausdorff(coords1, coords2)[0],
                directed_hausdorff(coords2, coords1)[0]
            )
        else:
            hausdorff = np.inf
    except Exception:
        hausdorff = np.inf
    
    return {
        'iou': iou,
        'hausdorff': hausdorff,
        'area1': mask1.sum(),
        'area2': mask2.sum()
    }
