"""
FRAP PDE Solver - Finite Element/PDE-Based Spatial FRAP Analysis

This module implements next-generation FRAP analysis using Partial Differential
Equation (PDE) modeling to solve the actual diffusion-reaction equations numerically.

Advantages over analytical models:
- Handles irregular ROI shapes (not just circles)
- Accounts for finite cell boundaries
- Models complex geometries (nucleolus, ER, etc.)
- More accurate for small compartments

The reaction-diffusion equation solved:
    ∂C/∂t = D∇²C - k_on·C + k_off·(C_total - C)

Where:
    C = concentration of free/mobile protein
    D = diffusion coefficient
    k_on = binding rate constant
    k_off = unbinding rate constant
    C_total = total protein concentration

Author: FRAP Analysis Suite
"""

import numpy as np
from scipy import ndimage, optimize
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BoundaryCondition(Enum):
    """Boundary condition types for PDE solver."""
    NEUMANN = "neumann"  # No flux (∂C/∂n = 0) - most common for cells
    DIRICHLET = "dirichlet"  # Fixed concentration at boundary
    PERIODIC = "periodic"  # Periodic boundaries


@dataclass
class PDEParameters:
    """Parameters for PDE-based FRAP fitting."""
    D: float  # Diffusion coefficient (µm²/s)
    k_on: float = 0.0  # Binding rate (s⁻¹)
    k_off: float = 0.0  # Unbinding rate (s⁻¹)
    immobile_fraction: float = 0.0  # Fraction of immobile protein (0-1)
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to array for optimization."""
        return np.array([self.D, self.k_on, self.k_off, self.immobile_fraction])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'PDEParameters':
        """Create parameters from array."""
        return cls(
            D=arr[0],
            k_on=arr[1] if len(arr) > 1 else 0.0,
            k_off=arr[2] if len(arr) > 2 else 0.0,
            immobile_fraction=arr[3] if len(arr) > 3 else 0.0
        )


class GeometryExtractor:
    """Extract cell and ROI geometry from images."""
    
    @staticmethod
    def extract_cell_mask(image: np.ndarray, 
                          threshold: Optional[float] = None,
                          min_size: int = 100) -> np.ndarray:
        """
        Extract cell boundary mask from pre-bleach image.
        
        Parameters:
        -----------
        image : np.ndarray
            Pre-bleach image (2D)
        threshold : float, optional
            Intensity threshold. If None, uses Otsu's method
        min_size : int
            Minimum object size to keep
            
        Returns:
        --------
        np.ndarray
            Binary mask of cell region
        """
        from scipy.ndimage import binary_fill_holes, binary_opening
        
        # Normalize image
        img_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)
        
        # Threshold
        if threshold is None:
            # Otsu's method
            hist, bin_edges = np.histogram(img_norm.ravel(), bins=256)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Calculate between-class variance
            weight1 = np.cumsum(hist)
            weight2 = np.cumsum(hist[::-1])[::-1]
            mean1 = np.cumsum(hist * bin_centers) / (weight1 + 1e-10)
            mean2 = (np.cumsum((hist * bin_centers)[::-1]) / (weight2[::-1] + 1e-10))[::-1]
            
            variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
            threshold = bin_centers[np.argmax(variance)]
        
        # Create binary mask
        mask = img_norm > threshold
        
        # Clean up mask
        mask = binary_fill_holes(mask)
        mask = binary_opening(mask, iterations=2)
        
        # Remove small objects
        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            mask_sizes = sizes > min_size
            remove_pixel = mask_sizes[labeled - 1]
            remove_pixel[labeled == 0] = False
            mask = remove_pixel
        
        return mask.astype(bool)
    
    @staticmethod
    def extract_bleach_roi(pre_image: np.ndarray, 
                           post_image: np.ndarray,
                           intensity_drop_threshold: float = 0.3) -> np.ndarray:
        """
        Detect bleach ROI by comparing pre and post-bleach images.
        
        Parameters:
        -----------
        pre_image : np.ndarray
            Image before bleaching
        post_image : np.ndarray
            Image immediately after bleaching
        intensity_drop_threshold : float
            Minimum fractional intensity drop to consider as bleached
            
        Returns:
        --------
        np.ndarray
            Binary mask of bleached region
        """
        # Normalize images
        pre_norm = pre_image / (np.max(pre_image) + 1e-10)
        post_norm = post_image / (np.max(pre_image) + 1e-10)  # Use pre-max for reference
        
        # Calculate intensity drop
        intensity_drop = (pre_norm - post_norm) / (pre_norm + 1e-10)
        
        # Threshold to find bleached region
        bleach_mask = intensity_drop > intensity_drop_threshold
        
        # Clean up
        bleach_mask = ndimage.binary_opening(bleach_mask, iterations=1)
        bleach_mask = ndimage.binary_closing(bleach_mask, iterations=2)
        
        return bleach_mask.astype(bool)
    
    @staticmethod
    def create_mesh_2d(mask: np.ndarray, 
                       pixel_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a simple regular mesh from a binary mask.
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask defining the domain
        pixel_size : float
            Physical size of each pixel (µm)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            x and y coordinate arrays
        """
        ny, nx = mask.shape
        x = np.arange(nx) * pixel_size
        y = np.arange(ny) * pixel_size
        return np.meshgrid(x, y)


class FiniteDifferenceSolver:
    """
    Finite Difference solver for 2D reaction-diffusion equation.
    
    Uses implicit Crank-Nicolson scheme for stability.
    """
    
    def __init__(self, 
                 cell_mask: np.ndarray,
                 bleach_mask: np.ndarray,
                 pixel_size: float = 1.0,
                 boundary_condition: BoundaryCondition = BoundaryCondition.NEUMANN):
        """
        Initialize the solver.
        
        Parameters:
        -----------
        cell_mask : np.ndarray
            Binary mask of cell region
        bleach_mask : np.ndarray
            Binary mask of bleached region
        pixel_size : float
            Physical size of each pixel (µm)
        boundary_condition : BoundaryCondition
            Type of boundary condition to apply
        """
        self.cell_mask = cell_mask.astype(bool)
        self.bleach_mask = bleach_mask.astype(bool)
        self.pixel_size = pixel_size
        self.dx = pixel_size
        self.dy = pixel_size
        self.boundary_condition = boundary_condition
        
        self.ny, self.nx = cell_mask.shape
        self.n_points = np.sum(cell_mask)
        
        # Create mapping from 2D indices to 1D indices for interior points
        self._create_index_mapping()
        
    def _create_index_mapping(self):
        """Create mapping between 2D grid and 1D vector for linear algebra."""
        self.idx_2d_to_1d = np.full((self.ny, self.nx), -1, dtype=int)
        self.idx_1d_to_2d = []
        
        idx = 0
        for j in range(self.ny):
            for i in range(self.nx):
                if self.cell_mask[j, i]:
                    self.idx_2d_to_1d[j, i] = idx
                    self.idx_1d_to_2d.append((j, i))
                    idx += 1
        
        self.idx_1d_to_2d = np.array(self.idx_1d_to_2d)
    
    def _build_laplacian_matrix(self, D: float) -> csr_matrix:
        """
        Build the Laplacian matrix for diffusion.
        
        Uses 5-point stencil: ∇²C ≈ (C[i+1,j] + C[i-1,j] + C[i,j+1] + C[i,j-1] - 4*C[i,j]) / h²
        """
        n = self.n_points
        
        # Coefficient for diffusion
        coeff = D / (self.dx * self.dy)
        
        # Build sparse matrix
        row_idx = []
        col_idx = []
        values = []
        
        for idx in range(n):
            j, i = self.idx_1d_to_2d[idx]
            
            # Diagonal term
            diagonal_val = -4.0 * coeff
            neighbor_count = 0
            
            # Check neighbors
            neighbors = [(j-1, i), (j+1, i), (j, i-1), (j, i+1)]
            
            for nj, ni in neighbors:
                if 0 <= nj < self.ny and 0 <= ni < self.nx:
                    if self.cell_mask[nj, ni]:
                        # Interior neighbor
                        neighbor_idx = self.idx_2d_to_1d[nj, ni]
                        row_idx.append(idx)
                        col_idx.append(neighbor_idx)
                        values.append(coeff)
                        neighbor_count += 1
                    elif self.boundary_condition == BoundaryCondition.NEUMANN:
                        # Neumann BC: reflect back to current point
                        diagonal_val += coeff
            
            # Adjust diagonal for missing neighbors (boundary)
            if self.boundary_condition == BoundaryCondition.NEUMANN:
                diagonal_val = -neighbor_count * coeff
            
            row_idx.append(idx)
            col_idx.append(idx)
            values.append(diagonal_val)
        
        return csr_matrix((values, (row_idx, col_idx)), shape=(n, n))
    
    def _build_reaction_matrix(self, k_on: float, k_off: float) -> csr_matrix:
        """Build the reaction term matrix."""
        n = self.n_points
        
        # Reaction: -k_on * C + k_off * (1 - C) for normalized concentration
        # Simplified: -(k_on + k_off) * C + k_off
        diagonal = np.full(n, -(k_on + k_off))
        
        return diags(diagonal, 0, shape=(n, n), format='csr')
    
    def simulate(self, 
                 params: PDEParameters,
                 time_points: np.ndarray,
                 initial_condition: Optional[np.ndarray] = None,
                 bleach_depth: float = 0.2) -> Dict[str, Any]:
        """
        Simulate FRAP recovery.
        
        Parameters:
        -----------
        params : PDEParameters
            Physical parameters (D, k_on, k_off, immobile_fraction)
        time_points : np.ndarray
            Time points for output (s)
        initial_condition : np.ndarray, optional
            Initial concentration field. If None, creates from bleach_mask
        bleach_depth : float
            Relative intensity immediately after bleach (0-1)
            
        Returns:
        --------
        Dict containing:
            'recovery_curve': Mean intensity in bleach ROI over time
            'concentration_fields': Full 2D concentration at each time point
            'time': Time points
        """
        n = self.n_points
        dt = np.min(np.diff(time_points)) if len(time_points) > 1 else 0.1
        
        # Set up initial condition
        if initial_condition is None:
            C = np.ones(n)
            # Apply bleaching
            for idx in range(n):
                j, i = self.idx_1d_to_2d[idx]
                if self.bleach_mask[j, i]:
                    C[idx] = bleach_depth
        else:
            C = initial_condition[self.cell_mask].flatten()
        
        # Account for immobile fraction
        mobile_fraction = 1.0 - params.immobile_fraction
        C_mobile = C * mobile_fraction
        C_immobile = np.ones(n) * params.immobile_fraction  # Immobile doesn't recover
        
        # Build matrices
        L = self._build_laplacian_matrix(params.D)
        
        if params.k_on > 0 or params.k_off > 0:
            R = self._build_reaction_matrix(params.k_on, params.k_off)
            A = L + R
        else:
            A = L
        
        # Crank-Nicolson matrices
        I = diags(np.ones(n), 0, format='csr')
        A_implicit = I - 0.5 * dt * A
        A_explicit = I + 0.5 * dt * A
        
        # Source term for reaction (k_off term)
        source = np.zeros(n)
        if params.k_off > 0:
            source = np.full(n, params.k_off * dt)
        
        # Storage for results
        recovery_curve = []
        concentration_fields = []
        
        # Get indices of bleach ROI for averaging
        bleach_indices = []
        for idx in range(n):
            j, i = self.idx_1d_to_2d[idx]
            if self.bleach_mask[j, i]:
                bleach_indices.append(idx)
        bleach_indices = np.array(bleach_indices)
        
        # Time stepping
        current_time = 0.0
        time_idx = 0
        
        while time_idx < len(time_points):
            target_time = time_points[time_idx]
            
            # Step forward until we reach target time
            while current_time < target_time:
                step_dt = min(dt, target_time - current_time)
                
                # Update matrices if dt changed
                if abs(step_dt - dt) > 1e-10:
                    A_implicit = I - 0.5 * step_dt * A
                    A_explicit = I + 0.5 * step_dt * A
                    if params.k_off > 0:
                        source = np.full(n, params.k_off * step_dt)
                
                # Crank-Nicolson step
                rhs = A_explicit @ C_mobile + source
                C_mobile = spsolve(A_implicit, rhs)
                
                # Enforce bounds
                C_mobile = np.clip(C_mobile, 0, mobile_fraction)
                
                current_time += step_dt
            
            # Total concentration = mobile + immobile
            C_total = C_mobile + C_immobile
            
            # Record results
            if len(bleach_indices) > 0:
                mean_intensity = np.mean(C_total[bleach_indices])
            else:
                mean_intensity = np.mean(C_total)
            recovery_curve.append(mean_intensity)
            
            # Store full field
            field_2d = np.zeros((self.ny, self.nx))
            for idx in range(n):
                j, i = self.idx_1d_to_2d[idx]
                field_2d[j, i] = C_total[idx]
            concentration_fields.append(field_2d.copy())
            
            time_idx += 1
        
        return {
            'recovery_curve': np.array(recovery_curve),
            'concentration_fields': np.array(concentration_fields),
            'time': time_points
        }


class PDEFRAPFitter:
    """
    Fit FRAP data using PDE-based model.
    
    This class handles the optimization to find best-fit parameters.
    """
    
    def __init__(self, 
                 cell_mask: np.ndarray,
                 bleach_mask: np.ndarray,
                 pixel_size: float = 1.0):
        """
        Initialize the fitter.
        
        Parameters:
        -----------
        cell_mask : np.ndarray
            Binary mask of cell region
        bleach_mask : np.ndarray
            Binary mask of bleached region  
        pixel_size : float
            Physical size of each pixel (µm)
        """
        self.solver = FiniteDifferenceSolver(cell_mask, bleach_mask, pixel_size)
        self.cell_mask = cell_mask
        self.bleach_mask = bleach_mask
        self.pixel_size = pixel_size
        
    def _objective(self, 
                   param_array: np.ndarray,
                   time_data: np.ndarray,
                   intensity_data: np.ndarray,
                   model_type: str,
                   bleach_depth: float) -> float:
        """Objective function for optimization."""
        # Unpack parameters based on model type
        if model_type == 'diffusion_only':
            params = PDEParameters(D=param_array[0])
        elif model_type == 'diffusion_binding':
            params = PDEParameters(D=param_array[0], k_on=param_array[1], k_off=param_array[2])
        elif model_type == 'diffusion_immobile':
            params = PDEParameters(D=param_array[0], immobile_fraction=param_array[1])
        else:  # full model
            params = PDEParameters(
                D=param_array[0], 
                k_on=param_array[1], 
                k_off=param_array[2],
                immobile_fraction=param_array[3]
            )
        
        # Simulate
        try:
            result = self.solver.simulate(params, time_data, bleach_depth=bleach_depth)
            simulated = result['recovery_curve']
            
            # Calculate residual sum of squares
            residuals = intensity_data - simulated
            rss = np.sum(residuals ** 2)
            
            return rss
            
        except Exception as e:
            logger.warning(f"Simulation failed: {e}")
            return 1e10
    
    def fit(self,
            time_data: np.ndarray,
            intensity_data: np.ndarray,
            model_type: str = 'diffusion_only',
            bleach_depth: Optional[float] = None,
            initial_guess: Optional[Dict[str, float]] = None,
            bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Fit PDE model to FRAP recovery data.
        
        Parameters:
        -----------
        time_data : np.ndarray
            Time points (s)
        intensity_data : np.ndarray
            Normalized intensity values (0-1)
        model_type : str
            One of: 'diffusion_only', 'diffusion_binding', 'diffusion_immobile', 'full'
        bleach_depth : float, optional
            Initial intensity after bleach. If None, estimated from data
        initial_guess : dict, optional
            Initial parameter guesses
        bounds : dict, optional
            Parameter bounds
            
        Returns:
        --------
        Dict containing:
            'params': Fitted PDEParameters
            'fitted_curve': Fitted recovery curve
            'r2': R-squared
            'aic': Akaike Information Criterion
            'residuals': Fit residuals
        """
        # Estimate bleach depth if not provided
        if bleach_depth is None:
            bleach_depth = intensity_data[0]
        
        # Set up parameter bounds and initial guesses
        if model_type == 'diffusion_only':
            n_params = 1
            default_guess = [1.0]  # D in µm²/s
            default_bounds = [(0.001, 100.0)]
        elif model_type == 'diffusion_binding':
            n_params = 3
            default_guess = [1.0, 0.1, 0.1]  # D, k_on, k_off
            default_bounds = [(0.001, 100.0), (0.0, 10.0), (0.0, 10.0)]
        elif model_type == 'diffusion_immobile':
            n_params = 2
            default_guess = [1.0, 0.2]  # D, immobile_fraction
            default_bounds = [(0.001, 100.0), (0.0, 0.9)]
        else:  # full
            n_params = 4
            default_guess = [1.0, 0.1, 0.1, 0.2]
            default_bounds = [(0.001, 100.0), (0.0, 10.0), (0.0, 10.0), (0.0, 0.9)]
        
        # Apply user overrides
        x0 = np.array(default_guess)
        param_bounds = default_bounds
        
        if initial_guess:
            param_names = ['D', 'k_on', 'k_off', 'immobile_fraction']
            for i, name in enumerate(param_names[:n_params]):
                if name in initial_guess:
                    x0[i] = initial_guess[name]
        
        if bounds:
            param_names = ['D', 'k_on', 'k_off', 'immobile_fraction']
            for i, name in enumerate(param_names[:n_params]):
                if name in bounds:
                    param_bounds[i] = bounds[name]
        
        # Optimize
        result = optimize.minimize(
            self._objective,
            x0,
            args=(time_data, intensity_data, model_type, bleach_depth),
            method='L-BFGS-B',
            bounds=param_bounds,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        # Extract fitted parameters
        if model_type == 'diffusion_only':
            fitted_params = PDEParameters(D=result.x[0])
        elif model_type == 'diffusion_binding':
            fitted_params = PDEParameters(D=result.x[0], k_on=result.x[1], k_off=result.x[2])
        elif model_type == 'diffusion_immobile':
            fitted_params = PDEParameters(D=result.x[0], immobile_fraction=result.x[1])
        else:
            fitted_params = PDEParameters(
                D=result.x[0], k_on=result.x[1], k_off=result.x[2], immobile_fraction=result.x[3]
            )
        
        # Generate fitted curve
        sim_result = self.solver.simulate(fitted_params, time_data, bleach_depth=bleach_depth)
        fitted_curve = sim_result['recovery_curve']
        
        # Calculate statistics
        residuals = intensity_data - fitted_curve
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((intensity_data - np.mean(intensity_data)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # AIC
        n = len(intensity_data)
        if ss_res > 0:
            aic = n * np.log(ss_res / n) + 2 * n_params
        else:
            aic = -np.inf
        
        return {
            'params': fitted_params,
            'fitted_curve': fitted_curve,
            'concentration_fields': sim_result['concentration_fields'],
            'r2': r2,
            'aic': aic,
            'residuals': residuals,
            'success': result.success,
            'message': result.message
        }


def fit_frap_with_pde(image_stack: np.ndarray,
                      time_points: np.ndarray,
                      bleach_frame: int,
                      pixel_size: float = 1.0,
                      model_type: str = 'diffusion_only',
                      cell_threshold: Optional[float] = None,
                      bleach_threshold: float = 0.3) -> Dict[str, Any]:
    """
    High-level function to fit FRAP data using PDE model.
    
    Parameters:
    -----------
    image_stack : np.ndarray
        3D array (time, y, x) of FRAP images
    time_points : np.ndarray
        Time points corresponding to each frame (s)
    bleach_frame : int
        Index of first post-bleach frame
    pixel_size : float
        Physical size of each pixel (µm)
    model_type : str
        Model type: 'diffusion_only', 'diffusion_binding', 'diffusion_immobile', 'full'
    cell_threshold : float, optional
        Threshold for cell detection
    bleach_threshold : float
        Threshold for bleach region detection
        
    Returns:
    --------
    Dict containing fit results
    """
    # Extract geometry
    extractor = GeometryExtractor()
    
    # Use average of pre-bleach frames
    pre_bleach = np.mean(image_stack[:bleach_frame], axis=0)
    post_bleach = image_stack[bleach_frame]
    
    # Get masks
    cell_mask = extractor.extract_cell_mask(pre_bleach, threshold=cell_threshold)
    bleach_mask = extractor.extract_bleach_roi(pre_bleach, post_bleach, bleach_threshold)
    
    # Ensure bleach mask is within cell
    bleach_mask = bleach_mask & cell_mask
    
    if np.sum(bleach_mask) == 0:
        raise ValueError("No bleach region detected within cell mask")
    
    # Extract recovery curve from images
    recovery_curve = []
    for t in range(bleach_frame, len(image_stack)):
        frame = image_stack[t]
        mean_intensity = np.mean(frame[bleach_mask])
        recovery_curve.append(mean_intensity)
    
    recovery_curve = np.array(recovery_curve)
    
    # Normalize
    pre_intensity = np.mean(pre_bleach[bleach_mask])
    recovery_curve = recovery_curve / pre_intensity
    
    # Get corresponding time points
    recovery_time = time_points[bleach_frame:] - time_points[bleach_frame]
    
    # Fit
    fitter = PDEFRAPFitter(cell_mask, bleach_mask, pixel_size)
    result = fitter.fit(recovery_time, recovery_curve, model_type=model_type)
    
    # Add geometry info
    result['cell_mask'] = cell_mask
    result['bleach_mask'] = bleach_mask
    result['recovery_time'] = recovery_time
    result['recovery_data'] = recovery_curve
    
    return result


# Simplified 1D PDE solver for when full 2D is not needed
class SimplifiedPDESolver:
    """
    1D radial PDE solver for circular bleach spots.
    
    More efficient than full 2D when geometry is approximately circular.
    Solves: ∂C/∂t = D(∂²C/∂r² + (1/r)∂C/∂r) - k_on*C + k_off*(1-C)
    """
    
    def __init__(self, 
                 bleach_radius: float,
                 cell_radius: float,
                 n_points: int = 100):
        """
        Initialize 1D radial solver.
        
        Parameters:
        -----------
        bleach_radius : float
            Radius of bleach spot (µm)
        cell_radius : float
            Effective radius of cell/compartment (µm)
        n_points : int
            Number of radial grid points
        """
        self.bleach_radius = bleach_radius
        self.cell_radius = cell_radius
        self.n_points = n_points
        
        # Create radial grid
        self.r = np.linspace(0, cell_radius, n_points)
        self.dr = self.r[1] - self.r[0]
        
    def simulate(self,
                 D: float,
                 k_on: float = 0.0,
                 k_off: float = 0.0,
                 time_points: np.ndarray = None,
                 bleach_depth: float = 0.2) -> Dict[str, Any]:
        """
        Simulate radial FRAP recovery.
        
        Parameters:
        -----------
        D : float
            Diffusion coefficient (µm²/s)
        k_on : float
            Binding rate (s⁻¹)
        k_off : float
            Unbinding rate (s⁻¹)
        time_points : np.ndarray
            Output time points
        bleach_depth : float
            Initial intensity in bleach spot
            
        Returns:
        --------
        Dict with recovery curve and concentration profiles
        """
        if time_points is None:
            time_points = np.linspace(0, 10, 100)
        
        n = self.n_points
        r = self.r
        dr = self.dr
        
        # Initial condition
        C = np.ones(n)
        C[r <= self.bleach_radius] = bleach_depth
        
        # Time stepping (implicit scheme)
        dt = 0.01  # Small time step for stability
        
        recovery_curve = []
        profiles = []
        
        current_time = 0.0
        time_idx = 0
        
        while time_idx < len(time_points):
            target_time = time_points[time_idx]
            
            while current_time < target_time - 1e-10:
                step_dt = min(dt, target_time - current_time)
                
                # Build tridiagonal system for Crank-Nicolson
                # At r=0: use L'Hopital, (1/r)∂C/∂r → ∂²C/∂r²
                # Boundary at r=R: no flux ∂C/∂r = 0
                
                C_new = np.copy(C)
                
                for _ in range(3):  # A few iterations for implicit
                    # Interior points
                    for i in range(1, n-1):
                        ri = r[i]
                        
                        # Diffusion term
                        d2Cdr2 = (C[i+1] - 2*C[i] + C[i-1]) / dr**2
                        dCdr = (C[i+1] - C[i-1]) / (2 * dr)
                        laplacian = d2Cdr2 + dCdr / ri
                        
                        # Reaction term
                        reaction = -k_on * C[i] + k_off * (1 - C[i])
                        
                        # Update
                        C_new[i] = C[i] + step_dt * (D * laplacian + reaction)
                    
                    # r=0 boundary (symmetry)
                    d2Cdr2_0 = 2 * (C[1] - C[0]) / dr**2  # L'Hopital
                    reaction_0 = -k_on * C[0] + k_off * (1 - C[0])
                    C_new[0] = C[0] + step_dt * (D * d2Cdr2_0 + reaction_0)
                    
                    # r=R boundary (no flux)
                    C_new[-1] = C_new[-2]
                    
                    C = np.copy(C_new)
                
                # Enforce bounds
                C = np.clip(C, 0, 1)
                current_time += step_dt
            
            # Calculate mean intensity in bleach region (area-weighted)
            bleach_mask = r <= self.bleach_radius
            if np.any(bleach_mask):
                r_bleach = r[bleach_mask]
                C_bleach = C[bleach_mask]
                # Area-weighted average: ∫C·2πr·dr / ∫2πr·dr
                weights = r_bleach + 1e-10  # Avoid division by zero
                mean_intensity = np.sum(C_bleach * weights) / np.sum(weights)
            else:
                mean_intensity = C[0]
            
            recovery_curve.append(mean_intensity)
            profiles.append(C.copy())
            time_idx += 1
        
        return {
            'recovery_curve': np.array(recovery_curve),
            'concentration_profiles': np.array(profiles),
            'radial_positions': r,
            'time': time_points
        }


def quick_pde_fit(time_data: np.ndarray,
                  intensity_data: np.ndarray,
                  bleach_radius: float = 1.0,
                  cell_radius: float = 10.0,
                  model_type: str = 'diffusion_only') -> Dict[str, Any]:
    """
    Quick PDE-based fitting using simplified 1D radial model.
    
    This is faster than full 2D PDE and suitable for approximately circular bleach spots.
    
    Parameters:
    -----------
    time_data : np.ndarray
        Time points (s)
    intensity_data : np.ndarray  
        Normalized intensity (0-1)
    bleach_radius : float
        Radius of bleach spot (µm)
    cell_radius : float
        Effective cell/compartment radius (µm)
    model_type : str
        'diffusion_only' or 'diffusion_binding'
        
    Returns:
    --------
    Dict with fitted parameters and curve
    """
    solver = SimplifiedPDESolver(bleach_radius, cell_radius)
    bleach_depth = intensity_data[0]
    
    def objective(params):
        if model_type == 'diffusion_only':
            D = params[0]
            result = solver.simulate(D, 0, 0, time_data, bleach_depth)
        else:
            D, k_on, k_off = params
            result = solver.simulate(D, k_on, k_off, time_data, bleach_depth)
        
        simulated = result['recovery_curve']
        return np.sum((intensity_data - simulated) ** 2)
    
    # Optimize
    if model_type == 'diffusion_only':
        x0 = [1.0]
        bounds = [(0.01, 100)]
    else:
        x0 = [1.0, 0.1, 0.1]
        bounds = [(0.01, 100), (0, 10), (0, 10)]
    
    result = optimize.minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    
    # Get fitted curve
    if model_type == 'diffusion_only':
        D = result.x[0]
        sim = solver.simulate(D, 0, 0, time_data, bleach_depth)
        params_dict = {'D': D}
    else:
        D, k_on, k_off = result.x
        sim = solver.simulate(D, k_on, k_off, time_data, bleach_depth)
        params_dict = {'D': D, 'k_on': k_on, 'k_off': k_off}
    
    fitted_curve = sim['recovery_curve']
    
    # Statistics
    residuals = intensity_data - fitted_curve
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((intensity_data - np.mean(intensity_data)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    n = len(intensity_data)
    n_params = len(result.x)
    aic = n * np.log(ss_res / n) + 2 * n_params if ss_res > 0 else -np.inf
    
    return {
        'params': params_dict,
        'fitted_curve': fitted_curve,
        'r2': r2,
        'aic': aic,
        'residuals': residuals,
        'success': result.success,
        'concentration_profiles': sim['concentration_profiles'],
        'radial_positions': sim['radial_positions']
    }
