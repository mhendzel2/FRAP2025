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
from scipy.sparse import diags, csr_matrix, bmat
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
    k_on2: float = 0.0  # Binding rate 2 (s⁻¹)
    k_off2: float = 0.0  # Unbinding rate 2 (s⁻¹)
    immobile_fraction: float = 0.0  # Fraction of immobile protein (0-1)
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to array for optimization."""
        return np.array([self.D, self.k_on, self.k_off, self.k_on2, self.k_off2, self.immobile_fraction])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'PDEParameters':
        """Create parameters from array."""
        return cls(
            D=arr[0],
            k_on=arr[1] if len(arr) > 1 else 0.0,
            k_off=arr[2] if len(arr) > 2 else 0.0,
            k_on2=arr[3] if len(arr) > 3 else 0.0,
            k_off2=arr[4] if len(arr) > 4 else 0.0,
            immobile_fraction=arr[5] if len(arr) > 5 else 0.0
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


class ReactionDiffusionSystemSolver:
    """
    Solver for coupled systems of reaction-diffusion equations.
    Supports 1 or 2 binding components.

    Model:
    u: Free diffusing protein
    v1: Bound protein 1 (immobile)
    v2: Bound protein 2 (immobile)

    Equations:
    ∂u/∂t = D∇²u - (kon1 + kon2)u + koff1*v1 + koff2*v2
    ∂v1/∂t = kon1*u - koff1*v1
    ∂v2/∂t = kon2*u - koff2*v2
    """

    def __init__(self,
                 cell_mask: np.ndarray,
                 bleach_mask: np.ndarray,
                 pixel_size: float = 1.0,
                 boundary_condition: BoundaryCondition = BoundaryCondition.NEUMANN):
        self.helper = FiniteDifferenceSolver(cell_mask, bleach_mask, pixel_size, boundary_condition)
        self.n = self.helper.n_points
        self.bleach_mask = bleach_mask
        self.cell_mask = cell_mask
        self.ny, self.nx = cell_mask.shape
        self.helper._create_index_mapping()
        self.idx_1d_to_2d = self.helper.idx_1d_to_2d

    def simulate(self,
                 params: PDEParameters,
                 time_points: np.ndarray,
                 model_type: str = 'reaction_diffusion_1',
                 bleach_depth: float = 0.0) -> Dict[str, Any]:

        n = self.n
        dt = np.min(np.diff(time_points)) if len(time_points) > 1 else 0.1

        # Calculate equilibrium fractions (sum = 1.0)
        # u_eq + v1_eq + v2_eq = 1.0
        # v1_eq = (kon1/koff1) * u_eq = K1 * u_eq
        # v2_eq = (kon2/koff2) * u_eq = K2 * u_eq

        K1 = params.k_on / params.k_off if params.k_off > 1e-9 else 0

        if model_type == 'reaction_diffusion_2':
            K2 = params.k_on2 / params.k_off2 if params.k_off2 > 1e-9 else 0
            num_species = 3
        else:
            K2 = 0
            num_species = 2

        u_eq = 1.0 / (1.0 + K1 + K2)
        v1_eq = K1 * u_eq
        v2_eq = K2 * u_eq

        # Initial conditions: apply bleach
        # u(0) = u_eq * bleach
        # v1(0) = v1_eq * bleach
        # v2(0) = v2_eq * bleach

        u = np.full(n, u_eq)
        v1 = np.full(n, v1_eq)
        v2 = np.full(n, v2_eq)

        for idx in range(n):
            j, i = self.idx_1d_to_2d[idx]
            if self.bleach_mask[j, i]:
                u[idx] *= bleach_depth
                v1[idx] *= bleach_depth
                v2[idx] *= bleach_depth

        # Stack state vector X = [u, v1, v2]
        if num_species == 3:
            X = np.concatenate([u, v1, v2])
        else:
            X = np.concatenate([u, v1])

        # Build block matrices
        # Laplacian block for u
        L_u = self.helper._build_laplacian_matrix(params.D)
        Z = csr_matrix((n, n))
        I = diags(np.ones(n), 0, format='csr')

        # Reaction blocks
        # R11: u -> u term: -(kon1 + kon2)
        rate_u_loss = -(params.k_on + (params.k_on2 if num_species == 3 else 0))
        R_uu = diags(np.full(n, rate_u_loss), 0, format='csr')

        # R12: v1 -> u term: +koff1
        R_uv1 = diags(np.full(n, params.k_off), 0, format='csr')

        # R21: u -> v1 term: +kon1
        R_v1u = diags(np.full(n, params.k_on), 0, format='csr')

        # R22: v1 -> v1 term: -koff1
        R_v1v1 = diags(np.full(n, -params.k_off), 0, format='csr')

        if num_species == 3:
            # R13: v2 -> u term: +koff2
            R_uv2 = diags(np.full(n, params.k_off2), 0, format='csr')

            # R31: u -> v2 term: +kon2
            R_v2u = diags(np.full(n, params.k_on2), 0, format='csr')

            # R33: v2 -> v2 term: -koff2
            R_v2v2 = diags(np.full(n, -params.k_off2), 0, format='csr')

            # System matrix A
            # [ L + Ruu   Ruv1    Ruv2 ]
            # [ Rv1u      Rv1v1   0    ]
            # [ Rv2u      0       Rv2v2]

            A_blocks = [
                [L_u + R_uu, R_uv1, R_uv2],
                [R_v1u,      R_v1v1, Z],
                [R_v2u,      Z,      R_v2v2]
            ]
        else:
            # System matrix A
            # [ L + Ruu   Ruv1  ]
            # [ Rv1u      Rv1v1 ]

            A_blocks = [
                [L_u + R_uu, R_uv1],
                [R_v1u,      R_v1v1]
            ]

        A = bmat(A_blocks, format='csr')

        # Crank-Nicolson matrices
        total_size = num_species * n
        I_total = diags(np.ones(total_size), 0, format='csr')

        A_implicit = I_total - 0.5 * dt * A
        A_explicit = I_total + 0.5 * dt * A

        # Bleach indices for averaging
        bleach_indices = []
        for idx in range(n):
            j, i = self.idx_1d_to_2d[idx]
            if self.bleach_mask[j, i]:
                bleach_indices.append(idx)
        bleach_indices = np.array(bleach_indices)

        recovery_curve = []
        concentration_fields = []

        current_time = 0.0
        time_idx = 0

        while time_idx < len(time_points):
            target_time = time_points[time_idx]

            while current_time < target_time:
                step_dt = min(dt, target_time - current_time)

                if abs(step_dt - dt) > 1e-10:
                    A_implicit = I_total - 0.5 * step_dt * A
                    A_explicit = I_total + 0.5 * step_dt * A

                rhs = A_explicit @ X
                X = spsolve(A_implicit, rhs)

                # Clip to physical range [0, 1]?
                # Actually sum should be <= 1 (approx).
                X = np.maximum(X, 0)

                current_time += step_dt

            # Extract components
            u_curr = X[:n]
            v1_curr = X[n:2*n]
            if num_species == 3:
                v2_curr = X[2*n:]
                total = u_curr + v1_curr + v2_curr
            else:
                total = u_curr + v1_curr

            # Record
            if len(bleach_indices) > 0:
                mean_int = np.mean(total[bleach_indices])
            else:
                mean_int = np.mean(total)
            recovery_curve.append(mean_int)

            # Store field (total concentration)
            field_2d = np.zeros((self.ny, self.nx))
            for idx in range(n):
                j, i = self.idx_1d_to_2d[idx]
                field_2d[j, i] = total[idx]
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
        self.system_solver = ReactionDiffusionSystemSolver(cell_mask, bleach_mask, pixel_size)
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
        elif model_type == 'reaction_diffusion_1':
            params = PDEParameters(D=param_array[0], k_on=param_array[1], k_off=param_array[2])
        elif model_type == 'reaction_diffusion_2':
            params = PDEParameters(
                D=param_array[0],
                k_on=param_array[1],
                k_off=param_array[2],
                k_on2=param_array[3],
                k_off2=param_array[4]
            )
        else:  # full model
            params = PDEParameters(
                D=param_array[0], 
                k_on=param_array[1], 
                k_off=param_array[2],
                immobile_fraction=param_array[3]
            )
        
        # Simulate
        try:
            if model_type in ['reaction_diffusion_1', 'reaction_diffusion_2']:
                result = self.system_solver.simulate(params, time_data, model_type, bleach_depth)
            else:
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
            One of: 'diffusion_only', 'diffusion_binding', 'diffusion_immobile', 'full',
                    'reaction_diffusion_1', 'reaction_diffusion_2', 'auto_binding'
        bleach_depth : float, optional
            Initial intensity after bleach. If None, estimated from data
        initial_guess : dict, optional
            Initial parameter guesses
        bounds : dict, optional
            Parameter bounds
            
        Returns:
        --------
        Dict containing fit results
        """
        # Handle automatic model selection
        if model_type == 'auto_binding':
            logger.info("Running auto model selection for binding components...")

            # Fit 1-component
            res1 = self.fit(time_data, intensity_data, model_type='reaction_diffusion_1',
                           bleach_depth=bleach_depth, initial_guess=initial_guess, bounds=bounds)

            # Fit 2-component
            res2 = self.fit(time_data, intensity_data, model_type='reaction_diffusion_2',
                           bleach_depth=bleach_depth, initial_guess=initial_guess, bounds=bounds)

            # Model selection using BIC
            n = len(intensity_data)
            k1 = 3 # D, kon, koff
            k2 = 5 # D, kon1, koff1, kon2, koff2

            # BIC = AIC - 2k + k*ln(n)
            bic1 = res1['aic'] - 2*k1 + k1*np.log(n)
            bic2 = res2['aic'] - 2*k2 + k2*np.log(n)

            logger.info(f"Model Selection: 1-comp BIC={bic1:.2f}, 2-comp BIC={bic2:.2f}")

            # Require significant improvement to pick more complex model (BIC diff > 10)
            if bic2 < bic1 - 10:
                logger.info("Selected 2-component model")
                best_res = res2
            else:
                logger.info("Selected 1-component model")
                best_res = res1

            best_res['model_comparison'] = {
                'bic1': bic1,
                'bic2': bic2,
                'selected': best_res['model_type']
            }
            return best_res

        # Estimate bleach depth if not provided
        if bleach_depth is None:
            bleach_depth = intensity_data[0]
        
        # Set up parameter bounds and initial guesses
        if model_type == 'diffusion_only':
            n_params = 1
            default_guess = [1.0]  # D
            default_bounds = [(0.001, 100.0)]
        elif model_type in ['diffusion_binding', 'reaction_diffusion_1']:
            n_params = 3
            default_guess = [1.0, 0.1, 0.1]  # D, k_on, k_off
            default_bounds = [(0.001, 100.0), (0.0, 10.0), (0.0, 10.0)]
        elif model_type == 'diffusion_immobile':
            n_params = 2
            default_guess = [1.0, 0.2]  # D, immobile_fraction
            default_bounds = [(0.001, 100.0), (0.0, 0.9)]
        elif model_type == 'reaction_diffusion_2':
            n_params = 5
            # D, k_on1, k_off1, k_on2, k_off2
            default_guess = [1.0, 0.1, 0.1, 0.1, 0.01]
            default_bounds = [(0.001, 100.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
        else:  # full
            n_params = 4
            default_guess = [1.0, 0.1, 0.1, 0.2]
            default_bounds = [(0.001, 100.0), (0.0, 10.0), (0.0, 10.0), (0.0, 0.9)]
        
        # Apply user overrides
        x0 = np.array(default_guess)
        param_bounds = default_bounds
        
        # Determine parameter names based on model type
        if model_type == 'diffusion_only':
            param_names = ['D']
        elif model_type in ['diffusion_binding', 'reaction_diffusion_1']:
            param_names = ['D', 'k_on', 'k_off']
        elif model_type == 'diffusion_immobile':
            param_names = ['D', 'immobile_fraction']
        elif model_type == 'reaction_diffusion_2':
            param_names = ['D', 'k_on', 'k_off', 'k_on2', 'k_off2']
        else: # full
            param_names = ['D', 'k_on', 'k_off', 'immobile_fraction']

        if initial_guess:
            for i, name in enumerate(param_names):
                if name in initial_guess:
                    x0[i] = initial_guess[name]
        
        if bounds:
            for i, name in enumerate(param_names):
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
        elif model_type in ['diffusion_binding', 'reaction_diffusion_1']:
            fitted_params = PDEParameters(D=result.x[0], k_on=result.x[1], k_off=result.x[2])
        elif model_type == 'diffusion_immobile':
            fitted_params = PDEParameters(D=result.x[0], immobile_fraction=result.x[1])
        elif model_type == 'reaction_diffusion_2':
            fitted_params = PDEParameters(
                D=result.x[0], k_on=result.x[1], k_off=result.x[2],
                k_on2=result.x[3], k_off2=result.x[4]
            )
        else:
            fitted_params = PDEParameters(
                D=result.x[0], k_on=result.x[1], k_off=result.x[2], immobile_fraction=result.x[3]
            )
        
        # Generate fitted curve
        if model_type in ['reaction_diffusion_1', 'reaction_diffusion_2']:
            sim_result = self.system_solver.simulate(fitted_params, time_data, model_type, bleach_depth)
        else:
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
            'message': result.message,
            'model_type': model_type
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
        Model type. If 'auto_binding', compares 1 vs 2 binding components.
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


# -----------------------------------------------------------------------------
# MULTI-COMPONENT REACTION-DIFFUSION SOLVER (Two-State Binding)
# -----------------------------------------------------------------------------


@dataclass
class CoupledPDEParameters:
    """Parameters for coupled multi-state FRAP fitting (two binding interactions)."""

    D: float              # Diffusion coefficient (µm²/s)
    k_on1: float = 0.0    # Binding rate 1 (s⁻¹)
    k_off1: float = 0.0   # Unbinding rate 1 (s⁻¹)
    k_on2: float = 0.0    # Binding rate 2 (s⁻¹)
    k_off2: float = 0.0   # Unbinding rate 2 (s⁻¹)

    @property
    def fractions(self) -> Tuple[float, float, float]:
        """Return equilibrium fractions (Mobile, Bound1, Bound2).

        At steady state (local kinetics only):
        V1 = (k_on1/k_off1) * U, V2 = (k_on2/k_off2) * U, with U+V1+V2 = 1.
        """
        K1 = (self.k_on1 / self.k_off1) if self.k_off1 > 1e-12 else 0.0
        K2 = (self.k_on2 / self.k_off2) if self.k_off2 > 1e-12 else 0.0
        total = 1.0 + K1 + K2
        if total <= 0:
            return 1.0, 0.0, 0.0
        return 1.0 / total, K1 / total, K2 / total


class CoupledPDESolver:
    """Solve the coupled U/V1/V2 reaction–diffusion FRAP system.

    Mobile species U diffuses; bound states V1 and V2 are immobile.

    Operator splitting per step:
    - Reaction (local kinetics) via a backward-Euler (semi-implicit) update
    - Diffusion of U via Crank–Nicolson
    """

    def __init__(
        self,
        cell_mask: np.ndarray,
        bleach_mask: np.ndarray,
        pixel_size: float = 1.0,
        boundary_condition: BoundaryCondition = BoundaryCondition.NEUMANN,
    ):
        self.cell_mask = cell_mask.astype(bool)
        self.bleach_mask = bleach_mask.astype(bool)
        self.pixel_size = float(pixel_size)
        self.boundary_condition = boundary_condition

        self.ny, self.nx = self.cell_mask.shape
        self.n_points = int(np.sum(self.cell_mask))

        self.idx_2d_to_1d = np.full((self.ny, self.nx), -1, dtype=int)
        self.idx_1d_to_2d: np.ndarray

        idx_pairs = []
        idx = 0
        for j in range(self.ny):
            for i in range(self.nx):
                if self.cell_mask[j, i]:
                    self.idx_2d_to_1d[j, i] = idx
                    idx_pairs.append((j, i))
                    idx += 1
        self.idx_1d_to_2d = np.array(idx_pairs, dtype=int)

        bleach_indices = []
        for k in range(self.n_points):
            j, i = self.idx_1d_to_2d[k]
            if self.bleach_mask[j, i]:
                bleach_indices.append(k)
        self.bleach_indices = np.array(bleach_indices, dtype=int)

    def _build_laplacian(self, D: float) -> csr_matrix:
        """Standard 5-point Laplacian matrix for the masked domain."""
        n = self.n_points
        h2 = self.pixel_size ** 2
        coeff = float(D) / (h2 + 1e-20)

        row_idx: List[int] = []
        col_idx: List[int] = []
        values: List[float] = []

        for idx in range(n):
            j, i = self.idx_1d_to_2d[idx]

            diag = -4.0 * coeff
            neighbor_count = 0
            neighbors = [(j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1)]
            for nj, ni in neighbors:
                if 0 <= nj < self.ny and 0 <= ni < self.nx:
                    if self.cell_mask[nj, ni]:
                        n_idx = int(self.idx_2d_to_1d[nj, ni])
                        row_idx.append(idx)
                        col_idx.append(n_idx)
                        values.append(coeff)
                        neighbor_count += 1
                    elif self.boundary_condition == BoundaryCondition.NEUMANN:
                        diag += coeff

            if self.boundary_condition == BoundaryCondition.NEUMANN:
                diag = -neighbor_count * coeff

            row_idx.append(idx)
            col_idx.append(idx)
            values.append(diag)

        return csr_matrix((values, (row_idx, col_idx)), shape=(n, n))

    @staticmethod
    def _reaction_step_backward_euler(
        U: np.ndarray,
        V1: np.ndarray,
        V2: np.ndarray,
        dt: float,
        p: CoupledPDEParameters,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stable local kinetics update using backward Euler.

        Solves per-voxel linear system:
        dU/dt = -(k1+k2)U + koff1 V1 + koff2 V2
        dV1/dt = kon1 U - koff1 V1
        dV2/dt = kon2 U - koff2 V2
        """
        dt = float(dt)
        k1on = float(p.k_on1)
        k1off = float(p.k_off1)
        k2on = float(p.k_on2)
        k2off = float(p.k_off2)

        a1 = 1.0 + dt * k1off
        a2 = 1.0 + dt * k2off
        # Coefficient for U_new after eliminating V1_new/V2_new
        coef = (
            1.0
            + dt * (k1on + k2on)
            - (dt * dt * k1off * k1on) / a1
            - (dt * dt * k2off * k2on) / a2
        )
        coef = max(coef, 1e-12)

        rhs = U + (dt * k1off / a1) * V1 + (dt * k2off / a2) * V2
        U_new = rhs / coef
        V1_new = (V1 + dt * k1on * U_new) / a1
        V2_new = (V2 + dt * k2on * U_new) / a2

        # Physical bounds (concentrations / fractions)
        U_new = np.clip(U_new, 0.0, 1.0)
        V1_new = np.clip(V1_new, 0.0, 1.0)
        V2_new = np.clip(V2_new, 0.0, 1.0)
        return U_new, V1_new, V2_new

    def simulate(
        self,
        params: CoupledPDEParameters,
        time_points: np.ndarray,
        bleach_depth: float = 0.0,
        dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Simulate recovery curve for the coupled binding model.

        Parameters
        ----------
        params : CoupledPDEParameters
            Physical parameters.
        time_points : np.ndarray
            Time points (s) at which to sample output, starting at 0.
        bleach_depth : float
            Fraction remaining immediately after bleach in ROI.
        dt : float, optional
            Internal integration step. If None, chosen from time_points.
        """
        time_points = np.asarray(time_points, dtype=float)
        if time_points.ndim != 1 or len(time_points) == 0:
            raise ValueError("time_points must be a 1D non-empty array")

        n = self.n_points
        if n <= 0:
            raise ValueError("Empty cell mask domain")

        # 1) Pre-bleach equilibrium initial conditions
        f_m, f_b1, f_b2 = params.fractions
        U = np.full(n, f_m, dtype=float)
        V1 = np.full(n, f_b1, dtype=float)
        V2 = np.full(n, f_b2, dtype=float)

        # 2) Apply bleach (all species affected)
        bleach_depth = float(bleach_depth)
        if len(self.bleach_indices) > 0:
            U[self.bleach_indices] *= bleach_depth
            V1[self.bleach_indices] *= bleach_depth
            V2[self.bleach_indices] *= bleach_depth

        # Diffusion operator (U only). Note: CN matrices depend on dt (built per step).
        L = self._build_laplacian(params.D)
        I = diags(np.ones(n), 0, format='csr')

        recovery_curve: List[float] = []

        def _record() -> None:
            total = U + V1 + V2
            if len(self.bleach_indices) > 0:
                recovery_curve.append(float(np.mean(total[self.bleach_indices])))
            else:
                recovery_curve.append(float(np.mean(total)))

        # Record at the first time point
        _record()

        # Step interval-by-interval between observation time points
        for k in range(len(time_points) - 1):
            delta_t = float(time_points[k + 1] - time_points[k])
            if delta_t <= 0:
                _record()
                continue

            # Optional sub-stepping for accuracy if dt is provided
            if dt is None or dt >= delta_t:
                n_sub = 1
                dt_sub = delta_t
            else:
                n_sub = int(np.ceil(delta_t / float(dt)))
                n_sub = max(n_sub, 1)
                dt_sub = delta_t / n_sub

            for _ in range(n_sub):
                # Reaction
                U, V1, V2 = self._reaction_step_backward_euler(U, V1, V2, dt_sub, params)

                # Diffusion (Crank–Nicolson)
                A_implicit = I - 0.5 * dt_sub * L
                A_explicit = I + 0.5 * dt_sub * L
                rhs = A_explicit @ U
                U = np.asarray(spsolve(A_implicit, rhs), dtype=float)
                U = np.clip(U, 0.0, 1.0)

            _record()

        return {
            'time': time_points,
            'recovery': np.asarray(recovery_curve, dtype=float),
            'final_distribution': (U, V1, V2),
        }


def compare_binding_models(
    image_stack: np.ndarray,
    time_points: np.ndarray,
    bleach_frame: int,
    pixel_size: float = 1.0,
    bleach_threshold: float = 0.3,
    cell_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Fit and compare 1-binding vs 2-binding coupled PDE models using AIC.

    Returns a dict containing per-model parameters and AIC/RSS, plus a preferred model.
    """
    # Extract geometry
    extractor = GeometryExtractor()
    pre = np.mean(image_stack[:bleach_frame], axis=0)
    post = image_stack[bleach_frame]

    cell_mask = extractor.extract_cell_mask(pre, threshold=cell_threshold)
    bleach_mask = extractor.extract_bleach_roi(pre, post, intensity_drop_threshold=bleach_threshold)
    bleach_mask = bleach_mask & cell_mask
    if np.sum(bleach_mask) == 0:
        raise ValueError("No bleach region detected within cell mask")

    # Normalized recovery curve in bleach ROI
    y_data = np.array([np.mean(frame[bleach_mask]) for frame in image_stack[bleach_frame:]], dtype=float)
    pre_int = float(np.mean(pre[bleach_mask]))
    if pre_int <= 0:
        raise ValueError("Invalid pre-bleach intensity for normalization")
    y_data = y_data / pre_int
    bleach_depth = float(y_data[0])

    t_data = np.asarray(time_points[bleach_frame:], dtype=float) - float(time_points[bleach_frame])
    solver = CoupledPDESolver(cell_mask, bleach_mask, pixel_size=pixel_size)

    # --- Model 1: Single binding (k_on2 = k_off2 = 0) ---
    def obj_1state(x):
        D, kon, koff = x
        if D <= 0 or kon < 0 or koff < 0:
            return 1e12
        p = CoupledPDEParameters(D=D, k_on1=kon, k_off1=koff, k_on2=0.0, k_off2=0.0)
        sim = solver.simulate(p, t_data, bleach_depth=bleach_depth)
        return float(np.sum((y_data - sim['recovery']) ** 2))

    res1 = optimize.minimize(obj_1state, [1.0, 0.1, 0.1], method='Nelder-Mead')
    rss1 = float(res1.fun)
    aic1 = len(y_data) * np.log(max(rss1, 1e-30) / len(y_data)) + 2 * 3

    # --- Model 2: Two binding ---
    def obj_2state(x):
        D, k1on, k1off, k2on, k2off = x
        if any(v < 0 for v in x) or D <= 0:
            return 1e12
        p = CoupledPDEParameters(D=D, k_on1=k1on, k_off1=k1off, k_on2=k2on, k_off2=k2off)
        sim = solver.simulate(p, t_data, bleach_depth=bleach_depth)
        return float(np.sum((y_data - sim['recovery']) ** 2))

    d_guess, k_guess, koff_guess = (res1.x if hasattr(res1, 'x') else [1.0, 0.1, 0.1])
    guess2 = [float(d_guess), float(k_guess), float(koff_guess), 0.05, 0.05]
    res2 = optimize.minimize(obj_2state, guess2, method='Nelder-Mead')
    rss2 = float(res2.fun)
    aic2 = len(y_data) * np.log(max(rss2, 1e-30) / len(y_data)) + 2 * 5

    preferred = "Two-State" if aic2 < aic1 - 2 else "Single-State"

    return {
        "model_1": {
            "D": float(res1.x[0]),
            "k_on": float(res1.x[1]),
            "k_off": float(res1.x[2]),
            "AIC": float(aic1),
            "RSS": float(rss1),
        },
        "model_2": {
            "D": float(res2.x[0]),
            "k_on1": float(res2.x[1]),
            "k_off1": float(res2.x[2]),
            "k_on2": float(res2.x[3]),
            "k_off2": float(res2.x[4]),
            "AIC": float(aic2),
            "RSS": float(rss2),
        },
        "preferred_model": preferred,
        "delta_aic": float(aic1 - aic2),
    }
