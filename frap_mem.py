"""
FRAP Maximum Entropy Method (MEM) - Continuous Distribution Analysis

This module implements Maximum Entropy reconstruction for FRAP analysis,
allowing extraction of continuous distributions of rate constants without
assuming a fixed number of exponential components.

Theory:
-------
The FRAP recovery can be expressed as a Laplace transform:
    y(t) = ∫ P(k) · (1 - e^(-kt)) dk

Where P(k) is the distribution of rate constants. This is an ill-posed
inverse problem requiring regularization.

Methods implemented:
1. Non-Negative Least Squares (NNLS) with Tikhonov regularization
2. Maximum Entropy regularization
3. L1 regularization (LASSO) for sparse solutions

The resulting spectrum reveals:
- Number of distinct populations (peaks)
- Width of each population (peak broadness)
- Continuous binding state distributions

Author: FRAP Analysis Suite
"""

import numpy as np
from scipy import optimize
from scipy.linalg import lstsq
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logger.info("cvxpy not available. Using scipy-based solvers.")

try:
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.info("scikit-learn not available. Using basic solvers.")


@dataclass
class MEMResult:
    """Results from Maximum Entropy Method analysis."""
    rate_constants: np.ndarray  # k values (logarithmic grid)
    distribution: np.ndarray  # P(k) amplitudes
    fitted_curve: np.ndarray  # Reconstructed recovery curve
    time: np.ndarray  # Time points
    r2: float  # Goodness of fit
    aic: float  # Akaike Information Criterion
    peaks: List[Dict[str, float]]  # Detected peaks
    entropy: float  # Entropy of solution
    regularization_param: float  # λ used
    
    def get_mobile_fraction(self) -> float:
        """Calculate total mobile fraction from distribution."""
        return np.trapz(self.distribution, np.log10(self.rate_constants)) * 100
    
    def get_mean_rate(self) -> float:
        """Calculate amplitude-weighted mean rate constant."""
        if np.sum(self.distribution) > 0:
            weights = self.distribution / np.sum(self.distribution)
            return np.sum(weights * self.rate_constants)
        return 0.0
    
    def get_half_time(self) -> float:
        """Calculate effective half-time from mean rate."""
        mean_k = self.get_mean_rate()
        return np.log(2) / mean_k if mean_k > 0 else np.inf


class RateConstantBasis:
    """
    Generate basis functions for rate constant distribution fitting.
    
    Creates a set of exponential recovery functions with rates logarithmically
    spaced to cover the expected range of biological rate constants.
    """
    
    def __init__(self,
                 k_min: float = 1e-3,
                 k_max: float = 10.0,
                 n_basis: int = 100,
                 spacing: str = 'log'):
        """
        Initialize basis function generator.
        
        Parameters:
        -----------
        k_min : float
            Minimum rate constant (s⁻¹)
        k_max : float
            Maximum rate constant (s⁻¹)
        n_basis : int
            Number of basis functions
        spacing : str
            'log' for logarithmic spacing, 'linear' for linear
        """
        self.k_min = k_min
        self.k_max = k_max
        self.n_basis = n_basis
        self.spacing = spacing
        
        # Generate rate constant grid
        if spacing == 'log':
            self.k_values = np.logspace(np.log10(k_min), np.log10(k_max), n_basis)
        else:
            self.k_values = np.linspace(k_min, k_max, n_basis)
    
    def generate_matrix(self, time_points: np.ndarray) -> np.ndarray:
        """
        Generate the basis function matrix A.
        
        Each column j is: A[:,j] = 1 - exp(-k_j * t)
        
        Parameters:
        -----------
        time_points : np.ndarray
            Time points for evaluation
            
        Returns:
        --------
        np.ndarray
            Matrix of shape (n_timepoints, n_basis)
        """
        n_t = len(time_points)
        A = np.zeros((n_t, self.n_basis))
        
        for j, k in enumerate(self.k_values):
            A[:, j] = 1.0 - np.exp(-k * time_points)
        
        return A
    
    def generate_derivative_matrix(self, time_points: np.ndarray) -> np.ndarray:
        """
        Generate derivative matrix for smoothness regularization.
        
        Returns second derivative operator for Tikhonov regularization.
        """
        n = self.n_basis
        # Second difference matrix (approximates second derivative)
        D = np.zeros((n-2, n))
        for i in range(n-2):
            D[i, i] = 1
            D[i, i+1] = -2
            D[i, i+2] = 1
        
        return D


class NNLSSolver:
    """
    Non-Negative Least Squares solver with various regularization options.
    
    Solves: min ||Ax - b||² + λ·R(x) subject to x ≥ 0
    
    Where R(x) is a regularization term.
    """
    
    @staticmethod
    def solve_basic_nnls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Basic NNLS without regularization."""
        from scipy.optimize import nnls
        x, residual = nnls(A, b)
        return x
    
    @staticmethod
    def solve_tikhonov_nnls(A: np.ndarray, 
                           b: np.ndarray, 
                           alpha: float = 0.01,
                           D: Optional[np.ndarray] = None) -> np.ndarray:
        """
        NNLS with Tikhonov (Ridge) regularization.
        
        Minimizes: ||Ax - b||² + α||Dx||²
        
        Parameters:
        -----------
        A : np.ndarray
            Design matrix
        b : np.ndarray
            Target values
        alpha : float
            Regularization strength
        D : np.ndarray, optional
            Regularization matrix (default: identity)
            
        Returns:
        --------
        np.ndarray
            Non-negative solution vector
        """
        n = A.shape[1]
        
        if D is None:
            D = np.eye(n)
        
        # Augmented system
        A_aug = np.vstack([A, np.sqrt(alpha) * D])
        b_aug = np.concatenate([b, np.zeros(D.shape[0])])
        
        from scipy.optimize import nnls
        x, residual = nnls(A_aug, b_aug)
        
        return x
    
    @staticmethod
    def solve_elastic_net_nnls(A: np.ndarray,
                               b: np.ndarray,
                               alpha: float = 0.01,
                               l1_ratio: float = 0.5) -> np.ndarray:
        """
        NNLS with Elastic Net regularization (L1 + L2).
        
        Combines sparsity (L1) with smoothness (L2).
        """
        n = A.shape[1]
        
        def objective(x):
            x = np.maximum(x, 0)  # Enforce non-negativity
            residual = np.sum((A @ x - b) ** 2)
            l1_penalty = alpha * l1_ratio * np.sum(np.abs(x))
            l2_penalty = alpha * (1 - l1_ratio) * np.sum(x ** 2)
            return residual + l1_penalty + l2_penalty
        
        # Initialize with basic NNLS
        x0 = NNLSSolver.solve_basic_nnls(A, b)
        
        # Optimize with bounds
        result = optimize.minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=[(0, None) for _ in range(n)]
        )
        
        return np.maximum(result.x, 0)


class MaximumEntropySolver:
    """
    Maximum Entropy solver for distribution reconstruction.
    
    Maximizes entropy: S = -∫ P(k) log(P(k)/m(k)) dk
    Subject to: χ² constraint (fit quality)
    
    Where m(k) is a prior distribution (default: flat).
    """
    
    def __init__(self, 
                 prior: Optional[np.ndarray] = None,
                 target_chi2: float = 1.0):
        """
        Initialize Maximum Entropy solver.
        
        Parameters:
        -----------
        prior : np.ndarray, optional
            Prior distribution m(k). Default is flat.
        target_chi2 : float
            Target reduced chi-squared for data fit
        """
        self.prior = prior
        self.target_chi2 = target_chi2
    
    def _entropy(self, P: np.ndarray, m: np.ndarray) -> float:
        """Calculate entropy: S = -∑ P_i log(P_i/m_i)"""
        # Avoid log(0)
        P_safe = np.maximum(P, 1e-20)
        m_safe = np.maximum(m, 1e-20)
        
        return -np.sum(P_safe * np.log(P_safe / m_safe))
    
    def solve(self,
              A: np.ndarray,
              b: np.ndarray,
              noise_estimate: Optional[float] = None,
              max_iter: int = 100) -> Tuple[np.ndarray, float]:
        """
        Solve using Maximum Entropy method.
        
        Parameters:
        -----------
        A : np.ndarray
            Design matrix
        b : np.ndarray
            Target values
        noise_estimate : float, optional
            Estimated noise level. If None, estimated from data.
        max_iter : int
            Maximum iterations
            
        Returns:
        --------
        Tuple[np.ndarray, float]
            Solution vector and regularization parameter used
        """
        n_data, n_basis = A.shape
        
        # Set prior
        if self.prior is None:
            m = np.ones(n_basis) / n_basis
        else:
            m = self.prior / np.sum(self.prior)
        
        # Estimate noise if not provided
        if noise_estimate is None:
            # Use MAD estimator
            residuals = b - np.mean(b)
            noise_estimate = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
            noise_estimate = max(noise_estimate, 0.01)
        
        sigma2 = noise_estimate ** 2
        
        # Lagrange multiplier search
        def objective(log_lambda):
            lam = np.exp(log_lambda)
            
            # Solve regularized problem
            P = self._solve_fixed_lambda(A, b, m, lam)
            
            # Calculate chi-squared
            residuals = A @ P - b
            chi2 = np.sum(residuals ** 2) / sigma2 / n_data
            
            # Target: chi2 ≈ 1
            return (chi2 - self.target_chi2) ** 2
        
        # Find optimal lambda
        result = optimize.minimize_scalar(
            objective,
            bounds=(-10, 10),
            method='bounded'
        )
        
        best_lambda = np.exp(result.x)
        P = self._solve_fixed_lambda(A, b, m, best_lambda)
        
        return P, best_lambda
    
    def _solve_fixed_lambda(self,
                            A: np.ndarray,
                            b: np.ndarray,
                            m: np.ndarray,
                            lam: float) -> np.ndarray:
        """Solve MEM problem for fixed regularization parameter."""
        n_basis = A.shape[1]
        
        # Objective: minimize ||Ax-b||² - λ·S(x)
        # Use iterative approach
        
        # Initialize with NNLS
        P = NNLSSolver.solve_basic_nnls(A, b)
        P = np.maximum(P, 1e-10)
        P = P / np.sum(P) * np.sum(b)  # Scale
        
        for iteration in range(50):
            P_old = P.copy()
            
            # Gradient of chi-squared term
            grad_chi2 = 2 * A.T @ (A @ P - b)
            
            # Gradient of entropy term: -log(P/m) - 1
            grad_S = -np.log(P / m + 1e-20) - 1
            
            # Combined gradient
            grad = grad_chi2 - lam * grad_S
            
            # Step size (simple line search)
            step = 0.1 / (1 + iteration * 0.1)
            
            # Update with projection to non-negative
            P = P - step * grad
            P = np.maximum(P, 1e-10)
            
            # Check convergence
            if np.max(np.abs(P - P_old)) < 1e-8:
                break
        
        return P


class CVXPYSolver:
    """
    CVXPY-based solvers for convex optimization formulations.
    
    Provides more robust solutions using proper convex optimization.
    """
    
    @staticmethod
    def solve_nnls_l2(A: np.ndarray, 
                      b: np.ndarray, 
                      alpha: float = 0.01) -> np.ndarray:
        """
        NNLS with L2 regularization using CVXPY.
        
        min ||Ax - b||² + α||x||²  s.t. x ≥ 0
        """
        if not CVXPY_AVAILABLE:
            return NNLSSolver.solve_tikhonov_nnls(A, b, alpha)
        
        n = A.shape[1]
        x = cp.Variable(n, nonneg=True)
        
        objective = cp.Minimize(
            cp.sum_squares(A @ x - b) + alpha * cp.sum_squares(x)
        )
        
        problem = cp.Problem(objective)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            return np.array(x.value).flatten()
        except Exception:
            return NNLSSolver.solve_tikhonov_nnls(A, b, alpha)
    
    @staticmethod
    def solve_nnls_l1(A: np.ndarray,
                      b: np.ndarray,
                      alpha: float = 0.01) -> np.ndarray:
        """
        NNLS with L1 regularization (sparse solution).
        
        min ||Ax - b||² + α||x||₁  s.t. x ≥ 0
        """
        if not CVXPY_AVAILABLE:
            return NNLSSolver.solve_elastic_net_nnls(A, b, alpha, l1_ratio=1.0)
        
        n = A.shape[1]
        x = cp.Variable(n, nonneg=True)
        
        objective = cp.Minimize(
            cp.sum_squares(A @ x - b) + alpha * cp.norm(x, 1)
        )
        
        problem = cp.Problem(objective)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            return np.array(x.value).flatten()
        except Exception:
            return NNLSSolver.solve_elastic_net_nnls(A, b, alpha, l1_ratio=1.0)
    
    @staticmethod
    def solve_entropy_regularized(A: np.ndarray,
                                  b: np.ndarray,
                                  alpha: float = 0.01) -> np.ndarray:
        """
        Solve with entropy regularization (approximate).
        
        Uses log-barrier approximation to entropy.
        """
        if not CVXPY_AVAILABLE:
            solver = MaximumEntropySolver()
            P, _ = solver.solve(A, b)
            return P
        
        n = A.shape[1]
        x = cp.Variable(n, pos=True)  # Strictly positive
        
        # Entropy: -sum(x * log(x))
        # Use relative entropy from uniform: sum(x * log(x/m)) where m=1/n
        # = sum(x * log(x)) + sum(x) * log(n)
        entropy_term = cp.sum(cp.entr(x))  # -x*log(x)
        
        objective = cp.Minimize(
            cp.sum_squares(A @ x - b) - alpha * entropy_term
        )
        
        problem = cp.Problem(objective)
        
        try:
            problem.solve(solver=cp.SCS, verbose=False)
            return np.array(x.value).flatten()
        except Exception:
            solver = MaximumEntropySolver()
            P, _ = solver.solve(A, b)
            return P


class PeakDetector:
    """Detect and characterize peaks in the rate constant distribution."""
    
    @staticmethod
    def find_peaks(k_values: np.ndarray,
                   distribution: np.ndarray,
                   min_height: float = 0.05,
                   min_prominence: float = 0.02) -> List[Dict[str, float]]:
        """
        Find peaks in the distribution.
        
        Parameters:
        -----------
        k_values : np.ndarray
            Rate constant values
        distribution : np.ndarray
            Distribution amplitudes
        min_height : float
            Minimum peak height (fraction of max)
        min_prominence : float
            Minimum peak prominence
            
        Returns:
        --------
        List of dicts with peak properties
        """
        from scipy.signal import find_peaks, peak_widths
        
        # Normalize
        dist_norm = distribution / (np.max(distribution) + 1e-10)
        
        # Find peaks
        peaks, properties = find_peaks(
            dist_norm,
            height=min_height,
            prominence=min_prominence,
            distance=3
        )
        
        if len(peaks) == 0:
            # Return single "peak" at maximum
            max_idx = np.argmax(distribution)
            return [{
                'k': k_values[max_idx],
                'amplitude': distribution[max_idx],
                'half_time': np.log(2) / k_values[max_idx],
                'width': 0,
                'fraction': 1.0
            }]
        
        # Calculate widths
        try:
            widths, width_heights, left_ips, right_ips = peak_widths(
                dist_norm, peaks, rel_height=0.5
            )
        except Exception:
            widths = np.ones(len(peaks))
            left_ips = peaks - 1
            right_ips = peaks + 1
        
        # Compile results
        total_area = np.trapz(distribution, np.log10(k_values))
        
        peak_list = []
        for i, peak_idx in enumerate(peaks):
            # Peak rate constant
            k_peak = k_values[peak_idx]
            
            # Amplitude
            amplitude = distribution[peak_idx]
            
            # Width in log space
            left_k = k_values[max(0, int(left_ips[i]))]
            right_k = k_values[min(len(k_values)-1, int(right_ips[i]))]
            width_log = np.log10(right_k / left_k)
            
            # Estimate fraction (integrate around peak)
            left_idx = max(0, int(left_ips[i]))
            right_idx = min(len(k_values)-1, int(right_ips[i]))
            peak_area = np.trapz(
                distribution[left_idx:right_idx+1],
                np.log10(k_values[left_idx:right_idx+1])
            )
            fraction = peak_area / total_area if total_area > 0 else 0
            
            peak_list.append({
                'k': k_peak,
                'amplitude': amplitude,
                'half_time': np.log(2) / k_peak,
                'width_log': width_log,
                'fraction': fraction,
                'left_k': left_k,
                'right_k': right_k
            })
        
        # Sort by amplitude
        peak_list.sort(key=lambda x: x['amplitude'], reverse=True)
        
        return peak_list


class MEMAnalyzer:
    """
    Main class for Maximum Entropy Method FRAP analysis.
    
    Provides high-level interface for distribution reconstruction.
    """
    
    def __init__(self,
                 k_min: float = 1e-3,
                 k_max: float = 10.0,
                 n_basis: int = 100,
                 method: str = 'tikhonov'):
        """
        Initialize MEM analyzer.
        
        Parameters:
        -----------
        k_min : float
            Minimum rate constant (s⁻¹)
        k_max : float
            Maximum rate constant (s⁻¹)
        n_basis : int
            Number of basis functions
        method : str
            Regularization method: 'tikhonov', 'entropy', 'l1', 'elastic'
        """
        self.basis = RateConstantBasis(k_min, k_max, n_basis)
        self.method = method
        self.k_values = self.basis.k_values
        
    def fit(self,
            time_data: np.ndarray,
            intensity_data: np.ndarray,
            alpha: Optional[float] = None,
            noise_estimate: Optional[float] = None) -> MEMResult:
        """
        Fit FRAP data to extract rate constant distribution.
        
        Parameters:
        -----------
        time_data : np.ndarray
            Time points (s)
        intensity_data : np.ndarray
            Normalized intensity (0-1)
        alpha : float, optional
            Regularization parameter. If None, selected by cross-validation.
        noise_estimate : float, optional
            Estimated noise level
            
        Returns:
        --------
        MEMResult
            Fitted distribution and statistics
        """
        # Generate basis matrix
        A = self.basis.generate_matrix(time_data)
        b = intensity_data
        
        # Auto-select regularization if needed
        if alpha is None:
            alpha = self._select_alpha(A, b)
        
        # Solve based on method
        if self.method == 'tikhonov':
            D = self.basis.generate_derivative_matrix(time_data)
            distribution = NNLSSolver.solve_tikhonov_nnls(A, b, alpha, D)
        elif self.method == 'entropy':
            if CVXPY_AVAILABLE:
                distribution = CVXPYSolver.solve_entropy_regularized(A, b, alpha)
            else:
                solver = MaximumEntropySolver()
                distribution, _ = solver.solve(A, b, noise_estimate)
        elif self.method == 'l1':
            if CVXPY_AVAILABLE:
                distribution = CVXPYSolver.solve_nnls_l1(A, b, alpha)
            else:
                distribution = NNLSSolver.solve_elastic_net_nnls(A, b, alpha, l1_ratio=1.0)
        elif self.method == 'elastic':
            distribution = NNLSSolver.solve_elastic_net_nnls(A, b, alpha, l1_ratio=0.5)
        else:
            distribution = NNLSSolver.solve_basic_nnls(A, b)
        
        # Ensure non-negative
        distribution = np.maximum(distribution, 0)
        
        # Compute fitted curve
        fitted_curve = A @ distribution
        
        # Statistics
        residuals = b - fitted_curve
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((b - np.mean(b)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # AIC (approximate, using effective DOF)
        n = len(b)
        k_eff = np.sum(distribution > 0.01 * np.max(distribution))  # Effective parameters
        aic = n * np.log(ss_res / n + 1e-10) + 2 * k_eff
        
        # Entropy
        dist_norm = distribution / (np.sum(distribution) + 1e-10)
        entropy = -np.sum(dist_norm * np.log(dist_norm + 1e-20))
        
        # Find peaks
        peaks = PeakDetector.find_peaks(self.k_values, distribution)
        
        return MEMResult(
            rate_constants=self.k_values,
            distribution=distribution,
            fitted_curve=fitted_curve,
            time=time_data,
            r2=r2,
            aic=aic,
            peaks=peaks,
            entropy=entropy,
            regularization_param=alpha
        )
    
    def _select_alpha(self, A: np.ndarray, b: np.ndarray) -> float:
        """Select regularization parameter using L-curve or GCV."""
        # Try several alpha values
        alphas = np.logspace(-4, 1, 20)
        
        residuals = []
        norms = []
        
        D = self.basis.generate_derivative_matrix(np.zeros(A.shape[0]))
        
        for alpha in alphas:
            x = NNLSSolver.solve_tikhonov_nnls(A, b, alpha, D)
            res = np.sum((A @ x - b) ** 2)
            norm = np.sum(x ** 2)
            residuals.append(res)
            norms.append(norm)
        
        residuals = np.array(residuals)
        norms = np.array(norms)
        
        # L-curve criterion: maximize curvature
        log_res = np.log10(residuals + 1e-10)
        log_norm = np.log10(norms + 1e-10)
        
        # Approximate curvature
        curvature = np.zeros(len(alphas))
        for i in range(1, len(alphas) - 1):
            dx = log_res[i+1] - log_res[i-1]
            dy = log_norm[i+1] - log_norm[i-1]
            d2x = log_res[i+1] - 2*log_res[i] + log_res[i-1]
            d2y = log_norm[i+1] - 2*log_norm[i] + log_norm[i-1]
            curvature[i] = abs(dx*d2y - dy*d2x) / (dx**2 + dy**2 + 1e-10)**1.5
        
        best_idx = np.argmax(curvature)
        return alphas[best_idx]
    
    def compare_methods(self,
                        time_data: np.ndarray,
                        intensity_data: np.ndarray) -> Dict[str, MEMResult]:
        """
        Compare different regularization methods.
        
        Returns results for each method for comparison.
        """
        methods = ['tikhonov', 'l1', 'elastic']
        if CVXPY_AVAILABLE:
            methods.append('entropy')
        
        results = {}
        for method in methods:
            analyzer = MEMAnalyzer(
                k_min=self.basis.k_min,
                k_max=self.basis.k_max,
                n_basis=self.basis.n_basis,
                method=method
            )
            results[method] = analyzer.fit(time_data, intensity_data)
        
        return results


def fit_frap_mem(time_data: np.ndarray,
                 intensity_data: np.ndarray,
                 k_min: float = 1e-3,
                 k_max: float = 10.0,
                 n_basis: int = 100,
                 method: str = 'tikhonov',
                 alpha: Optional[float] = None) -> MEMResult:
    """
    High-level function to fit FRAP data using Maximum Entropy Method.
    
    Parameters:
    -----------
    time_data : np.ndarray
        Time points (s)
    intensity_data : np.ndarray
        Normalized intensity (0-1)
    k_min : float
        Minimum rate constant (s⁻¹)
    k_max : float
        Maximum rate constant (s⁻¹)
    n_basis : int
        Number of basis functions
    method : str
        Regularization method: 'tikhonov', 'entropy', 'l1', 'elastic'
    alpha : float, optional
        Regularization parameter
        
    Returns:
    --------
    MEMResult
        Fitted distribution and analysis
    """
    analyzer = MEMAnalyzer(k_min, k_max, n_basis, method)
    return analyzer.fit(time_data, intensity_data, alpha)


def plot_mem_result(result: MEMResult,
                    ax_dist=None,
                    ax_fit=None,
                    show_peaks: bool = True) -> None:
    """
    Plot MEM analysis results.
    
    Parameters:
    -----------
    result : MEMResult
        Results from MEM analysis
    ax_dist : matplotlib axes, optional
        Axes for distribution plot
    ax_fit : matplotlib axes, optional
        Axes for fit plot
    show_peaks : bool
        Whether to mark detected peaks
    """
    import matplotlib.pyplot as plt
    
    if ax_dist is None and ax_fit is None:
        fig, (ax_dist, ax_fit) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Distribution plot
    if ax_dist is not None:
        ax_dist.semilogx(result.rate_constants, result.distribution, 'b-', linewidth=2)
        ax_dist.fill_between(result.rate_constants, result.distribution, alpha=0.3)
        ax_dist.set_xlabel('Rate Constant k (s⁻¹)')
        ax_dist.set_ylabel('Amplitude P(k)')
        ax_dist.set_title('Rate Constant Distribution')
        
        if show_peaks and result.peaks:
            for peak in result.peaks:
                ax_dist.axvline(peak['k'], color='r', linestyle='--', alpha=0.7)
                ax_dist.annotate(
                    f"k={peak['k']:.3f}\nt½={peak['half_time']:.2f}s",
                    xy=(peak['k'], peak['amplitude']),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=8
                )
    
    # Fit plot
    if ax_fit is not None:
        ax_fit.plot(result.time, result.fitted_curve, 'r-', label='MEM Fit', linewidth=2)
        ax_fit.set_xlabel('Time (s)')
        ax_fit.set_ylabel('Normalized Intensity')
        ax_fit.set_title(f'Recovery Fit (R² = {result.r2:.4f})')
        ax_fit.legend()
    
    plt.tight_layout()


def generate_spectrum_report(result: MEMResult) -> str:
    """
    Generate text report of MEM analysis.
    
    Parameters:
    -----------
    result : MEMResult
        Results from MEM analysis
        
    Returns:
    --------
    str
        Formatted report
    """
    report = []
    report.append("=" * 60)
    report.append("MAXIMUM ENTROPY METHOD (MEM) FRAP ANALYSIS")
    report.append("=" * 60)
    report.append("")
    report.append(f"Fit Quality:")
    report.append(f"  R² = {result.r2:.4f}")
    report.append(f"  AIC = {result.aic:.2f}")
    report.append(f"  Entropy = {result.entropy:.4f}")
    report.append(f"  Regularization λ = {result.regularization_param:.2e}")
    report.append("")
    report.append(f"Overall Parameters:")
    report.append(f"  Total Mobile Fraction = {result.get_mobile_fraction():.1f}%")
    report.append(f"  Mean Rate Constant = {result.get_mean_rate():.4f} s⁻¹")
    report.append(f"  Effective Half-Time = {result.get_half_time():.2f} s")
    report.append("")
    report.append(f"Detected Populations ({len(result.peaks)}):")
    report.append("-" * 50)
    
    for i, peak in enumerate(result.peaks, 1):
        report.append(f"  Population {i}:")
        report.append(f"    Rate constant k = {peak['k']:.4f} s⁻¹")
        report.append(f"    Half-time t½ = {peak['half_time']:.2f} s")
        report.append(f"    Fraction = {peak['fraction']*100:.1f}%")
        if 'width_log' in peak:
            report.append(f"    Width (log) = {peak['width_log']:.2f}")
        report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)
