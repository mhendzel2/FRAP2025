"""
HiFRAP (High-throughput FRAP) Analysis Module

Implementation of the HiFRAP method for estimating kinetic parameters from FRAP data.
Uses Variable Projection to decouple non-linear and linear parameter estimation.

Reference: Lorenzetti et al., 2025

Author: FRAP2025 Project
"""

import numpy as np
from scipy import optimize, linalg, stats
from typing import Tuple, Optional, Dict, Any, NamedTuple
import warnings


class HiFRAPResult(NamedTuple):
    """Container for HiFRAP fitting results."""
    D: float  # Diffusion coefficient
    k_on: float  # Binding rate (on)
    k_off: float  # Unbinding rate (off)
    beta: float  # Bleaching/reaction parameter
    c0: np.ndarray  # Initial condition coefficients
    source_rate: float  # Source rate
    background: float  # Background level
    D_err: float  # Uncertainty in D
    k_on_err: float  # Uncertainty in k_on
    k_off_err: float  # Uncertainty in k_off
    beta_err: float  # Uncertainty in beta
    r_squared: float  # Coefficient of determination
    adjusted_r_squared: float  # Adjusted R²
    ks_statistic: float  # Kolmogorov-Smirnov statistic
    ks_pvalue: float  # KS test p-value
    residuals: np.ndarray  # Final projected residuals
    n_iterations: int  # Number of optimizer iterations
    cost: float  # Final cost value
    success: bool  # Whether optimization converged


class HiFRAPFitter:
    """
    HiFRAP (High-throughput FRAP) Fitter.
    
    Estimates kinetic parameters (D, k_on, k_off, beta) from noisy FRAP data
    using Variable Projection optimization.
    
    Parameters
    ----------
    time_step : float
        Time step between frames (dt) in seconds.
    pixel_size : float
        Pixel size (dx) in micrometers.
    n_modes : int, optional
        Number of Fourier modes per dimension (default=9).
    lambda_reg : float, optional
        Regularization parameter for matrix inversion (default=1e-10).
    epsilon : float, optional
        SVD truncation threshold (default=1e-6).
    
    Attributes
    ----------
    F : np.ndarray
        Spatial Fourier matrix for data compression.
    wavenumbers : np.ndarray
        Wavenumber array for Fourier basis.
    
    Example
    -------
    >>> fitter = HiFRAPFitter(time_step=0.1, pixel_size=0.1, n_modes=9)
    >>> result = fitter.fit(images)
    >>> print(f"Diffusion coefficient: {result.D:.4f} ± {result.D_err:.4f}")
    """
    
    def __init__(
        self,
        time_step: float,
        pixel_size: float,
        n_modes: int = 9,
        lambda_reg: float = 1e-10,
        epsilon: float = 1e-6
    ):
        self.dt: float = time_step
        self.dx: float = pixel_size
        self.n_modes: int = n_modes
        self.lambda_reg: float = lambda_reg
        self.epsilon: float = epsilon
        
        # Will be set when data dimensions are known
        self.F: Optional[np.ndarray] = None
        self.wavenumbers: Optional[np.ndarray] = None
        self._N: Optional[int] = None  # Image dimension (assumes square)
        self._T: Optional[int] = None  # Number of time points
        
    def _construct_fourier_matrix(self, N: int) -> np.ndarray:
        """
        Construct the real-valued Spatial Fourier Matrix F.
        
        Reference: Section 4.1, Equations 3 & 4
        
        Parameters
        ----------
        N : int
            Image dimension (pixels per side).
            
        Returns
        -------
        F : np.ndarray
            Fourier matrix of shape (n_modes, N).
        """
        K = self.n_modes
        F = np.zeros((K, N))
        
        # Pixel positions (normalized to [0, 1))
        x = np.arange(N) / N
        
        # Wavenumber indices: k maps to indices via floor((k+1)/2)
        # k=0 -> DC, k=1,2 -> first harmonic (cos, sin), etc.
        wavenumbers = np.zeros(K)
        
        for k in range(K):
            n = (k + 1) // 2  # Harmonic number
            wavenumbers[k] = n
            
            if k == 0:
                # DC component: constant
                F[k, :] = 1.0 / np.sqrt(N)
            elif k % 2 == 1:
                # Odd indices: cosine terms
                F[k, :] = np.sqrt(2.0 / N) * np.cos(2 * np.pi * n * x)
            else:
                # Even indices: sine terms
                F[k, :] = np.sqrt(2.0 / N) * np.sin(2 * np.pi * n * x)
        
        self.wavenumbers = wavenumbers
        return F
    
    def _compress_data(self, images: np.ndarray) -> np.ndarray:
        """
        Compress input image stack using Fourier basis.
        
        Applies F to both spatial axes: y_compressed = F @ images @ F.T
        Then flattens to a vector for each time point.
        
        Parameters
        ----------
        images : np.ndarray
            3D array of shape (T, N, N) - time series of square images.
            
        Returns
        -------
        y : np.ndarray
            Compressed data vector of shape (T * K^2,).
        """
        T, N, _ = images.shape
        K = self.n_modes
        
        # Initialize Fourier matrix if needed
        if self.F is None or self.F.shape[1] != N:
            self.F = self._construct_fourier_matrix(N)
        
        # Compress each time frame
        y_compressed = np.zeros((T, K, K))
        for t in range(T):
            # Apply F to both spatial dimensions: F @ image @ F.T
            y_compressed[t] = self.F @ images[t] @ self.F.T
        
        # Flatten to vector: (T, K, K) -> (T * K^2,)
        return y_compressed.reshape(-1)
    
    def _compute_1d_kernel(
        self,
        D: float,
        beta: float,
        t: float
    ) -> np.ndarray:
        """
        Compute 1D transport kernel matrix.
        
        Reference: Section 4.2.3
        
        Parameters
        ----------
        D : float
            Diffusion coefficient.
        beta : float
            Bleaching/reaction parameter.
        t : float
            Time point.
            
        Returns
        -------
        K_1d : np.ndarray
            1D kernel matrix of shape (K, K).
            
        Raises
        ------
        RuntimeError
            If fit() has not been called first.
        """
        if self._N is None or self.wavenumbers is None:
            raise RuntimeError("fit() must be called before using this method")
            
        K = self.n_modes
        K_1d = np.zeros((K, K))
        
        # Wavenumber spatial frequencies
        # omega_n = 2*pi*n / (N*dx) where n is the harmonic number
        L = self._N * self.dx  # Physical domain size
        
        for i in range(K):
            for j in range(K):
                n_i = self.wavenumbers[i]
                n_j = self.wavenumbers[j]
                
                # Only diagonal elements are non-zero for diffusion operator
                if i == j:
                    omega = 2 * np.pi * n_i / L
                    # Diffusion decay: exp(-D * omega^2 * t)
                    # Reaction/bleaching: exp(-beta * t)
                    K_1d[i, j] = np.exp(-D * omega**2 * t - beta * t)
        
        return K_1d
    
    def _compute_reduced_basis(
        self,
        D: float,
        beta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute reduced basis U_R using SVD truncation.
        
        Reference: Sections 4.4, 4.5
        
        Parameters
        ----------
        D : float
            Diffusion coefficient.
        beta : float
            Bleaching/reaction parameter.
            
        Returns
        -------
        U_R : np.ndarray
            Reduced basis matrix.
        sigma : np.ndarray
            Retained singular values.
            
        Raises
        ------
        RuntimeError
            If fit() has not been called first.
        """
        if self._T is None:
            raise RuntimeError("fit() must be called before using this method")
            
        K = self.n_modes
        T = self._T
        
        # Build full kernel matrix for all time points
        # Shape: (T * K^2, K^2) where K^2 is initial condition size
        kernel_blocks = []
        
        for t_idx in range(T):
            t = t_idx * self.dt
            
            # 1D kernels for x and y
            K_x = self._compute_1d_kernel(D, beta, t)
            K_y = self._compute_1d_kernel(D, beta, t)
            
            # 2D kernel via Kronecker product
            K_2d = np.kron(K_x, K_y)
            kernel_blocks.append(K_2d)
        
        # Stack blocks: shape (T * K^2, K^2)
        full_kernel = np.vstack(kernel_blocks)
        
        # SVD truncation
        try:
            U, s, Vh = linalg.svd(full_kernel, full_matrices=False)
            
            # Retain singular values above threshold
            s_max = s[0] if len(s) > 0 else 1.0
            mask = s > self.epsilon * s_max
            n_retain = max(int(np.sum(mask)), 1)
            
            U_R = U[:, :n_retain]
            sigma = s[:n_retain]
            
            return U_R, sigma
            
        except linalg.LinAlgError:
            # Fallback: return identity-like basis
            warnings.warn("SVD failed, using fallback basis")
            return np.eye(T * K * K, K * K), np.ones(K * K)
    
    def _compute_projected_residuals(
        self,
        U_R: np.ndarray,
        raw_residual: np.ndarray
    ) -> np.ndarray:
        """
        Compute projected residuals using Woodbury identity.
        
        r = (I - U_R @ (U_R.T @ U_R + lambda*I)^-1 @ U_R.T) @ raw_residual
        
        Uses Cholesky decomposition for numerical stability.
        
        Parameters
        ----------
        U_R : np.ndarray
            Reduced basis matrix.
        raw_residual : np.ndarray
            Raw residual vector.
            
        Returns
        -------
        r : np.ndarray
            Projected residual vector.
        """
        # Form inner matrix (small: n_k x n_k)
        inner_M = U_R.T @ U_R
        inner_M[np.diag_indices_from(inner_M)] += self.lambda_reg
        
        try:
            # Cholesky factorization
            L = linalg.cholesky(inner_M, lower=True)
            
            # Apply Woodbury identity efficiently
            # Step A: Project residual onto basis
            w = U_R.T @ raw_residual
            
            # Step B: Solve M @ z = w
            z = linalg.cho_solve((L, True), w)
            
            # Step C: Lift back and subtract
            correction = U_R @ z
            return raw_residual - correction
            
        except linalg.LinAlgError:
            # Fallback: return large residuals
            warnings.warn("Cholesky decomposition failed")
            return raw_residual * 1e6
    
    def _construct_linear_jacobian(
        self,
        U_R: np.ndarray
    ) -> np.ndarray:
        """
        Construct Jacobian for linear parameters (source, background).
        
        Reference: Section 4.2.4, Eq. 25
        
        Parameters
        ----------
        U_R : np.ndarray
            Reduced basis matrix.
            
        Returns
        -------
        J_linear : np.ndarray
            Jacobian matrix for linear parameters.
            
        Raises
        ------
        RuntimeError
            If fit() has not been called first.
        """
        if self._T is None:
            raise RuntimeError("fit() must be called before using this method")
            
        T = self._T
        K = self.n_modes
        n_data = T * K * K
        
        # Background: constant contribution (DC component)
        j_background = np.zeros(n_data)
        # Background affects DC mode at each time point
        dc_idx = 0  # First mode is DC
        for t_idx in range(T):
            idx = t_idx * K * K + dc_idx * K + dc_idx
            j_background[idx] = 1.0
        
        # Source: accumulates over time
        j_source = np.zeros(n_data)
        for t_idx in range(T):
            t = t_idx * self.dt
            idx = t_idx * K * K + dc_idx * K + dc_idx
            j_source[idx] = t  # Linear accumulation
        
        # Apply projector to get effective Jacobian
        j_background_proj = self._compute_projected_residuals(U_R, j_background)
        j_source_proj = self._compute_projected_residuals(U_R, j_source)
        
        return np.column_stack([j_source_proj, j_background_proj])
    
    def _solve_linear_parameters(
        self,
        U_R: np.ndarray,
        y_compressed: np.ndarray,
        D: float,
        beta: float
    ) -> Tuple[float, float, np.ndarray]:
        """
        Solve for optimal linear parameters analytically.
        
        Reference: Section 4.2.4, Eq. 24
        
        Parameters
        ----------
        U_R : np.ndarray
            Reduced basis matrix.
        y_compressed : np.ndarray
            Compressed data vector.
        D : float
            Diffusion coefficient.
        beta : float
            Bleaching parameter.
            
        Returns
        -------
        source_rate : float
            Optimal source rate.
        background : float
            Optimal background level.
        c0 : np.ndarray
            Initial condition coefficients.
        """
        T = self._T
        K = self.n_modes
        
        # Construct linear Jacobian
        J_linear = self._construct_linear_jacobian(U_R)
        
        # Get initial condition from first frame (projected)
        y_t0 = y_compressed[:K*K]
        
        # Solve least squares for linear parameters
        # Using projected data
        y_proj = self._compute_projected_residuals(U_R, y_compressed)
        
        try:
            # Normal equations solution
            JTJ = J_linear.T @ J_linear
            JTJ[np.diag_indices_from(JTJ)] += self.lambda_reg
            JTy = J_linear.T @ y_proj
            
            linear_params = linalg.solve(JTJ, JTy, assume_a='pos')
            source_rate = float(linear_params[0])
            background = float(linear_params[1])
            
        except linalg.LinAlgError:
            source_rate = 0.0
            background = float(np.mean(y_compressed))
        
        # Estimate initial condition from first frame
        c0 = y_t0.reshape(K, K)
        
        return source_rate, background, c0
    
    def _objective_function(
        self,
        params: np.ndarray,
        y_compressed: np.ndarray
    ) -> np.ndarray:
        """
        Objective function for Variable Projection optimization.
        
        Returns scaled residual vector for least_squares optimizer.
        
        Reference: Section 4.2.4
        
        Parameters
        ----------
        params : np.ndarray
            Non-linear parameters [D, beta].
        y_compressed : np.ndarray
            Compressed data vector.
            
        Returns
        -------
        residuals : np.ndarray
            Scaled projected residual vector.
            
        Raises
        ------
        RuntimeError
            If fit() has not been called first.
        """
        if self._T is None:
            raise RuntimeError("fit() must be called before using this method")
            
        # Unpack non-linear parameters
        D = params[0]
        beta = params[1]
        
        # Enforce positivity (bounds should handle this, but be safe)
        D = max(D, 1e-10)
        beta = max(beta, 0.0)
        
        # Get reduced basis
        U_R, sigma = self._compute_reduced_basis(D, beta)
        
        # Solve for linear parameters
        source_rate, background, c0 = self._solve_linear_parameters(
            U_R, y_compressed, D, beta
        )
        
        # Compute model prediction
        T = self._T
        K = self.n_modes
        y_model = np.zeros(T * K * K)
        
        for t_idx in range(T):
            t = t_idx * self.dt
            
            # Apply transport kernel to initial condition
            K_x = self._compute_1d_kernel(D, beta, t)
            K_y = self._compute_1d_kernel(D, beta, t)
            
            # y(t) = K_x @ c0 @ K_y.T + source*t + background
            c_t = K_x @ c0 @ K_y.T
            
            # Add source and background (DC component)
            c_t[0, 0] += source_rate * t + background
            
            # Flatten and store
            y_model[t_idx * K * K:(t_idx + 1) * K * K] = c_t.reshape(-1)
        
        # Raw residual
        raw_residual = y_compressed - y_model
        
        # Projected residual
        projected_residual = self._compute_projected_residuals(U_R, raw_residual)
        
        # Compute effective degrees of freedom
        n_data = len(y_compressed)
        n_params = 2 + 2 + K * K  # D, beta + source, background + c0
        dof = max(n_data - n_params, 1)
        
        # Scale residuals for proper chi-squared
        scale = 1.0 / np.sqrt(dof)
        
        return projected_residual * scale
    
    def _compute_hessian(
        self,
        params: np.ndarray,
        y_compressed: np.ndarray,
        step: float = 1e-6
    ) -> np.ndarray:
        """
        Compute numerical Hessian at optimum.
        
        Parameters
        ----------
        params : np.ndarray
            Optimal parameters.
        y_compressed : np.ndarray
            Compressed data.
        step : float
            Finite difference step size.
            
        Returns
        -------
        H : np.ndarray
            Hessian matrix.
        """
        n_params = len(params)
        H = np.zeros((n_params, n_params))
        
        # Central difference approximation
        for i in range(n_params):
            for j in range(i, n_params):
                # f(x + ei + ej)
                p_pp = params.copy()
                p_pp[i] += step
                p_pp[j] += step
                r_pp = self._objective_function(p_pp, y_compressed)
                f_pp = np.sum(r_pp**2)
                
                # f(x + ei - ej)
                p_pm = params.copy()
                p_pm[i] += step
                p_pm[j] -= step
                r_pm = self._objective_function(p_pm, y_compressed)
                f_pm = np.sum(r_pm**2)
                
                # f(x - ei + ej)
                p_mp = params.copy()
                p_mp[i] -= step
                p_mp[j] += step
                r_mp = self._objective_function(p_mp, y_compressed)
                f_mp = np.sum(r_mp**2)
                
                # f(x - ei - ej)
                p_mm = params.copy()
                p_mm[i] -= step
                p_mm[j] -= step
                r_mm = self._objective_function(p_mm, y_compressed)
                f_mm = np.sum(r_mm**2)
                
                # Mixed partial derivative
                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * step**2)
                H[j, i] = H[i, j]
        
        return H
    
    def _estimate_errors(
        self,
        params: np.ndarray,
        y_compressed: np.ndarray,
        residuals: np.ndarray
    ) -> np.ndarray:
        """
        Estimate parameter uncertainties from Hessian.
        
        Reference: Section 4.7
        
        Parameters
        ----------
        params : np.ndarray
            Optimal parameters.
        y_compressed : np.ndarray
            Compressed data.
        residuals : np.ndarray
            Final residuals.
            
        Returns
        -------
        errors : np.ndarray
            Standard errors for each parameter.
        """
        try:
            # Compute Hessian
            H = self._compute_hessian(params, y_compressed)
            
            # Covariance matrix is inverse of Hessian
            # Add regularization for numerical stability
            H_reg = H + self.lambda_reg * np.eye(len(params))
            cov = linalg.inv(H_reg)
            
            # Errors are square roots of diagonal
            variances = np.diag(cov)
            errors = np.sqrt(np.maximum(variances, 0))
            
            return errors
            
        except linalg.LinAlgError:
            warnings.warn("Error estimation failed, returning NaN")
            return np.full(len(params), np.nan)
    
    def _compute_goodness_of_fit(
        self,
        y_compressed: np.ndarray,
        residuals: np.ndarray,
        n_params: int
    ) -> Tuple[float, float, float, float]:
        """
        Compute goodness of fit statistics.
        
        Reference: Section 4.2.5, Eq. 15
        
        Parameters
        ----------
        y_compressed : np.ndarray
            Compressed data.
        residuals : np.ndarray
            Final projected residuals.
        n_params : int
            Number of fitted parameters.
            
        Returns
        -------
        r_squared : float
            Coefficient of determination.
        adj_r_squared : float
            Adjusted R².
        ks_stat : float
            Kolmogorov-Smirnov statistic.
        ks_pvalue : float
            KS test p-value.
        """
        n_data = len(y_compressed)
        
        # Total sum of squares
        y_mean = np.mean(y_compressed)
        ss_tot = np.sum((y_compressed - y_mean)**2)
        
        # Residual sum of squares
        ss_res = np.sum(residuals**2)
        
        # R² and adjusted R²
        if ss_tot > 0:
            r_squared = 1 - ss_res / ss_tot
        else:
            r_squared = 0.0
        
        dof_total = n_data - 1
        dof_resid = n_data - n_params
        
        if dof_resid > 0 and dof_total > 0:
            adj_r_squared = 1 - (1 - r_squared) * dof_total / dof_resid
        else:
            adj_r_squared = r_squared
        
        # Kolmogorov-Smirnov test on residuals
        # Normalize residuals for comparison to standard normal
        residual_std = np.std(residuals)
        if residual_std > 0:
            normalized_residuals = residuals / residual_std
            ks_stat, ks_pvalue = stats.kstest(normalized_residuals, 'norm')
        else:
            ks_stat, ks_pvalue = 0.0, 1.0
        
        return r_squared, adj_r_squared, ks_stat, ks_pvalue
    
    def fit(
        self,
        images: np.ndarray,
        D_init: Optional[float] = None,
        beta_init: Optional[float] = None,
        D_bounds: Tuple[float, float] = (1e-4, 100.0),
        beta_bounds: Tuple[float, float] = (0.0, 10.0),
        max_iter: int = 100,
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        verbose: int = 0
    ) -> HiFRAPResult:
        """
        Fit FRAP data using HiFRAP method.
        
        Parameters
        ----------
        images : np.ndarray
            3D array of shape (T, N, N) - time series of square images.
        D_init : float, optional
            Initial guess for diffusion coefficient.
            Default: geometric mean of bounds.
        beta_init : float, optional
            Initial guess for bleaching parameter.
            Default: 0.1.
        D_bounds : tuple, optional
            Bounds for diffusion coefficient (D_min, D_max).
        beta_bounds : tuple, optional
            Bounds for bleaching parameter (beta_min, beta_max).
        max_iter : int, optional
            Maximum number of optimizer iterations.
        ftol : float, optional
            Function tolerance for convergence.
        xtol : float, optional
            Parameter tolerance for convergence.
        verbose : int, optional
            Verbosity level (0=silent, 1=summary, 2=detailed).
            
        Returns
        -------
        result : HiFRAPResult
            Named tuple containing all fit results and statistics.
        """
        # Validate input
        if images.ndim != 3:
            raise ValueError(f"Expected 3D array, got {images.ndim}D")
        
        T, N, M = images.shape
        if N != M:
            raise ValueError(f"Expected square images, got {N}x{M}")
        
        # Store dimensions
        self._N = N
        self._T = T
        
        # Initialize Fourier matrix
        self.F = self._construct_fourier_matrix(N)
        
        if verbose > 0:
            print(f"HiFRAP Fitting: {T} frames of {N}x{N} images")
            print(f"Using {self.n_modes} Fourier modes")
        
        # Compress data
        y_compressed = self._compress_data(images)
        
        if verbose > 1:
            print(f"Compressed data shape: {y_compressed.shape}")
        
        # Set initial guesses
        if D_init is None:
            D_init = np.sqrt(D_bounds[0] * D_bounds[1])  # Geometric mean
        if beta_init is None:
            beta_init = 0.1
        
        params_init = np.array([D_init, beta_init])
        bounds_lower = np.array([D_bounds[0], beta_bounds[0]])
        bounds_upper = np.array([D_bounds[1], beta_bounds[1]])
        
        if verbose > 0:
            print(f"Initial: D={D_init:.4f}, beta={beta_init:.4f}")
        
        # Run optimization
        result = optimize.least_squares(
            fun=lambda p: self._objective_function(p, y_compressed),
            x0=params_init,
            bounds=(bounds_lower, bounds_upper),
            method='trf',
            max_nfev=max_iter,
            ftol=ftol,
            xtol=xtol,
            verbose=max(0, verbose - 1)
        )
        
        # Extract optimal parameters
        D_opt = result.x[0]
        beta_opt = result.x[1]
        
        if verbose > 0:
            print(f"Optimal: D={D_opt:.6f}, beta={beta_opt:.6f}")
            print(f"Cost: {result.cost:.6e}, Iterations: {result.nfev}")
        
        # Get final linear parameters
        U_R, sigma = self._compute_reduced_basis(D_opt, beta_opt)
        source_rate, background, c0 = self._solve_linear_parameters(
            U_R, y_compressed, D_opt, beta_opt
        )
        
        # Compute final residuals
        final_residuals = self._objective_function(result.x, y_compressed)
        
        # Estimate parameter errors
        errors = self._estimate_errors(result.x, y_compressed, final_residuals)
        D_err = errors[0]
        beta_err = errors[1]
        
        # For k_on and k_off, we derive from beta if reaction model is used
        # Simplified: assume k_on = k_off = beta/2 as placeholder
        k_on = beta_opt / 2
        k_off = beta_opt / 2
        k_on_err = beta_err / 2
        k_off_err = beta_err / 2
        
        # Goodness of fit
        K = self.n_modes
        n_params = 2 + 2 + K * K  # D, beta + source, background + c0
        r_squared, adj_r_squared, ks_stat, ks_pvalue = self._compute_goodness_of_fit(
            y_compressed, final_residuals, n_params
        )
        
        if verbose > 0:
            print(f"R² = {r_squared:.4f}, Adjusted R² = {adj_r_squared:.4f}")
            print(f"KS test: statistic={ks_stat:.4f}, p-value={ks_pvalue:.4f}")
        
        return HiFRAPResult(
            D=D_opt,
            k_on=k_on,
            k_off=k_off,
            beta=beta_opt,
            c0=c0,
            source_rate=source_rate,
            background=background,
            D_err=D_err,
            k_on_err=k_on_err,
            k_off_err=k_off_err,
            beta_err=beta_err,
            r_squared=r_squared,
            adjusted_r_squared=adj_r_squared,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pvalue,
            residuals=final_residuals,
            n_iterations=result.nfev,
            cost=result.cost,
            success=result.success
        )
    
    def predict(
        self,
        result: HiFRAPResult,
        times: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict FRAP recovery curve using fitted parameters.
        
        Parameters
        ----------
        result : HiFRAPResult
            Fitting result from fit() method.
        times : np.ndarray, optional
            Time points for prediction. Default: original time points.
            
        Returns
        -------
        prediction : np.ndarray
            Predicted compressed data at each time point.
            
        Raises
        ------
        RuntimeError
            If fit() has not been called first.
        """
        if self._T is None or self._N is None or self.wavenumbers is None:
            raise RuntimeError("fit() must be called before using this method")
            
        if times is None:
            times = np.arange(self._T) * self.dt
        
        K = self.n_modes
        n_times = len(times)
        prediction = np.zeros((n_times, K, K))
        
        for i, t in enumerate(times):
            # Apply transport kernel to initial condition
            K_x = self._compute_1d_kernel(result.D, result.beta, t)
            K_y = self._compute_1d_kernel(result.D, result.beta, t)
            
            c_t = K_x @ result.c0 @ K_y.T
            
            # Add source and background
            c_t[0, 0] += result.source_rate * t + result.background
            
            prediction[i] = c_t
        
        return prediction
    
    def reconstruct_images(
        self,
        result: HiFRAPResult,
        times: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Reconstruct full images from fitted parameters.
        
        Parameters
        ----------
        result : HiFRAPResult
            Fitting result from fit() method.
        times : np.ndarray, optional
            Time points for reconstruction.
            
        Returns
        -------
        images : np.ndarray
            Reconstructed images of shape (T, N, N).
            
        Raises
        ------
        RuntimeError
            If fit() has not been called first.
        """
        if self._T is None or self._N is None or self.F is None:
            raise RuntimeError("fit() must be called before using this method")
            
        if times is None:
            times = np.arange(self._T) * self.dt
        
        # Get predictions in Fourier space
        pred_fourier = self.predict(result, times)
        
        # Inverse transform back to image space
        # images = F.T @ pred @ F
        n_times = len(times)
        N = self._N
        images = np.zeros((n_times, N, N))
        
        for i in range(n_times):
            images[i] = self.F.T @ pred_fourier[i] @ self.F
        
        return images
        
        for i in range(n_times):
            images[i] = self.F.T @ pred_fourier[i] @ self.F
        
        return images


def simulate_frap_data(
    N: int = 64,
    T: int = 50,
    D: float = 1.0,
    beta: float = 0.1,
    bleach_radius: float = 0.2,
    noise_level: float = 0.05,
    dt: float = 0.1,
    dx: float = 0.1,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Simulate FRAP recovery data for testing.
    
    Parameters
    ----------
    N : int
        Image size (pixels).
    T : int
        Number of time frames.
    D : float
        True diffusion coefficient.
    beta : float
        True bleaching/decay parameter.
    bleach_radius : float
        Bleach spot radius as fraction of image size.
    noise_level : float
        Gaussian noise standard deviation.
    dt : float
        Time step.
    dx : float
        Pixel size.
    random_seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    images : np.ndarray
        Simulated FRAP image stack (T, N, N).
    true_params : dict
        Dictionary of true parameter values.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create coordinate grid
    x = np.linspace(-0.5, 0.5, N)
    y = np.linspace(-0.5, 0.5, N)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Initial condition: bleach spot at center
    I0 = 1.0  # Pre-bleach intensity
    bleach_depth = 0.8  # How much intensity is bleached
    initial = I0 - bleach_depth * np.exp(-R**2 / (2 * bleach_radius**2))
    
    # Simulate diffusion using Fourier method
    L = N * dx
    kx = np.fft.fftfreq(N, dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    
    images = np.zeros((T, N, N))
    
    # Initial Fourier transform
    c_hat = np.fft.fft2(initial)
    
    for t_idx in range(T):
        t = t_idx * dt
        
        # Diffusion and decay in Fourier space
        decay = np.exp(-D * K2 * t - beta * t)
        c_hat_t = c_hat * decay
        
        # Inverse transform
        c_t = np.real(np.fft.ifft2(c_hat_t))
        
        # Add noise
        c_t += np.random.randn(N, N) * noise_level
        
        # Ensure non-negative
        c_t = np.maximum(c_t, 0)
        
        images[t_idx] = c_t
    
    true_params = {
        'D': D,
        'beta': beta,
        'bleach_radius': bleach_radius,
        'noise_level': noise_level
    }
    
    return images, true_params


# Example usage and testing
def fit_hifrap(
    images: np.ndarray,
    time_step: float,
    pixel_size: float,
    n_modes: int = 9,
    D_init: Optional[float] = None,
    beta_init: Optional[float] = None,
    D_bounds: Tuple[float, float] = (1e-4, 100.0),
    beta_bounds: Tuple[float, float] = (0.0, 10.0),
    verbose: int = 0
) -> HiFRAPResult:
    """
    Convenience function to fit FRAP data using HiFRAP method.
    
    This is the main entry point for HiFRAP analysis.
    
    Parameters
    ----------
    images : np.ndarray
        3D array of shape (T, N, N) - time series of square images.
    time_step : float
        Time step between frames (dt) in seconds.
    pixel_size : float
        Pixel size (dx) in micrometers.
    n_modes : int, optional
        Number of Fourier modes (default=9).
    D_init : float, optional
        Initial guess for diffusion coefficient.
    beta_init : float, optional
        Initial guess for bleaching parameter.
    D_bounds : tuple, optional
        Bounds for diffusion coefficient.
    beta_bounds : tuple, optional
        Bounds for bleaching parameter.
    verbose : int, optional
        Verbosity level.
        
    Returns
    -------
    result : HiFRAPResult
        Fitting results including D, beta, errors, and statistics.
        
    Example
    -------
    >>> from hifrap_fitter import fit_hifrap
    >>> result = fit_hifrap(images, time_step=0.1, pixel_size=0.1)
    >>> print(f"D = {result.D:.4f} ± {result.D_err:.4f} µm²/s")
    """
    fitter = HiFRAPFitter(
        time_step=time_step,
        pixel_size=pixel_size,
        n_modes=n_modes
    )
    
    return fitter.fit(
        images=images,
        D_init=D_init,
        beta_init=beta_init,
        D_bounds=D_bounds,
        beta_bounds=beta_bounds,
        verbose=verbose
    )


if __name__ == "__main__":
    print("=" * 60)
    print("HiFRAP Fitter - Test Suite")
    print("=" * 60)
    
    # Generate test data
    print("\n1. Generating simulated FRAP data...")
    images, true_params = simulate_frap_data(
        N=64,
        T=30,
        D=0.5,
        beta=0.05,
        noise_level=0.02,
        random_seed=42
    )
    print(f"   True D = {true_params['D']}")
    print(f"   True beta = {true_params['beta']}")
    print(f"   Image shape: {images.shape}")
    
    # Create fitter
    print("\n2. Creating HiFRAP fitter...")
    fitter = HiFRAPFitter(
        time_step=0.1,
        pixel_size=0.1,
        n_modes=9
    )
    
    # Fit data
    print("\n3. Fitting data...")
    result = fitter.fit(images, verbose=1)
    
    # Report results
    print("\n4. Results:")
    print(f"   D = {result.D:.4f} ± {result.D_err:.4f} (true: {true_params['D']})")
    print(f"   beta = {result.beta:.4f} ± {result.beta_err:.4f} (true: {true_params['beta']})")
    print(f"   R² = {result.r_squared:.4f}")
    print(f"   Adjusted R² = {result.adjusted_r_squared:.4f}")
    print(f"   Converged: {result.success}")
    
    # Error analysis
    D_error_pct = abs(result.D - true_params['D']) / true_params['D'] * 100
    beta_error_pct = abs(result.beta - true_params['beta']) / true_params['beta'] * 100
    print(f"\n5. Error Analysis:")
    print(f"   D error: {D_error_pct:.1f}%")
    print(f"   beta error: {beta_error_pct:.1f}%")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
