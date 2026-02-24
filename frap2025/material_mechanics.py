"""Material mechanics utilities for condensate biophysics.

Includes capillary-wave spectroscopy of condensate boundaries.
"""

from __future__ import annotations

import warnings

import numpy as np

BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23


def _validate_contour(contour: np.ndarray, index: int) -> np.ndarray:
    arr = np.asarray(contour, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Contour {index} must have shape (N, 2).")
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if arr.shape[0] < 8:
        raise ValueError(f"Contour {index} must contain at least 8 finite points.")
    return arr


def _radial_profile_on_uniform_theta(contour_xy: np.ndarray, n_theta_samples: int) -> tuple[np.ndarray, np.ndarray, float]:
    centroid = contour_xy.mean(axis=0)
    rel = contour_xy - centroid

    theta = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2.0 * np.pi)
    radius = np.linalg.norm(rel, axis=1)

    order = np.argsort(theta)
    theta_sorted = theta[order]
    radius_sorted = radius[order]

    theta_aug = np.concatenate([theta_sorted, theta_sorted[:1] + 2.0 * np.pi])
    radius_aug = np.concatenate([radius_sorted, radius_sorted[:1]])

    theta_uniform = np.linspace(0.0, 2.0 * np.pi, n_theta_samples, endpoint=False)
    radius_uniform = np.interp(theta_uniform, theta_aug, radius_aug)
    mean_radius = float(np.mean(radius_uniform))

    return theta_uniform, radius_uniform, mean_radius


def compute_capillary_waves(
    contour_timeseries: list[np.ndarray],
    pixel_size: float,
    temperature: float,
    *,
    n_theta_samples: int = 256,
    min_mode: int = 2,
    max_mode: int | None = None,
) -> dict[str, float | np.ndarray | int]:
    """Compute capillary-wave mode spectra and infer relative surface tension.

    Parameters
    ----------
    contour_timeseries : list of ndarray
        Sequence of contours over time. Each contour must be an ``(N, 2)`` array of
        boundary coordinates in pixel units ``(x, y)``.
    pixel_size : float
        Pixel size in meters/pixel, used to convert geometry to SI units.
    temperature : float
        Absolute temperature in kelvin.
    n_theta_samples : int, default=256
        Number of angular samples for interpolating ``r(theta)`` before FFT.
    min_mode : int, default=2
        Minimum Fourier mode included in fitting (mode 0 is mean radius, mode 1
        often reflects translation artifacts).
    max_mode : int or None, default=None
        Maximum Fourier mode included in fitting. If ``None``, uses Nyquist-limited
        maximum from ``n_theta_samples``.

    Returns
    -------
    dict
        Dictionary with spectral and fit outputs:

        - ``q_modes``: physical wavenumbers in ``1/m``.
        - ``mode_indices``: integer Fourier mode indices.
        - ``variance_uq``: time-averaged mode variances ``<|u_q|^2>`` in ``m^2``.
        - ``gamma_estimate_n_per_m``: inferred relative surface tension ``gamma`` in ``N/m``.
        - ``gamma_per_mode_n_per_m``: per-mode gamma estimates from equipartition.
        - ``slope_fit``: fitted proportionality constant in
          ``<|u_q|^2> = slope_fit / q^2``.
        - ``r_squared``: coefficient of determination for the linearized fit.
        - ``n_timepoints_used``: number of contours used.

    Notes
    -----
    For each timepoint, the contour is represented as a radial function
    ``r(theta)`` around the centroid and decomposed as

    ``u(theta) = r(theta) - <r(theta)>``.

    The Fourier coefficients ``u_q`` are obtained from the 1D FFT of ``u(theta)``.
    Capillary-wave equipartition predicts

    ``<|u_q|^2> = (k_B T) / (gamma q^2)``,

    where ``k_B`` is Boltzmann's constant and ``gamma`` is interfacial tension.
    Rearranging gives per-mode estimates

    ``gamma_q = (k_B T) / (<|u_q|^2> q^2)``.

    A global ``gamma`` is estimated by a constrained least-squares fit of
    ``<|u_q|^2>`` against ``1/q^2`` with zero intercept.
    """
    if not contour_timeseries:
        raise ValueError("contour_timeseries must contain at least one contour.")
    if pixel_size <= 0:
        raise ValueError("pixel_size must be > 0.")
    if temperature <= 0:
        raise ValueError("temperature must be > 0 K.")
    if n_theta_samples < 16:
        raise ValueError("n_theta_samples must be >= 16.")

    validated = [_validate_contour(contour, idx) for idx, contour in enumerate(contour_timeseries)]

    max_allowed_mode = n_theta_samples // 2
    if max_mode is None:
        max_mode = max_allowed_mode
    if min_mode < 1 or min_mode >= max_mode:
        raise ValueError("Require 1 <= min_mode < max_mode.")

    modal_amplitudes: list[np.ndarray] = []
    mean_radii_m: list[float] = []

    for contour in validated:
        _, radius_px, mean_radius_px = _radial_profile_on_uniform_theta(contour, n_theta_samples=n_theta_samples)

        radius_m = radius_px * pixel_size
        mean_radius_m = mean_radius_px * pixel_size
        fluctuation = radius_m - np.mean(radius_m)

        fft_coeff = np.fft.rfft(fluctuation) / n_theta_samples
        modal_amplitudes.append(fft_coeff)
        mean_radii_m.append(mean_radius_m)

    amplitudes = np.vstack(modal_amplitudes)
    variances = np.mean(np.abs(amplitudes) ** 2, axis=0)

    mode_indices = np.arange(0, variances.size, dtype=int)
    effective_radius_m = float(np.mean(mean_radii_m))
    if effective_radius_m <= 0:
        raise ValueError("Computed non-positive effective radius; check contour geometry and pixel size.")

    q_modes = mode_indices / effective_radius_m

    fit_mask = (
        (mode_indices >= int(min_mode))
        & (mode_indices <= int(max_mode))
        & np.isfinite(variances)
        & (variances > 0)
        & (q_modes > 0)
    )

    if np.sum(fit_mask) < 3:
        raise ValueError("Insufficient valid Fourier modes for capillary-wave fitting.")

    q_fit = q_modes[fit_mask]
    var_fit = variances[fit_mask]

    x = 1.0 / (q_fit**2)
    slope = float(np.dot(x, var_fit) / np.dot(x, x))
    predicted = slope * x

    ss_res = float(np.sum((var_fit - predicted) ** 2))
    ss_tot = float(np.sum((var_fit - np.mean(var_fit)) ** 2))
    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else np.nan

    kbt = BOLTZMANN_CONSTANT_J_PER_K * float(temperature)
    if slope <= 0:
        warnings.warn(
            "Non-positive fitted slope for capillary-wave spectrum. Returning NaN gamma estimate.",
            RuntimeWarning,
            stacklevel=2,
        )
        gamma_estimate = np.nan
    else:
        gamma_estimate = float(kbt / slope)

    gamma_per_mode = np.full_like(q_fit, np.nan, dtype=float)
    positive_mask = (var_fit > 0) & (q_fit > 0)
    gamma_per_mode[positive_mask] = kbt / (var_fit[positive_mask] * q_fit[positive_mask] ** 2)

    return {
        "q_modes": q_fit,
        "mode_indices": mode_indices[fit_mask],
        "variance_uq": var_fit,
        "gamma_estimate_n_per_m": gamma_estimate,
        "gamma_per_mode_n_per_m": gamma_per_mode,
        "slope_fit": slope,
        "r_squared": r_squared,
        "effective_radius_m": effective_radius_m,
        "n_timepoints_used": int(len(validated)),
    }


__all__ = ["BOLTZMANN_CONSTANT_J_PER_K", "compute_capillary_waves"]
