"""
Verifies diffusion coefficient formulas against known ground truth.
ALL TESTS IN THIS FILE MUST PASS BEFORE OTHER CHANGES ARE MADE.
"""

from __future__ import annotations

import numpy as np
from scipy.special import iv


def verify_diffusion_formula() -> dict[str, float]:
    """
    End-to-end formula verification using synthetic noise-free FRAP curves.

    Convention verified here and used in code:
    - Exponential fits use k_s = 1/tau_s (from exp(-k_s * t_s)).
    - Therefore D_um2_per_s = w_um^2 * k_s / 4 = w_um^2 / (4*tau_s).
    - If starting from half-time instead: D_um2_per_s = w_um^2 * ln(2) / (4*t_half_s).
    """
    from frap2025.core.frap_fitting import (
        fit_double_exponential,
        fit_single_exponential,
        fit_soumpasis,
    )

    # Single exponential synthetic trace.
    w_um = 1.0
    d_single_true_um2_per_s = 1.0
    tau_single_true_s = w_um**2 / (4 * d_single_true_um2_per_s)
    t_single_s = np.linspace(0, 5, 300)
    intensity_single = 0.8 * (1 - np.exp(-t_single_s / tau_single_true_s))
    single_fit = fit_single_exponential(t_single_s, intensity_single, bleach_radius_um=w_um)
    d_single_recovered_um2_per_s = w_um**2 / (4 * single_fit["tau_s"])

    # Double exponential synthetic trace (two known components).
    d_fast_true_um2_per_s = 0.5
    d_slow_true_um2_per_s = 0.05
    tau_fast_true_s = w_um**2 / (4 * d_fast_true_um2_per_s)
    tau_slow_true_s = w_um**2 / (4 * d_slow_true_um2_per_s)
    t_double_s = np.linspace(0, 80, 1200)
    intensity_double = 0.55 * (1 - np.exp(-t_double_s / tau_fast_true_s)) + 0.30 * (
        1 - np.exp(-t_double_s / tau_slow_true_s)
    )
    double_fit = fit_double_exponential(t_double_s, intensity_double, bleach_radius_um=w_um)
    components = sorted(double_fit["components"], key=lambda c: c["tau_s"])
    d_fast_recovered_um2_per_s = w_um**2 / (4 * components[0]["tau_s"])
    d_slow_recovered_um2_per_s = w_um**2 / (4 * components[1]["tau_s"])

    # Soumpasis synthetic trace.
    d_soumpasis_true_um2_per_s = 0.5
    tau_D_true_s = w_um**2 / (4 * d_soumpasis_true_um2_per_s)
    t_soumpasis_s = np.linspace(0.001, 10, 500)
    z = tau_D_true_s / (2 * t_soumpasis_s)
    intensity_soumpasis = np.exp(-z) * (iv(0, z) + iv(1, z))
    soumpasis_fit = fit_soumpasis(t_soumpasis_s, intensity_soumpasis, w_um=w_um)
    d_soumpasis_recovered_um2_per_s = soumpasis_fit["D_um2_per_s"]

    rel_single = abs(d_single_recovered_um2_per_s - d_single_true_um2_per_s) / d_single_true_um2_per_s
    rel_fast = abs(d_fast_recovered_um2_per_s - d_fast_true_um2_per_s) / d_fast_true_um2_per_s
    rel_slow = abs(d_slow_recovered_um2_per_s - d_slow_true_um2_per_s) / d_slow_true_um2_per_s
    rel_soumpasis = abs(d_soumpasis_recovered_um2_per_s - d_soumpasis_true_um2_per_s) / d_soumpasis_true_um2_per_s

    assert rel_single < 0.01
    assert rel_fast < 0.01
    assert rel_slow < 0.01
    assert rel_soumpasis < 0.01

    return {
        "single_rel_error": rel_single,
        "double_fast_rel_error": rel_fast,
        "double_slow_rel_error": rel_slow,
        "soumpasis_rel_error": rel_soumpasis,
    }


def test_verify_diffusion_formula():
    stats = verify_diffusion_formula()
    assert stats["single_rel_error"] < 0.01
    assert stats["double_fast_rel_error"] < 0.01
    assert stats["double_slow_rel_error"] < 0.01
    assert stats["soumpasis_rel_error"] < 0.01


def test_single_exp_D_recovery(frap_single_exp):
    """D recovered from tau_s must equal D_true within 1% on noise-free data."""
    from frap2025.core.frap_fitting import fit_single_exponential

    result = fit_single_exponential(
        frap_single_exp["t_s"],
        frap_single_exp["intensity"],
        bleach_radius_um=frap_single_exp["w_um"],
    )
    tau_fit_s = result["tau_s"]
    w_um = frap_single_exp["w_um"]
    d_recovered_um2_per_s = w_um**2 / (4 * tau_fit_s)  # D = w^2/(4*tau_s), k_s = 1/tau_s
    d_true_um2_per_s = frap_single_exp["D_true_um2_per_s"]
    assert abs(d_recovered_um2_per_s - d_true_um2_per_s) / d_true_um2_per_s < 0.01


def test_soumpasis_D_recovery(frap_soumpasis):
    """Soumpasis model must recover D within 2% on noise-free data."""
    from frap2025.core.frap_fitting import fit_soumpasis

    result = fit_soumpasis(
        frap_soumpasis["t_s"],
        frap_soumpasis["intensity"],
        w_um=frap_soumpasis["w_um"],
    )
    d_recovered_um2_per_s = result["D_um2_per_s"]
    d_true_um2_per_s = frap_soumpasis["D_true_um2_per_s"]
    assert abs(d_recovered_um2_per_s - d_true_um2_per_s) / d_true_um2_per_s < 0.02


def test_half_time_to_D_conversion_consistency():
    """Verify ln(2) convention: t_half_s = tau_s*ln(2), D from t_half_s needs ln(2)."""
    tau_s = 0.25
    t_half_s = tau_s * np.log(2)
    w_um = 1.0
    d_from_tau = w_um**2 / (4 * tau_s)
    d_from_t_half = w_um**2 * np.log(2) / (4 * t_half_s)
    assert abs(d_from_tau - d_from_t_half) < 1e-12


def test_mobile_fraction_bounds(frap_single_exp):
    """Mobile fraction must be in (0, 1]."""
    from frap2025.core.frap_fitting import fit_single_exponential

    result = fit_single_exponential(frap_single_exp["t_s"], frap_single_exp["intensity"])
    assert 0 < result["mobile_fraction"] <= 1.0


def test_mobile_fraction_recovery(frap_single_exp):
    """Mobile fraction must recover to within 2% on noise-free data."""
    from frap2025.core.frap_fitting import fit_single_exponential

    result = fit_single_exponential(frap_single_exp["t_s"], frap_single_exp["intensity"])
    mf_true = frap_single_exp["mobile_fraction_true"]
    assert abs(result["mobile_fraction"] - mf_true) < 0.02

