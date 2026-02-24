"""Synthetic-data fitting tests with parameter recovery checks."""

from __future__ import annotations

import numpy as np
import pytest


def test_double_exp_component_separation(frap_double_exp):
    """Well-separated components (tau1=2s, tau2=40s) must be recovered."""
    from frap2025.core.frap_fitting import fit_double_exponential

    result = fit_double_exponential(frap_double_exp["t_s"], frap_double_exp["intensity"])
    components = sorted(result["components"], key=lambda c: c["tau_s"])

    assert abs(components[0]["tau_s"] - frap_double_exp["tau1_s"]) / frap_double_exp["tau1_s"] < 0.05
    assert abs(components[1]["tau_s"] - frap_double_exp["tau2_s"]) / frap_double_exp["tau2_s"] < 0.05


def test_double_exp_mobile_fraction_sum(frap_double_exp):
    """Sum of component amplitudes must match true total mobile fraction."""
    from frap2025.core.frap_fitting import fit_double_exponential

    result = fit_double_exponential(frap_double_exp["t_s"], frap_double_exp["intensity"])
    amp_sum = sum(component["amplitude"] for component in result["components"])
    assert abs(amp_sum - frap_double_exp["mobile_fraction_true"]) < 0.02


def test_model_selection_prefers_single_for_single_data(frap_single_exp):
    """AIC/BIC selection should favor one component on one-component data."""
    from frap2025.core.frap_advanced_fitting import select_best_model

    best = select_best_model(frap_single_exp["t_s"], frap_single_exp["intensity"], max_components=3)
    assert best["n_components"] == 1


def test_fit_confidence_intervals_present(noisy_frap):
    """Single-component fit must include confidence intervals."""
    from frap2025.core.frap_fitting import fit_single_exponential

    result = fit_single_exponential(noisy_frap["t_s"], noisy_frap["intensity"])

    assert "tau_s_ci95" in result
    assert "mobile_fraction_ci95" in result
    low, high = result["tau_s_ci95"]
    assert low < result["tau_s"] < high


def test_poor_fit_raises_warning():
    """Flat curves should emit FitQualityWarning rather than silently returning garbage."""
    from frap2025.core.frap_fitting import FitQualityWarning, fit_single_exponential

    t_s = np.linspace(0, 10, 100)
    rng = np.random.default_rng(0)
    flat = np.ones(100) * 0.1 + rng.normal(0, 0.01, 100)

    with pytest.warns(FitQualityWarning):
        fit_single_exponential(t_s, flat)


def test_delta_aic_interpretation_buckets():
    """AIC-interpretation helper should map to expected textual buckets."""
    from frap2025.core.frap_advanced_fitting import _delta_aic_interpretation

    assert _delta_aic_interpretation(1.0) == "substantial support"
    assert _delta_aic_interpretation(5.0) == "considerably less support"
    assert _delta_aic_interpretation(12.0) == "essentially no support"
    assert _delta_aic_interpretation(3.0) == "moderate support"
