import numpy as np
import pytest


@pytest.fixture
def frap_single_exp():
    """Single-component FRAP curve with known ground truth."""
    # D_true = 1.0 um^2/s, w_um = 1.0 um -> tau_s = w_um^2/(4*D) = 0.25 s
    t_s = np.linspace(0, 5, 200)
    tau_s = 0.25
    mobile_fraction = 0.80
    intensity = mobile_fraction * (1 - np.exp(-t_s / tau_s))
    return {
        "t_s": t_s,
        "intensity": intensity,
        "tau_true_s": tau_s,
        "mobile_fraction_true": mobile_fraction,
        "D_true_um2_per_s": 1.0,
        "w_um": 1.0,
    }


@pytest.fixture
def frap_double_exp():
    """Double-component FRAP with well-separated components."""
    t_s = np.linspace(0, 120, 500)
    amp1, tau1_s = 0.50, 2.0
    amp2, tau2_s = 0.30, 40.0
    intensity = amp1 * (1 - np.exp(-t_s / tau1_s)) + amp2 * (1 - np.exp(-t_s / tau2_s))
    return {
        "t_s": t_s,
        "intensity": intensity,
        "amp1": amp1,
        "tau1_s": tau1_s,
        "amp2": amp2,
        "tau2_s": tau2_s,
        "mobile_fraction_true": amp1 + amp2,
    }


@pytest.fixture
def frap_soumpasis():
    """FRAP curve following the Soumpasis model."""
    from scipy.special import iv

    t_s = np.linspace(0.001, 10, 300)
    d_true_um2_per_s = 0.5
    w_um = 1.0
    tau_D_s = w_um**2 / (4 * d_true_um2_per_s)
    z = tau_D_s / (2 * t_s)
    intensity = np.exp(-z) * (iv(0, z) + iv(1, z))
    return {
        "t_s": t_s,
        "intensity": intensity,
        "D_true_um2_per_s": d_true_um2_per_s,
        "w_um": w_um,
        "tau_D_true_s": tau_D_s,
    }


@pytest.fixture
def noisy_frap(frap_single_exp):
    """Single-component FRAP with 2% Gaussian noise."""
    rng = np.random.default_rng(42)
    noisy_intensity = frap_single_exp["intensity"] + rng.normal(0, 0.02, len(frap_single_exp["intensity"]))
    return {**frap_single_exp, "intensity": noisy_intensity}

