"""Normalization-mode behavior tests."""

from __future__ import annotations

import numpy as np
import pytest


def test_simple_normalization_range():
    """Simple normalization should anchor pre-bleach~1 and first post-bleach near 0."""
    from frap2025.core.frap_core import normalize_simple

    t_s = np.linspace(-2.0, 30.0, 200)
    intensity = np.where(
        t_s < 0,
        1000.0,
        80.0 + 720.0 * (1.0 - np.exp(-np.maximum(t_s, 0.0) / 8.0)),
    )

    f_norm = normalize_simple(t_s, intensity, background=50.0)

    post_idx = np.where(t_s >= 0)[0]
    assert f_norm[post_idx[0]] < 0.05

    pre_idx = np.where(t_s < 0)[0]
    assert abs(float(np.mean(f_norm[pre_idx])) - 1.0) < 0.01


def test_double_normalization_corrects_bleaching():
    """Double normalization should remove monotonic acquisition photobleaching."""
    from frap2025.core.frap_core import normalize_double

    t_s = np.linspace(0, 60, 300)
    tau_frap_s = 5.0
    tau_bleach_s = 200.0

    f_roi = 0.8 * (1 - np.exp(-t_s / tau_frap_s)) * np.exp(-t_s / tau_bleach_s)
    f_ref = np.exp(-t_s / tau_bleach_s)

    corrected = normalize_double(
        t_s,
        f_roi,
        f_ref,
        F_roi_pre=1.0,
        F_ref_pre=1.0,
    )

    plateau = float(np.mean(corrected[t_s > 40]))
    assert abs(plateau - 0.8) < 0.05


def test_full_scale_normalization_range():
    """Full-scale normalization should map first post-bleach to ~0 and pre-bleach mean to ~1."""
    from frap2025.core.frap_core import normalize_full_scale

    t_s = np.linspace(-2.0, 20.0, 150)
    intensity = np.where(
        t_s < 0,
        1000.0,
        120.0 + 760.0 * (1.0 - np.exp(-np.maximum(t_s, 0.0) / 6.0)),
    )

    f_norm = normalize_full_scale(t_s, intensity)
    pre_idx = np.where(t_s < 0)[0]
    post_idx = np.where(t_s >= 0)[0]
    assert abs(float(np.mean(f_norm[pre_idx])) - 1.0) < 0.01
    assert abs(float(f_norm[post_idx[0]])) < 0.03


def test_normalize_frap_curve_dispatch():
    """Dispatch API should route to all explicit normalization modes."""
    from frap2025.core.frap_core import normalize_frap_curve

    t_s = np.linspace(-1.0, 10.0, 60)
    intensity = np.where(t_s < 0, 500.0, 80.0 + 360.0 * (1 - np.exp(-np.maximum(t_s, 0.0) / 2.5)))
    ref = np.linspace(1.0, 0.95, t_s.size)

    simple = normalize_frap_curve(t_s, intensity, mode="simple", F_bg=20.0)
    full_scale = normalize_frap_curve(t_s, intensity, mode="full_scale")
    double = normalize_frap_curve(t_s, intensity, mode="double", F_ref=ref, F_roi_pre=1.0, F_ref_pre=1.0)

    assert simple.shape == intensity.shape
    assert full_scale.shape == intensity.shape
    assert double.shape == intensity.shape


def test_normalization_errors():
    """Physiologically implausible inputs should raise NormalizationError."""
    from frap2025.core.frap_core import NormalizationError, normalize_double, normalize_frap_curve, normalize_simple

    t_s = np.linspace(0, 10, 20)
    with pytest.raises(NormalizationError):
        normalize_simple(t_s, np.ones_like(t_s), background=2.0)

    with pytest.raises(NormalizationError):
        normalize_double(t_s, np.ones_like(t_s), np.zeros_like(t_s), F_roi_pre=1.0, F_ref_pre=1.0)

    with pytest.raises(NormalizationError):
        normalize_frap_curve(t_s, np.ones_like(t_s), mode="invalid")
