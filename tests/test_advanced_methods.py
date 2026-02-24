"""Tests for 2025/2026 advanced analysis hooks with graceful degradation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def test_compare_populations_wasserstein_returns_valid_metrics():
    from frap2025.population_analysis import compare_populations_wasserstein

    rng = np.random.default_rng(0)
    df_a = pd.DataFrame(
        {
            "D": rng.normal(1.0, 0.1, 32),
            "alpha": rng.normal(0.8, 0.05, 32),
            "radius": rng.normal(2.0, 0.15, 32),
        }
    )
    df_b = pd.DataFrame(
        {
            "D": rng.normal(1.2, 0.1, 32),
            "alpha": rng.normal(0.9, 0.05, 32),
            "radius": rng.normal(2.2, 0.15, 32),
        }
    )

    result = compare_populations_wasserstein(
        df_a,
        df_b,
        features=["D", "alpha", "radius"],
        n_permutations=40,
        random_state=1,
    )

    assert result["method"] in {"pot_emd2", "featurewise_wasserstein_fallback"}
    distance = float(result["distance"])
    p_value = float(result["p_value"])
    n_perm = int(result["n_permutations"])

    assert distance >= 0
    assert 0.0 <= p_value <= 1.0
    assert n_perm == 40


def test_compute_capillary_waves_returns_finite_spectrum():
    from frap2025.material_mechanics import compute_capillary_waves

    rng = np.random.default_rng(3)
    angles = np.linspace(0.0, 2.0 * np.pi, 160, endpoint=False)

    contours = []
    for _ in range(16):
        r = 35.0 + 1.0 * np.cos(2 * angles + rng.uniform(0, 2 * np.pi))
        r += 0.7 * np.cos(3 * angles + rng.uniform(0, 2 * np.pi))
        r += rng.normal(0.0, 0.15, size=angles.size)
        x = 90.0 + r * np.cos(angles)
        y = 70.0 + r * np.sin(angles)
        contours.append(np.column_stack([x, y]))

    result = compute_capillary_waves(contours, pixel_size=1e-7, temperature=298.15)

    assert result["n_timepoints_used"] == len(contours)
    assert np.isfinite(float(result["effective_radius_m"]))
    q_modes = np.asarray(result["q_modes"], dtype=float)
    variance = np.asarray(result["variance_uq"], dtype=float)
    mode_idx = np.asarray(result["mode_indices"], dtype=int)
    assert len(q_modes) == len(variance) == len(mode_idx)
    assert np.all(q_modes > 0)
    assert np.all(variance > 0)


def test_hdp_hmm_analyzer_outputs_consistent_shapes():
    from frap2025.spt_models.bayesian import HDPHMM_Analyzer

    rng = np.random.default_rng(10)
    increments = np.column_stack([rng.normal(0.25, 0.15, 140), rng.normal(0.10, 0.15, 140)])
    trajectory = np.cumsum(increments, axis=0)

    result = HDPHMM_Analyzer(max_states=5, num_steps=25, random_state=0).analyze(trajectory)

    n_states = int(cast(int, result["optimal_num_states"]))
    transition = np.asarray(result["transition_matrix"], dtype=float)
    state_sequence = np.asarray(result["state_sequence"], dtype=int)

    assert n_states >= 1
    assert transition.shape == (n_states, n_states)
    assert len(state_sequence) == trajectory.shape[0] - 1
    assert result["backend"] in {"pyro_hdp_hmm", "gmm_fallback"}


def test_fit_frap_pinn_returns_parameters_or_fallback():
    from frap2025.frap_fitting.fit_pinn import fit_frap_pinn

    rng = np.random.default_rng(4)
    t_size, h_size, w_size = 10, 20, 20
    stack = np.ones((t_size, h_size, w_size), dtype=float)
    y_grid, x_grid = np.mgrid[:h_size, :w_size]
    bleach = (x_grid - 10) ** 2 + (y_grid - 10) ** 2 <= 9

    for t in range(t_size):
        recovery = 0.2 + 0.7 * (1.0 - np.exp(-t / 3.5))
        stack[t, bleach] = recovery
        stack[t] += rng.normal(0.0, 0.005, size=(h_size, w_size))

    mask = np.ones((h_size, w_size), dtype=bool)

    result = fit_frap_pinn(stack, mask, bleach, epochs=10)

    assert {"D", "k_on", "k_off", "backend", "epochs"}.issubset(result.keys())
    backend = cast(str, result["backend"])
    assert backend in {"torch_pinn", "classical_reaction_diffusion_fallback"}


def test_compute_deformation_field_backend_interface():
    from frap2025.optical_flow_analysis import compute_deformation_field

    h_size, w_size = 64, 64
    y_grid, x_grid = np.mgrid[:h_size, :w_size]
    img1 = np.exp(-((x_grid - 24) ** 2 + (y_grid - 22) ** 2) / (2 * 6.0**2))
    img2 = np.exp(-((x_grid - 27) ** 2 + (y_grid - 24) ** 2) / (2 * 6.0**2))

    flow_farneback = compute_deformation_field(img1, img2, method="farneback")
    flow_raft_or_fallback = compute_deformation_field(img1, img2, method="raft")

    assert flow_farneback.shape == (h_size, w_size, 2)
    assert flow_raft_or_fallback.shape == (h_size, w_size, 2)
    assert np.all(np.isfinite(flow_farneback))
