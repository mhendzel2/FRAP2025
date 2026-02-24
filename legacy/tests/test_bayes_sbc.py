"""
Simulation-based calibration scaffold for Bayesian FRAP models.

This is intentionally lightweight so it can run in CI with small draws while
still validating rank-uniformity plumbing.
"""

import numpy as np
import pandas as pd
import pytest

from frap_bayes_data import build_bayesian_dataset
from frap_bayes_fit import fit_hierarchical_bayes
from frap_bayes_models import PYMC_AVAILABLE


def _simulate_single_exp_trace(k: float, n_time: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 6.0, n_time)
    mu = 1.0 - np.exp(-k * t)
    y = mu + rng.normal(0.0, 0.02, size=t.size)
    return t, y


def _make_simulated_tables(n_cells: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)

    trace_rows = []
    feature_rows = []

    for cell_id in range(1, n_cells + 1):
        k_true = float(rng.lognormal(mean=-1.0, sigma=0.25))
        condition = "A" if cell_id <= (n_cells // 2) else "B"

        t, y = _simulate_single_exp_trace(k_true, seed=seed + cell_id)

        for frame, (tt, yy) in enumerate(zip(t, y)):
            trace_rows.append(
                {
                    "exp_id": "exp0",
                    "movie_id": "mov0",
                    "cell_id": cell_id,
                    "frame": frame,
                    "t": float(tt),
                    "x": 0.0,
                    "y": 0.0,
                    "radius": 3.0,
                    "signal_raw": yy,
                    "signal_bg": 0.0,
                    "signal_corr": yy,
                    "signal_norm": yy,
                    "qc_motion": False,
                    "qc_reason": "",
                }
            )

        feature_rows.append(
            {
                "exp_id": "exp0",
                "movie_id": "mov0",
                "cell_id": cell_id,
                "condition": condition,
                "bleach_qc": True,
                "k": k_true,
                "I0": 0.0,
                "I_inf": 1.0,
            }
        )

    return pd.DataFrame(trace_rows), pd.DataFrame(feature_rows)


def _rank_statistic(samples: np.ndarray, truth: float) -> int:
    """Rank of the true value among posterior samples (0..n_samples)."""
    samples = np.asarray(samples, dtype=float)
    return int(np.sum(samples < truth))


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC unavailable")
def test_sbc_rank_statistic_scaffold_runs():
    traces, features = _make_simulated_tables(n_cells=4)
    dataset = build_bayesian_dataset(traces, features)

    fit = fit_hierarchical_bayes(
        dataset=dataset,
        model_name="hb_1exp",
        draws=80,
        tune=80,
        chains=2,
        target_accept=0.9,
        sample_prior_predictive_draws=0,
        sample_posterior_predictive=False,
    )

    posterior_k = np.asarray(fit.idata.posterior["k"]).reshape(-1, dataset.n_cells)
    cell0_samples = posterior_k[:, 0]
    k_true_cell0 = float(features.iloc[0]["k"])

    rank = _rank_statistic(cell0_samples, k_true_cell0)
    assert 0 <= rank <= cell0_samples.size
