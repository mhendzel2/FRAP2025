"""Statistical-validation tests for bootstrap and multiple-testing routines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_bootstrap_summary_fields(frap_single_exp):
    """Bootstrap helper should return finite summary fields for valid input."""
    from frap2025.core.frap_bootstrap import bootstrap_frap_fit

    summary = bootstrap_frap_fit(
        frap_single_exp["t_s"],
        frap_single_exp["intensity"],
        bleach_radius_um=frap_single_exp["w_um"],
        n_bootstrap=60,
        random_seed=7,
    )

    assert summary["n_bootstrap"] > 0
    low, high = summary["tau_s_ci95"]
    assert np.isfinite(low) and np.isfinite(high)
    assert low < high


def test_bootstrap_invalid_iteration_count_raises(frap_single_exp):
    """Bootstrap should enforce a minimum iteration count for stable intervals."""
    from frap2025.core.frap_bootstrap import bootstrap_frap_fit

    with pytest.raises(ValueError):
        bootstrap_frap_fit(
            frap_single_exp["t_s"],
            frap_single_exp["intensity"],
            n_bootstrap=10,
        )


@pytest.mark.slow
def test_bootstrap_ci_coverage():
    """Bootstrap 95% CI should contain true tau in most repeated noisy fits."""
    from frap2025.core.frap_bootstrap import bootstrap_frap_fit

    rng = np.random.default_rng(123)
    t_s = np.linspace(0, 5, 200)
    tau_true_s = 0.25
    mobile_fraction_true = 0.8
    clean = mobile_fraction_true * (1.0 - np.exp(-t_s / tau_true_s))

    covered = 0
    n_trials = 24
    for trial in range(n_trials):
        noisy = clean + rng.normal(0.0, 0.02, size=clean.size)
        summary = bootstrap_frap_fit(
            t_s,
            noisy,
            bleach_radius_um=1.0,
            n_bootstrap=120,
            random_seed=trial,
        )
        low, high = summary["tau_s_ci95"]
        if np.isfinite(low) and np.isfinite(high) and low <= tau_true_s <= high:
            covered += 1

    coverage = covered / n_trials
    assert coverage >= 0.80


def test_fdr_correction_applied_to_pvalues():
    """BH-adjusted p-values should be monotone-corrected and capped at 1."""
    from frap2025.stats.frap_group_stats import compare_groups_with_fdr

    rng = np.random.default_rng(42)
    n = 22
    df = pd.DataFrame(
        {
            "group": np.repeat(["A", "B", "C", "D"], n),
            "value": np.concatenate(
                [
                    rng.normal(0.00, 1.0, n),
                    rng.normal(0.05, 1.0, n),
                    rng.normal(0.15, 1.0, n),
                    rng.normal(0.80, 1.0, n),
                ]
            ),
        }
    )

    out = compare_groups_with_fdr(df, correction="fdr_bh")

    assert (out["p_value_adjusted"] <= 1.0).all()
    assert (out["p_value_adjusted"] >= 0.0).all()
    # Correct BH adjustment should not reduce p-values below raw p-values.
    assert (out["p_value_adjusted"] >= out["p_value"] - 1e-12).all()


def test_aic_bic_prefer_simpler_model(frap_single_exp):
    """For one-component synthetic data, best AIC model should be one-component."""
    from frap2025.core.frap_advanced_fitting import select_best_model

    best = select_best_model(frap_single_exp["t_s"], frap_single_exp["intensity"], max_components=3)
    assert best["n_components"] == 1

    ranked = best["ranked_models"]
    assert len(ranked) >= 2
    assert ranked[0]["aic"] <= ranked[1]["aic"]
