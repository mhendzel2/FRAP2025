"""Estimation-plot utilities (Ho et al. style) with optional `dabest` backend."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class EstimationResult:
    """Mean-difference estimate with bootstrap confidence interval."""

    mean_difference: float
    ci95: tuple[float, float]
    n_bootstrap: int
    method: str


def _bootstrap_mean_difference(
    group_a: np.ndarray,
    group_b: np.ndarray,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> EstimationResult:
    rng = np.random.default_rng(random_seed)
    diffs = np.empty(n_bootstrap, dtype=float)

    for idx in range(n_bootstrap):
        a = rng.choice(group_a, size=group_a.size, replace=True)
        b = rng.choice(group_b, size=group_b.size, replace=True)
        diffs[idx] = float(np.mean(a) - np.mean(b))

    mean_diff = float(np.mean(group_a) - np.mean(group_b))
    ci95 = (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)))
    return EstimationResult(mean_difference=mean_diff, ci95=ci95, n_bootstrap=n_bootstrap, method="bootstrap")


def create_estimation_plot(
    data: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    group_a: str,
    group_b: str,
    n_bootstrap: int = 5000,
    random_seed: int = 42,
):
    """Create a two-panel Cumming-style estimation plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    result : dict
        Includes mean difference and 95% CI.
    """
    if n_bootstrap < 5000:
        raise ValueError("n_bootstrap must be >= 5000 for estimation plots.")

    subset = data[[group_col, value_col]].dropna()
    a = np.asarray(subset.loc[subset[group_col] == group_a, value_col], dtype=float)
    b = np.asarray(subset.loc[subset[group_col] == group_b, value_col], dtype=float)

    if a.size == 0 or b.size == 0:
        raise ValueError("Both groups must contain at least one value.")

    used_dabest = False
    try:
        import dabest  # type: ignore

        dab = dabest.load(subset, x=group_col, y=value_col, idx=(group_a, group_b))
        db_res = dab.mean_diff.results.iloc[0]
        estimate = EstimationResult(
            mean_difference=float(db_res["difference"]),
            ci95=(float(db_res["bca_low"]), float(db_res["bca_high"])),
            n_bootstrap=n_bootstrap,
            method="dabest",
        )
        used_dabest = True
    except Exception:
        estimate = _bootstrap_mean_difference(a, b, n_bootstrap=n_bootstrap, random_seed=random_seed)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(7.5, 7.0), height_ratios=[2, 1], constrained_layout=True)

    jitter_rng = np.random.default_rng(random_seed)
    x_a = np.zeros(a.size) + jitter_rng.uniform(-0.08, 0.08, a.size)
    x_b = np.ones(b.size) + jitter_rng.uniform(-0.08, 0.08, b.size)

    ax_top.scatter(x_a, a, alpha=0.75, s=28, label=group_a)
    ax_top.scatter(x_b, b, alpha=0.75, s=28, label=group_b)
    ax_top.plot([0, 1], [np.mean(a), np.mean(b)], color="black", linewidth=1.2)
    ax_top.set_xticks([0, 1], [group_a, group_b])
    ax_top.set_ylabel(value_col)
    ax_top.set_title("Cumming Estimation Plot")

    md = estimate.mean_difference
    ci_low, ci_high = estimate.ci95
    ax_bottom.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax_bottom.errorbar(
        [0],
        [md],
        yerr=[[md - ci_low], [ci_high - md]],
        fmt="o",
        color="black",
        capsize=4,
    )
    ax_bottom.set_xlim(-0.5, 0.5)
    ax_bottom.set_xticks([0], [f"{group_a} - {group_b}"])
    ax_bottom.set_ylabel(f"Δ {value_col}")

    summary = {
        "mean_difference": md,
        "ci95": estimate.ci95,
        "n_bootstrap": estimate.n_bootstrap,
        "method": estimate.method,
        "label": f"Δ {value_col} = {md:.4g} [{ci_low:.4g}, {ci_high:.4g}]",
        "used_dabest": used_dabest,
    }
    return fig, summary


__all__ = ["create_estimation_plot", "EstimationResult"]
