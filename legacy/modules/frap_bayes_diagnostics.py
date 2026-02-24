"""
Diagnostics utilities for hierarchical Bayesian FRAP fitting.
"""

from __future__ import annotations

from typing import Any, Optional
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import arviz as az

    ARVIZ_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent
    az = None
    ARVIZ_AVAILABLE = False


def _require_arviz() -> None:
    if not ARVIZ_AVAILABLE:
        raise ImportError("arviz is required for Bayesian diagnostics")


def summarize_diagnostics(
    idata: Any,
    var_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Return convergence summary table and key scalar metrics."""
    _require_arviz()

    summary = az.summary(idata, var_names=var_names)

    rhat_col = "r_hat" if "r_hat" in summary.columns else "rhat"
    ess_bulk_col = "ess_bulk" if "ess_bulk" in summary.columns else None
    ess_tail_col = "ess_tail" if "ess_tail" in summary.columns else None

    max_rhat = float(summary[rhat_col].max()) if rhat_col in summary.columns else float("nan")
    min_ess_bulk = float(summary[ess_bulk_col].min()) if ess_bulk_col else float("nan")
    min_ess_tail = float(summary[ess_tail_col].min()) if ess_tail_col else float("nan")

    divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())

    return {
        "summary_table": summary,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
        "divergences": divergences,
    }


def diagnostics_gate(
    idata: Any,
    *,
    rhat_max: float = 1.01,
    ess_bulk_min: float = 400.0,
    max_divergences: int = 0,
    var_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Apply a simple diagnostics acceptance gate.

    Returns pass/fail plus raw diagnostics.
    """
    info = summarize_diagnostics(idata, var_names=var_names)

    checks = {
        "rhat_ok": bool(np.isfinite(info["max_rhat"]) and info["max_rhat"] <= rhat_max),
        "ess_ok": bool(np.isfinite(info["min_ess_bulk"]) and info["min_ess_bulk"] >= ess_bulk_min),
        "divergence_ok": bool(info["divergences"] <= max_divergences),
    }

    passed = all(checks.values())

    return {
        **info,
        "thresholds": {
            "rhat_max": rhat_max,
            "ess_bulk_min": ess_bulk_min,
            "max_divergences": max_divergences,
        },
        "checks": checks,
        "passed": passed,
    }


def compute_information_criteria(idata: Any) -> dict[str, Any]:
    """Compute PSIS-LOO and WAIC summaries from InferenceData."""
    _require_arviz()

    results: dict[str, Any] = {}

    try:
        loo = az.loo(idata, pointwise=True)
        results["loo"] = {
            "elpd": float(getattr(loo, "elpd_loo", np.nan)),
            "p": float(getattr(loo, "p_loo", np.nan)),
            "se": float(getattr(loo, "se", np.nan)),
            "pareto_k_max": float(np.nanmax(np.asarray(loo.pareto_k))),
        }
    except Exception as exc:
        logger.warning("LOO computation failed: %s", exc)
        results["loo_error"] = str(exc)

    try:
        waic = az.waic(idata, pointwise=True)
        results["waic"] = {
            "elpd": float(getattr(waic, "elpd_waic", np.nan)),
            "p": float(getattr(waic, "p_waic", np.nan)),
            "se": float(getattr(waic, "se", np.nan)),
        }
    except Exception as exc:
        logger.warning("WAIC computation failed: %s", exc)
        results["waic_error"] = str(exc)

    return results


def compare_model_criteria(idata_by_model: dict[str, Any], ic: str = "loo") -> pd.DataFrame:
    """Compare fitted Bayesian models with ArviZ `compare`."""
    _require_arviz()

    if not idata_by_model:
        return pd.DataFrame()

    return az.compare(idata_by_model, ic=ic)


def ppc_summary(idata: Any, observed_var: str = "y") -> dict[str, float]:
    """Compute compact posterior predictive fit metrics."""
    if not hasattr(idata, "posterior_predictive"):
        return {}
    if observed_var not in getattr(idata, "posterior_predictive", {}):
        return {}
    if observed_var not in getattr(idata, "observed_data", {}):
        return {}

    y_obs = np.asarray(idata.observed_data[observed_var]).reshape(-1)
    y_pp = np.asarray(idata.posterior_predictive[observed_var])

    if y_pp.ndim < 2:
        return {}

    # Expected shape: (chain, draw, obs) or (draw, obs)
    if y_pp.ndim == 2:
        samples = y_pp
    else:
        samples = y_pp.reshape(-1, y_pp.shape[-1])

    pred_mean = np.mean(samples, axis=0)
    rmse = float(np.sqrt(np.mean((pred_mean - y_obs) ** 2)))
    mae = float(np.mean(np.abs(pred_mean - y_obs)))

    lo = np.quantile(samples, 0.025, axis=0)
    hi = np.quantile(samples, 0.975, axis=0)
    coverage = float(np.mean((y_obs >= lo) & (y_obs <= hi)))

    return {
        "rmse": rmse,
        "mae": mae,
        "coverage_95": coverage,
    }
