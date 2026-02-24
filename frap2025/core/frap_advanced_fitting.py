"""Model-selection utilities for FRAP fits."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .frap_fitting import (
    FitQualityWarning,
    fit_double_exponential,
    fit_single_exponential,
    fit_soumpasis,
    fit_triple_exponential,
)


def _delta_aic_interpretation(delta_aic: float) -> str:
    if delta_aic < 2:
        return "substantial support"
    if 4 <= delta_aic <= 7:
        return "considerably less support"
    if delta_aic > 10:
        return "essentially no support"
    return "moderate support"


def select_best_model(
    time_s: np.ndarray | list[float],
    intensity: np.ndarray | list[float],
    *,
    max_components: int = 3,
    bleach_radius_um: float = 1.0,
) -> dict[str, Any]:
    """
    Fit candidate models and rank by AIC, using BIC as a secondary criterion.
    """
    candidates: list[dict[str, Any]] = []

    def _as_dict(result: Any) -> dict[str, Any]:
        if hasattr(result, "to_dict"):
            return result.to_dict()
        if isinstance(result, dict):
            return result
        raise TypeError(f"Unsupported result type: {type(result)!r}")

    try:
        candidates.append(_as_dict(fit_single_exponential(time_s, intensity, bleach_radius_um=bleach_radius_um)))
    except Exception:
        pass
    if max_components >= 2:
        try:
            candidates.append(_as_dict(fit_double_exponential(time_s, intensity, bleach_radius_um=bleach_radius_um)))
        except Exception:
            pass
    if max_components >= 3:
        try:
            candidates.append(_as_dict(fit_triple_exponential(time_s, intensity, bleach_radius_um=bleach_radius_um)))
        except Exception:
            pass
    try:
        candidates.append(_as_dict(fit_soumpasis(time_s, intensity, w_um=bleach_radius_um)))
    except Exception:
        pass

    if not candidates:
        raise ValueError("No model fit succeeded.")

    ranked = sorted(candidates, key=lambda r: (r.get("aic", np.inf), r.get("bic", np.inf)))
    best = ranked[0]
    best_aic = best.get("aic", np.inf)

    for row in ranked:
        delta_aic = float(row.get("aic", np.inf) - best_aic)
        row["delta_aic"] = delta_aic
        row["delta_aic_interpretation"] = _delta_aic_interpretation(delta_aic)

    if best.get("chi2_reduced", np.nan) > 2:
        warnings.warn(
            "Best model by AIC has reduced chi-squared > 2. Fit quality is poor regardless of complexity.",
            FitQualityWarning,
            stacklevel=2,
        )

    return {
        **best,
        "ranked_models": ranked,
        "aic_preferred": True,
        "bic_secondary": True,
    }
