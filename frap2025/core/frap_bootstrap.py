"""Bootstrap utilities for FRAP parameter uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from .frap_fitting import fit_single_exponential

LOGGER = logging.getLogger(__name__)


@dataclass
class BootstrapSummary:
    """Bootstrap summary for single-exponential FRAP parameters."""

    n_bootstrap: int
    tau_s_median: float
    tau_s_ci95: tuple[float, float]
    mobile_fraction_median: float
    mobile_fraction_ci95: tuple[float, float]
    D_um2_per_s_median: float | None
    D_um2_per_s_ci95: tuple[float, float] | None

    def to_dict(self) -> dict[str, float | int | tuple[float, float] | None]:
        return {
            "n_bootstrap": self.n_bootstrap,
            "tau_s_median": self.tau_s_median,
            "tau_s_ci95": self.tau_s_ci95,
            "mobile_fraction_median": self.mobile_fraction_median,
            "mobile_fraction_ci95": self.mobile_fraction_ci95,
            "D_um2_per_s_median": self.D_um2_per_s_median,
            "D_um2_per_s_ci95": self.D_um2_per_s_ci95,
        }


def _single_model(t_s: np.ndarray, mobile_fraction: float, tau_s: float, baseline: float) -> np.ndarray:
    return baseline + mobile_fraction * (1.0 - np.exp(-t_s / tau_s))


def bootstrap_frap_fit(
    time_s: np.ndarray | list[float],
    intensity: np.ndarray | list[float],
    *,
    bleach_radius_um: float | None = None,
    n_bootstrap: int = 500,
    random_seed: int = 42,
) -> dict[str, float | int | tuple[float, float] | None]:
    """Parametric residual bootstrap for single-exponential FRAP fits."""
    if n_bootstrap < 50:
        raise ValueError("n_bootstrap must be >= 50 for stable confidence intervals.")

    t_s = np.asarray(time_s, dtype=float).reshape(-1)
    y = np.asarray(intensity, dtype=float).reshape(-1)
    if t_s.size != y.size:
        raise ValueError("time_s and intensity must have same length.")

    base = fit_single_exponential(t_s, y, bleach_radius_um=bleach_radius_um)
    tau_hat_s = float(base.tau_s)
    mf_hat = float(base.mobile_fraction)
    baseline = float(base.components[0].get("baseline", np.nan)) if base.components else np.nan

    if not np.isfinite(tau_hat_s) or tau_hat_s <= 0 or not np.isfinite(mf_hat) or not np.isfinite(baseline):
        LOGGER.warning("Bootstrap skipped because base fit is invalid.")
        return BootstrapSummary(
            n_bootstrap=0,
            tau_s_median=np.nan,
            tau_s_ci95=(np.nan, np.nan),
            mobile_fraction_median=np.nan,
            mobile_fraction_ci95=(np.nan, np.nan),
            D_um2_per_s_median=np.nan if bleach_radius_um is not None else None,
            D_um2_per_s_ci95=(np.nan, np.nan) if bleach_radius_um is not None else None,
        ).to_dict()

    t_shift_s = t_s - np.min(t_s)
    y_hat = _single_model(t_shift_s, mf_hat, tau_hat_s, baseline)
    residuals = y - y_hat

    rng = np.random.default_rng(random_seed)
    tau_samples: list[float] = []
    mf_samples: list[float] = []
    d_samples: list[float] = []

    for _ in range(int(n_bootstrap)):
        bootstrap_noise = rng.choice(residuals, size=residuals.size, replace=True)
        y_boot = y_hat + bootstrap_noise
        fit = fit_single_exponential(t_s, y_boot, bleach_radius_um=bleach_radius_um)
        if np.isfinite(fit.tau_s) and fit.tau_s > 0 and np.isfinite(fit.mobile_fraction):
            tau_samples.append(float(fit.tau_s))
            mf_samples.append(float(fit.mobile_fraction))
            if fit.D_um2_per_s is not None and np.isfinite(fit.D_um2_per_s):
                d_samples.append(float(fit.D_um2_per_s))

    if not tau_samples:
        LOGGER.warning("All bootstrap iterations failed.")
        return BootstrapSummary(
            n_bootstrap=0,
            tau_s_median=np.nan,
            tau_s_ci95=(np.nan, np.nan),
            mobile_fraction_median=np.nan,
            mobile_fraction_ci95=(np.nan, np.nan),
            D_um2_per_s_median=np.nan if bleach_radius_um is not None else None,
            D_um2_per_s_ci95=(np.nan, np.nan) if bleach_radius_um is not None else None,
        ).to_dict()

    tau_arr = np.asarray(tau_samples, dtype=float)
    mf_arr = np.asarray(mf_samples, dtype=float)
    d_arr = np.asarray(d_samples, dtype=float) if d_samples else np.asarray([], dtype=float)

    summary = BootstrapSummary(
        n_bootstrap=int(tau_arr.size),
        tau_s_median=float(np.median(tau_arr)),
        tau_s_ci95=(float(np.quantile(tau_arr, 0.025)), float(np.quantile(tau_arr, 0.975))),
        mobile_fraction_median=float(np.median(mf_arr)),
        mobile_fraction_ci95=(float(np.quantile(mf_arr, 0.025)), float(np.quantile(mf_arr, 0.975))),
        D_um2_per_s_median=float(np.median(d_arr)) if d_arr.size else (np.nan if bleach_radius_um is not None else None),
        D_um2_per_s_ci95=(float(np.quantile(d_arr, 0.025)), float(np.quantile(d_arr, 0.975)))
        if d_arr.size
        else ((np.nan, np.nan) if bleach_radius_um is not None else None),
    )
    return summary.to_dict()


__all__ = ["bootstrap_frap_fit", "BootstrapSummary"]
