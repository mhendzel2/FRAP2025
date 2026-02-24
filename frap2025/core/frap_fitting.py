"""Physics-aware FRAP fitting models.

All diffusion conversions in this module use the explicit convention:
- Exponential rate constant ``k_s = 1 / tau_s`` for terms of the form ``exp(-k_s * t_s)``.
- Therefore ``D_um2_per_s = w_um**2 * k_s / 4 = w_um**2 / (4 * tau_s)``.
- If using half-time directly, include ``ln(2)``: ``D_um2_per_s = w_um**2 * ln(2) / (4 * t_half_s)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable
import logging
import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import ive

LOGGER = logging.getLogger(__name__)


class FitQualityWarning(UserWarning):
    """Warning raised when fit quality is poor or physically implausible."""


@dataclass
class FRAPFitResult:
    """Structured FRAP fit result with uncertainties and model metadata."""

    model: str
    mobile_fraction: float
    immobile_fraction: float
    tau_s: float
    tau_s_ci95: tuple[float, float]
    t_half_s: float
    D_um2_per_s: float | None
    D_um2_per_s_ci95: tuple[float, float] | None
    r_squared: float
    chi2_reduced: float
    aic: float
    bic: float
    n_components: int
    components: list[dict[str, Any]]
    normalization_mode: str
    bleach_radius_um: float | None
    pixel_size_um: float | None
    notes: list[str] = field(default_factory=list)
    mobile_fraction_ci95: tuple[float, float] = (np.nan, np.nan)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tau"] = self.tau_s
        payload["tau_ci95"] = self.tau_s_ci95
        payload["k_s"] = (1.0 / self.tau_s) if np.isfinite(self.tau_s) and self.tau_s > 0 else np.nan
        payload["k"] = payload["k_s"]
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.to_dict()

    def __iter__(self):
        return iter(self.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def keys(self) -> Iterable[str]:
        return self.to_dict().keys()

    def items(self) -> Iterable[tuple[str, Any]]:
        return self.to_dict().items()


def _as_clean_trace(time_s: np.ndarray | list[float], intensity: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
    t_s = np.asarray(time_s, dtype=float).reshape(-1)
    y = np.asarray(intensity, dtype=float).reshape(-1)
    if t_s.size != y.size:
        raise ValueError("time_s and intensity must have the same length.")
    finite_mask = np.isfinite(t_s) & np.isfinite(y)
    t_s = t_s[finite_mask]
    y = y[finite_mask]
    if t_s.size < 5:
        raise ValueError("At least 5 finite points are required for fitting.")
    order = np.argsort(t_s)
    return t_s[order], y[order]


def _single_exp_model(t_s: np.ndarray, mobile_fraction: float, tau_s: float, baseline: float) -> np.ndarray:
    return baseline + mobile_fraction * (1.0 - np.exp(-t_s / tau_s))


def _double_exp_model(
    t_s: np.ndarray,
    amp1: float,
    tau1_s: float,
    amp2: float,
    tau2_s: float,
    baseline: float,
) -> np.ndarray:
    return baseline + amp1 * (1.0 - np.exp(-t_s / tau1_s)) + amp2 * (1.0 - np.exp(-t_s / tau2_s))


def _triple_exp_model(
    t_s: np.ndarray,
    amp1: float,
    tau1_s: float,
    amp2: float,
    tau2_s: float,
    amp3: float,
    tau3_s: float,
    baseline: float,
) -> np.ndarray:
    return (
        baseline
        + amp1 * (1.0 - np.exp(-t_s / tau1_s))
        + amp2 * (1.0 - np.exp(-t_s / tau2_s))
        + amp3 * (1.0 - np.exp(-t_s / tau3_s))
    )


def _soumpasis_model(t_s: np.ndarray, tau_D_s: float, mobile_fraction: float, baseline: float) -> np.ndarray:
    t_safe_s = np.maximum(t_s, 1e-12)
    z = tau_D_s / (2.0 * t_safe_s)
    # ive(v, z) = exp(-z) * iv(v, z) for z>0, which avoids overflow in iv.
    kernel = ive(0, z) + ive(1, z)
    return baseline + mobile_fraction * kernel


def _reaction_diffusion_model(
    t_s: np.ndarray,
    mobile_fraction: float,
    k_on_s: float,
    k_off_s: float,
    k_D_s: float,
    baseline: float,
) -> np.ndarray:
    denom = np.maximum(k_on_s + k_off_s, 1e-12)
    binding_term = (k_off_s / denom) * np.exp(-denom * t_s)
    diffusion_term = (k_on_s / denom) * np.exp(-k_D_s * t_s)
    recovery = 1.0 - binding_term - diffusion_term
    return baseline + mobile_fraction * recovery


def _ci95_from_covariance(value: float, cov: np.ndarray | None, index: int) -> tuple[float, float]:
    if cov is None or index >= cov.shape[0]:
        return (np.nan, np.nan)
    var = cov[index, index]
    if not np.isfinite(var) or var < 0:
        return (np.nan, np.nan)
    delta = 1.96 * float(np.sqrt(var))
    return (float(value - delta), float(value + delta))


def _metrics(y_true: np.ndarray, y_fit: np.ndarray, n_params: int) -> tuple[float, float, float, float]:
    residual = y_true - y_fit
    rss = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r_squared = 1.0 - (rss / ss_tot) if ss_tot > 0 else np.nan
    dof = max(len(y_true) - n_params, 1)
    chi2_reduced = rss / dof
    if rss <= 0:
        return r_squared, chi2_reduced, -np.inf, -np.inf
    n = len(y_true)
    aic = n * np.log(rss / n) + 2 * n_params
    bic = n * np.log(rss / n) + n_params * np.log(n)
    return r_squared, chi2_reduced, float(aic), float(bic)


def _d_from_tau(
    tau_s: float,
    tau_s_ci95: tuple[float, float],
    bleach_radius_um: float | None,
) -> tuple[float | None, tuple[float, float] | None]:
    if bleach_radius_um is None or not np.isfinite(bleach_radius_um) or bleach_radius_um <= 0:
        return None, None
    if not np.isfinite(tau_s) or tau_s <= 0:
        return np.nan, (np.nan, np.nan)

    # k_s = 1/tau_s from exp(-k_s * t_s), so D_um2_per_s = w_um^2 * k_s / 4.
    d_um2_per_s = (bleach_radius_um**2) / (4.0 * tau_s)
    tau_low_s, tau_high_s = tau_s_ci95
    if tau_low_s > 0 and tau_high_s > 0:
        d_ci95 = ((bleach_radius_um**2) / (4.0 * tau_high_s), (bleach_radius_um**2) / (4.0 * tau_low_s))
    else:
        d_ci95 = (np.nan, np.nan)
    return float(d_um2_per_s), d_ci95


def _warn_if_poor_quality(
    *,
    success: bool,
    r_squared: float,
    tau_s: float,
    frame_interval_s: float,
    mobile_fraction: float,
) -> list[str]:
    notes: list[str] = []
    if not success:
        notes.append("Fit did not converge.")
    if np.isfinite(r_squared) and r_squared < 0.95:
        notes.append("R-squared < 0.95.")
    if np.isfinite(tau_s) and np.isfinite(frame_interval_s) and tau_s < frame_interval_s:
        notes.append("tau_s is shorter than frame interval.")
    if np.isfinite(mobile_fraction) and not (0.0 < mobile_fraction <= 1.0):
        notes.append("mobile_fraction is outside (0, 1].")

    for note in notes:
        warnings.warn(note, FitQualityWarning, stacklevel=2)
    return notes


def _failed_result(model: str, n_components: int, normalization_mode: str, bleach_radius_um: float | None, pixel_size_um: float | None) -> FRAPFitResult:
    notes = _warn_if_poor_quality(
        success=False,
        r_squared=np.nan,
        tau_s=np.nan,
        frame_interval_s=np.nan,
        mobile_fraction=np.nan,
    )
    return FRAPFitResult(
        model=model,
        mobile_fraction=np.nan,
        immobile_fraction=np.nan,
        tau_s=np.nan,
        tau_s_ci95=(np.nan, np.nan),
        t_half_s=np.nan,
        D_um2_per_s=np.nan if bleach_radius_um is not None else None,
        D_um2_per_s_ci95=(np.nan, np.nan) if bleach_radius_um is not None else None,
        r_squared=np.nan,
        chi2_reduced=np.inf,
        aic=np.inf,
        bic=np.inf,
        n_components=n_components,
        components=[],
        normalization_mode=normalization_mode,
        bleach_radius_um=bleach_radius_um,
        pixel_size_um=pixel_size_um,
        notes=notes,
        mobile_fraction_ci95=(np.nan, np.nan),
    )


def fit_single_exponential(
    time_s: np.ndarray | list[float],
    intensity: np.ndarray | list[float],
    *,
    bleach_radius_um: float | None = None,
    normalization_mode: str = "simple",
    pixel_size_um: float | None = None,
) -> FRAPFitResult:
    """Fit a single-component FRAP exponential model.

    Notes
    -----
    Model form: ``F(t) = baseline + mobile_fraction * (1 - exp(-t / tau_s))``.

    Diffusion conversion (Axelrod-style exponential approximation):
    ``D_um2_per_s = w_um**2 / (4 * tau_s)`` where ``w_um`` is bleach e^-2 radius.
    """
    t_s, y = _as_clean_trace(time_s, intensity)
    t_shift_s = t_s - float(np.min(t_s))
    positive_t_s = t_shift_s[t_shift_s > 0]
    frame_interval_s = float(np.min(np.diff(t_shift_s))) if t_shift_s.size > 1 else np.nan

    baseline0 = float(np.min(y))
    plateau0 = float(np.mean(y[-max(3, len(y) // 10) :]))
    mobile0 = max(plateau0 - baseline0, 1e-3)
    tau0_s = float(np.median(positive_t_s)) if positive_t_s.size else 1.0
    tau0_s = max(tau0_s, frame_interval_s if np.isfinite(frame_interval_s) else 1e-3)

    y_range = max(float(np.max(y) - np.min(y)), 1.0)
    bounds = ([0.0, 1e-9, baseline0 - 2.0 * y_range], [1.5, max(t_shift_s) * 100.0 + 1e-9, baseline0 + 2.0 * y_range])

    try:
        popt, pcov = curve_fit(
            _single_exp_model,
            t_shift_s,
            y,
            p0=[mobile0, tau0_s, baseline0],
            bounds=bounds,
            maxfev=50000,
        )
        success = True
    except Exception as exc:  # pragma: no cover - exercised by warning-path test
        LOGGER.warning("Single exponential fit failed: %s", exc)
        return _failed_result("single_exp", 1, normalization_mode, bleach_radius_um, pixel_size_um)

    mobile_fraction, tau_s, baseline = [float(v) for v in popt]
    y_fit = _single_exp_model(t_shift_s, mobile_fraction, tau_s, baseline)
    r_squared, chi2_reduced, aic, bic = _metrics(y, y_fit, n_params=3)

    tau_s_ci95 = _ci95_from_covariance(tau_s, pcov, 1)
    mobile_fraction_ci95 = _ci95_from_covariance(mobile_fraction, pcov, 0)
    t_half_s = float(tau_s * np.log(2.0)) if np.isfinite(tau_s) else np.nan
    d_um2_per_s, d_ci95 = _d_from_tau(tau_s, tau_s_ci95, bleach_radius_um)

    notes = _warn_if_poor_quality(
        success=success,
        r_squared=r_squared,
        tau_s=tau_s,
        frame_interval_s=frame_interval_s,
        mobile_fraction=mobile_fraction,
    )

    return FRAPFitResult(
        model="single_exp",
        mobile_fraction=mobile_fraction,
        immobile_fraction=1.0 - mobile_fraction if np.isfinite(mobile_fraction) else np.nan,
        tau_s=tau_s,
        tau_s_ci95=tau_s_ci95,
        t_half_s=t_half_s,
        D_um2_per_s=d_um2_per_s,
        D_um2_per_s_ci95=d_ci95,
        r_squared=r_squared,
        chi2_reduced=chi2_reduced,
        aic=aic,
        bic=bic,
        n_components=1,
        components=[
            {
                "baseline": baseline,
                "amplitude": mobile_fraction,
                "amplitude_ci95": mobile_fraction_ci95,
                "tau_s": tau_s,
                "tau_s_ci95": tau_s_ci95,
                "k_s": (1.0 / tau_s) if tau_s > 0 else np.nan,
                "t_half_s": t_half_s,
                "D_um2_per_s": d_um2_per_s,
                "D_um2_per_s_ci95": d_ci95,
            }
        ],
        normalization_mode=normalization_mode,
        bleach_radius_um=bleach_radius_um,
        pixel_size_um=pixel_size_um,
        notes=notes,
        mobile_fraction_ci95=mobile_fraction_ci95,
    )


def fit_double_exponential(
    time_s: np.ndarray | list[float],
    intensity: np.ndarray | list[float],
    *,
    bleach_radius_um: float | None = None,
    normalization_mode: str = "simple",
    pixel_size_um: float | None = None,
) -> FRAPFitResult:
    """Fit a two-component exponential FRAP model.

    Notes
    -----
    Model form:
    ``F(t) = baseline + A1*(1-exp(-t/tau1_s)) + A2*(1-exp(-t/tau2_s))``.

    Component diffusion conversion:
    ``D1_um2_per_s = w_um**2/(4*tau1_s)``, ``D2_um2_per_s = w_um**2/(4*tau2_s)``.
    """
    t_s, y = _as_clean_trace(time_s, intensity)
    t_shift_s = t_s - float(np.min(t_s))
    positive_t_s = t_shift_s[t_shift_s > 0]
    frame_interval_s = float(np.min(np.diff(t_shift_s))) if t_shift_s.size > 1 else np.nan

    baseline0 = float(np.min(y))
    plateau0 = float(np.mean(y[-max(3, len(y) // 10) :]))
    total_amp0 = max(plateau0 - baseline0, 1e-3)

    if positive_t_s.size:
        tau_fast0_s = max(float(np.quantile(positive_t_s, 0.1)), frame_interval_s if np.isfinite(frame_interval_s) else 1e-3)
        tau_slow0_s = max(float(np.quantile(positive_t_s, 0.8)), tau_fast0_s * 5.0)
    else:
        tau_fast0_s, tau_slow0_s = 1.0, 10.0

    y_range = max(float(np.max(y) - np.min(y)), 1.0)
    bounds = (
        [0.0, 1e-9, 0.0, 1e-9, baseline0 - 2.0 * y_range],
        [1.5, max(t_shift_s) * 200.0 + 1e-9, 1.5, max(t_shift_s) * 200.0 + 1e-9, baseline0 + 2.0 * y_range],
    )

    try:
        popt, pcov = curve_fit(
            _double_exp_model,
            t_shift_s,
            y,
            p0=[0.6 * total_amp0, tau_fast0_s, 0.4 * total_amp0, tau_slow0_s, baseline0],
            bounds=bounds,
            maxfev=80000,
        )
        success = True
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Double exponential fit failed: %s", exc)
        return _failed_result("double_exp", 2, normalization_mode, bleach_radius_um, pixel_size_um)

    amp1, tau1_s, amp2, tau2_s, baseline = [float(v) for v in popt]
    y_fit = _double_exp_model(t_shift_s, amp1, tau1_s, amp2, tau2_s, baseline)
    r_squared, chi2_reduced, aic, bic = _metrics(y, y_fit, n_params=5)

    tau1_ci95 = _ci95_from_covariance(tau1_s, pcov, 1)
    tau2_ci95 = _ci95_from_covariance(tau2_s, pcov, 3)
    amp1_ci95 = _ci95_from_covariance(amp1, pcov, 0)
    amp2_ci95 = _ci95_from_covariance(amp2, pcov, 2)

    mobile_fraction = amp1 + amp2
    if pcov is not None and pcov.shape[0] >= 3:
        mobile_var = pcov[0, 0] + pcov[2, 2] + 2.0 * pcov[0, 2]
        if np.isfinite(mobile_var) and mobile_var >= 0:
            mobile_delta = 1.96 * np.sqrt(mobile_var)
            mobile_fraction_ci95 = (float(mobile_fraction - mobile_delta), float(mobile_fraction + mobile_delta))
        else:
            mobile_fraction_ci95 = (np.nan, np.nan)
    else:
        mobile_fraction_ci95 = (np.nan, np.nan)

    component_rows: list[dict[str, Any]] = []
    for amp, amp_ci95, tau_s, tau_ci95 in [
        (amp1, amp1_ci95, tau1_s, tau1_ci95),
        (amp2, amp2_ci95, tau2_s, tau2_ci95),
    ]:
        d_um2_per_s, d_ci95 = _d_from_tau(tau_s, tau_ci95, bleach_radius_um)
        component_rows.append(
            {
                "baseline": baseline,
                "amplitude": amp,
                "amplitude_ci95": amp_ci95,
                "tau_s": tau_s,
                "tau_s_ci95": tau_ci95,
                "k_s": (1.0 / tau_s) if tau_s > 0 else np.nan,
                "t_half_s": tau_s * np.log(2.0) if np.isfinite(tau_s) else np.nan,
                "D_um2_per_s": d_um2_per_s,
                "D_um2_per_s_ci95": d_ci95,
            }
        )
    component_rows.sort(key=lambda row: row["tau_s"])

    dominant = component_rows[0]
    tau_s = float(dominant["tau_s"])
    tau_s_ci95 = dominant["tau_s_ci95"]
    t_half_s = tau_s * np.log(2.0) if np.isfinite(tau_s) else np.nan
    d_um2_per_s = dominant.get("D_um2_per_s")
    d_ci95 = dominant.get("D_um2_per_s_ci95")

    notes = _warn_if_poor_quality(
        success=success,
        r_squared=r_squared,
        tau_s=tau_s,
        frame_interval_s=frame_interval_s,
        mobile_fraction=mobile_fraction,
    )

    return FRAPFitResult(
        model="double_exp",
        mobile_fraction=mobile_fraction,
        immobile_fraction=1.0 - mobile_fraction if np.isfinite(mobile_fraction) else np.nan,
        tau_s=tau_s,
        tau_s_ci95=tau_s_ci95,
        t_half_s=float(t_half_s),
        D_um2_per_s=d_um2_per_s,
        D_um2_per_s_ci95=d_ci95,
        r_squared=r_squared,
        chi2_reduced=chi2_reduced,
        aic=aic,
        bic=bic,
        n_components=2,
        components=component_rows,
        normalization_mode=normalization_mode,
        bleach_radius_um=bleach_radius_um,
        pixel_size_um=pixel_size_um,
        notes=notes,
        mobile_fraction_ci95=mobile_fraction_ci95,
    )


def fit_triple_exponential(
    time_s: np.ndarray | list[float],
    intensity: np.ndarray | list[float],
    *,
    bleach_radius_um: float | None = None,
    normalization_mode: str = "simple",
    pixel_size_um: float | None = None,
) -> FRAPFitResult:
    """Fit a three-component exponential FRAP model."""
    t_s, y = _as_clean_trace(time_s, intensity)
    t_shift_s = t_s - float(np.min(t_s))
    positive_t_s = t_shift_s[t_shift_s > 0]
    frame_interval_s = float(np.min(np.diff(t_shift_s))) if t_shift_s.size > 1 else np.nan

    baseline0 = float(np.min(y))
    plateau0 = float(np.mean(y[-max(3, len(y) // 10) :]))
    total_amp0 = max(plateau0 - baseline0, 1e-3)

    if positive_t_s.size:
        tau1_0_s = max(float(np.quantile(positive_t_s, 0.05)), frame_interval_s if np.isfinite(frame_interval_s) else 1e-3)
        tau2_0_s = max(float(np.quantile(positive_t_s, 0.45)), tau1_0_s * 3.0)
        tau3_0_s = max(float(np.quantile(positive_t_s, 0.85)), tau2_0_s * 3.0)
    else:
        tau1_0_s, tau2_0_s, tau3_0_s = 1.0, 8.0, 30.0

    y_range = max(float(np.max(y) - np.min(y)), 1.0)
    bounds = (
        [0.0, 1e-9, 0.0, 1e-9, 0.0, 1e-9, baseline0 - 2.0 * y_range],
        [1.5, max(t_shift_s) * 200.0 + 1e-9, 1.5, max(t_shift_s) * 200.0 + 1e-9, 1.5, max(t_shift_s) * 200.0 + 1e-9, baseline0 + 2.0 * y_range],
    )

    try:
        popt, pcov = curve_fit(
            _triple_exp_model,
            t_shift_s,
            y,
            p0=[0.4 * total_amp0, tau1_0_s, 0.35 * total_amp0, tau2_0_s, 0.25 * total_amp0, tau3_0_s, baseline0],
            bounds=bounds,
            maxfev=120000,
        )
        success = True
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Triple exponential fit failed: %s", exc)
        return _failed_result("triple_exp", 3, normalization_mode, bleach_radius_um, pixel_size_um)

    amp1, tau1_s, amp2, tau2_s, amp3, tau3_s, baseline = [float(v) for v in popt]
    y_fit = _triple_exp_model(t_shift_s, amp1, tau1_s, amp2, tau2_s, amp3, tau3_s, baseline)
    r_squared, chi2_reduced, aic, bic = _metrics(y, y_fit, n_params=7)

    components: list[dict[str, Any]] = []
    for amp, amp_idx, tau_s, tau_idx in [(amp1, 0, tau1_s, 1), (amp2, 2, tau2_s, 3), (amp3, 4, tau3_s, 5)]:
        amp_ci95 = _ci95_from_covariance(amp, pcov, amp_idx)
        tau_ci95 = _ci95_from_covariance(tau_s, pcov, tau_idx)
        d_um2_per_s, d_ci95 = _d_from_tau(tau_s, tau_ci95, bleach_radius_um)
        components.append(
            {
                "baseline": baseline,
                "amplitude": amp,
                "amplitude_ci95": amp_ci95,
                "tau_s": tau_s,
                "tau_s_ci95": tau_ci95,
                "k_s": (1.0 / tau_s) if tau_s > 0 else np.nan,
                "t_half_s": tau_s * np.log(2.0) if np.isfinite(tau_s) else np.nan,
                "D_um2_per_s": d_um2_per_s,
                "D_um2_per_s_ci95": d_ci95,
            }
        )
    components.sort(key=lambda row: row["tau_s"])

    mobile_fraction = amp1 + amp2 + amp3
    if pcov is not None and pcov.shape[0] >= 5:
        mobile_var = (
            pcov[0, 0]
            + pcov[2, 2]
            + pcov[4, 4]
            + 2.0 * (pcov[0, 2] + pcov[0, 4] + pcov[2, 4])
        )
        if np.isfinite(mobile_var) and mobile_var >= 0:
            mobile_delta = 1.96 * np.sqrt(mobile_var)
            mobile_fraction_ci95 = (float(mobile_fraction - mobile_delta), float(mobile_fraction + mobile_delta))
        else:
            mobile_fraction_ci95 = (np.nan, np.nan)
    else:
        mobile_fraction_ci95 = (np.nan, np.nan)

    dominant = components[0]
    tau_s = float(dominant["tau_s"])
    tau_s_ci95 = dominant["tau_s_ci95"]
    t_half_s = tau_s * np.log(2.0) if np.isfinite(tau_s) else np.nan

    notes = _warn_if_poor_quality(
        success=success,
        r_squared=r_squared,
        tau_s=tau_s,
        frame_interval_s=frame_interval_s,
        mobile_fraction=mobile_fraction,
    )

    return FRAPFitResult(
        model="triple_exp",
        mobile_fraction=mobile_fraction,
        immobile_fraction=1.0 - mobile_fraction if np.isfinite(mobile_fraction) else np.nan,
        tau_s=tau_s,
        tau_s_ci95=tau_s_ci95,
        t_half_s=float(t_half_s),
        D_um2_per_s=dominant.get("D_um2_per_s"),
        D_um2_per_s_ci95=dominant.get("D_um2_per_s_ci95"),
        r_squared=r_squared,
        chi2_reduced=chi2_reduced,
        aic=aic,
        bic=bic,
        n_components=3,
        components=components,
        normalization_mode=normalization_mode,
        bleach_radius_um=bleach_radius_um,
        pixel_size_um=pixel_size_um,
        notes=notes,
        mobile_fraction_ci95=mobile_fraction_ci95,
    )


def fit_soumpasis(
    time_s: np.ndarray | list[float],
    intensity: np.ndarray | list[float],
    *,
    w_um: float,
    normalization_mode: str = "simple",
    pixel_size_um: float | None = None,
) -> FRAPFitResult:
    """Fit Soumpasis (1983) exact circular diffusion model.

    References
    ----------
    Soumpasis, D. M. (1983). Theoretical analysis of fluorescence photobleaching recovery experiments.
    *Biophysical Journal*, 41(1), 95-97. DOI: 10.1016/S0006-3495(83)84410-5.

    Notes
    -----
    Parameterization here fits ``tau_D_s`` in
    ``F(t)=baseline + mobile_fraction*exp(-tau_D_s/(2t))*(I0(z)+I1(z))`` where ``z=tau_D_s/(2t)``.

    ``tau_D_s = w_um**2/(4*D_um2_per_s)``, so ``D_um2_per_s = w_um**2/(4*tau_D_s)``.
    Equivalent half-time relation (for circular Gaussian bleach) is
    ``D_um2_per_s ~= 0.2240 * w_um**2 / t_half_s``.
    """
    if not np.isfinite(w_um) or w_um <= 0:
        raise ValueError("w_um must be positive.")

    t_s, y = _as_clean_trace(time_s, intensity)
    t_fit_s = np.asarray(t_s, dtype=float)
    min_positive_s = float(np.min(t_fit_s[t_fit_s > 0])) if np.any(t_fit_s > 0) else 1e-6
    t_fit_s = np.where(t_fit_s <= 0, min_positive_s * 0.5, t_fit_s)
    frame_interval_s = float(np.min(np.diff(np.sort(t_fit_s)))) if t_fit_s.size > 1 else np.nan

    baseline0 = float(np.min(y))
    mobile0 = max(float(np.max(y) - baseline0), 0.2)
    tau_D0_s = max(float(np.median(t_fit_s)), frame_interval_s if np.isfinite(frame_interval_s) else 1e-3)

    y_range = max(float(np.max(y) - np.min(y)), 1.0)
    bounds = ([1e-9, 0.0, baseline0 - 2.0 * y_range], [max(t_fit_s) * 300.0 + 1e-9, 1.5, baseline0 + 2.0 * y_range])

    try:
        popt, pcov = curve_fit(
            _soumpasis_model,
            t_fit_s,
            y,
            p0=[tau_D0_s, mobile0, baseline0],
            bounds=bounds,
            maxfev=120000,
        )
        success = True
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Soumpasis fit failed: %s", exc)
        return _failed_result("soumpasis", 1, normalization_mode, w_um, pixel_size_um)

    tau_D_s, mobile_fraction, baseline = [float(v) for v in popt]
    y_fit = _soumpasis_model(t_fit_s, tau_D_s, mobile_fraction, baseline)
    r_squared, chi2_reduced, aic, bic = _metrics(y, y_fit, n_params=3)

    tau_s_ci95 = _ci95_from_covariance(tau_D_s, pcov, 0)
    mobile_fraction_ci95 = _ci95_from_covariance(mobile_fraction, pcov, 1)

    d_um2_per_s, d_ci95 = _d_from_tau(tau_D_s, tau_s_ci95, w_um)

    # Numerical half-time from model trajectory (Soumpasis t_half does not map to tau_D via ln(2)).
    t_grid_s = np.geomspace(max(float(np.min(t_fit_s)), 1e-6), max(float(np.max(t_fit_s)) * 50.0, 1e-4), 25000)
    y_grid = _soumpasis_model(t_grid_s, tau_D_s, mobile_fraction, baseline)
    half_target = baseline + 0.5 * mobile_fraction
    t_half_s = float(t_grid_s[int(np.argmin(np.abs(y_grid - half_target)))])

    notes = _warn_if_poor_quality(
        success=success,
        r_squared=r_squared,
        tau_s=tau_D_s,
        frame_interval_s=frame_interval_s,
        mobile_fraction=mobile_fraction,
    )
    notes.append("Soumpasis relation: D_um2_per_s = w_um^2/(4*tau_D_s) ~= 0.2240*w_um^2/t_half_s.")

    return FRAPFitResult(
        model="soumpasis",
        mobile_fraction=mobile_fraction,
        immobile_fraction=1.0 - mobile_fraction if np.isfinite(mobile_fraction) else np.nan,
        tau_s=tau_D_s,
        tau_s_ci95=tau_s_ci95,
        t_half_s=t_half_s,
        D_um2_per_s=d_um2_per_s,
        D_um2_per_s_ci95=d_ci95,
        r_squared=r_squared,
        chi2_reduced=chi2_reduced,
        aic=aic,
        bic=bic,
        n_components=1,
        components=[
            {
                "baseline": baseline,
                "amplitude": mobile_fraction,
                "amplitude_ci95": mobile_fraction_ci95,
                "tau_s": tau_D_s,
                "tau_s_ci95": tau_s_ci95,
                "k_s": (1.0 / tau_D_s) if tau_D_s > 0 else np.nan,
                "t_half_s": t_half_s,
                "D_um2_per_s": d_um2_per_s,
                "D_um2_per_s_ci95": d_ci95,
            }
        ],
        normalization_mode=normalization_mode,
        bleach_radius_um=w_um,
        pixel_size_um=pixel_size_um,
        notes=notes,
        mobile_fraction_ci95=mobile_fraction_ci95,
    )


def fit_reaction_diffusion(
    time_s: np.ndarray | list[float],
    intensity: np.ndarray | list[float],
    *,
    bleach_radius_um: float | None = None,
    normalization_mode: str = "simple",
    pixel_size_um: float | None = None,
) -> FRAPFitResult:
    """Fit a simplified reaction-diffusion FRAP model.

    References
    ----------
    Sprague et al. (2004), Biophys J 86:3473-3495. DOI: 10.1529/biophysj.103.026765.

    Notes
    -----
    Model:
    ``F(t)=baseline + mobile_fraction * [1 - (k_off/(k_on+k_off))*exp(-(k_on+k_off)t)
           - (k_on/(k_on+k_off))*exp(-k_D*t)]``

    Here ``k_D_s`` is treated as the diffusion-like inverse timescale (1/s). If a bleach radius
    is supplied, an effective diffusion estimate uses ``D_um2_per_s = w_um**2 * k_D_s / 4``.
    """
    t_s, y = _as_clean_trace(time_s, intensity)
    t_shift_s = t_s - float(np.min(t_s))
    frame_interval_s = float(np.min(np.diff(t_shift_s))) if t_shift_s.size > 1 else np.nan

    baseline0 = float(np.min(y))
    plateau0 = float(np.mean(y[-max(3, len(y) // 10) :]))
    mobile0 = max(plateau0 - baseline0, 1e-3)

    y_range = max(float(np.max(y) - np.min(y)), 1.0)
    bounds = (
        [0.0, 1e-6, 1e-6, 1e-6, baseline0 - 2.0 * y_range],
        [1.5, 200.0, 200.0, 200.0, baseline0 + 2.0 * y_range],
    )

    try:
        popt, pcov = curve_fit(
            _reaction_diffusion_model,
            t_shift_s,
            y,
            p0=[mobile0, 0.2, 0.2, 0.5, baseline0],
            bounds=bounds,
            maxfev=120000,
        )
        success = True
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Reaction-diffusion fit failed: %s", exc)
        return _failed_result("reaction_diffusion", 3, normalization_mode, bleach_radius_um, pixel_size_um)

    mobile_fraction, k_on_s, k_off_s, k_D_s, baseline = [float(v) for v in popt]
    y_fit = _reaction_diffusion_model(t_shift_s, mobile_fraction, k_on_s, k_off_s, k_D_s, baseline)
    r_squared, chi2_reduced, aic, bic = _metrics(y, y_fit, n_params=5)

    k_D_ci95 = _ci95_from_covariance(k_D_s, pcov, 3)
    mobile_fraction_ci95 = _ci95_from_covariance(mobile_fraction, pcov, 0)

    tau_s = (1.0 / k_D_s) if k_D_s > 0 else np.nan
    tau_s_ci95 = (
        (1.0 / k_D_ci95[1], 1.0 / k_D_ci95[0]) if k_D_ci95[0] > 0 and k_D_ci95[1] > 0 else (np.nan, np.nan)
    )
    t_half_s = tau_s * np.log(2.0) if np.isfinite(tau_s) else np.nan

    if bleach_radius_um is not None and np.isfinite(bleach_radius_um) and bleach_radius_um > 0 and k_D_s > 0:
        d_um2_per_s = (bleach_radius_um**2) * k_D_s / 4.0
        if k_D_ci95[0] > 0 and k_D_ci95[1] > 0:
            d_ci95 = ((bleach_radius_um**2) * k_D_ci95[0] / 4.0, (bleach_radius_um**2) * k_D_ci95[1] / 4.0)
        else:
            d_ci95 = (np.nan, np.nan)
    else:
        d_um2_per_s = None
        d_ci95 = None

    notes = _warn_if_poor_quality(
        success=success,
        r_squared=r_squared,
        tau_s=tau_s,
        frame_interval_s=frame_interval_s,
        mobile_fraction=mobile_fraction,
    )

    return FRAPFitResult(
        model="reaction_diffusion",
        mobile_fraction=mobile_fraction,
        immobile_fraction=1.0 - mobile_fraction if np.isfinite(mobile_fraction) else np.nan,
        tau_s=tau_s,
        tau_s_ci95=tau_s_ci95,
        t_half_s=t_half_s,
        D_um2_per_s=d_um2_per_s,
        D_um2_per_s_ci95=d_ci95,
        r_squared=r_squared,
        chi2_reduced=chi2_reduced,
        aic=aic,
        bic=bic,
        n_components=3,
        components=[
            {
                "baseline": baseline,
                "k_on_s": k_on_s,
                "k_on_s_ci95": _ci95_from_covariance(k_on_s, pcov, 1),
                "k_off_s": k_off_s,
                "k_off_s_ci95": _ci95_from_covariance(k_off_s, pcov, 2),
                "k_D_s": k_D_s,
                "k_D_s_ci95": k_D_ci95,
                "tau_s": tau_s,
                "tau_s_ci95": tau_s_ci95,
                "D_um2_per_s": d_um2_per_s,
                "D_um2_per_s_ci95": d_ci95,
            }
        ],
        normalization_mode=normalization_mode,
        bleach_radius_um=bleach_radius_um,
        pixel_size_um=pixel_size_um,
        notes=notes,
        mobile_fraction_ci95=mobile_fraction_ci95,
    )


__all__ = [
    "FRAPFitResult",
    "FitQualityWarning",
    "fit_single_exponential",
    "fit_double_exponential",
    "fit_triple_exponential",
    "fit_soumpasis",
    "fit_reaction_diffusion",
]
