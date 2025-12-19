"""Special FRAP models for advanced / non-batch workflows.

These routines are intentionally *not* wired into automatic batch processing.
They are designed for targeted experiments where you explicitly opt-in.

Included:
- Global multi-spot fitting (shared parameters across bleach radii)
- Effective diffusion (fast-exchange) approximation
- CTRW-style anomalous subdiffusion (Mittag–Leffler) with geometry-aware baseline
- Pixel-by-pixel parameter mapping (fast, non-iterative)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize

from frap_pde_solver import (
    CoupledPDEParameters,
    CoupledPDESolver,
    FiniteDifferenceSolver,
    GeometryExtractor,
    PDEParameters,
)


def _aic(n: int, rss: float, k: int) -> float:
    n = int(n)
    k = int(k)
    rss = float(rss)
    if n <= 0:
        return float('nan')
    return n * float(np.log(max(rss, 1e-30) / n)) + 2.0 * k


def _summarize_t_half(time: np.ndarray, curve: np.ndarray) -> float:
    """Return approximate half-time for a monotone-ish recovery curve."""
    time = np.asarray(time, dtype=float)
    curve = np.asarray(curve, dtype=float)
    if len(time) == 0 or len(curve) == 0 or len(time) != len(curve):
        return float('nan')

    y0 = float(curve[0])
    yinf = float(np.mean(curve[-min(len(curve), 5):]))
    target = y0 + 0.5 * (yinf - y0)
    idx = np.where(curve >= target)[0]
    return float(time[idx[0]]) if len(idx) else float('nan')


@dataclass(frozen=True)
class MultiSpotDataset:
    """One FRAP acquisition for global multi-spot fitting."""

    image_stack: np.ndarray
    time_points: np.ndarray
    bleach_frame: int
    pixel_size: float = 1.0
    bleach_threshold: float = 0.3
    cell_threshold: Optional[float] = None


def _prepare_dataset(ds: MultiSpotDataset) -> Dict[str, Any]:
    """Extract masks and normalized ROI recovery data for a dataset."""

    extractor = GeometryExtractor()

    stack = np.asarray(ds.image_stack)
    if stack.ndim != 3:
        raise ValueError("image_stack must be a 3D array: (t, y, x)")

    time_points = np.asarray(ds.time_points, dtype=float)
    if time_points.ndim != 1 or len(time_points) != stack.shape[0]:
        raise ValueError("time_points must be 1D and match image_stack length")

    bleach_frame = int(ds.bleach_frame)
    if bleach_frame <= 0 or bleach_frame >= stack.shape[0] - 1:
        raise ValueError("bleach_frame must be within stack range")

    pre = np.mean(stack[:bleach_frame], axis=0)
    post = stack[bleach_frame]

    cell_mask = extractor.extract_cell_mask(pre, threshold=ds.cell_threshold)
    bleach_mask = extractor.extract_bleach_roi(pre, post, intensity_drop_threshold=float(ds.bleach_threshold))
    bleach_mask = bleach_mask & cell_mask
    if np.sum(bleach_mask) == 0:
        raise ValueError("No bleach region detected within cell mask")

    # ROI recovery curve in bleach mask
    roi = np.array([np.mean(frame[bleach_mask]) for frame in stack[bleach_frame:]], dtype=float)
    pre_int = float(np.mean(pre[bleach_mask]))
    if pre_int <= 0:
        raise ValueError("Invalid pre-bleach intensity for normalization")

    y = roi / pre_int
    bleach_depth = float(y[0])

    t = time_points[bleach_frame:] - float(time_points[bleach_frame])

    area_px = float(np.sum(bleach_mask))
    radius_um = np.sqrt(area_px / np.pi) * float(ds.pixel_size)

    return {
        "cell_mask": cell_mask,
        "bleach_mask": bleach_mask,
        "t": np.asarray(t, dtype=float),
        "y": np.asarray(y, dtype=float),
        "bleach_depth": bleach_depth,
        "bleach_radius_um": radius_um,
        "pixel_size": float(ds.pixel_size),
    }


def global_multi_spot_fit(
    datasets: Sequence[MultiSpotDataset],
    *,
    model: str = "compare",
    initial: Optional[Sequence[float]] = None,
    method: str = "Nelder-Mead",
) -> Dict[str, Any]:
    """Global objective fit across multiple bleach spot sizes.

    Physics leveraged:
    - Diffusion: characteristic times scale with spot size (approximately \\propto w^2)
    - Binding: characteristic times less sensitive to spot size (depends on k_off)

    Parameters
    ----------
    datasets:
        A list of acquisitions with different bleach radii.
    model (5 options):
        - 'diffusion_only': shared (D) using original scalar PDE
        - 'reaction_diffusion': shared (D, k_on, k_off) using original scalar PDE
        - 'reaction_diffusion_immobile': shared (D, k_on, k_off, immobile_fraction) using original scalar PDE
        - 'fast_exchange_plus_specific': shared (D_eff, k_on, k_off) using coupled solver (reduced vs 2-binding)
        - 'two_binding': shared (D, k_on1, k_off1, k_on2, k_off2) using coupled solver
    initial:
        Optional initial guess vector.
    method:
        Passed to scipy.optimize.minimize.

    Returns
    -------
    Dict with fitted params and per-dataset fit curves.

    Notes
    -----
    This is intentionally not called by the batch pipeline.
    """

    if len(datasets) < 2:
        raise ValueError("Provide at least 2 datasets (different spot sizes) for global fitting")

    model = str(model)
    if model in {"compare", "all"}:
        return global_multi_spot_compare_models(datasets, method=method)

    prepared = [_prepare_dataset(ds) for ds in datasets]

    scalar_solvers = [
        FiniteDifferenceSolver(p["cell_mask"], p["bleach_mask"], pixel_size=p["pixel_size"])
        for p in prepared
    ]
    coupled_solvers = [
        CoupledPDESolver(p["cell_mask"], p["bleach_mask"], pixel_size=p["pixel_size"])
        for p in prepared
    ]

    valid = {
        "diffusion_only",
        "reaction_diffusion",
        "reaction_diffusion_immobile",
        "fast_exchange_plus_specific",
        "two_binding",
    }
    if model not in valid:
        raise ValueError(f"model must be one of: {sorted(valid)}")

    # Define objective per model
    if model == "diffusion_only":
        x0 = [1.0] if initial is None else list(initial)
        k_params = 1

        def obj(x: Sequence[float]) -> float:
            D = float(x[0])
            if D <= 0:
                return 1e12
            rss = 0.0
            for p, solver in zip(prepared, scalar_solvers, strict=False):
                sim = solver.simulate(
                    PDEParameters(D=D, k_on=0.0, k_off=0.0, immobile_fraction=0.0),
                    p["t"],
                    bleach_depth=float(p["bleach_depth"]),
                )
                rss += float(np.sum((p["y"] - np.asarray(sim["recovery_curve"])) ** 2))
            return rss

    elif model == "reaction_diffusion":
        x0 = [1.0, 0.1, 0.1] if initial is None else list(initial)
        k_params = 3

        def obj(x: Sequence[float]) -> float:
            D, k_on, k_off = map(float, x)
            if D <= 0 or k_on < 0 or k_off < 0:
                return 1e12
            rss = 0.0
            for p, solver in zip(prepared, scalar_solvers, strict=False):
                sim = solver.simulate(
                    PDEParameters(D=D, k_on=k_on, k_off=k_off, immobile_fraction=0.0),
                    p["t"],
                    bleach_depth=float(p["bleach_depth"]),
                )
                rss += float(np.sum((p["y"] - np.asarray(sim["recovery_curve"])) ** 2))
            return rss

    elif model == "reaction_diffusion_immobile":
        x0 = [1.0, 0.1, 0.1, 0.2] if initial is None else list(initial)
        k_params = 4

        def obj(x: Sequence[float]) -> float:
            D, k_on, k_off, immobile = map(float, x)
            if D <= 0 or k_on < 0 or k_off < 0 or not (0.0 <= immobile <= 0.95):
                return 1e12
            rss = 0.0
            for p, solver in zip(prepared, scalar_solvers, strict=False):
                sim = solver.simulate(
                    PDEParameters(D=D, k_on=k_on, k_off=k_off, immobile_fraction=immobile),
                    p["t"],
                    bleach_depth=float(p["bleach_depth"]),
                )
                rss += float(np.sum((p["y"] - np.asarray(sim["recovery_curve"])) ** 2))
            return rss

    elif model == "fast_exchange_plus_specific":
        x0 = [1.0, 0.1, 0.1] if initial is None else list(initial)
        k_params = 3

        def obj(x: Sequence[float]) -> float:
            D_eff, k_on, k_off = map(float, x)
            if D_eff <= 0 or k_on < 0 or k_off < 0:
                return 1e12
            rss = 0.0
            for p, solver in zip(prepared, coupled_solvers, strict=False):
                sim = solver.simulate(
                    CoupledPDEParameters(D=D_eff, k_on1=k_on, k_off1=k_off, k_on2=0.0, k_off2=0.0),
                    p["t"],
                    bleach_depth=float(p["bleach_depth"]),
                )
                rss += float(np.sum((p["y"] - np.asarray(sim["recovery"])) ** 2))
            return rss

    else:  # two_binding
        x0 = [1.0, 0.1, 0.1, 0.05, 0.05] if initial is None else list(initial)
        k_params = 5

        def obj(x: Sequence[float]) -> float:
            D, k1on, k1off, k2on, k2off = map(float, x)
            if D <= 0 or any(v < 0 for v in [k1on, k1off, k2on, k2off]):
                return 1e12
            rss = 0.0
            for p, solver in zip(prepared, coupled_solvers, strict=False):
                sim = solver.simulate(
                    CoupledPDEParameters(D=D, k_on1=k1on, k_off1=k1off, k_on2=k2on, k_off2=k2off),
                    p["t"],
                    bleach_depth=float(p["bleach_depth"]),
                )
                rss += float(np.sum((p["y"] - np.asarray(sim["recovery"])) ** 2))
            return rss

    result = optimize.minimize(obj, x0, method=method)
    x_hat = np.asarray(result.x, dtype=float)

    # Build per-dataset predictions
    per: List[Dict[str, Any]] = []
    for p, solver_scalar, solver_coupled in zip(prepared, scalar_solvers, coupled_solvers, strict=False):
        if model == "diffusion_only":
            sim = solver_scalar.simulate(
                PDEParameters(D=float(x_hat[0]), k_on=0.0, k_off=0.0, immobile_fraction=0.0),
                p["t"],
                bleach_depth=float(p["bleach_depth"]),
            )
            fit = np.asarray(sim["recovery_curve"], dtype=float)
        elif model == "reaction_diffusion":
            sim = solver_scalar.simulate(
                PDEParameters(D=float(x_hat[0]), k_on=float(x_hat[1]), k_off=float(x_hat[2]), immobile_fraction=0.0),
                p["t"],
                bleach_depth=float(p["bleach_depth"]),
            )
            fit = np.asarray(sim["recovery_curve"], dtype=float)
        elif model == "reaction_diffusion_immobile":
            sim = solver_scalar.simulate(
                PDEParameters(D=float(x_hat[0]), k_on=float(x_hat[1]), k_off=float(x_hat[2]), immobile_fraction=float(x_hat[3])),
                p["t"],
                bleach_depth=float(p["bleach_depth"]),
            )
            fit = np.asarray(sim["recovery_curve"], dtype=float)
        elif model == "fast_exchange_plus_specific":
            sim = solver_coupled.simulate(
                CoupledPDEParameters(D=float(x_hat[0]), k_on1=float(x_hat[1]), k_off1=float(x_hat[2]), k_on2=0.0, k_off2=0.0),
                p["t"],
                bleach_depth=float(p["bleach_depth"]),
            )
            fit = np.asarray(sim["recovery"], dtype=float)
        else:
            sim = solver_coupled.simulate(
                CoupledPDEParameters(D=float(x_hat[0]), k_on1=float(x_hat[1]), k_off1=float(x_hat[2]), k_on2=float(x_hat[3]), k_off2=float(x_hat[4])),
                p["t"],
                bleach_depth=float(p["bleach_depth"]),
            )
            fit = np.asarray(sim["recovery"], dtype=float)

        per.append(
            {
                "bleach_radius_um": p["bleach_radius_um"],
                "time": p["t"],
                "data": p["y"],
                "fit": fit,
                "t_half": _summarize_t_half(p["t"], fit),
            }
        )

    n_total = int(sum(len(p["y"]) for p in prepared))
    rss = float(result.fun)

    scaling = [
        {
            "bleach_radius_um": float(item["bleach_radius_um"]),
            "bleach_radius_um2": float(item["bleach_radius_um"]) ** 2,
            "t_half": float(item["t_half"]),
        }
        for item in per
    ]

    out: Dict[str, Any] = {
        "model": model,
        "success": bool(result.success),
        "message": str(result.message),
        "rss": rss,
        "n_points": n_total,
        "aic": _aic(n_total, rss, k_params),
        "datasets": per,
        "scaling": scaling,
    }

    if model == "diffusion_only":
        out["params"] = {"D": float(x_hat[0])}
    elif model == "reaction_diffusion":
        out["params"] = {"D": float(x_hat[0]), "k_on": float(x_hat[1]), "k_off": float(x_hat[2])}
    elif model == "reaction_diffusion_immobile":
        out["params"] = {
            "D": float(x_hat[0]),
            "k_on": float(x_hat[1]),
            "k_off": float(x_hat[2]),
            "immobile_fraction": float(x_hat[3]),
        }
    elif model == "fast_exchange_plus_specific":
        out["params"] = {"D_eff": float(x_hat[0]), "k_on": float(x_hat[1]), "k_off": float(x_hat[2])}
    else:
        out["params"] = {
            "D": float(x_hat[0]),
            "k_on1": float(x_hat[1]),
            "k_off1": float(x_hat[2]),
            "k_on2": float(x_hat[3]),
            "k_off2": float(x_hat[4]),
        }

    return out


def global_multi_spot_compare_models(
    datasets: Sequence[MultiSpotDataset],
    *,
    method: str = "Nelder-Mead",
) -> Dict[str, Any]:
    """Fit all 5 global models and pick the statistically best (AIC)."""

    models = [
        "diffusion_only",
        "reaction_diffusion",
        "reaction_diffusion_immobile",
        "fast_exchange_plus_specific",
        "two_binding",
    ]

    results: Dict[str, Dict[str, Any]] = {}
    for m in models:
        results[m] = global_multi_spot_fit(datasets, model=m, method=method)

    # Choose best by AIC among successful fits
    candidates = [(m, r) for m, r in results.items() if np.isfinite(r.get("aic", np.nan))]
    best_model = min(candidates, key=lambda mr: float(mr[1]["aic"]))[0] if candidates else None
    best_aic = float(results[best_model]["aic"]) if best_model else float('nan')

    for m, r in results.items():
        r["delta_aic"] = float(r.get("aic", np.nan) - best_aic) if best_model else float('nan')

    return {
        "best_model": best_model,
        "results": results,
    }


def format_global_multi_spot_report(compare: Dict[str, Any]) -> str:
    """Render a simple markdown report with separate sections per model."""

    best_model = compare.get("best_model")
    results = compare.get("results") or {}

    lines: List[str] = []
    lines.append("# Global Multi-Spot Fitting Report")
    lines.append("")
    lines.append(f"**Best model (AIC):** {best_model}")
    lines.append("")

    order = [
        "diffusion_only",
        "reaction_diffusion",
        "reaction_diffusion_immobile",
        "fast_exchange_plus_specific",
        "two_binding",
    ]

    for m in order:
        r = results.get(m)
        if not isinstance(r, dict):
            continue
        lines.append(f"## Model: {m}")
        lines.append(f"- Success: {r.get('success')} ({r.get('message')})")
        lines.append(f"- RSS: {r.get('rss'):.6g}")
        lines.append(f"- AIC: {r.get('aic'):.3f}")
        if "delta_aic" in r:
            lines.append(f"- ΔAIC vs best: {r.get('delta_aic'):.3f}")
        lines.append(f"- Params: {r.get('params')}")
        lines.append("")
        lines.append("### Spot-size scaling (fit curves)")
        for row in r.get("scaling", []) or []:
            try:
                lines.append(
                    f"- w={row['bleach_radius_um']:.3f} µm, w²={row['bleach_radius_um2']:.3f}, t½={row['t_half']:.3g} s"
                )
            except Exception:
                continue
        lines.append("")

    return "\n".join(lines)


def attach_global_multispot_to_group(data_manager: Any, group_name: str, compare: Dict[str, Any]) -> None:
    """Store global multi-spot comparison results on a group for HTML/PDF export.

    This is a small convenience helper. It does not run any fitting.

    Stores:
    - group['global_multispot_compare']: raw comparison dict
    - group['global_multispot_report_md']: formatted markdown (for quick display)
    """

    if not hasattr(data_manager, 'groups'):
        raise AttributeError("data_manager must have a 'groups' attribute")
    if group_name not in data_manager.groups:
        raise KeyError(f"Group {group_name} not found")
    group = data_manager.groups[group_name]
    group['global_multispot_compare'] = compare
    try:
        group['global_multispot_report_md'] = format_global_multi_spot_report(compare)
    except Exception:
        group['global_multispot_report_md'] = None


def effective_diffusion(D_free: float, k_on_star: float, k_off_star: float) -> float:
    """Fast-exchange effective diffusion approximation.

    D_eff = D_free / (1 + k_on*/k_off*)
    """

    D_free = float(D_free)
    k_on_star = float(k_on_star)
    k_off_star = float(k_off_star)
    if D_free < 0:
        raise ValueError("D_free must be >= 0")
    if k_on_star < 0 or k_off_star <= 0:
        raise ValueError("k_on_star must be >=0 and k_off_star must be > 0")
    return D_free / (1.0 + (k_on_star / k_off_star))


def fit_fast_exchange_plus_specific_binding(
    image_stack: np.ndarray,
    time_points: np.ndarray,
    bleach_frame: int,
    *,
    pixel_size: float = 1.0,
    bleach_threshold: float = 0.3,
    cell_threshold: Optional[float] = None,
    method: str = "Nelder-Mead",
) -> Dict[str, Any]:
    """3-parameter model: (D_eff, k_on, k_off) with geometry-aware simulation.

    This is a reduced model vs the 2-binding system, intended for fast-exchange
    non-specific interactions + a single specific interaction.
    """

    ds = MultiSpotDataset(
        image_stack=np.asarray(image_stack),
        time_points=np.asarray(time_points),
        bleach_frame=int(bleach_frame),
        pixel_size=float(pixel_size),
        bleach_threshold=float(bleach_threshold),
        cell_threshold=cell_threshold,
    )
    p = _prepare_dataset(ds)
    solver = CoupledPDESolver(p["cell_mask"], p["bleach_mask"], pixel_size=p["pixel_size"])

    def obj(x: Sequence[float]) -> float:
        D_eff, k_on, k_off = map(float, x)
        if D_eff <= 0 or k_on < 0 or k_off < 0:
            return 1e12
        sim = solver.simulate(
            CoupledPDEParameters(D=D_eff, k_on1=k_on, k_off1=k_off, k_on2=0.0, k_off2=0.0),
            p["t"],
            bleach_depth=p["bleach_depth"],
        )
        return float(np.sum((p["y"] - sim["recovery"]) ** 2))

    res = optimize.minimize(obj, [1.0, 0.1, 0.1], method=method)
    D_eff, k_on, k_off = map(float, res.x)
    sim = solver.simulate(
        CoupledPDEParameters(D=D_eff, k_on1=k_on, k_off1=k_off, k_on2=0.0, k_off2=0.0),
        p["t"],
        bleach_depth=p["bleach_depth"],
    )

    rss = float(res.fun)
    aic = len(p["y"]) * np.log(max(rss, 1e-30) / len(p["y"])) + 2 * 3
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "params": {"D_eff": D_eff, "k_on": k_on, "k_off": k_off},
        "rss": rss,
        "aic": float(aic),
        "time": p["t"],
        "data": p["y"],
        "fit": sim["recovery"],
    }


def _mittag_leffler_E(alpha: float, z: np.ndarray) -> np.ndarray:
    """E_alpha(z) with SciPy fallback."""

    alpha = float(alpha)
    z = np.asarray(z, dtype=float)

    try:
        from scipy import special

        ml = getattr(special, "mittag_leffler", None)
        if ml is not None:
            # scipy.special.mittag_leffler(alpha, beta, z) with beta default=1
            return np.asarray(ml(alpha, 1.0, z), dtype=float)
    except Exception:
        pass

    # Fallback: stretched exponential approximation (common practical surrogate)
    # E_alpha(-x) ~ exp(-x) when alpha=1 and heavier tail for alpha<1.
    x = np.clip(-z, 0.0, None)
    return np.exp(-np.power(x, alpha))


def ctrw_mittag_leffler_recovery(
    t: np.ndarray,
    *,
    bleach_depth: float,
    mobile_fraction: float,
    tau: float,
    alpha: float,
) -> np.ndarray:
    """CTRW recovery curve based on Mittag–Leffler relaxation.

    Model:
        y(t) = 1 - mobile_fraction * (1 - bleach_depth) * E_alpha( -(t/tau)^alpha )

    - At t=0: y(0) = bleach_depth
    - As t→∞: y(∞) = 1 - (1-mobile_fraction) * (1-bleach_depth)

    This is a 1D curve model (no geometry); see fit_ctrw_geometry_corrected
    for a geometry-aware variant.
    """

    t = np.asarray(t, dtype=float)
    tau = max(float(tau), 1e-12)
    alpha = float(alpha)
    alpha = float(np.clip(alpha, 1e-3, 1.0))
    mobile_fraction = float(np.clip(mobile_fraction, 0.0, 1.0))
    bleach_depth = float(np.clip(bleach_depth, 0.0, 1.0))

    z = -np.power(np.clip(t / tau, 0.0, None), alpha)
    E = _mittag_leffler_E(alpha, z)
    return 1.0 - mobile_fraction * (1.0 - bleach_depth) * E


def fit_ctrw_geometry_corrected(
    image_stack: np.ndarray,
    time_points: np.ndarray,
    bleach_frame: int,
    *,
    pixel_size: float = 1.0,
    bleach_threshold: float = 0.3,
    cell_threshold: Optional[float] = None,
    method: str = "Nelder-Mead",
) -> Dict[str, Any]:
    """Geometry-aware CTRW-style anomalous diffusion fit via time-warping.

    Approach:
    - Simulate a geometry-aware *normal diffusion* baseline curve r0(t) with D=1.
    - Use CTRW time-warping: r(t) ≈ r0(D_alpha * t^alpha).

    Fits (D_alpha, alpha, mobile_fraction). Not run in batch.
    """

    ds = MultiSpotDataset(
        image_stack=np.asarray(image_stack),
        time_points=np.asarray(time_points),
        bleach_frame=int(bleach_frame),
        pixel_size=float(pixel_size),
        bleach_threshold=float(bleach_threshold),
        cell_threshold=cell_threshold,
    )
    p = _prepare_dataset(ds)

    # Baseline diffusion-only solver (D=1). We'll simulate on-demand on warped times.
    solver = FiniteDifferenceSolver(p["cell_mask"], p["bleach_mask"], pixel_size=p["pixel_size"])

    def simulate_baseline(warped_t: np.ndarray) -> np.ndarray:
        # Ensure strictly non-decreasing for solver stability
        warped_t = np.asarray(warped_t, dtype=float)
        warped_t = np.clip(warped_t, 0.0, None)
        warped_t = np.maximum.accumulate(warped_t)
        sim = solver.simulate(
            PDEParameters(D=1.0, k_on=0.0, k_off=0.0, immobile_fraction=0.0),
            warped_t,
            bleach_depth=float(p["bleach_depth"]),
        )
        return np.asarray(sim["recovery_curve"], dtype=float)

    def obj(x: Sequence[float]) -> float:
        D_alpha, alpha, mobile_fraction = map(float, x)
        if D_alpha <= 0 or not (0 < alpha <= 1.0) or not (0 <= mobile_fraction <= 1.0):
            return 1e12
        warped = D_alpha * np.power(np.clip(p["t"], 0.0, None), alpha)
        base = simulate_baseline(warped)
        # Scale by mobile fraction: simple mixture with immobile plateau at 1.0
        pred = (1.0 - mobile_fraction) * 1.0 + mobile_fraction * base
        return float(np.sum((p["y"] - pred) ** 2))

    res = optimize.minimize(obj, [1.0, 0.8, 1.0], method=method)
    D_alpha, alpha, mobile_fraction = map(float, res.x)
    warped = D_alpha * np.power(np.clip(p["t"], 0.0, None), alpha)
    base = simulate_baseline(warped)
    pred = (1.0 - mobile_fraction) * 1.0 + mobile_fraction * base

    rss = float(res.fun)
    aic = len(p["y"]) * np.log(max(rss, 1e-30) / len(p["y"])) + 2 * 3

    return {
        "success": bool(res.success),
        "message": str(res.message),
        "params": {"D_alpha": D_alpha, "alpha": alpha, "mobile_fraction": mobile_fraction},
        "rss": rss,
        "aic": float(aic),
        "time": p["t"],
        "data": p["y"],
        "fit": pred,
    }


def pixel_by_pixel_parameter_mapping(
    image_stack: np.ndarray,
    time_points: np.ndarray,
    bleach_frame: int,
    *,
    pixel_size: float = 1.0,
    cell_mask: Optional[np.ndarray] = None,
    pre_frames: Optional[int] = None,
    end_frames: int = 5,
    bleach_radius_um: Optional[float] = None,
) -> Dict[str, Any]:
    """Fast pixel-wise maps of mobile fraction and an approximate D.

    This intentionally avoids iterative fitting per pixel (too slow) and instead
    uses direct, robust estimators:

    - Mobile fraction per pixel: (I_end - I_bleach) / (I_pre - I_bleach)
    - Half-time per pixel: first time reaching half-recovery
    - Rate constant: k = ln(2) / t_half
    - Approx D (optional): D ≈ w^2 / (4 * t_half)

    Returns NaNs where estimation is undefined.
    """

    stack = np.asarray(image_stack, dtype=float)
    if stack.ndim != 3:
        raise ValueError("image_stack must be a 3D array: (t, y, x)")

    time_points = np.asarray(time_points, dtype=float)
    if time_points.ndim != 1 or len(time_points) != stack.shape[0]:
        raise ValueError("time_points must be 1D and match image_stack length")

    bleach_frame = int(bleach_frame)
    if bleach_frame <= 0 or bleach_frame >= stack.shape[0] - 1:
        raise ValueError("bleach_frame must be within stack range")

    if pre_frames is None:
        pre_frames = bleach_frame
    pre_frames = max(int(pre_frames), 1)

    if cell_mask is None:
        pre = np.mean(stack[:bleach_frame], axis=0)
        cell_mask = GeometryExtractor.extract_cell_mask(pre)
    else:
        cell_mask = np.asarray(cell_mask).astype(bool)

    y0 = np.mean(stack[:pre_frames], axis=0)
    yb = stack[bleach_frame]
    yend = np.mean(stack[-max(int(end_frames), 1):], axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = (y0 - yb)
        mobile_fraction_map = (yend - yb) / denom

    mobile_fraction_map = np.clip(mobile_fraction_map, 0.0, 1.0)
    mobile_fraction_map[~np.isfinite(mobile_fraction_map)] = np.nan

    # Normalize time series per pixel (using y0)
    eps = 1e-12
    y0_safe = np.where(np.abs(y0) < eps, np.nan, y0)
    norm_stack = stack / y0_safe

    # Build half-time map: first time reaching half recovery
    n_t, ny, nx = norm_stack.shape
    t_rel = time_points[bleach_frame:] - float(time_points[bleach_frame])
    post = norm_stack[bleach_frame:]

    # Targets per pixel
    yb_n = post[0]
    yend_n = np.nanmean(post[-max(int(end_frames), 1):], axis=0)
    target = yb_n + 0.5 * (yend_n - yb_n)

    t_half_map = np.full((ny, nx), np.nan, dtype=float)

    # Vectorized-ish search: iterate time, fill first crossing
    crossed = np.zeros((ny, nx), dtype=bool)
    for k in range(1, post.shape[0]):
        yk = post[k]
        hit = (~crossed) & np.isfinite(target) & (yk >= target)
        if np.any(hit):
            t_half_map[hit] = float(t_rel[k])
            crossed[hit] = True
        if np.all(crossed | ~np.isfinite(target)):
            break

    k_map = np.full((ny, nx), np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_map = np.log(2.0) / t_half_map

    D_map = np.full((ny, nx), np.nan, dtype=float)
    if bleach_radius_um is not None:
        w = float(bleach_radius_um)
        with np.errstate(divide="ignore", invalid="ignore"):
            D_map = (w * w) / (4.0 * t_half_map)

    # Apply cell mask
    mobile_fraction_map[~cell_mask] = np.nan
    t_half_map[~cell_mask] = np.nan
    k_map[~cell_mask] = np.nan
    D_map[~cell_mask] = np.nan

    return {
        "cell_mask": cell_mask,
        "mobile_fraction_map": mobile_fraction_map,
        "t_half_map": t_half_map,
        "k_map": k_map,
        "D_map": D_map,
        "time_relative": t_rel,
    }
