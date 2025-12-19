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
    model: str = "single_binding",
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
    model:
        - 'single_binding': shared (D, k_on, k_off) using a coupled U+V model
        - 'diffusion_only': shared (D)
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

    prepared = [_prepare_dataset(ds) for ds in datasets]

    solvers = [
        CoupledPDESolver(p["cell_mask"], p["bleach_mask"], pixel_size=p["pixel_size"])
        for p in prepared
    ]

    model = str(model)
    if model not in {"single_binding", "diffusion_only"}:
        raise ValueError("model must be 'single_binding' or 'diffusion_only'")

    if model == "diffusion_only":
        x0 = [1.0] if initial is None else list(initial)

        def obj(x: Sequence[float]) -> float:
            D = float(x[0])
            if D <= 0:
                return 1e12
            rss = 0.0
            for p, solver in zip(prepared, solvers, strict=False):
                sim = solver.simulate(
                    CoupledPDEParameters(D=D, k_on1=0.0, k_off1=0.0, k_on2=0.0, k_off2=0.0),
                    p["t"],
                    bleach_depth=p["bleach_depth"],
                )
                rss += float(np.sum((p["y"] - sim["recovery"]) ** 2))
            return rss

    else:
        x0 = [1.0, 0.1, 0.1] if initial is None else list(initial)

        def obj(x: Sequence[float]) -> float:
            D, k_on, k_off = map(float, x)
            if D <= 0 or k_on < 0 or k_off < 0:
                return 1e12
            rss = 0.0
            for p, solver in zip(prepared, solvers, strict=False):
                sim = solver.simulate(
                    CoupledPDEParameters(D=D, k_on1=k_on, k_off1=k_off, k_on2=0.0, k_off2=0.0),
                    p["t"],
                    bleach_depth=p["bleach_depth"],
                )
                rss += float(np.sum((p["y"] - sim["recovery"]) ** 2))
            return rss

    result = optimize.minimize(obj, x0, method=method)

    # Build per-dataset predictions
    x_hat = result.x
    per = []
    for p, solver in zip(prepared, solvers, strict=False):
        if model == "diffusion_only":
            params = CoupledPDEParameters(D=float(x_hat[0]))
        else:
            params = CoupledPDEParameters(D=float(x_hat[0]), k_on1=float(x_hat[1]), k_off1=float(x_hat[2]))
        sim = solver.simulate(params, p["t"], bleach_depth=p["bleach_depth"])
        per.append(
            {
                "bleach_radius_um": p["bleach_radius_um"],
                "time": p["t"],
                "data": p["y"],
                "fit": sim["recovery"],
            }
        )

    # Rough scaling diagnostic: t_half vs radius^2
    scaling = []
    for item in per:
        y = item["fit"]
        t = item["time"]
        y0 = float(y[0])
        yinf = float(np.mean(y[-min(len(y), 5):]))
        target = y0 + 0.5 * (yinf - y0)
        idx = np.where(y >= target)[0]
        t_half = float(t[idx[0]]) if len(idx) else float("nan")
        scaling.append(
            {
                "bleach_radius_um": float(item["bleach_radius_um"]),
                "bleach_radius_um2": float(item["bleach_radius_um"]) ** 2,
                "t_half": t_half,
            }
        )

    out: Dict[str, Any] = {
        "model": model,
        "success": bool(result.success),
        "message": str(result.message),
        "rss": float(result.fun),
        "datasets": per,
        "scaling": scaling,
    }
    if model == "diffusion_only":
        out["params"] = {"D": float(x_hat[0])}
    else:
        out["params"] = {"D": float(x_hat[0]), "k_on": float(x_hat[1]), "k_off": float(x_hat[2])}
    return out


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

        if hasattr(special, "mittag_leffler"):
            # scipy.special.mittag_leffler(alpha, beta, z) with beta default=1
            return np.asarray(special.mittag_leffler(alpha, 1.0, z), dtype=float)
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
