"""Normalization routines for FRAP traces."""

from __future__ import annotations

import numpy as np


class NormalizationError(ValueError):
    """Raised when normalization inputs are physiologically implausible."""


def _to_1d_float(array_like: np.ndarray | list[float], name: str) -> np.ndarray:
    arr = np.asarray(array_like, dtype=float).reshape(-1)
    if arr.size == 0:
        raise NormalizationError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise NormalizationError(f"{name} contains non-finite values.")
    return arr


def _pre_bleach_mean(t_s: np.ndarray, signal: np.ndarray) -> float:
    pre_mask = t_s < 0
    if np.any(pre_mask):
        return float(np.mean(signal[pre_mask]))
    # Fallback for traces without negative-time pre-bleach points.
    n_head = max(3, int(0.1 * len(signal)))
    return float(np.mean(signal[:n_head]))


def normalize_simple(
    t_s: np.ndarray | list[float],
    intensity: np.ndarray | list[float],
    *,
    background: float,
) -> np.ndarray:
    """
    Simple FRAP normalization:

    F_norm(t) = [F(t) - F_bg] / [F_pre - F_bg]
    """
    t_s_arr = _to_1d_float(t_s, "t_s")
    intensity_arr = _to_1d_float(intensity, "intensity")
    if len(t_s_arr) != len(intensity_arr):
        raise NormalizationError("t_s and intensity must have the same length.")
    if not np.isfinite(background):
        raise NormalizationError("background must be a finite scalar.")

    f_pre = _pre_bleach_mean(t_s_arr, intensity_arr)
    if f_pre <= background:
        raise NormalizationError(
            "Invalid simple normalization: pre-bleach mean must be greater than background."
        )

    f_norm = (intensity_arr - background) / (f_pre - background)

    post_idx = np.where(t_s_arr >= 0)[0]
    if post_idx.size:
        first_post_value = f_norm[post_idx[0]]
        if first_post_value > 0.85:
            raise NormalizationError(
                "Invalid bleach event: immediate post-bleach signal exceeds 85% of pre-bleach."
            )

    return f_norm


def normalize_double(
    t_s: np.ndarray | list[float],
    f_roi: np.ndarray | list[float],
    f_ref: np.ndarray | list[float],
    *,
    F_roi_pre: float,
    F_ref_pre: float,
) -> np.ndarray:
    """
    Double normalization (photobleaching correction):

    F_norm(t) = [F_ROI(t)/F_ref(t)] / [F_ROI_pre/F_ref_pre]
    """
    t_s_arr = _to_1d_float(t_s, "t_s")
    f_roi_arr = _to_1d_float(f_roi, "f_roi")
    f_ref_arr = _to_1d_float(f_ref, "f_ref")
    if len(t_s_arr) != len(f_roi_arr) or len(t_s_arr) != len(f_ref_arr):
        raise NormalizationError("t_s, f_roi, and f_ref must have the same length.")
    if not np.isfinite(F_roi_pre) or not np.isfinite(F_ref_pre):
        raise NormalizationError("F_roi_pre and F_ref_pre must be finite scalars.")
    if F_ref_pre <= 0 or F_roi_pre <= 0:
        raise NormalizationError("F_roi_pre and F_ref_pre must be > 0.")
    if np.any(f_ref_arr <= 0):
        raise NormalizationError("f_ref contains non-positive values; cannot divide by reference ROI.")

    scale = F_roi_pre / F_ref_pre
    if scale <= 0:
        raise NormalizationError("F_roi_pre/F_ref_pre must be > 0.")
    return (f_roi_arr / f_ref_arr) / scale


def normalize_full_scale(
    t_s: np.ndarray | list[float],
    intensity: np.ndarray | list[float],
) -> np.ndarray:
    """
    Full-scale normalization:

    F_norm(t) = [F(t) - F_post_min] / [F_pre - F_post_min]
    """
    t_s_arr = _to_1d_float(t_s, "t_s")
    intensity_arr = _to_1d_float(intensity, "intensity")
    if len(t_s_arr) != len(intensity_arr):
        raise NormalizationError("t_s and intensity must have the same length.")

    f_pre = _pre_bleach_mean(t_s_arr, intensity_arr)
    post_idx = np.where(t_s_arr >= 0)[0]
    if post_idx.size == 0:
        raise NormalizationError("full_scale normalization requires at least one post-bleach timepoint (t >= 0).")

    f_post_min = float(np.min(intensity_arr[post_idx]))
    denom = f_pre - f_post_min
    if denom <= 0:
        raise NormalizationError("Invalid full_scale normalization: pre-bleach must exceed post-bleach minimum.")
    return (intensity_arr - f_post_min) / denom


def normalize_frap_curve(
    t_s: np.ndarray | list[float],
    intensity: np.ndarray | list[float],
    *,
    mode: str,
    **kwargs,
) -> np.ndarray:
    """
    Unified normalization entry point. `mode` is required and has no default.
    """
    if mode == "simple":
        if "F_bg" in kwargs:
            background = kwargs["F_bg"]
        elif "background" in kwargs:
            background = kwargs["background"]
        else:
            raise NormalizationError("mode='simple' requires `F_bg` (or `background`).")
        return normalize_simple(t_s, intensity, background=float(background))

    if mode == "double":
        f_ref = kwargs.get("F_ref")
        f_roi_pre = kwargs.get("F_roi_pre")
        f_ref_pre = kwargs.get("F_ref_pre")
        if f_ref is None or f_roi_pre is None or f_ref_pre is None:
            raise NormalizationError(
                "mode='double' requires F_ref, F_roi_pre, and F_ref_pre."
            )
        return normalize_double(
            t_s,
            intensity,
            f_ref,
            F_roi_pre=float(f_roi_pre),
            F_ref_pre=float(f_ref_pre),
        )

    if mode == "full_scale":
        return normalize_full_scale(t_s, intensity)

    raise NormalizationError("mode must be one of: 'simple', 'double', 'full_scale'.")

