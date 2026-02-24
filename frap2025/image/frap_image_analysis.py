"""Image-analysis primitives used by FRAP model fitting and QC."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import warnings

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class BleachRadiusResult:
    """Measured bleach radius and circularity diagnostics."""

    w_um: float
    w_px: float
    fit_quality: float
    center_xy: tuple[float, float]
    ellipticity: float

    def to_dict(self) -> dict[str, float | tuple[float, float]]:
        return {
            "w_um": self.w_um,
            "w_px": self.w_px,
            "fit_quality": self.fit_quality,
            "center_xy": self.center_xy,
            "ellipticity": self.ellipticity,
        }


def _as_2d(frame: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(frame, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D image array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _cross_correlation_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[int, int]:
    """Return integer shift `(dy, dx)` maximizing cross-correlation."""
    ref = reference - np.mean(reference)
    mov = moving - np.mean(moving)

    fft_ref = np.fft.fft2(ref)
    fft_mov = np.fft.fft2(mov)
    corr = np.fft.ifft2(fft_ref * np.conj(fft_mov))
    corr = np.fft.fftshift(np.real(corr))

    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    center_y = corr.shape[0] // 2
    center_x = corr.shape[1] // 2
    return int(peak_y - center_y), int(peak_x - center_x)


def _shift_image(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    shifted = np.roll(frame, shift=dy, axis=0)
    shifted = np.roll(shifted, shift=dx, axis=1)
    return shifted


def measure_bleach_radius_from_image(
    post_bleach_frame: np.ndarray,
    pre_bleach_frame: np.ndarray,
    pixel_size_um: float,
) -> dict[str, float | tuple[float, float]]:
    """Estimate bleach e^-2 radius from pre/post bleach image pair.

    The bleach profile is approximated with a 2D Gaussian from image moments on
    `pre_bleach - post_bleach`. For a Gaussian profile
    ``exp(-2 r^2 / w^2)``, the e^-2 radius is ``w_px = 2 * sigma_px``.
    """
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be > 0.")

    post = _as_2d(post_bleach_frame, "post_bleach_frame")
    pre = _as_2d(pre_bleach_frame, "pre_bleach_frame")
    if post.shape != pre.shape:
        raise ValueError("post_bleach_frame and pre_bleach_frame must have identical shape.")

    bleach_depth = np.maximum(pre - post, 0.0)
    total = float(np.sum(bleach_depth))
    if total <= 0:
        raise ValueError("No measurable bleach depth in image pair.")

    yy, xx = np.indices(bleach_depth.shape)
    x0 = float(np.sum(xx * bleach_depth) / total)
    y0 = float(np.sum(yy * bleach_depth) / total)

    var_x = float(np.sum(((xx - x0) ** 2) * bleach_depth) / total)
    var_y = float(np.sum(((yy - y0) ** 2) * bleach_depth) / total)
    sigma_x = max(np.sqrt(max(var_x, 1e-12)), 1e-6)
    sigma_y = max(np.sqrt(max(var_y, 1e-12)), 1e-6)

    sigma_mean = float(np.sqrt(0.5 * (sigma_x**2 + sigma_y**2)))
    w_px = float(2.0 * sigma_mean)
    w_um = float(w_px * pixel_size_um)

    ellipticity = float(max(sigma_x, sigma_y) / max(min(sigma_x, sigma_y), 1e-12))

    model = np.exp(-((xx - x0) ** 2 / (2.0 * sigma_x**2) + (yy - y0) ** 2 / (2.0 * sigma_y**2)))
    model *= float(np.max(bleach_depth))
    residual = bleach_depth - model
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((bleach_depth - np.mean(bleach_depth)) ** 2))
    fit_quality = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    if ellipticity > 1.2:
        warnings.warn(
            "Bleach ellipticity > 1.2; circular FRAP assumptions may be violated.",
            UserWarning,
            stacklevel=2,
        )

    return BleachRadiusResult(
        w_um=w_um,
        w_px=w_px,
        fit_quality=fit_quality,
        center_xy=(x0, y0),
        ellipticity=ellipticity,
    ).to_dict()


def apply_drift_correction(frames: np.ndarray, *, reference_frame_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Correct translational drift using frame-wise cross-correlation alignment.

    Returns
    -------
    corrected_frames : np.ndarray
        Aligned stack with same shape as input `(n_frames, height, width)`.
    shifts_dy_dx : np.ndarray
        Integer shifts per frame with shape `(n_frames, 2)` as `(dy, dx)`.
    """
    stack = np.asarray(frames, dtype=float)
    if stack.ndim != 3:
        raise ValueError("frames must have shape (n_frames, height, width).")
    if not np.all(np.isfinite(stack)):
        raise ValueError("frames contains non-finite values.")

    ref = stack[int(reference_frame_index)]
    corrected = np.empty_like(stack)
    shifts = np.zeros((stack.shape[0], 2), dtype=int)

    for idx in range(stack.shape[0]):
        dy, dx = _cross_correlation_shift(ref, stack[idx])
        corrected[idx] = _shift_image(stack[idx], dy=dy, dx=dx)
        shifts[idx] = (dy, dx)

    return corrected, shifts


def detect_photobleaching(pre_bleach_frames: np.ndarray) -> dict[str, float | bool | str]:
    """Detect acquisition-phase photobleaching from pre-bleach intensity trend.

    A warning condition is triggered when absolute linear slope magnitude exceeds
    1% of initial intensity per frame.
    """
    arr = np.asarray(pre_bleach_frames, dtype=float)
    if arr.ndim == 1:
        series = arr
    elif arr.ndim >= 2:
        series = np.mean(arr.reshape(arr.shape[0], -1), axis=1)
    else:
        raise ValueError("pre_bleach_frames must be 1D or higher.")

    if series.size < 3:
        raise ValueError("At least 3 pre-bleach frames are required.")
    if not np.all(np.isfinite(series)):
        raise ValueError("pre_bleach_frames contains non-finite values.")

    x = np.arange(series.size, dtype=float)
    slope, intercept = np.polyfit(x, series, 1)

    baseline = float(max(abs(series[0]), 1e-12))
    slope_pct_per_frame = float(100.0 * slope / baseline)

    if np.any(series <= 0):
        exp_rate_per_frame = np.nan
    else:
        exp_rate_per_frame = float(-np.polyfit(x, np.log(series), 1)[0])

    flagged = abs(slope_pct_per_frame) > 1.0
    if flagged:
        warnings.warn(
            "Pre-bleach trend exceeds 1%/frame; significant photobleaching detected.",
            UserWarning,
            stacklevel=2,
        )

    return {
        "flagged": flagged,
        "slope_pct_per_frame": slope_pct_per_frame,
        "model": "linear",
        "exp_rate_per_frame": exp_rate_per_frame,
    }


def extract_multi_roi_timeseries(
    image_stack: np.ndarray,
    bleach_roi_mask: np.ndarray,
    reference_roi_mask: np.ndarray,
    background_roi_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Extract bleach/reference/background ROI traces from one stack."""
    stack = np.asarray(image_stack, dtype=float)
    if stack.ndim != 3:
        raise ValueError("image_stack must have shape (n_frames, height, width).")

    masks = {
        "bleach": np.asarray(bleach_roi_mask, dtype=bool),
        "reference": np.asarray(reference_roi_mask, dtype=bool),
        "background": np.asarray(background_roi_mask, dtype=bool),
    }

    for name, mask in masks.items():
        if mask.shape != stack.shape[1:]:
            raise ValueError(f"{name}_roi_mask has shape {mask.shape}; expected {stack.shape[1:]}.")
        if np.count_nonzero(mask) == 0:
            raise ValueError(f"{name}_roi_mask must include at least one pixel.")

    bleach_ts = stack[:, masks["bleach"]].mean(axis=1)
    reference_ts = stack[:, masks["reference"]].mean(axis=1)
    background_ts = stack[:, masks["background"]].mean(axis=1)

    return {
        "bleach_roi": bleach_ts,
        "reference_roi": reference_ts,
        "background_roi": background_ts,
    }


__all__ = [
    "apply_drift_correction",
    "detect_photobleaching",
    "extract_multi_roi_timeseries",
    "measure_bleach_radius_from_image",
]
