"""
Motion compensation utilities for FRAP image stacks.

APIs:
- register_global(stack) -> dict(transform_params, aligned_stack, method, success_flags)
- track_spot(stack, init_center, radius, ...) -> dict(centroids, radii, success_flags, metrics)
- stabilize_roi(stack, init_center, radius, ...) -> dict(stabilized_stack, roi_trace, drift_um, warnings, details)

Notes:
- Pure, unit-testable functions. No UI dependencies.
- Input stack shape: (T, H, W) grayscale. If uint types, will convert to float32 internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import map_coordinates
import importlib

# Try modern and legacy scikit-image names for the phase-correlation function.
_phase_cross_correlation = None
_for_names = [
    ("skimage.registration", "phase_cross_correlation"),  # skimage >= 0.17
    ("skimage.feature", "register_translation"),          # older skimage
]
for _mod_name, _attr in _for_names:
    try:
        _mod = importlib.import_module(_mod_name)
        _phase_cross_correlation = getattr(_mod, _attr)
        break
    except Exception:
        continue

if _phase_cross_correlation is None:
    from typing import Tuple, Any

    def phase_cross_correlation(*args, **kwargs) -> Tuple[np.ndarray, float, Any]:
        raise ImportError(
            "scikit-image is required for phase_cross_correlation; "
            "please install scikit-image>=0.17 or a compatible version."
        )
else:
    phase_cross_correlation = _phase_cross_correlation
try:
    import cv2  # type: ignore
except Exception:
    from scipy import ndimage as _ndimage
    import numpy as np

    class _CV2Fallback:
        # minimal constants used by the code
        MOTION_TRANSLATION = 0
        TERM_CRITERIA_EPS = 1
        TERM_CRITERIA_COUNT = 2
        BORDER_REFLECT = 'reflect'
        INTER_LINEAR = 1

        def warpAffine(self, img, M, dsize, flags=None, borderMode=None):
            # translation-only fallback using scipy.ndimage.shift; M expected [[1,0,dx],[0,1,dy]]
            dx = float(M[0, 2])
            dy = float(M[1, 2])
            # _ndimage.shift expects shift=(shift_y, shift_x)
            return _ndimage.shift(img, shift=(dy, dx), order=1, mode='reflect')

        def calcOpticalFlowPyrLK(self, prev, curr, p0, *args, **kwargs):
            # Optical flow not available in fallback; return None so callers handle it gracefully
            return None, None, None

        def findTransformECC(self, template, image, warp_matrix, warp_mode, criteria):
            # ECC not available; raise so caller falls back to phase correlation
            raise NotImplementedError("cv2.findTransformECC is not available in this environment")

    cv2 = _CV2Fallback()


# -------------------------- Gaussian utilities -------------------------- #

def _gaussian2d(coords, amp, x0, y0, sx, sy, offset):
    x, y = coords
    g = amp * np.exp(-(((x - x0) ** 2) / (2 * sx ** 2) + ((y - y0) ** 2) / (2 * sy ** 2))) + offset
    return g.ravel()


def fit_gaussian_2d(
    image: np.ndarray,
    init_center: Tuple[float, float],
    window_radius: int = 12,
    invert: bool = True,
) -> Tuple[Tuple[float, float], Tuple[float, float], Dict[str, float]]:
    """
    Fit a 2D Gaussian around an expected center within a square window.

    Returns (center_x, center_y), (sx, sy), metrics dict with keys:
    - amplitude, offset, snr, rmse, n_points, success (1.0 or 0.0)
    """
    h, w = image.shape
    cx, cy = init_center
    cx_i, cy_i = int(round(cx)), int(round(cy))
    x_min = max(0, cx_i - window_radius)
    x_max = min(w, cx_i + window_radius + 1)
    y_min = max(0, cy_i - window_radius)
    y_max = min(h, cy_i + window_radius + 1)
    roi = image[y_min:y_max, x_min:x_max].astype(np.float32)
    if roi.size < 25:  # too small
        return (cx, cy), (np.nan, np.nan), {"success": 0.0, "rmse": np.inf, "snr": 0.0, "amplitude": 0.0, "offset": float(np.nan), "n_points": float(roi.size)}

    if invert:
        # FRAP spot is dark; invert to make it bright to stabilize fit
        roi = roi.max() - roi

    # Build coordinate grids in image coordinates
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    x = xx.astype(np.float32)
    y = yy.astype(np.float32)

    # Initial guesses
    amp0 = float(roi.max() - roi.min() + 1e-6)
    off0 = float(roi.min())
    x0 = float(np.clip(cx, x_min + 1, x_max - 2))
    y0 = float(np.clip(cy, y_min + 1, y_max - 2))
    sx0 = sy0 = max(2.0, min(window_radius / 2.0, 10.0))

    p0 = [amp0, x0, y0, sx0, sy0, off0]
    bounds = (
        [0.0, x_min, y_min, 0.5, 0.5, -np.inf],
        [np.inf, x_max - 1, y_max - 1, window_radius * 2.0, window_radius * 2.0, np.inf],
    )

    try:
        popt, pcov = curve_fit(_gaussian2d, (x, y), roi.ravel(), p0=p0, bounds=bounds, maxfev=10000)
        amp, x0, y0, sx, sy, off = popt
        fit_vals = _gaussian2d((x, y), *popt).reshape(roi.shape)
        resid = (roi - fit_vals)
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        noise = float(np.std(roi)) + 1e-6
        snr = float(amp / noise)
        success = 1.0 if (snr > 2.0 and rmse < max(2.0, 0.25 * amp)) else 0.0
        return (float(x0), float(y0)), (float(sx), float(sy)), {
            "success": success,
            "rmse": rmse,
            "snr": snr,
            "amplitude": float(amp),
            "offset": float(off),
            "n_points": float(roi.size),
        }
    except Exception:
        return (cx, cy), (np.nan, np.nan), {"success": 0.0, "rmse": np.inf, "snr": 0.0, "amplitude": 0.0, "offset": float(np.nan), "n_points": float(roi.size)}


# ---------------------- Global registration (translation) ---------------------- #

def register_global(
    stack: np.ndarray,
    *,
    upsample_factor: int = 10,
    use_ecc_if_poor: bool = True,
    ecc_iterations: int = 100,
    ecc_eps: float = 1e-4,
) -> Dict[str, Any]:
    """
    Estimate and apply per-frame global translation using phase correlation.
    Optionally refine with ECC when correlation is poor.

    Returns dict with keys:
    - transform_params: list of {dx, dy, method}
    - aligned_stack: np.ndarray of same shape
    - method: 'phase_corr' or 'phase_corr+ecc'
    - success_flags: list[bool]
    """
    if stack.ndim != 3:
        raise ValueError("stack must be 3D (T, H, W)")
    T, H, W = stack.shape
    ref = stack[0].astype(np.float32)
    aligned = np.empty_like(stack)
    aligned[0] = stack[0]
    transforms: List[Dict[str, Any]] = [{"dx": 0.0, "dy": 0.0, "method": "identity"}]
    success: List[bool] = [True]
    ecc_used_any = False

    for t in range(1, T):
        img = stack[t].astype(np.float32)
        shift, error, _ = phase_cross_correlation(ref, img, upsample_factor=upsample_factor, normalization=None)
        dy, dx = float(shift[0]), float(shift[1])  # skimage returns (row, col)
        used = "phase_corr"
        ok = bool(np.isfinite(error))

        if use_ecc_if_poor and (not ok or error > 0.5):  # lower error is better for pcc (MSE of phase?)
            # Try ECC translation model
            try:
                warp_mode = cv2.MOTION_TRANSLATION
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iterations, ecc_eps)
                cc, warp_matrix = cv2.findTransformECC(ref, img, warp_matrix, warp_mode, criteria)
                dx_ecc = float(warp_matrix[0, 2])
                dy_ecc = float(warp_matrix[1, 2])
                dx, dy = dx_ecc, dy_ecc
                used = "phase_corr+ecc"
                ecc_used_any = True
                ok = np.isfinite(cc)
            except Exception:
                # keep phase correlation result
                pass

        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        aligned[t] = cv2.warpAffine(stack[t], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        transforms.append({"dx": dx, "dy": dy, "method": used})
        success.append(ok)

    return {
        "transform_params": transforms,
        "aligned_stack": aligned,
        "method": "phase_corr+ecc" if ecc_used_any else "phase_corr",
        "success_flags": success,
    }


# ---------------------- Local tracking and stabilization ---------------------- #

@dataclass
class KalmanParams:
    q_pos: float = 1e-2  # process noise for position
    q_vel: float = 1e-1  # process noise for velocity
    r_pos: float = 1.0   # measurement noise


def _kalman_smooth_2d(centers: List[Tuple[float, float]], dt: float = 1.0, params: KalmanParams = KalmanParams()) -> List[Tuple[float, float]]:
    """Constant-velocity Kalman smoothing of 2D positions."""
    if not centers:
        return []
    x0, y0 = centers[0]
    # State: [x, y, vx, vy]
    x = np.array([[x0], [y0], [0.0], [0.0]], dtype=float)
    A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    Q = np.diag([params.q_pos, params.q_pos, params.q_vel, params.q_vel])
    R = np.diag([params.r_pos, params.r_pos])
    P = np.eye(4)

    smoothed: List[Tuple[float, float]] = []
    for (zx, zy) in centers:
        # Predict
        x = A @ x
        P = A @ P @ A.T + Q
        # Update
        z = np.array([[zx], [zy]])
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(4) - K @ H) @ P
        smoothed.append((float(x[0, 0]), float(x[1, 0])))
    return smoothed


def _avg_lk_flow(
    prev: np.ndarray,
    curr: np.ndarray,
    center: Tuple[float, float],
    window: int = 32,
    grid_step: int = 8,
) -> Tuple[float, float, int]:
    """Compute average Lucas–Kanade optical flow within a window centered at center.
    Returns (dx, dy, n_tracked) in image coordinates (x right, y down)."""
    H, W = prev.shape
    cx, cy = center
    cx_i, cy_i = int(round(cx)), int(round(cy))
    x_min = max(0, cx_i - window)
    x_max = min(W - 1, cx_i + window)
    y_min = max(0, cy_i - window)
    y_max = min(H - 1, cy_i + window)
    if x_max <= x_min or y_max <= y_min:
        return 0.0, 0.0, 0
    # sample points on grid
    xs = np.arange(x_min, x_max + 1, grid_step)
    ys = np.arange(y_min, y_max + 1, grid_step)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    p0 = pts.reshape(-1, 1, 2)

    prev_8u = cv2.normalize(prev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    curr_8u = cv2.normalize(curr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_8u, curr_8u, p0, None)
    if p1 is None or st is None:
        return 0.0, 0.0, 0
    st = st.reshape(-1)
    valid = st == 1
    if not np.any(valid):
        return 0.0, 0.0, 0
    flow = (p1.reshape(-1, 2) - p0.reshape(-1, 2))[valid]
    dx = float(np.median(flow[:, 0]))
    dy = float(np.median(flow[:, 1]))
    return dx, dy, int(valid.sum())


def track_spot(
    stack: np.ndarray,
    init_center: Tuple[float, float],
    radius: float,
    *,
    window_radius: int = 24,
    use_optical_flow: bool = False,
    flow_window: int = 32,
    kalman: bool = True,
) -> Dict[str, Any]:
    """
    Track bleach spot center with subpixel precision using 2D Gaussian fits.
    Optionally augment step-to-step prediction with Lucas–Kanade optical flow.

    Returns dict with keys:
    - centroids: list[(x,y)] per frame (float)
    - radii: list[float] per frame (applied radius in pixels)
    - success_flags: list[bool]
    - metrics: list[dict] per frame with fitting/flow metrics
    """
    if stack.ndim != 3:
        raise ValueError("stack must be 3D (T, H, W)")
    T, H, W = stack.shape
    center_pred = init_center
    centroids: List[Tuple[float, float]] = []
    radii: List[float] = []
    flags: List[bool] = []
    metrics: List[Dict[str, Any]] = []

    prev_frame = stack[0].astype(np.float32)
    # Fit first frame
    c0, (sx0, sy0), m0 = fit_gaussian_2d(prev_frame, init_center, window_radius)
    ok0 = bool(m0.get("success", 0.0) > 0.0)
    centroids.append(c0)
    radii.append(float(radius))
    flags.append(ok0)
    metrics.append({"frame": 0, **m0})
    center_pred = c0

    for t in range(1, T):
        frame = stack[t].astype(np.float32)
        # Predict next center using optional LK flow
        if use_optical_flow:
            dx, dy, n = _avg_lk_flow(prev_frame, frame, center_pred, window=flow_window)
            center_guess = (center_pred[0] + dx, center_pred[1] + dy)
        else:
            center_guess = center_pred

        c, (sx, sy), m = fit_gaussian_2d(frame, center_guess, window_radius)
        ok = bool(m.get("success", 0.0) > 0.0)
        centroids.append(c)
        radii.append(float(radius))
        flags.append(ok)
        m.update({"frame": t})
        if use_optical_flow:
            m.update({"flow_used": True})
        metrics.append(m)
        center_pred = c
        prev_frame = frame

    # Optional trajectory smoothing
    if kalman:
        smoothed = _kalman_smooth_2d(centroids)
        centroids = smoothed

    return {"centroids": centroids, "radii": radii, "success_flags": flags, "metrics": metrics}


def _translate_frame(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    H, W = frame.shape
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def stabilize_roi(
    stack: np.ndarray,
    init_center: Tuple[float, float],
    radius: float,
    *,
    pixel_size_um: Optional[float] = None,
    use_optical_flow: bool = True,
    window_radius: int = 24,
    flow_window: int = 32,
    kalman: bool = True,
    do_global: bool = True,
) -> Dict[str, Any]:
    """
    Full stabilization pipeline:
    1) Optional global translation alignment (phase correlation, ECC fallback)
    2) Spot recentering via 2D Gaussian fits (subpixel)
    3) Optional local optical flow to aid prediction and a Kalman smoother

    Returns dict with keys:
    - stabilized_stack: stack after global and local recentering (translation only)
    - roi_trace: list[dict] with frame, centroid(x,y), radius, displacement_px
    - drift_um: total drift in micrometers (if pixel_size provided), else None
    - warnings: list[str]
    - details: nested dict containing global and tracking outputs
    """
    warnings_list: List[str] = []

    if stack.ndim != 3:
        raise ValueError("stack must be 3D (T, H, W)")

    T, H, W = stack.shape
    work_stack = stack.copy()
    global_out: Optional[Dict[str, Any]] = None

    if do_global:
        global_out = register_global(work_stack)
        work_stack = global_out["aligned_stack"]
        if not all(global_out["success_flags"]):
            warnings_list.append("Global registration had low confidence on some frames.")

    track_out = track_spot(
        work_stack,
        init_center=init_center,
        radius=float(radius),
        window_radius=window_radius,
        use_optical_flow=use_optical_flow,
        flow_window=flow_window,
        kalman=kalman,
    )

    centroids = track_out["centroids"]
    success_flags = track_out["success_flags"]
    if not all(success_flags):
        n_bad = int(np.sum(~np.array(success_flags)))
        warnings_list.append(f"Spot fit confidence low on {n_bad} frame(s).")

    # Recentering: translate each frame so centroid is at initial center
    target = init_center
    stabilized = np.empty_like(work_stack)
    displacements: List[Tuple[float, float]] = []
    for t in range(T):
        cx, cy = centroids[t]
        dx = target[0] - cx
        dy = target[1] - cy
        stabilized[t] = _translate_frame(work_stack[t], dx, dy)
        displacements.append((dx, dy))

    # Build trace and drift
    roi_trace = []
    total_disp_px = 0.0
    for t, (c, r, (dx, dy)) in enumerate(zip(centroids, track_out["radii"], displacements)):
        disp_px = float(np.hypot(dx, dy))
        total_disp_px += disp_px
        roi_trace.append({
            "frame": t,
            "centroid": {"x": float(c[0]), "y": float(c[1])},
            "applied_radius": float(r),
            "displacement_px": disp_px,
        })

    drift_um = (total_disp_px * pixel_size_um) if (pixel_size_um is not None) else None

    return {
        "stabilized_stack": stabilized,
        "roi_trace": roi_trace,
        "drift_um": drift_um,
        "warnings": warnings_list,
        "details": {"global": global_out, "tracking": track_out},
    }


__all__ = [
    "register_global",
    "track_spot",
    "stabilize_roi",
    "fit_gaussian_2d",
    "KalmanParams",
]
