"""Optical flow analysis backends for deformation-field estimation."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any
import warnings

import numpy as np

_HAS_CV2 = importlib.util.find_spec("cv2") is not None
_HAS_RAFT = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None


def _get_cv2() -> Any:
    if not _HAS_CV2:
        raise ImportError("OpenCV is required for optical-flow computation.")
    return importlib.import_module("cv2")


def _ensure_gray_float(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        cv2 = _get_cv2()
        arr = cv2.cvtColor(arr.astype(np.float32), cv2.COLOR_BGR2GRAY)
    elif arr.ndim != 2:
        raise ValueError("Input image must be 2D grayscale or 3-channel RGB/BGR array.")

    arr = arr.astype(np.float32)
    if not np.any(np.isfinite(arr)):
        raise ValueError("Input image contains no finite pixels.")

    min_val = float(np.nanmin(arr))
    max_val = float(np.nanmax(arr))
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr


def _compute_farneback(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    cv2 = _get_cv2()
    initial_flow = np.zeros((img1.shape[0], img1.shape[1], 2), dtype=np.float32)
    flow = cv2.calcOpticalFlowFarneback(
        prev=img1,
        next=img2,
        flow=initial_flow,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return np.asarray(flow, dtype=np.float32)


def _compute_raft(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    if not _HAS_RAFT:
        raise ImportError("RAFT backend requires torch + torchvision.")

    torch = importlib.import_module("torch")
    optical_flow_mod = importlib.import_module("torchvision.models.optical_flow")
    raft_large = getattr(optical_flow_mod, "raft_large")
    raft_weights = getattr(optical_flow_mod, "Raft_Large_Weights")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h, w = img1.shape
    img1_rgb = np.repeat(img1[..., None], repeats=3, axis=2)
    img2_rgb = np.repeat(img2[..., None], repeats=3, axis=2)

    t1 = torch.from_numpy(img1_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(dev)
    t2 = torch.from_numpy(img2_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(dev)

    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8
    if pad_h or pad_w:
        t1 = torch.nn.functional.pad(t1, (0, pad_w, 0, pad_h), mode="replicate")
        t2 = torch.nn.functional.pad(t2, (0, pad_w, 0, pad_h), mode="replicate")

    model = raft_large(weights=raft_weights.DEFAULT, progress=False).to(dev)
    model.eval()

    with torch.no_grad():
        flow_predictions = model(t1, t2)
        flow = flow_predictions[-1][0].detach().cpu().numpy()

    flow = flow[:, :h, :w]
    return np.moveaxis(flow, 0, -1).astype(np.float32)


def compute_deformation_field(img1: np.ndarray, img2: np.ndarray, method: str = "farneback") -> np.ndarray:
    """Compute pixel-wise deformation field between two images.

    Parameters
    ----------
    img1, img2 : ndarray
        Input images (grayscale ``H x W`` or RGB ``H x W x 3``).
    method : {'farneback', 'raft'}, default='farneback'
        Optical flow backend.

    Returns
    -------
    ndarray
        Dense displacement field with shape ``(H, W, 2)`` in ``(dx, dy)``.
    """
    g1 = _ensure_gray_float(img1)
    g2 = _ensure_gray_float(img2)
    if g1.shape != g2.shape:
        raise ValueError("img1 and img2 must have identical shape.")

    method_key = str(method).strip().lower()
    if method_key == "farneback":
        return _compute_farneback(g1, g2)

    if method_key == "raft":
        try:
            return _compute_raft(g1, g2)
        except Exception as exc:
            warnings.warn(
                f"RAFT backend unavailable ({exc}). Falling back to farneback.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _compute_farneback(g1, g2)

    raise ValueError("method must be one of {'farneback', 'raft'}")


__all__ = ["compute_deformation_field"]
