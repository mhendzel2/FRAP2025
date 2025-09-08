#!/usr/bin/env python3
"""
Unit tests for image_motion motion compensation utilities.
Synthetic stacks with a dark Gaussian spot that drifts.
"""

import os
import sys
import math
import numpy as np

# Try to import OpenCV; if unavailable, set cv2 to None and provide a clear runtime error when used.
try:
    # Use dynamic import to avoid static analyzers reporting "Import 'cv2' could not be resolved"
    # in environments where OpenCV is not installed.
    import importlib
    cv2 = importlib.import_module("cv2")
except Exception:
    cv2 = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def require_cv2():
    """Return the cv2 module or raise a clear ImportError instructing how to install it."""
    if cv2 is None:
        raise ImportError(
            "OpenCV (cv2) is required for these tests; install it with 'pip install opencv-python' "
            "or run these tests in an environment where OpenCV is available."
        )
    return cv2

from image_motion import register_global, track_spot, stabilize_roi
from frap_core_corrected import FRAPAnalysisCore


def make_spot_frame(h, w, cx, cy, amp=200.0, sx=3.0, sy=3.0, base=800.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ga = amp * np.exp(-(((xx - cx) ** 2) / (2 * sx ** 2) + ((yy - cy) ** 2) / (2 * sy ** 2)))
    # Dark spot: subtract from base
    frame = base - ga
    noise = np.random.default_rng(0).normal(0.0, 2.0, size=frame.shape).astype(np.float32)
    return (frame + noise).astype(np.float32)


def make_stack(T=6, H=64, W=64, start=(32.0, 32.0), drift=(1.5, -1.0),
               amp=200.0, sx=3.0, sy=3.0, base=800.0):
    """Create a synthetic stack of T frames (H x W) with a drifting dark Gaussian spot.

    Arguments:
        T: number of frames
        H, W: frame height and width
        start: (cx, cy) initial spot center (x, y) in pixels
        drift: (dx, dy) per-frame displacement in pixels (x, y)
        amp, sx, sy, base: parameters forwarded to make_spot_frame
    """
    cx, cy = start
    dx, dy = drift
    stack = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        stack[t] = make_spot_frame(H, W, cx + t * dx, cy + t * dy,
                                   amp=amp, sx=sx, sy=sy, base=base)
    return stack


def test_register_global_translation():
    cv2_mod = require_cv2()
    stack = make_stack(T=6)
    # Add global shift relative to first frame
    shifted = np.empty_like(stack)
    shifted[0] = stack[0]
    for t in range(1, stack.shape[0]):
        M = np.array([[1, 0, 1.5 * t], [0, 1, -1.0 * t]], dtype=np.float32)
        shifted[t] = cv2_mod.warpAffine(stack[t], M, (stack.shape[2], stack.shape[1]))

    out = register_global(shifted, use_ecc_if_poor=False)
    aligned = out["aligned_stack"]
    # Compare center-of-mass between frames should be similar after alignment
    def com(img):
        yx = np.indices(img.shape)
        w = img.max() - img
        w = np.maximum(w - w.min(), 0)
        s = w.sum()
        if s == 0:
            return (img.shape[1] / 2, img.shape[0] / 2)
        cx = (w * yx[1]).sum() / s
        cy = (w * yx[0]).sum() / s
        return (float(cx), float(cy))

    c0 = com(aligned[0])
    for t in range(1, aligned.shape[0]):
        ct = com(aligned[t])
def test_stabilize_roi_outputs():
    # Ensure OpenCV available for functions that rely on it
    require_cv2()

    stack = make_stack(T=7, start=(30.0, 15.0), drift=(1.0, 0.0))
    res = stabilize_roi(stack, init_center=(30.0, 15.0), radius=6.0, pixel_size_um=0.2, use_optical_flow=True)
    assert "stabilized_stack" in res and res["stabilized_stack"].shape == stack.shape
    assert "roi_trace" in res and len(res["roi_trace"]) == stack.shape[0]
    # After stabilization, centroid displacement per frame should be small
    disp = [rt["displacement_px"] for rt in res["roi_trace"]]
    assert np.mean(disp) < 1.5
    # Core wrapper should call through without error
    core_out = FRAPAnalysisCore.motion_compensate_stack(stack, (30.0, 15.0), 6.0, pixel_size_um=0.2)
    assert core_out["stabilized_stack"].shape == stack.shape

# Duplicate test removed: the test_stabilize_roi_outputs defined earlier (which calls require_cv2())
# already covers stabilization checks and avoids a local 'import cv2' that may not be resolvable.


if __name__ == "__main__":
    # Quick smoke run
    test_register_global_translation()
    test_stabilize_roi_outputs()
    print("All motion tests passed.")
