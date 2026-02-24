"""Population-level heterogeneity analysis utilities.

This module provides distribution-level comparisons between biological populations,
including Wasserstein/Earth Mover's distance over multivariate feature clouds.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler

try:
    import ot  # type: ignore

    _HAS_POT = True
except ImportError:  # pragma: no cover - depends on environment
    ot = None
    _HAS_POT = False


@dataclass(frozen=True)
class PopulationComparisonResult:
    """Result container for a multivariate population comparison.

    Attributes
    ----------
    distance : float
        Observed distribution distance between population A and B.
    p_value : float
        Empirical permutation-test p-value.
    n_permutations : int
        Number of permutations used for hypothesis testing.
    method : str
        Backend used for distance computation (``"pot_emd2"`` or
        ``"featurewise_wasserstein_fallback"``).
    """

    distance: float
    p_value: float
    n_permutations: int
    method: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "distance": float(self.distance),
            "p_value": float(self.p_value),
            "n_permutations": int(self.n_permutations),
            "method": self.method,
        }


def _coerce_feature_frame(data: pd.DataFrame | np.ndarray, features: list[str], label: str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        missing = [column for column in features if column not in data.columns]
        if missing:
            raise ValueError(f"{label} is missing required feature columns: {missing}")
        frame = data.loc[:, features].copy()
    else:
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{label} must be a 2D array when passed as ndarray.")
        if arr.shape[1] != len(features):
            raise ValueError(
                f"{label} has {arr.shape[1]} columns but {len(features)} features were provided."
            )
        frame = pd.DataFrame(arr, columns=features)

    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if frame.empty:
        raise ValueError(f"{label} has no finite rows after filtering.")
    return frame


def _distance_with_backend(xa: np.ndarray, xb: np.ndarray, *, warn_on_fallback: bool = True) -> tuple[float, str]:
    n_a, n_b = xa.shape[0], xb.shape[0]
    if _HAS_POT and ot is not None:
        a = np.full(n_a, 1.0 / n_a, dtype=float)
        b = np.full(n_b, 1.0 / n_b, dtype=float)
        cost_matrix = ot.dist(xa, xb, metric="euclidean")
        distance = float(ot.emd2(a, b, cost_matrix))
        return distance, "pot_emd2"

    if warn_on_fallback:
        warnings.warn(
            "POT is not installed. Falling back to feature-wise 1D Wasserstein mean distance.",
            RuntimeWarning,
            stacklevel=2,
        )
    per_feature = [wasserstein_distance(xa[:, idx], xb[:, idx]) for idx in range(xa.shape[1])]
    return float(np.mean(per_feature)), "featurewise_wasserstein_fallback"


def compare_populations_wasserstein(
    df_A: pd.DataFrame | np.ndarray,
    df_B: pd.DataFrame | np.ndarray,
    features: list[str],
    *,
    n_permutations: int = 500,
    random_state: int | None = 42,
) -> dict[str, float | int | str]:
    """Compare two populations using multivariate Wasserstein geometry.

    Parameters
    ----------
    df_A, df_B : pandas.DataFrame or numpy.ndarray
        Two populations represented as feature matrices. If a DataFrame is passed,
        ``features`` are selected by name; if an ndarray is passed, columns are
        assumed to align with ``features``.
    features : list of str
        Ordered feature names defining the comparison space (e.g. diffusion
        coefficients, anomalous exponents, condensate radii).
    n_permutations : int, default=500
        Number of label permutations for the empirical significance test.
    random_state : int or None, default=42
        Seed for permutation reproducibility.

    Returns
    -------
    dict
        Dictionary with keys: ``distance``, ``p_value``, ``n_permutations``, and
        ``method``.

    Notes
    -----
    Let ``x_i in R^d`` and ``y_j in R^d`` denote standardized samples from
    populations A and B. The Earth Mover's objective is

    ``W = min_P sum_{i,j} P_{ij} c(x_i, y_j)``

    subject to row/column marginals matching uniform sample weights. Here
    ``c(x_i, y_j)`` is Euclidean transport cost. The permutation p-value is

    ``p = (1 + sum_k I[W_k >= W_obs]) / (n_permutations + 1)``.
    """
    if not features:
        raise ValueError("features must contain at least one feature name.")
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1.")

    frame_a = _coerce_feature_frame(df_A, features, "df_A")
    frame_b = _coerce_feature_frame(df_B, features, "df_B")

    merged = pd.concat([frame_a, frame_b], axis=0, ignore_index=True)
    scaler = StandardScaler()
    merged_scaled = scaler.fit_transform(merged.to_numpy(dtype=float))

    n_a = frame_a.shape[0]
    xa = merged_scaled[:n_a]
    xb = merged_scaled[n_a:]

    observed_distance, method = _distance_with_backend(xa, xb)

    rng = np.random.default_rng(random_state)
    n_total = merged_scaled.shape[0]
    permuted_distances = np.empty(n_permutations, dtype=float)

    for idx in range(n_permutations):
        perm = rng.permutation(n_total)
        xa_perm = merged_scaled[perm[:n_a]]
        xb_perm = merged_scaled[perm[n_a:]]
        permuted_distances[idx], _ = _distance_with_backend(xa_perm, xb_perm, warn_on_fallback=False)

    p_value = float((1.0 + np.sum(permuted_distances >= observed_distance)) / (n_permutations + 1.0))

    result = PopulationComparisonResult(
        distance=float(observed_distance),
        p_value=p_value,
        n_permutations=int(n_permutations),
        method=method,
    )
    return result.to_dict()


__all__ = ["PopulationComparisonResult", "compare_populations_wasserstein"]
