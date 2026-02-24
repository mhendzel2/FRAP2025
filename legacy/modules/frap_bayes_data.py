"""
Hierarchical Bayesian data preparation utilities for FRAP.

This module provides a long-format contract that mirrors the intent of
`UnifiedModelWorkflow.prepare_data()` while adding FRAP-specific preprocessing
choices (bleach alignment, optional normalization fallback, ragged handling,
and optional interpolation to a shared time grid).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BayesianPreprocessingRecipe:
    """Record of preprocessing decisions used to build Bayesian inputs."""

    mode: str = "deterministic"  # "deterministic" or "semi_generative"
    intensity_source: str = "signal_norm"
    fallback_intensity_source: str = "signal_corr"
    align_to_bleach_minimum: bool = True
    normalize_if_needed: bool = True
    interpolation: str = "none"  # "none" or "common_grid"
    common_grid_size: Optional[int] = None
    qc_only: bool = True
    min_points_per_cell: int = 4
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BayesianDataset:
    """Canonical dataset consumed by hierarchical Bayesian model builders."""

    long_df: pd.DataFrame
    t: np.ndarray
    y: np.ndarray
    obs_cell_idx: np.ndarray
    cell_condition_idx: np.ndarray
    cell_batch_idx: np.ndarray
    cell_ids: list[str]
    condition_levels: list[str]
    batch_levels: list[str]
    recipe: dict[str, Any]
    ragged: bool
    common_time_grid: Optional[np.ndarray] = None

    @property
    def n_obs(self) -> int:
        return int(self.t.size)

    @property
    def n_cells(self) -> int:
        return int(len(self.cell_ids))

    @property
    def n_conditions(self) -> int:
        return int(len(self.condition_levels))

    @property
    def n_batches(self) -> int:
        return int(len(self.batch_levels))

    def as_model_data(self) -> dict[str, Any]:
        """Return arrays and dimensions for probabilistic model construction."""
        return {
            "t": self.t,
            "y": self.y,
            "obs_cell_idx": self.obs_cell_idx,
            "cell_condition_idx": self.cell_condition_idx,
            "cell_batch_idx": self.cell_batch_idx,
            "n_obs": self.n_obs,
            "n_cells": self.n_cells,
            "n_conditions": self.n_conditions,
            "n_batches": self.n_batches,
        }


def prepare_long_format(
    df: pd.DataFrame,
    time_col: str = "Time",
    intensity_col: str = "Intensity",
    curve_col: str = "CurveID",
    group_col: str = "GroupID",
    batch_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Mirror-style long-format builder inspired by UnifiedModelWorkflow.prepare_data.

    Returns a normalized schema with columns:
    `cell_uid`, `condition`, `batch`, `time`, `intensity`.
    """
    required = [time_col, intensity_col, curve_col, group_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for long-format conversion: {missing}")

    out = df[[curve_col, group_col, time_col, intensity_col]].copy()
    out = out.rename(
        columns={
            curve_col: "cell_uid",
            group_col: "condition",
            time_col: "time",
            intensity_col: "intensity",
        }
    )

    if batch_col and batch_col in df.columns:
        out["batch"] = df[batch_col].astype(str)
    else:
        out["batch"] = "batch_0"

    out["cell_uid"] = out["cell_uid"].astype(str)
    out["condition"] = out["condition"].astype(str)
    out = out.sort_values(["condition", "cell_uid", "time"]).reset_index(drop=True)
    return out[["cell_uid", "condition", "batch", "time", "intensity"]]


def build_bayesian_dataset(
    roi_traces: pd.DataFrame,
    cell_features: Optional[pd.DataFrame] = None,
    recipe: Optional[BayesianPreprocessingRecipe] = None,
    *,
    time_col: str = "t",
    intensity_col: str = "signal_norm",
    fallback_intensity_col: str = "signal_corr",
    cell_col: str = "cell_id",
    condition_col: str = "condition",
    batch_col: str = "exp_id",
    movie_col: str = "movie_id",
    qc_col: str = "bleach_qc",
    interpolate_to_common_grid: Optional[bool] = None,
    common_time_grid: Optional[np.ndarray] = None,
    common_grid_size: Optional[int] = None,
) -> BayesianDataset:
    """
    Build a hierarchical Bayesian dataset from FRAP2025 single-cell tables.

    Mode A (default) uses deterministic preprocessing decisions captured in
    `recipe` and fits to corrected/normalized post-bleach traces.
    """
    if recipe is None:
        recipe = BayesianPreprocessingRecipe(
            intensity_source=intensity_col,
            fallback_intensity_source=fallback_intensity_col,
        )

    # Allow call-time override while still recording a single recipe object.
    if interpolate_to_common_grid is not None:
        recipe.interpolation = "common_grid" if interpolate_to_common_grid else "none"
    if common_grid_size is not None:
        recipe.common_grid_size = int(common_grid_size)

    if roi_traces is None or roi_traces.empty:
        raise ValueError("roi_traces is empty; cannot construct Bayesian dataset")

    _require_columns(roi_traces, [cell_col, time_col], "roi_traces")

    traces = roi_traces.copy()
    traces = _build_cell_uid_column(
        traces,
        cell_col=cell_col,
        batch_col=batch_col,
        movie_col=movie_col,
        uid_col="cell_uid",
    )

    features_meta = _build_features_metadata(
        cell_features=cell_features,
        cell_col=cell_col,
        condition_col=condition_col,
        batch_col=batch_col,
        movie_col=movie_col,
        qc_col=qc_col,
    )

    intensity_primary = recipe.intensity_source or intensity_col
    intensity_fallback = recipe.fallback_intensity_source or fallback_intensity_col

    trace_records: list[pd.DataFrame] = []
    skipped_cells = 0

    for cell_uid, group in traces.groupby("cell_uid", sort=False):
        group = group.sort_values(time_col)

        if cell_uid in features_meta.index:
            meta_row = features_meta.loc[cell_uid]
        else:
            meta_row = {
                "condition": str(group[condition_col].iloc[0]) if condition_col in group.columns else "all",
                "batch": str(group[batch_col].iloc[0]) if batch_col in group.columns else "batch_0",
                "qc_pass": True,
            }

        if recipe.qc_only and not bool(meta_row.get("qc_pass", True)):
            skipped_cells += 1
            continue

        t = _to_numeric_array(group[time_col].values)
        y, used_source = _select_intensity_series(
            group,
            primary_col=intensity_primary,
            fallback_col=intensity_fallback,
        )

        if y is None:
            skipped_cells += 1
            continue

        # Align bleach frame to t=0 and keep post-bleach domain.
        bleach_idx = int(np.nanargmin(y)) if recipe.align_to_bleach_minimum else 0
        t_aligned = t - t[bleach_idx]

        if used_source == intensity_fallback and recipe.normalize_if_needed:
            y = _normalize_from_bleach(y, bleach_idx)

        post_mask = np.isfinite(t_aligned) & np.isfinite(y) & (t_aligned >= 0)
        t_post = t_aligned[post_mask]
        y_post = y[post_mask]

        if t_post.size < int(recipe.min_points_per_cell):
            skipped_cells += 1
            continue

        cell_df = pd.DataFrame(
            {
                "cell_uid": cell_uid,
                "condition": str(meta_row.get("condition", "all")),
                "batch": str(meta_row.get("batch", "batch_0")),
                "time": t_post.astype(float),
                "intensity": y_post.astype(float),
            }
        )

        # Preserve optional bleach radius metadata for mechanistic gating.
        if "radius" in group.columns:
            cell_df["radius"] = float(np.nanmedian(group["radius"].values))

        trace_records.append(cell_df)

    if not trace_records:
        raise ValueError("No valid post-bleach traces available after filtering")

    long_df = pd.concat(trace_records, axis=0, ignore_index=True)
    long_df = long_df.sort_values(["condition", "cell_uid", "time"]).reset_index(drop=True)

    if recipe.interpolation == "common_grid":
        long_df, common_time_grid = _interpolate_common_grid(
            long_df,
            grid=common_time_grid,
            grid_size=recipe.common_grid_size,
        )
        ragged = False
    else:
        common_time_grid = None
        ragged = _is_ragged(long_df)

    dataset = build_bayesian_dataset_from_long(
        long_df,
        recipe=recipe.to_dict(),
        ragged=ragged,
        common_time_grid=common_time_grid,
    )

    if skipped_cells:
        logger.info("Skipped %d cells during Bayesian dataset construction", skipped_cells)

    return dataset


def build_bayesian_dataset_from_long(
    long_df: pd.DataFrame,
    recipe: Optional[dict[str, Any]] = None,
    ragged: bool = True,
    common_time_grid: Optional[np.ndarray] = None,
) -> BayesianDataset:
    """Construct `BayesianDataset` from a standardized long-format DataFrame."""
    required = ["cell_uid", "condition", "batch", "time", "intensity"]
    _require_columns(long_df, required, "long_df")

    clean = long_df.copy()
    clean["cell_uid"] = clean["cell_uid"].astype(str)
    clean["condition"] = clean["condition"].astype(str)
    clean["batch"] = clean["batch"].astype(str)
    clean["time"] = pd.to_numeric(clean["time"], errors="coerce")
    clean["intensity"] = pd.to_numeric(clean["intensity"], errors="coerce")
    clean = clean.dropna(subset=["time", "intensity"])
    clean = clean[clean["time"] >= 0]
    clean = clean.sort_values(["condition", "cell_uid", "time"]).reset_index(drop=True)

    if clean.empty:
        raise ValueError("No valid observations found after long-format cleaning")

    cell_ids = clean["cell_uid"].drop_duplicates().tolist()
    condition_levels = sorted(clean["condition"].unique().tolist())
    batch_levels = sorted(clean["batch"].unique().tolist())

    cell_to_idx = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
    condition_to_idx = {name: idx for idx, name in enumerate(condition_levels)}
    batch_to_idx = {name: idx for idx, name in enumerate(batch_levels)}

    obs_cell_idx = clean["cell_uid"].map(cell_to_idx).to_numpy(dtype=np.int64)

    cell_meta = clean.drop_duplicates("cell_uid")[["cell_uid", "condition", "batch"]].set_index("cell_uid")
    cell_condition_idx = np.array(
        [condition_to_idx[cell_meta.loc[cell_id, "condition"]] for cell_id in cell_ids],
        dtype=np.int64,
    )
    cell_batch_idx = np.array(
        [batch_to_idx[cell_meta.loc[cell_id, "batch"]] for cell_id in cell_ids],
        dtype=np.int64,
    )

    t = clean["time"].to_numpy(dtype=float)
    y = clean["intensity"].to_numpy(dtype=float)

    if (t < 0).any():
        raise ValueError("Bayesian dataset contains negative post-bleach times")

    return BayesianDataset(
        long_df=clean,
        t=t,
        y=y,
        obs_cell_idx=obs_cell_idx,
        cell_condition_idx=cell_condition_idx,
        cell_batch_idx=cell_batch_idx,
        cell_ids=cell_ids,
        condition_levels=condition_levels,
        batch_levels=batch_levels,
        recipe=recipe or {},
        ragged=bool(ragged),
        common_time_grid=None if common_time_grid is None else np.asarray(common_time_grid, dtype=float),
    )


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}")


def _build_features_metadata(
    cell_features: Optional[pd.DataFrame],
    cell_col: str,
    condition_col: str,
    batch_col: str,
    movie_col: str,
    qc_col: str,
) -> pd.DataFrame:
    if cell_features is None or cell_features.empty:
        return pd.DataFrame(columns=["condition", "batch", "qc_pass"])

    features = cell_features.copy()
    _require_columns(features, [cell_col], "cell_features")
    features = _build_cell_uid_column(
        features,
        cell_col=cell_col,
        batch_col=batch_col,
        movie_col=movie_col,
        uid_col="cell_uid",
    )

    if condition_col not in features.columns:
        features[condition_col] = "all"
    if batch_col not in features.columns:
        features[batch_col] = "batch_0"
    if qc_col not in features.columns:
        features[qc_col] = True

    meta = (
        features[["cell_uid", condition_col, batch_col, qc_col]]
        .drop_duplicates(subset=["cell_uid"])
        .rename(columns={condition_col: "condition", batch_col: "batch", qc_col: "qc_pass"})
        .set_index("cell_uid")
    )

    return meta


def _build_cell_uid_column(
    df: pd.DataFrame,
    *,
    cell_col: str,
    batch_col: str,
    movie_col: str,
    uid_col: str,
) -> pd.DataFrame:
    out = df.copy()

    if batch_col not in out.columns:
        out[batch_col] = "batch_0"
    if movie_col not in out.columns:
        out[movie_col] = "movie_0"

    out[uid_col] = (
        out[batch_col].astype(str)
        + "::"
        + out[movie_col].astype(str)
        + "::"
        + out[cell_col].astype(str)
    )
    return out


def _select_intensity_series(
    group: pd.DataFrame,
    primary_col: str,
    fallback_col: str,
) -> tuple[Optional[np.ndarray], str]:
    if primary_col in group.columns:
        y_primary = _to_numeric_array(group[primary_col].values)
        if np.isfinite(y_primary).sum() >= max(4, len(y_primary) // 2):
            return y_primary, primary_col

    if fallback_col in group.columns:
        y_fallback = _to_numeric_array(group[fallback_col].values)
        if np.isfinite(y_fallback).sum() >= 4:
            return y_fallback, fallback_col

    return None, ""


def _to_numeric_array(values: Any) -> np.ndarray:
    return pd.to_numeric(np.asarray(values), errors="coerce").astype(float)


def _normalize_from_bleach(y: np.ndarray, bleach_idx: int) -> np.ndarray:
    y = y.astype(float)
    if y.size == 0:
        return y

    if bleach_idx > 0:
        pre_bleach = float(np.nanmean(y[:bleach_idx]))
    else:
        pre_bleach = float(np.nanmax(y))

    i0 = float(y[bleach_idx])
    denom = pre_bleach - i0

    if abs(denom) < 1e-12:
        return y - i0

    return (y - i0) / denom


def _is_ragged(long_df: pd.DataFrame) -> bool:
    counts = long_df.groupby("cell_uid").size().to_numpy(dtype=int)
    return bool(np.unique(counts).size > 1)


def _interpolate_common_grid(
    long_df: pd.DataFrame,
    grid: Optional[np.ndarray] = None,
    grid_size: Optional[int] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    by_cell = list(long_df.groupby("cell_uid", sort=False))

    if not by_cell:
        raise ValueError("No cells available for interpolation")

    if grid is None:
        max_shared_time = min(float(cell_df["time"].max()) for _, cell_df in by_cell)
        if max_shared_time <= 0:
            raise ValueError("Cannot build common grid: shared max time is non-positive")

        if grid_size is None:
            lengths = [len(cell_df) for _, cell_df in by_cell]
            grid_size = int(np.clip(np.median(lengths), 8, 300))

        grid = np.linspace(0.0, max_shared_time, int(grid_size), dtype=float)
    else:
        grid = np.asarray(grid, dtype=float)
        grid = grid[np.isfinite(grid) & (grid >= 0)]
        if grid.size < 3:
            raise ValueError("Provided common_time_grid must contain at least 3 non-negative points")

    interpolated: list[pd.DataFrame] = []

    for cell_uid, cell_df in by_cell:
        t = cell_df["time"].to_numpy(dtype=float)
        y = cell_df["intensity"].to_numpy(dtype=float)

        order = np.argsort(t)
        t = t[order]
        y = y[order]

        # Keep only finite points and unique times for stable interpolation.
        finite = np.isfinite(t) & np.isfinite(y)
        t = t[finite]
        y = y[finite]

        if t.size < 2:
            continue

        uniq_t, uniq_idx = np.unique(t, return_index=True)
        uniq_y = y[uniq_idx]

        y_grid = np.interp(grid, uniq_t, uniq_y)

        row0 = cell_df.iloc[0]
        out = pd.DataFrame(
            {
                "cell_uid": cell_uid,
                "condition": str(row0["condition"]),
                "batch": str(row0["batch"]),
                "time": grid,
                "intensity": y_grid,
            }
        )

        if "radius" in cell_df.columns:
            out["radius"] = float(np.nanmedian(cell_df["radius"].values))

        interpolated.append(out)

    if not interpolated:
        raise ValueError("Interpolation failed: no cell had enough points")

    merged = pd.concat(interpolated, axis=0, ignore_index=True)
    merged = merged.sort_values(["condition", "cell_uid", "time"]).reset_index(drop=True)
    return merged, grid
