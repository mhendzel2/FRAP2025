"""
Reporting helpers for hierarchical Bayesian FRAP fits.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from frap_bayes_data import BayesianDataset

try:
    import arviz as az

    ARVIZ_AVAILABLE = True
except Exception:  # pragma: no cover
    az = None
    ARVIZ_AVAILABLE = False

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover
    go = None
    PLOTLY_AVAILABLE = False


def _require_arviz() -> None:
    if not ARVIZ_AVAILABLE:
        raise ImportError("arviz is required for Bayesian reporting")


def parameter_summary_table(
    idata: Any,
    var_names: Optional[list[str]] = None,
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    """Return ArviZ summary as a flat table."""
    _require_arviz()

    summary = az.summary(idata, var_names=var_names, hdi_prob=hdi_prob)
    summary = summary.reset_index().rename(columns={"index": "parameter"})
    return summary


def extract_condition_effects(idata: Any) -> pd.DataFrame:
    """Extract condition-level centered effects from posterior draws."""
    if not hasattr(idata, "posterior"):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for var_name in idata.posterior.data_vars:
        if not (var_name.startswith("delta_") and "_cond_centered" in var_name):
            continue

        arr = np.asarray(idata.posterior[var_name])
        if arr.ndim < 3:
            continue

        draws = arr.reshape(-1, arr.shape[-1])
        if "condition" in idata.posterior[var_name].coords:
            cond_labels = [str(v) for v in idata.posterior[var_name].coords["condition"].values]
        else:
            cond_labels = [f"cond_{i}" for i in range(draws.shape[1])]

        for idx, cond in enumerate(cond_labels):
            samples = draws[:, idx]
            rows.append(
                {
                    "effect": var_name,
                    "condition": cond,
                    "mean": float(np.mean(samples)),
                    "median": float(np.median(samples)),
                    "hdi_2.5%": float(np.quantile(samples, 0.025)),
                    "hdi_97.5%": float(np.quantile(samples, 0.975)),
                }
            )

    return pd.DataFrame(rows)


def extract_cell_parameter(idata: Any, parameter: str) -> pd.DataFrame:
    """Extract per-cell posterior summaries for a cell-level parameter."""
    if not hasattr(idata, "posterior") or parameter not in idata.posterior:
        return pd.DataFrame()

    arr = np.asarray(idata.posterior[parameter])
    if arr.ndim < 3:
        return pd.DataFrame()

    draws = arr.reshape(-1, arr.shape[-1])

    if "cell" in idata.posterior[parameter].coords:
        cell_labels = [str(v) for v in idata.posterior[parameter].coords["cell"].values]
    else:
        cell_labels = [str(i) for i in range(draws.shape[1])]

    rows: list[dict[str, Any]] = []
    for idx, cell_label in enumerate(cell_labels):
        samples = draws[:, idx]
        rows.append(
            {
                "cell": cell_label,
                "parameter": parameter,
                "mean": float(np.mean(samples)),
                "median": float(np.median(samples)),
                "hdi_2.5%": float(np.quantile(samples, 0.025)),
                "hdi_97.5%": float(np.quantile(samples, 0.975)),
            }
        )

    return pd.DataFrame(rows)


def make_condition_forest_plot(
    condition_effects: pd.DataFrame,
    *,
    effect_name: Optional[str] = None,
    title: str = "Condition Effects (Posterior)",
):
    """Build a simple forest-style plot for condition effects."""
    if not PLOTLY_AVAILABLE or condition_effects.empty:
        return None

    df = condition_effects.copy()
    if effect_name:
        df = df[df["effect"] == effect_name]
    if df.empty:
        return None

    df = df.sort_values("median")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["median"],
            y=df["condition"],
            mode="markers",
            error_x=dict(
                type="data",
                symmetric=False,
                array=(df["hdi_97.5%"] - df["median"]).to_numpy(),
                arrayminus=(df["median"] - df["hdi_2.5%"]).to_numpy(),
            ),
            marker=dict(size=8, color="#1f77b4"),
            showlegend=False,
        )
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color="gray")
    fig.update_layout(title=title, xaxis_title="Posterior effect", yaxis_title="Condition", height=350)
    return fig


def make_parameter_shrinkage_plot(
    posterior_cell_table: pd.DataFrame,
    deterministic_by_cell: Optional[pd.Series] = None,
    title: str = "Shrinkage Plot",
):
    """Plot deterministic estimates vs posterior medians."""
    if not PLOTLY_AVAILABLE or posterior_cell_table.empty:
        return None

    df = posterior_cell_table.copy()
    if deterministic_by_cell is not None and not deterministic_by_cell.empty:
        det_map = deterministic_by_cell.to_dict()
        df["deterministic"] = df["cell"].map(det_map)
    else:
        df["deterministic"] = np.nan

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["deterministic"],
            y=df["median"],
            mode="markers",
            marker=dict(size=7, color="#2ca02c", opacity=0.7),
            text=df["cell"],
            hovertemplate="Cell %{text}<br>Deterministic=%{x:.4f}<br>Posterior=%{y:.4f}<extra></extra>",
            showlegend=False,
        )
    )

    finite_x = np.isfinite(df["deterministic"].to_numpy(dtype=float))
    if finite_x.any():
        x_vals = df.loc[finite_x, "deterministic"].to_numpy(dtype=float)
        line_min = float(np.min(x_vals))
        line_max = float(np.max(x_vals))
        fig.add_trace(
            go.Scatter(
                x=[line_min, line_max],
                y=[line_min, line_max],
                mode="lines",
                line=dict(color="gray", dash="dash"),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Deterministic estimate",
        yaxis_title="Posterior median",
        height=350,
    )
    return fig


def make_ppc_overlay_plot(
    dataset: BayesianDataset,
    idata: Any,
    max_cells: int = 12,
):
    """Overlay observed traces and posterior predictive mean for selected cells."""
    if not PLOTLY_AVAILABLE:
        return None
    if not hasattr(idata, "posterior_predictive"):
        return None
    if "y" not in idata.posterior_predictive:
        return None

    y_pp = np.asarray(idata.posterior_predictive["y"])
    if y_pp.ndim == 2:
        pp_samples = y_pp
    else:
        pp_samples = y_pp.reshape(-1, y_pp.shape[-1])
    pp_mean = pp_samples.mean(axis=0)

    long_df = dataset.long_df.copy().reset_index(drop=True)
    if long_df.shape[0] != pp_mean.size:
        return None

    long_df["pp_mean"] = pp_mean

    cells = long_df["cell_uid"].drop_duplicates().tolist()[:max_cells]

    fig = go.Figure()
    for cell_uid in cells:
        cell_df = long_df[long_df["cell_uid"] == cell_uid].sort_values("time")
        fig.add_trace(
            go.Scatter(
                x=cell_df["time"],
                y=cell_df["intensity"],
                mode="markers",
                marker=dict(size=4, color="rgba(31,119,180,0.45)"),
                name=f"obs {cell_uid}",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cell_df["time"],
                y=cell_df["pp_mean"],
                mode="lines",
                line=dict(width=1.5, color="rgba(214,39,40,0.75)"),
                name=f"pp {cell_uid}",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Posterior Predictive Overlay",
        xaxis_title="Time",
        yaxis_title="Intensity",
        height=420,
    )
    return fig


def build_bayesian_report_tables(
    idata: Any,
    parameters: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """Create compact tables for report export and persistence."""
    parameters = parameters or ["k", "I0", "I_inf", "sigma"]

    tables: dict[str, pd.DataFrame] = {
        "parameter_summary": parameter_summary_table(idata, var_names=parameters),
        "condition_effects": extract_condition_effects(idata),
    }

    for param in parameters:
        cell_table = extract_cell_parameter(idata, param)
        if not cell_table.empty:
            tables[f"cell_{param}"] = cell_table

    return tables
