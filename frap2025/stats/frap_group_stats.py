"""Group statistics with automatic test selection and multiple-testing correction."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestSelection:
    """Selected inferential test and rationale."""

    test_name: str
    reason: str
    normality_p_values: dict[str, float]
    group_sizes: dict[str, int]
    min_n_warning: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "test_name": self.test_name,
            "reason": self.reason,
            "normality_p_values": self.normality_p_values,
            "group_sizes": self.group_sizes,
            "min_n_warning": self.min_n_warning,
        }


def _to_group_arrays(groups: dict[str, np.ndarray] | pd.DataFrame, value_col: str, group_col: str) -> dict[str, np.ndarray]:
    if isinstance(groups, pd.DataFrame):
        result: dict[str, np.ndarray] = {}
        for group_name, subset in groups.groupby(group_col):
            values = np.asarray(subset[value_col].dropna(), dtype=float)
            if values.size:
                result[str(group_name)] = values
        return result

    out: dict[str, np.ndarray] = {}
    for key, values in groups.items():
        arr = np.asarray(values, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            out[str(key)] = arr
    return out


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return np.nan
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1))
    if pooled == 0:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / pooled)


def _rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return np.nan
    return float(1.0 - (2.0 * u_stat) / (n1 * n2))


def _eta_squared_from_anova(f_stat: float, df_between: int, df_within: int) -> float:
    numerator = f_stat * df_between
    denominator = numerator + df_within
    if denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def apply_multiple_testing_correction(
    p_values: np.ndarray | list[float],
    *,
    method: str = "fdr_bh",
) -> np.ndarray:
    """Apply multiple-testing correction.

    Supported methods:
    - ``none``: no correction
    - ``bonferroni``: family-wise error rate control
    - ``fdr_bh``: Benjamini-Hochberg step-up FDR procedure
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be 1D.")

    if method == "none":
        return np.clip(p, 0.0, 1.0)

    m = p.size
    if m == 0:
        return p

    if method == "bonferroni":
        return np.clip(p * m, 0.0, 1.0)

    if method == "fdr_bh":
        order = np.argsort(p)
        p_sorted = p[order]
        ranks = np.arange(1, m + 1, dtype=float)

        adjusted_sorted = p_sorted * m / ranks
        adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
        adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

        adjusted = np.empty_like(adjusted_sorted)
        adjusted[order] = adjusted_sorted
        return adjusted

    raise ValueError("method must be one of: 'none', 'bonferroni', 'fdr_bh'.")


def choose_test_automatically(groups: dict[str, np.ndarray] | pd.DataFrame, *, value_col: str = "value", group_col: str = "group") -> dict[str, object]:
    """Choose a suitable test from group shape and normality checks."""
    group_arrays = _to_group_arrays(groups, value_col=value_col, group_col=group_col)
    if len(group_arrays) < 2:
        raise ValueError("At least two groups are required.")

    group_sizes = {name: int(arr.size) for name, arr in group_arrays.items()}
    normality_p_values: dict[str, float] = {}
    is_normal = True

    for name, arr in group_arrays.items():
        if arr.size >= 3:
            _, p_val = stats.shapiro(arr)
            normality_p_values[name] = float(p_val)
            if p_val <= 0.05:
                is_normal = False
        else:
            normality_p_values[name] = np.nan
            is_normal = False

    min_n = min(group_sizes.values())
    min_n_warning = min_n < 5
    if min_n_warning:
        warnings.warn(
            "Statistical comparison with N < 5 per group has low power. Results should be interpreted with caution.",
            UserWarning,
            stacklevel=2,
        )

    n_groups = len(group_arrays)
    if n_groups == 2:
        if is_normal and min_n >= 10:
            test_name = "unpaired_t_test"
            reason = "Two groups, each approximately normal by Shapiro-Wilk (p > 0.05), with N >= 10."
        else:
            test_name = "mann_whitney_u"
            reason = "Two groups with non-normal distribution or small sample size; using non-parametric test."
    else:
        if is_normal and min_n >= 10:
            test_name = "one_way_anova"
            reason = "Three or more groups with approximate normality and adequate sample size."
        else:
            test_name = "kruskal_wallis"
            reason = "Three or more groups with non-normality or small N; using rank-based test."

    return TestSelection(
        test_name=test_name,
        reason=reason,
        normality_p_values=normality_p_values,
        group_sizes=group_sizes,
        min_n_warning=min_n_warning,
    ).to_dict()


def compare_groups_with_fdr(
    data: pd.DataFrame,
    *,
    value_col: str = "value",
    group_col: str = "group",
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    """Pairwise group comparisons with BH/Bonferroni/none correction.

    Returns a tidy dataframe with raw/adjusted p-values and effect sizes.
    """
    if value_col not in data.columns or group_col not in data.columns:
        raise ValueError(f"Expected columns '{group_col}' and '{value_col}'.")

    group_arrays = _to_group_arrays(data, value_col=value_col, group_col=group_col)
    groups = list(group_arrays.keys())
    if len(groups) < 2:
        raise ValueError("At least two non-empty groups are required.")

    rows: list[dict[str, object]] = []
    for idx_a in range(len(groups)):
        for idx_b in range(idx_a + 1, len(groups)):
            group_a, group_b = groups[idx_a], groups[idx_b]
            x = group_arrays[group_a]
            y = group_arrays[group_b]

            pair_selection = choose_test_automatically({group_a: x, group_b: y})
            if pair_selection["test_name"] == "unpaired_t_test":
                stat, p_value = stats.ttest_ind(x, y, equal_var=False)
                effect_size = _cohens_d(x, y)
                effect_size_name = "cohens_d"
            else:
                stat, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
                effect_size = _rank_biserial_from_u(float(stat), x.size, y.size)
                effect_size_name = "rank_biserial_r"

            rows.append(
                {
                    "group_a": group_a,
                    "group_b": group_b,
                    "test_used": pair_selection["test_name"],
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "effect_size": float(effect_size),
                    "effect_size_name": effect_size_name,
                }
            )

    out = pd.DataFrame(rows)
    out["p_value_adjusted"] = apply_multiple_testing_correction(out["p_value"].to_numpy(dtype=float), method=correction)
    out["significant"] = out["p_value_adjusted"] < 0.05

    if len(groups) >= 3:
        # Report omnibus effect size in each row for convenience.
        arrays = [group_arrays[g] for g in groups]
        f_stat, _ = stats.f_oneway(*arrays)
        df_between = len(groups) - 1
        df_within = sum(arr.size for arr in arrays) - len(groups)
        out["eta_squared"] = _eta_squared_from_anova(float(f_stat), df_between=df_between, df_within=df_within)
    else:
        out["eta_squared"] = np.nan

    return out


__all__ = [
    "apply_multiple_testing_correction",
    "choose_test_automatically",
    "compare_groups_with_fdr",
]
