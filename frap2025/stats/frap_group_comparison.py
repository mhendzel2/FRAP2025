"""Group-comparison convenience wrapper."""

from __future__ import annotations

import pandas as pd

from .frap_group_stats import choose_test_automatically, compare_groups_with_fdr


def run_group_comparison(
    data: pd.DataFrame,
    *,
    value_col: str = "value",
    group_col: str = "group",
    correction: str = "fdr_bh",
) -> dict[str, object]:
    """Run automatic test selection plus pairwise comparisons."""
    selection = choose_test_automatically(data, value_col=value_col, group_col=group_col)
    pairwise = compare_groups_with_fdr(data, value_col=value_col, group_col=group_col, correction=correction)
    return {
        "selection": selection,
        "pairwise": pairwise,
    }


__all__ = ["run_group_comparison"]
