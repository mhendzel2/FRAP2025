import numpy as np
import pandas as pd

from frap_bayes_data import (
    BayesianPreprocessingRecipe,
    build_bayesian_dataset,
    prepare_long_format,
)


def _demo_tables():
    traces = pd.DataFrame(
        {
            "exp_id": ["e1"] * 7 + ["e1"] * 6,
            "movie_id": ["m1"] * 13,
            "cell_id": [1] * 7 + [2] * 6,
            "t": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5],
            "signal_corr": [1.0, 0.4, 0.55, 0.65, 0.74, 0.80, 0.84, 1.0, 0.45, 0.60, 0.70, 0.78, 0.82],
            "signal_norm": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                1.0,
                0.0,
                0.30,
                0.55,
                0.72,
                0.83,
            ],
            "radius": [3.0] * 13,
        }
    )

    features = pd.DataFrame(
        {
            "exp_id": ["e1", "e1"],
            "movie_id": ["m1", "m1"],
            "cell_id": [1, 2],
            "condition": ["ctrl", "drug"],
            "bleach_qc": [True, True],
            "k": [0.3, 0.2],
            "I0": [0.0, 0.0],
            "I_inf": [0.85, 0.83],
        }
    )

    return traces, features


def test_prepare_long_format_contract():
    df = pd.DataFrame(
        {
            "Time": [0, 1, 0, 1],
            "Intensity": [0.2, 0.4, 0.3, 0.5],
            "CurveID": ["c1", "c1", "c2", "c2"],
            "GroupID": ["A", "A", "B", "B"],
            "BatchID": ["b0", "b0", "b1", "b1"],
        }
    )
    out = prepare_long_format(df, batch_col="BatchID")
    assert list(out.columns) == ["cell_uid", "condition", "batch", "time", "intensity"]
    assert set(out["condition"]) == {"A", "B"}


def test_build_bayesian_dataset_ragged_default():
    traces, features = _demo_tables()

    recipe = BayesianPreprocessingRecipe(
        intensity_source="signal_norm",
        fallback_intensity_source="signal_corr",
        interpolation="none",
    )
    dataset = build_bayesian_dataset(traces, features, recipe=recipe)

    assert dataset.n_cells == 2
    assert dataset.n_obs >= 8
    assert dataset.ragged is True
    assert np.all(dataset.t >= 0)
    assert set(dataset.condition_levels) == {"ctrl", "drug"}


def test_build_bayesian_dataset_common_grid_interpolation():
    traces, features = _demo_tables()

    dataset = build_bayesian_dataset(
        traces,
        features,
        interpolate_to_common_grid=True,
        common_grid_size=16,
    )

    counts = dataset.long_df.groupby("cell_uid").size().to_numpy()
    assert np.unique(counts).size == 1
    assert dataset.ragged is False
    assert dataset.common_time_grid is not None
    assert dataset.common_time_grid.size == 16
