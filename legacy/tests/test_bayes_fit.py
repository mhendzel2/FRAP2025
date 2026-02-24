import pandas as pd
import pytest

from frap_bayes_data import build_bayesian_dataset
from frap_bayes_fit import (
    build_empirical_init_from_features,
    fit_hierarchical_bayes,
    merge_initializations,
)
from frap_bayes_models import PYMC_AVAILABLE


def _demo_dataset():
    traces = pd.DataFrame(
        {
            "exp_id": ["e1"] * 14,
            "movie_id": ["m1"] * 14,
            "cell_id": [1] * 7 + [2] * 7,
            "t": [0, 1, 2, 3, 4, 5, 6] * 2,
            "signal_corr": [1.0, 0.4, 0.55, 0.65, 0.74, 0.80, 0.85, 1.0, 0.45, 0.60, 0.71, 0.79, 0.83, 0.87],
            "signal_norm": [
                1.0,
                0.0,
                0.28,
                0.52,
                0.68,
                0.77,
                0.84,
                1.0,
                0.0,
                0.30,
                0.54,
                0.70,
                0.79,
                0.85,
            ],
            "radius": [2.5] * 14,
        }
    )

    features = pd.DataFrame(
        {
            "exp_id": ["e1", "e1"],
            "movie_id": ["m1", "m1"],
            "cell_id": [1, 2],
            "condition": ["ctrl", "drug"],
            "bleach_qc": [True, True],
            "k": [0.35, 0.22],
            "I0": [0.0, 0.0],
            "I_inf": [0.84, 0.86],
        }
    )

    dataset = build_bayesian_dataset(traces, features)
    return dataset, features


def test_build_empirical_init_from_features():
    dataset, features = _demo_dataset()
    init = build_empirical_init_from_features(dataset, features)
    assert "mu_logk_pop" in init
    assert "mu_logiti0_pop" in init
    assert "mu_logitrecoveryfrac_pop" in init


def test_merge_initializations_prefers_later_dict():
    merged = merge_initializations({"a": 1, "b": 2}, {"b": 9, "c": 3})
    assert merged == {"a": 1, "b": 9, "c": 3}


def test_fit_requires_pymc_when_unavailable():
    dataset, _ = _demo_dataset()

    if PYMC_AVAILABLE:
        pytest.skip("Environment has PyMC available; skip import-guard test.")

    with pytest.raises(ImportError):
        fit_hierarchical_bayes(
            dataset=dataset,
            model_name="hb_1exp",
            draws=50,
            tune=50,
            chains=2,
        )


def test_stan_pathway_is_scaffolded():
    dataset, _ = _demo_dataset()

    with pytest.raises((ImportError, NotImplementedError)):
        fit_hierarchical_bayes(
            dataset=dataset,
            model_name="hb_soumpasis",
            backend="stan",
            draws=50,
            tune=50,
            chains=2,
        )
