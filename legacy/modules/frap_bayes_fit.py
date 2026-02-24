"""
Hierarchical Bayesian fitting orchestrator for FRAP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import logging

import numpy as np
import pandas as pd

from frap_bayes_data import BayesianDataset
from frap_bayes_models import (
    HBModelConfig,
    PYMC_AVAILABLE,
    build_model,
    get_model_spec,
)
from frap_bayes_diagnostics import (
    compute_information_criteria,
    diagnostics_gate,
    ppc_summary,
)

logger = logging.getLogger(__name__)

try:
    import pymc as pm
except Exception:  # pragma: no cover - environment-dependent
    pm = None


@dataclass
class BayesianFitResult:
    """Container for Bayesian fit artifacts and summaries."""

    model_name: str
    backend: str
    dataset: BayesianDataset
    config: dict[str, Any]
    idata: Any = None
    prior_predictive: Any = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    information_criteria: dict[str, Any] = field(default_factory=dict)
    ppc_metrics: dict[str, Any] = field(default_factory=dict)
    initialization: dict[str, Any] = field(default_factory=dict)
    stan_fit: Any = None
    message: str = ""


def compile_bayesian_model(
    dataset: BayesianDataset,
    model_name: str = "hb_1exp",
    model_config: Optional[HBModelConfig | dict[str, Any]] = None,
):
    """Compile a PyMC model graph from dataset + model config."""
    config = _coerce_model_config(model_config)
    return build_model(model_name=model_name, dataset=dataset, config=config)


def fit_hierarchical_bayes(
    dataset: BayesianDataset,
    model_name: str = "hb_1exp",
    *,
    backend: str = "pymc",  # "pymc", "stan", or "auto"
    model_config: Optional[HBModelConfig | dict[str, Any]] = None,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: Optional[int] = 123,
    use_jax: bool = False,
    sample_prior_predictive_draws: int = 300,
    sample_posterior_predictive: bool = True,
    init_from_cell_features: Optional[pd.DataFrame] = None,
    init_from_global_fit: Optional[Any] = None,
) -> BayesianFitResult:
    """
    Fit a hierarchical Bayesian model and return diagnostics-ready artifacts.

    Initialization can optionally consume deterministic outputs from:
    - `frap_fitting.fit_with_model_selection` aggregated in `cell_features`
    - `frap_global_fitting` style fit result objects with `.parameters`
    """
    backend = (backend or "pymc").strip().lower()
    if backend == "auto":
        backend = "stan" if model_name == "hb_soumpasis" else "pymc"

    config = _coerce_model_config(model_config)

    if backend == "stan":
        return fit_hierarchical_bayes_stan(
            dataset=dataset,
            model_name=model_name,
            model_config=config,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
        )

    if backend != "pymc":
        raise ValueError(f"Unsupported backend: {backend}")

    if not PYMC_AVAILABLE:
        raise ImportError("PyMC is not available; cannot run NUTS sampling")

    spec = get_model_spec(model_name)
    if spec.requires_bleach_radius and "radius" not in dataset.long_df.columns:
        raise ValueError(
            "Model requires bleach radius metadata. Add `radius` per cell/time or use a phenomenological model."
        )

    model = compile_bayesian_model(dataset=dataset, model_name=model_name, model_config=config)

    empirical_init = merge_initializations(
        build_empirical_init_from_features(
            dataset,
            init_from_cell_features,
            i_max=float(config.i_max),
        ),
        build_empirical_init_from_global_fit(init_from_global_fit),
    )

    result = BayesianFitResult(
        model_name=model_name,
        backend="pymc",
        dataset=dataset,
        config={
            "mode": config.mode,
            "i_max": config.i_max,
            "alpha_upper": config.alpha_upper,
            "include_batch_effects": config.include_batch_effects,
            "noise_scale_prior": config.noise_scale_prior,
            "use_student_t": config.use_student_t,
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "target_accept": target_accept,
            "use_jax": use_jax,
        },
        initialization=empirical_init,
    )

    with model:
        sample_kwargs: dict[str, Any] = {
            "draws": int(draws),
            "tune": int(tune),
            "chains": int(chains),
            "target_accept": float(target_accept),
            "random_seed": random_seed,
            "idata_kwargs": {"log_likelihood": True},
            "return_inferencedata": True,
        }

        initvals = _filter_initvals_for_model(model, empirical_init)
        if initvals:
            sample_kwargs["initvals"] = initvals

        if use_jax:
            sample_kwargs["nuts_sampler"] = "numpyro"

        try:
            idata = pm.sample(**sample_kwargs)
        except TypeError:
            # Compatibility fallback for older PyMC versions without `nuts_sampler`.
            sample_kwargs.pop("nuts_sampler", None)
            idata = pm.sample(**sample_kwargs)

        prior = None
        if sample_prior_predictive_draws and sample_prior_predictive_draws > 0:
            try:
                prior = pm.sample_prior_predictive(
                    draws=int(sample_prior_predictive_draws),
                    random_seed=random_seed,
                )
            except TypeError:
                prior = pm.sample_prior_predictive(int(sample_prior_predictive_draws))

        if sample_posterior_predictive:
            try:
                idata = pm.sample_posterior_predictive(
                    idata,
                    random_seed=random_seed,
                    extend_inferencedata=True,
                )
            except TypeError:
                idata = pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    result.idata = idata
    result.prior_predictive = prior

    try:
        result.diagnostics = diagnostics_gate(idata)
    except Exception as exc:  # pragma: no cover
        logger.warning("Diagnostics failed: %s", exc)
        result.diagnostics = {"error": str(exc)}

    try:
        result.information_criteria = compute_information_criteria(idata)
    except Exception as exc:  # pragma: no cover
        logger.warning("Information-criteria computation failed: %s", exc)
        result.information_criteria = {"error": str(exc)}

    try:
        result.ppc_metrics = ppc_summary(idata)
    except Exception as exc:  # pragma: no cover
        logger.warning("PPC metrics failed: %s", exc)
        result.ppc_metrics = {"error": str(exc)}

    result.message = "Sampling completed"
    return result


def fit_hierarchical_bayes_stan(
    dataset: BayesianDataset,
    model_name: str = "hb_soumpasis",
    *,
    model_config: Optional[HBModelConfig | dict[str, Any]] = None,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    random_seed: Optional[int] = 123,
) -> BayesianFitResult:
    """
    Optional Stan pathway scaffold.

    This function exists to provide a stable integration point in FRAP2025.
    A complete Stan model file can be plugged in without changing call sites.
    """
    _ = dataset, model_config, draws, tune, chains, random_seed

    spec = get_model_spec(model_name)
    if not spec.supports_stan:
        raise NotImplementedError(f"Model '{model_name}' does not support Stan backend")

    try:
        import cmdstanpy  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "cmdstanpy is required for the Stan pathway. Install cmdstanpy and CmdStan first."
        ) from exc

    raise NotImplementedError(
        "Stan backend hook is available, but no compiled Stan program is bundled yet. "
        "Use PyMC backend models for now, or add a CmdStan model file and wire it here."
    )


def build_empirical_init_from_features(
    dataset: BayesianDataset,
    cell_features: Optional[pd.DataFrame],
    *,
    i_max: float = 1.2,
) -> dict[str, float]:
    """
    Build initialization values from deterministic per-cell feature table.

    Supports the most common FRAP2025 columns: `k`, `I0`, `I_inf`.
    """
    if cell_features is None or cell_features.empty:
        return {}

    features = cell_features.copy()
    feature_uid_col = _ensure_cell_uid_for_features(features)
    if feature_uid_col is None:
        return {}

    rows = features.set_index(feature_uid_col)
    matched = rows.loc[rows.index.intersection(dataset.cell_ids)]
    if matched.empty:
        return {}

    init: dict[str, float] = {}

    if "k" in matched.columns:
        k_vals = pd.to_numeric(matched["k"], errors="coerce").to_numpy(dtype=float)
        k_vals = k_vals[np.isfinite(k_vals) & (k_vals > 0)]
        if k_vals.size:
            init["mu_logk_pop"] = float(np.log(np.median(k_vals)))
            init["mu_logkoff_pop"] = float(np.log(np.median(k_vals)))
            init["mu_logkslow_pop"] = float(np.log(np.median(k_vals) * 0.5))
            init["mu_logdeltak_pop"] = float(np.log(max(np.median(k_vals) * 0.5, 1e-4)))
            init["mu_logtau_pop"] = float(np.log(1.0 / np.median(k_vals)))

    if "I0" in matched.columns:
        i0_vals = pd.to_numeric(matched["I0"], errors="coerce").to_numpy(dtype=float)
        i0_vals = i0_vals[np.isfinite(i0_vals)]
        if i0_vals.size:
            i0_frac = np.clip(i0_vals / max(i_max, 1e-6), 1e-6, 1 - 1e-6)
            init["mu_logiti0_pop"] = float(np.median(np.log(i0_frac / (1.0 - i0_frac))))

    if "I_inf" in matched.columns and "I0" in matched.columns:
        iinf_vals = pd.to_numeric(matched["I_inf"], errors="coerce").to_numpy(dtype=float)
        i0_vals = pd.to_numeric(matched["I0"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(iinf_vals) & np.isfinite(i0_vals)
        if valid.any():
            rec = (iinf_vals[valid] - i0_vals[valid]) / np.maximum(i_max - i0_vals[valid], 1e-6)
            rec = np.clip(rec, 1e-6, 1 - 1e-6)
            init["mu_logitrecoveryfrac_pop"] = float(np.median(np.log(rec / (1.0 - rec))))

    return init


def build_empirical_init_from_global_fit(global_fit_result: Optional[Any]) -> dict[str, float]:
    """Build initialization values from `frap_global_fitting` style result object."""
    if global_fit_result is None:
        return {}

    params: dict[str, Any] = {}

    if hasattr(global_fit_result, "parameters") and isinstance(global_fit_result.parameters, dict):
        params = global_fit_result.parameters
    elif isinstance(global_fit_result, dict):
        params = global_fit_result

    if not params:
        return {}

    def _extract_param(name: str) -> Optional[float]:
        if name not in params:
            return None
        value = params[name]
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        try:
            value_f = float(value)
        except Exception:
            return None
        return value_f if np.isfinite(value_f) else None

    init: dict[str, float] = {}

    k_val = _extract_param("k")
    if k_val is not None and k_val > 0:
        init["mu_logk_pop"] = float(np.log(k_val))
        init["mu_logkoff_pop"] = float(np.log(k_val))

    k_fast = _extract_param("k_fast")
    k_slow = _extract_param("k_slow")
    if k_slow is not None and k_slow > 0:
        init["mu_logkslow_pop"] = float(np.log(k_slow))
    if k_fast is not None and k_slow is not None and k_fast > k_slow > 0:
        init["mu_logdeltak_pop"] = float(np.log(k_fast - k_slow))

    return init


def merge_initializations(*parts: Optional[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        if not part:
            continue
        merged.update(part)
    return merged


def _coerce_model_config(config: Optional[HBModelConfig | dict[str, Any]]) -> HBModelConfig:
    if config is None:
        return HBModelConfig()
    if isinstance(config, HBModelConfig):
        return config
    if isinstance(config, dict):
        return HBModelConfig(**config)
    raise TypeError(f"Unsupported model_config type: {type(config)}")


def _filter_initvals_for_model(model: Any, initvals: dict[str, Any]) -> dict[str, Any]:
    if not initvals:
        return {}
    model_vars = set(getattr(model, "named_vars", {}).keys())
    return {k: v for k, v in initvals.items() if k in model_vars}


def _ensure_cell_uid_for_features(features: pd.DataFrame) -> Optional[str]:
    if "cell_uid" in features.columns:
        return "cell_uid"

    required = ["cell_id", "exp_id", "movie_id"]
    if not all(col in features.columns for col in required):
        return None

    features["cell_uid"] = (
        features["exp_id"].astype(str)
        + "::"
        + features["movie_id"].astype(str)
        + "::"
        + features["cell_id"].astype(str)
    )
    return "cell_uid"
