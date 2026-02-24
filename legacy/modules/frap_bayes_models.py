"""
Hierarchical Bayesian FRAP model library.

This module exposes a registry of forward models and PyMC builders for:
- hb_1exp
- hb_2exp_ordered
- hb_stretched
- hb_soumpasis (Stan-preferred placeholder in PyMC path)
- hb_reaction_dominant
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional
import logging

import numpy as np

from frap_bayes_data import BayesianDataset

logger = logging.getLogger(__name__)

try:
    import pymc as pm
    import pytensor.tensor as pt

    PYMC_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent
    pm = None
    pt = None
    PYMC_AVAILABLE = False


@dataclass(frozen=True)
class HBModelConfig:
    """Modeling options shared across hierarchical Bayesian builders."""

    mode: str = "deterministic"  # "deterministic" or "semi_generative"
    i_max: float = 1.2
    alpha_upper: float = 1.5
    include_batch_effects: bool = True
    noise_scale_prior: float = 0.05
    use_student_t: bool = True

    # Semi-generative nuisance priors (mode B)
    time_shift_sd: float = 0.15
    baseline_offset_sd: float = 0.03
    plateau_scale_sd: float = 0.10


@dataclass(frozen=True)
class HBModelSpec:
    name: str
    description: str
    builder: Optional[Callable[[BayesianDataset, HBModelConfig], Any]]
    forward: Callable[..., np.ndarray]
    supports_pymc: bool = True
    supports_stan: bool = False
    requires_bleach_radius: bool = False


def logit(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def inv_logit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def ordered_rate_constants(k_slow: np.ndarray, delta_k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return `(k_slow, k_fast)` with guaranteed `k_fast > k_slow`."""
    k_slow = np.asarray(k_slow, dtype=float)
    delta_k = np.asarray(delta_k, dtype=float)
    return k_slow, k_slow + np.maximum(delta_k, 1e-12)


def hb_1exp_forward(t: np.ndarray, I0: np.ndarray, I_inf: np.ndarray, k: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return I_inf - (I_inf - I0) * np.exp(-k * np.maximum(t, 0.0))


def hb_2exp_ordered_forward(
    t: np.ndarray,
    I0: np.ndarray,
    I_inf: np.ndarray,
    k_slow: np.ndarray,
    delta_k: np.ndarray,
    f_fast: np.ndarray,
) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    k_slow, k_fast = ordered_rate_constants(k_slow, delta_k)
    recover = f_fast * (1.0 - np.exp(-k_fast * np.maximum(t, 0.0)))
    recover += (1.0 - f_fast) * (1.0 - np.exp(-k_slow * np.maximum(t, 0.0)))
    return I0 + (I_inf - I0) * recover


def hb_stretched_forward(
    t: np.ndarray,
    I0: np.ndarray,
    I_inf: np.ndarray,
    tau: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    tau = np.maximum(np.asarray(tau, dtype=float), 1e-12)
    alpha = np.maximum(np.asarray(alpha, dtype=float), 1e-6)
    stretched = np.exp(-((np.maximum(t, 0.0) / tau) ** alpha))
    return I_inf - (I_inf - I0) * stretched


def hb_soumpasis_forward(
    t: np.ndarray,
    I0: np.ndarray,
    I_inf: np.ndarray,
    D: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
    """
    Soumpasis kernel forward model (numpy evaluation).

    Uses the numerically stable relationship:
    exp(-z) * (I0(z) + I1(z)) == i0e(z) + i1e(z)
    where z = r^2 / (2 D t).
    """
    from scipy.special import i0e, i1e

    t = np.maximum(np.asarray(t, dtype=float), 1e-12)
    D = np.maximum(np.asarray(D, dtype=float), 1e-12)
    radius = np.maximum(np.asarray(radius, dtype=float), 1e-12)

    z = (radius * radius) / (2.0 * D * t)
    kernel = i0e(z) + i1e(z)
    return I0 + (I_inf - I0) * kernel


def hb_reaction_dominant_forward(
    t: np.ndarray,
    I0: np.ndarray,
    I_inf: np.ndarray,
    k_off: np.ndarray,
) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return I_inf - (I_inf - I0) * np.exp(-k_off * np.maximum(t, 0.0))


def _require_pymc() -> None:
    if not PYMC_AVAILABLE:
        raise ImportError(
            "PyMC/pytensor is unavailable. Install pymc to use hierarchical Bayesian fitting."
        )


def _sanitize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _sigmoid(x: Any) -> Any:
    return 1.0 / (1.0 + pt.exp(-x))


def _build_positive_parameter(
    param_name: str,
    cond_idx_cell: Any,
    batch_idx_cell: Any,
    n_conditions: int,
    n_batches: int,
    n_cells: int,
    config: HBModelConfig,
    *,
    mu_loc: float = 0.0,
    mu_scale: float = 1.5,
    cond_scale: float = 0.7,
    sigma_cell_scale: float = 1.0,
    sigma_batch_scale: float = 0.5,
) -> Any:
    key = _sanitize_name(param_name)
    latent_name = f"log{key}"

    mu_pop = pm.Normal(f"mu_{latent_name}_pop", mu=mu_loc, sigma=mu_scale)
    delta_cond = pm.Normal(f"delta_{latent_name}_cond", mu=0.0, sigma=cond_scale, shape=n_conditions)
    delta_cond_centered = pm.Deterministic(
        f"delta_{latent_name}_cond_centered",
        delta_cond - pt.mean(delta_cond),
        dims="condition",
    )

    sigma_cell = pm.HalfNormal(f"sigma_{latent_name}_cell", sigma=sigma_cell_scale)
    z_cell = pm.Normal(f"z_{latent_name}_cell", mu=0.0, sigma=1.0, shape=n_cells)

    if config.include_batch_effects and n_batches > 1:
        sigma_batch = pm.HalfNormal(f"sigma_{latent_name}_batch", sigma=sigma_batch_scale)
        z_batch = pm.Normal(f"z_{latent_name}_batch", mu=0.0, sigma=1.0, shape=n_batches)
        batch_term = sigma_batch * z_batch[batch_idx_cell]
    else:
        batch_term = 0.0

    latent = mu_pop + delta_cond_centered[cond_idx_cell] + batch_term + sigma_cell * z_cell
    return pm.Deterministic(param_name, pt.exp(latent), dims="cell")


def _build_bounded_parameter(
    param_name: str,
    upper: float,
    cond_idx_cell: Any,
    batch_idx_cell: Any,
    n_conditions: int,
    n_batches: int,
    n_cells: int,
    config: HBModelConfig,
    *,
    mu_loc: float = 0.0,
    mu_scale: float = 1.5,
    cond_scale: float = 0.7,
    sigma_cell_scale: float = 1.0,
    sigma_batch_scale: float = 0.5,
) -> Any:
    key = _sanitize_name(param_name)
    latent_name = f"logit{key}"

    mu_pop = pm.Normal(f"mu_{latent_name}_pop", mu=mu_loc, sigma=mu_scale)
    delta_cond = pm.Normal(f"delta_{latent_name}_cond", mu=0.0, sigma=cond_scale, shape=n_conditions)
    delta_cond_centered = pm.Deterministic(
        f"delta_{latent_name}_cond_centered",
        delta_cond - pt.mean(delta_cond),
        dims="condition",
    )

    sigma_cell = pm.HalfNormal(f"sigma_{latent_name}_cell", sigma=sigma_cell_scale)
    z_cell = pm.Normal(f"z_{latent_name}_cell", mu=0.0, sigma=1.0, shape=n_cells)

    if config.include_batch_effects and n_batches > 1:
        sigma_batch = pm.HalfNormal(f"sigma_{latent_name}_batch", sigma=sigma_batch_scale)
        z_batch = pm.Normal(f"z_{latent_name}_batch", mu=0.0, sigma=1.0, shape=n_batches)
        batch_term = sigma_batch * z_batch[batch_idx_cell]
    else:
        batch_term = 0.0

    latent = mu_pop + delta_cond_centered[cond_idx_cell] + batch_term + sigma_cell * z_cell
    return pm.Deterministic(param_name, upper * _sigmoid(latent), dims="cell")


def _build_recovery_bounds(
    cond_idx_cell: Any,
    batch_idx_cell: Any,
    n_conditions: int,
    n_batches: int,
    n_cells: int,
    config: HBModelConfig,
) -> tuple[Any, Any]:
    I0 = _build_bounded_parameter(
        "I0",
        upper=config.i_max,
        cond_idx_cell=cond_idx_cell,
        batch_idx_cell=batch_idx_cell,
        n_conditions=n_conditions,
        n_batches=n_batches,
        n_cells=n_cells,
        config=config,
        mu_loc=-2.0,
    )
    recovery_frac = _build_bounded_parameter(
        "recovery_frac",
        upper=1.0,
        cond_idx_cell=cond_idx_cell,
        batch_idx_cell=batch_idx_cell,
        n_conditions=n_conditions,
        n_batches=n_batches,
        n_cells=n_cells,
        config=config,
        mu_loc=0.0,
    )
    I_inf = pm.Deterministic("I_inf", I0 + (config.i_max - I0) * recovery_frac, dims="cell")
    return I0, I_inf


def _build_effective_time(
    t_obs: Any,
    obs_cell_idx: Any,
    n_cells: int,
    config: HBModelConfig,
) -> Any:
    if config.mode != "semi_generative":
        return pt.maximum(t_obs, 0.0)

    delta_t = pm.Normal("delta_t", mu=0.0, sigma=config.time_shift_sd, shape=n_cells, dims="cell")
    return pt.maximum(t_obs + delta_t[obs_cell_idx], 0.0)


def _apply_nuisance_scaling(
    mu_raw: Any,
    obs_cell_idx: Any,
    n_cells: int,
    config: HBModelConfig,
) -> Any:
    if config.mode != "semi_generative":
        return mu_raw

    baseline_offset = pm.Normal(
        "baseline_offset",
        mu=0.0,
        sigma=config.baseline_offset_sd,
        shape=n_cells,
        dims="cell",
    )
    plateau_scale = pm.LogNormal(
        "plateau_scale",
        mu=0.0,
        sigma=config.plateau_scale_sd,
        shape=n_cells,
        dims="cell",
    )

    return baseline_offset[obs_cell_idx] + plateau_scale[obs_cell_idx] * mu_raw


def _add_likelihood(y_obs: Any, mu: Any, config: HBModelConfig) -> None:
    sigma = pm.HalfNormal("sigma", sigma=max(float(config.noise_scale_prior), 1e-6))

    if config.use_student_t:
        nu = pm.Exponential("nu_minus_1", lam=0.1) + 1.0
        pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=y_obs, dims="obs_id")
    else:
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs, dims="obs_id")


def _base_model_context(dataset: BayesianDataset) -> tuple[dict[str, Any], dict[str, Any]]:
    model_data = dataset.as_model_data()
    coords = {
        "obs_id": np.arange(dataset.n_obs, dtype=int),
        "cell": np.asarray(dataset.cell_ids, dtype=object),
        "condition": np.asarray(dataset.condition_levels, dtype=object),
        "batch": np.asarray(dataset.batch_levels, dtype=object),
    }
    return model_data, coords


def build_hb_1exp(dataset: BayesianDataset, config: Optional[HBModelConfig] = None) -> Any:
    _require_pymc()
    config = config or HBModelConfig()
    model_data, coords = _base_model_context(dataset)

    with pm.Model(coords=coords) as model:
        t_obs = pm.Data("t_obs", model_data["t"], dims="obs_id")
        y_obs = pm.Data("y_obs", model_data["y"], dims="obs_id")
        obs_cell_idx = pm.Data("obs_cell_idx", model_data["obs_cell_idx"], dims="obs_id")
        cond_idx_cell = pm.Data("cond_idx_cell", model_data["cell_condition_idx"], dims="cell")
        batch_idx_cell = pm.Data("batch_idx_cell", model_data["cell_batch_idx"], dims="cell")

        k = _build_positive_parameter(
            "k",
            cond_idx_cell,
            batch_idx_cell,
            model_data["n_conditions"],
            model_data["n_batches"],
            model_data["n_cells"],
            config,
            mu_loc=-1.0,
        )
        I0, I_inf = _build_recovery_bounds(
            cond_idx_cell,
            batch_idx_cell,
            model_data["n_conditions"],
            model_data["n_batches"],
            model_data["n_cells"],
            config,
        )

        t_eff = _build_effective_time(t_obs, obs_cell_idx, model_data["n_cells"], config)
        mu_raw = I_inf[obs_cell_idx] - (I_inf[obs_cell_idx] - I0[obs_cell_idx]) * pt.exp(-k[obs_cell_idx] * t_eff)
        mu = pm.Deterministic(
            "mu",
            _apply_nuisance_scaling(mu_raw, obs_cell_idx, model_data["n_cells"], config),
            dims="obs_id",
        )

        _add_likelihood(y_obs, mu, config)

    return model


def build_hb_2exp_ordered(dataset: BayesianDataset, config: Optional[HBModelConfig] = None) -> Any:
    _require_pymc()
    config = config or HBModelConfig()
    model_data, coords = _base_model_context(dataset)

    with pm.Model(coords=coords) as model:
        t_obs = pm.Data("t_obs", model_data["t"], dims="obs_id")
        y_obs = pm.Data("y_obs", model_data["y"], dims="obs_id")
        obs_cell_idx = pm.Data("obs_cell_idx", model_data["obs_cell_idx"], dims="obs_id")
        cond_idx_cell = pm.Data("cond_idx_cell", model_data["cell_condition_idx"], dims="cell")
        batch_idx_cell = pm.Data("batch_idx_cell", model_data["cell_batch_idx"], dims="cell")

        k_slow = _build_positive_parameter(
            "k_slow",
            cond_idx_cell,
            batch_idx_cell,
            model_data["n_conditions"],
            model_data["n_batches"],
            model_data["n_cells"],
            config,
            mu_loc=-1.3,
        )
        delta_k = _build_positive_parameter(
            "delta_k",
            cond_idx_cell,
            batch_idx_cell,
            model_data["n_conditions"],
            model_data["n_batches"],
            model_data["n_cells"],
            config,
            mu_loc=-1.6,
        )
        k_fast = pm.Deterministic("k_fast", k_slow + delta_k, dims="cell")

        f_fast = _build_bounded_parameter(
            "f_fast",
            upper=1.0,
            cond_idx_cell=cond_idx_cell,
            batch_idx_cell=batch_idx_cell,
            n_conditions=model_data["n_conditions"],
            n_batches=model_data["n_batches"],
            n_cells=model_data["n_cells"],
            config=config,
            mu_loc=0.0,
        )

        I0, I_inf = _build_recovery_bounds(
            cond_idx_cell,
            batch_idx_cell,
            model_data["n_conditions"],
            model_data["n_batches"],
            model_data["n_cells"],
            config,
        )

        t_eff = _build_effective_time(t_obs, obs_cell_idx, model_data["n_cells"], config)
        recover_fast = f_fast[obs_cell_idx] * (1.0 - pt.exp(-k_fast[obs_cell_idx] * t_eff))
        recover_slow = (1.0 - f_fast[obs_cell_idx]) * (1.0 - pt.exp(-k_slow[obs_cell_idx] * t_eff))
        mu_raw = I0[obs_cell_idx] + (I_inf[obs_cell_idx] - I0[obs_cell_idx]) * (recover_fast + recover_slow)
        mu = pm.Deterministic(
            "mu",
            _apply_nuisance_scaling(mu_raw, obs_cell_idx, model_data["n_cells"], config),
            dims="obs_id",
        )

        _add_likelihood(y_obs, mu, config)

    return model


def build_hb_stretched(dataset: BayesianDataset, config: Optional[HBModelConfig] = None) -> Any:
    _require_pymc()
    config = config or HBModelConfig()
    model_data, coords = _base_model_context(dataset)

    with pm.Model(coords=coords) as model:
        t_obs = pm.Data("t_obs", model_data["t"], dims="obs_id")
        y_obs = pm.Data("y_obs", model_data["y"], dims="obs_id")
        obs_cell_idx = pm.Data("obs_cell_idx", model_data["obs_cell_idx"], dims="obs_id")
        cond_idx_cell = pm.Data("cond_idx_cell", model_data["cell_condition_idx"], dims="cell")
        batch_idx_cell = pm.Data("batch_idx_cell", model_data["cell_batch_idx"], dims="cell")

        tau = _build_positive_parameter(
            "tau",
            cond_idx_cell,
            batch_idx_cell,
            model_data["n_conditions"],
            model_data["n_batches"],
            model_data["n_cells"],
            config,
            mu_loc=0.0,
        )

        alpha_unit = _build_bounded_parameter(
            "alpha_unit",
            upper=1.0,
            cond_idx_cell=cond_idx_cell,
            batch_idx_cell=batch_idx_cell,
            n_conditions=model_data["n_conditions"],
            n_batches=model_data["n_batches"],
            n_cells=model_data["n_cells"],
            config=config,
            mu_loc=0.5,
        )
        alpha = pm.Deterministic("alpha", 0.05 + (config.alpha_upper - 0.05) * alpha_unit, dims="cell")

        I0, I_inf = _build_recovery_bounds(
            cond_idx_cell,
            batch_idx_cell,
            model_data["n_conditions"],
            model_data["n_batches"],
            model_data["n_cells"],
            config,
        )

        t_eff = _build_effective_time(t_obs, obs_cell_idx, model_data["n_cells"], config)
        stretched = pt.exp(-((t_eff / tau[obs_cell_idx]) ** alpha[obs_cell_idx]))
        mu_raw = I_inf[obs_cell_idx] - (I_inf[obs_cell_idx] - I0[obs_cell_idx]) * stretched
        mu = pm.Deterministic(
            "mu",
            _apply_nuisance_scaling(mu_raw, obs_cell_idx, model_data["n_cells"], config),
            dims="obs_id",
        )

        _add_likelihood(y_obs, mu, config)

    return model


def build_hb_soumpasis(dataset: BayesianDataset, config: Optional[HBModelConfig] = None) -> Any:
    _require_pymc()
    _ = dataset, config
    raise NotImplementedError(
        "PyMC Soumpasis builder is intentionally disabled. Use backend='stan' "
        "for hb_soumpasis due to special-function stability and support."
    )


def build_hb_reaction_dominant(dataset: BayesianDataset, config: Optional[HBModelConfig] = None) -> Any:
    _require_pymc()
    config = config or HBModelConfig()
    model_data, coords = _base_model_context(dataset)

    with pm.Model(coords=coords) as model:
        t_obs = pm.Data("t_obs", model_data["t"], dims="obs_id")
        y_obs = pm.Data("y_obs", model_data["y"], dims="obs_id")
        obs_cell_idx = pm.Data("obs_cell_idx", model_data["obs_cell_idx"], dims="obs_id")
        cond_idx_cell = pm.Data("cond_idx_cell", model_data["cell_condition_idx"], dims="cell")
        batch_idx_cell = pm.Data("batch_idx_cell", model_data["cell_batch_idx"], dims="cell")

        k_off = _build_positive_parameter(
            "k_off",
            cond_idx_cell,
            batch_idx_cell,
            model_data["n_conditions"],
            model_data["n_batches"],
            model_data["n_cells"],
            config,
            mu_loc=-1.0,
        )

        I0, I_inf = _build_recovery_bounds(
            cond_idx_cell,
            batch_idx_cell,
            model_data["n_conditions"],
            model_data["n_batches"],
            model_data["n_cells"],
            config,
        )

        t_eff = _build_effective_time(t_obs, obs_cell_idx, model_data["n_cells"], config)
        mu_raw = I_inf[obs_cell_idx] - (I_inf[obs_cell_idx] - I0[obs_cell_idx]) * pt.exp(
            -k_off[obs_cell_idx] * t_eff
        )
        mu = pm.Deterministic(
            "mu",
            _apply_nuisance_scaling(mu_raw, obs_cell_idx, model_data["n_cells"], config),
            dims="obs_id",
        )

        _add_likelihood(y_obs, mu, config)

    return model


MODEL_REGISTRY: dict[str, HBModelSpec] = {
    "hb_1exp": HBModelSpec(
        name="hb_1exp",
        description="Hierarchical single-exponential recovery model",
        builder=build_hb_1exp,
        forward=hb_1exp_forward,
        supports_pymc=True,
        supports_stan=True,
    ),
    "hb_2exp_ordered": HBModelSpec(
        name="hb_2exp_ordered",
        description="Hierarchical ordered two-exponential recovery model",
        builder=build_hb_2exp_ordered,
        forward=hb_2exp_ordered_forward,
        supports_pymc=True,
        supports_stan=True,
    ),
    "hb_stretched": HBModelSpec(
        name="hb_stretched",
        description="Hierarchical stretched-exponential recovery model",
        builder=build_hb_stretched,
        forward=hb_stretched_forward,
        supports_pymc=True,
        supports_stan=True,
    ),
    "hb_soumpasis": HBModelSpec(
        name="hb_soumpasis",
        description="Soumpasis diffusion model (Stan-preferred)",
        builder=build_hb_soumpasis,
        forward=hb_soumpasis_forward,
        supports_pymc=False,
        supports_stan=True,
        requires_bleach_radius=True,
    ),
    "hb_reaction_dominant": HBModelSpec(
        name="hb_reaction_dominant",
        description="Hierarchical reaction-dominant binding proxy model",
        builder=build_hb_reaction_dominant,
        forward=hb_reaction_dominant_forward,
        supports_pymc=True,
        supports_stan=True,
    ),
}


def get_model_spec(model_name: str) -> HBModelSpec:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown Bayesian model: {model_name}")
    return MODEL_REGISTRY[model_name]


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def build_model(
    model_name: str,
    dataset: BayesianDataset,
    config: Optional[HBModelConfig] = None,
) -> Any:
    spec = get_model_spec(model_name)
    if not spec.supports_pymc:
        raise NotImplementedError(
            f"Model '{model_name}' is not enabled on the PyMC backend."
        )
    if spec.builder is None:
        raise RuntimeError(f"Model '{model_name}' has no registered builder")
    return spec.builder(dataset, config or HBModelConfig())
