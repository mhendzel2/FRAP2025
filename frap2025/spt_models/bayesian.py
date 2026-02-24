"""Bayesian single-particle trajectory analyzers.

This module provides an HDP-HMM-style analyzer for non-parametric state
inference on SPT trajectories with graceful degradation if Pyro is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

_HAS_PYRO = importlib.util.find_spec("pyro") is not None and importlib.util.find_spec("torch") is not None


@dataclass(frozen=True)
class HDPHMMResult:
    """Result object for HDP-HMM analysis."""

    optimal_num_states: int
    state_sequence: np.ndarray
    transition_matrix: np.ndarray
    state_weights: np.ndarray
    backend: str

    def to_dict(self) -> dict[str, object]:
        return {
            "optimal_num_states": int(self.optimal_num_states),
            "state_sequence": self.state_sequence.astype(int),
            "transition_matrix": self.transition_matrix.astype(float),
            "state_weights": self.state_weights.astype(float),
            "backend": self.backend,
        }


class HDPHMM_Analyzer:
    """Analyze SPT trajectories with a non-parametric HDP-HMM approximation.

    Parameters
    ----------
    max_states : int, default=8
        Truncation level for the stick-breaking approximation to the Dirichlet
        Process over hidden states.
    concentration_gamma : float, default=5.0
        Concentration parameter controlling global state richness in the
        stick-breaking prior.
    concentration_alpha : float, default=2.0
        Concentration for row-wise transition distributions around the global
        state weights.
    sticky_kappa : float, default=8.0
        Self-transition bias (sticky HDP-HMM term).
    num_steps : int, default=800
        Number of SVI optimization steps for the Pyro backend.
    lr : float, default=5e-3
        Learning rate for variational optimization.

    Notes
    -----
    The truncated stick-breaking representation uses:

    ``beta_k = v_k * prod_{j<k}(1-v_j)``, with ``v_k ~ Beta(1, gamma)``.

    Transition rows are modeled as:

    ``pi_k ~ Dirichlet(alpha * beta + kappa * e_k)``.

    Emissions are diagonal Gaussians over observations derived from trajectory
    increments (step length and step angle). If Pyro is unavailable, the class
    falls back to a BIC-selected Gaussian mixture with empirical transition
    estimation.
    """

    def __init__(
        self,
        *,
        max_states: int = 8,
        concentration_gamma: float = 5.0,
        concentration_alpha: float = 2.0,
        sticky_kappa: float = 8.0,
        num_steps: int = 800,
        lr: float = 5e-3,
        random_state: int = 42,
    ) -> None:
        if max_states < 2:
            raise ValueError("max_states must be >= 2")
        self.max_states = int(max_states)
        self.concentration_gamma = float(concentration_gamma)
        self.concentration_alpha = float(concentration_alpha)
        self.sticky_kappa = float(sticky_kappa)
        self.num_steps = int(num_steps)
        self.lr = float(lr)
        self.random_state = int(random_state)

    def analyze(self, trajectory: pd.DataFrame | np.ndarray) -> dict[str, object]:
        """Infer latent dynamic states from an SPT trajectory.

        Parameters
        ----------
        trajectory : pandas.DataFrame or numpy.ndarray
            Either:
            1) position trajectory ``(N,2)`` in ``x,y`` coordinates, or
            2) feature trajectory containing step descriptors. For DataFrames,
               preferred columns are ``step_length`` and ``step_angle``.

        Returns
        -------
        dict
            Dictionary with inferred ``optimal_num_states``, ``state_sequence``,
            ``transition_matrix``, ``state_weights``, and ``backend``.
        """
        observations = self._trajectory_to_observations(trajectory)
        if observations.shape[0] < 5:
            raise ValueError("Need at least 5 observations for HDPHMM analysis.")

        if _HAS_PYRO:
            try:
                return self._analyze_with_pyro(observations).to_dict()
            except Exception as exc:  # pragma: no cover - backend dependent
                warnings.warn(
                    f"Pyro HDP-HMM backend failed ({exc}); falling back to Gaussian-mixture approximation.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        warnings.warn(
            "Pyro is unavailable. Using Gaussian-mixture fallback for state inference.",
            RuntimeWarning,
            stacklevel=2,
        )
        return self._analyze_fallback(observations).to_dict()

    def _trajectory_to_observations(self, trajectory: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(trajectory, pd.DataFrame):
            frame = trajectory.copy()
            if {"step_length", "step_angle"}.issubset(frame.columns):
                obs = frame.loc[:, ["step_length", "step_angle"]].to_numpy(dtype=float)
                obs = obs[np.all(np.isfinite(obs), axis=1)]
                return obs

            if {"x", "y"}.issubset(frame.columns):
                xy = frame.loc[:, ["x", "y"]].to_numpy(dtype=float)
            else:
                if frame.shape[1] < 2:
                    raise ValueError("DataFrame trajectory must include x/y or step_length/step_angle columns.")
                xy = frame.iloc[:, :2].to_numpy(dtype=float)
        else:
            xy = np.asarray(trajectory, dtype=float)
            if xy.ndim != 2 or xy.shape[1] < 2:
                raise ValueError("trajectory ndarray must have shape (N,2+) for positions.")
            xy = xy[:, :2]

        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if xy.shape[0] < 3:
            raise ValueError("Position trajectory must have at least 3 finite points.")

        dxy = np.diff(xy, axis=0)
        step_length = np.linalg.norm(dxy, axis=1)
        step_angle = np.arctan2(dxy[:, 1], dxy[:, 0])
        obs = np.column_stack([step_length, step_angle])
        obs = obs[np.all(np.isfinite(obs), axis=1)]
        return obs

    def _analyze_fallback(self, observations: np.ndarray) -> HDPHMMResult:
        best_model: GaussianMixture | None = None
        best_bic = np.inf

        max_k = min(self.max_states, max(2, observations.shape[0] // 3))
        for n_components in range(1, max_k + 1):
            model = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                random_state=self.random_state,
                n_init=5,
            )
            model.fit(observations)
            bic = model.bic(observations)
            if bic < best_bic:
                best_bic = bic
                best_model = model

        if best_model is None:
            raise RuntimeError("Fallback state inference failed to fit Gaussian mixture model.")

        state_sequence = best_model.predict(observations)
        n_states = int(getattr(best_model, "n_components"))

        transition = self._estimate_transition_matrix(state_sequence, n_states)
        weights = np.bincount(state_sequence, minlength=n_states).astype(float)
        weights /= np.sum(weights)

        return HDPHMMResult(
            optimal_num_states=n_states,
            state_sequence=state_sequence,
            transition_matrix=transition,
            state_weights=weights,
            backend="gmm_fallback",
        )

    def _analyze_with_pyro(self, observations: np.ndarray) -> HDPHMMResult:
        if not _HAS_PYRO:
            raise RuntimeError("Pyro backend requested but pyro/torch imports are unavailable.")

        torch = importlib.import_module("torch")
        pyro = importlib.import_module("pyro")
        dist = importlib.import_module("pyro.distributions")
        infer_mod = importlib.import_module("pyro.infer")
        optim_mod = importlib.import_module("pyro.optim")

        SVI = getattr(infer_mod, "SVI")
        TraceEnum_ELBO = getattr(infer_mod, "TraceEnum_ELBO")
        Adam = getattr(optim_mod, "Adam")

        pyro.clear_param_store()
        pyro.set_rng_seed(self.random_state)

        x = torch.tensor(observations, dtype=torch.float32)
        T, dim = x.shape
        K = self.max_states

        alpha = self.concentration_alpha
        gamma = self.concentration_gamma
        kappa = self.sticky_kappa

        def stick_breaking(v: Any) -> Any:
            one = torch.tensor(1.0, dtype=v.dtype, device=v.device)
            pieces = [v[0:1]]
            prod = one - v[0:1]
            for idx in range(1, v.shape[0]):
                pieces.append(v[idx : idx + 1] * prod)
                prod = prod * (one - v[idx : idx + 1])
            pieces.append(prod)
            beta = torch.cat(pieces)
            beta = beta / beta.sum()
            return beta

        def model(obs: Any) -> None:
            v = pyro.sample("v", dist.Beta(torch.ones(K - 1), torch.full((K - 1,), gamma)).to_event(1))
            beta = stick_breaking(v)

            pi_rows = []
            for k in range(K):
                base = alpha * beta
                base = base + kappa * torch.nn.functional.one_hot(torch.tensor(k), num_classes=K).float()
                pi_k = pyro.sample(f"pi_{k}", dist.Dirichlet(base))
                pi_rows.append(pi_k)
            pi = torch.stack(pi_rows, dim=0)

            with pyro.plate("states", K):
                loc = pyro.sample("loc", dist.Normal(torch.zeros(dim), torch.ones(dim) * 3.0).to_event(1))
                scale = pyro.sample("scale", dist.LogNormal(torch.zeros(dim), torch.ones(dim) * 0.5).to_event(1))

            z_prev = pyro.sample("z_0", dist.Categorical(beta), infer={"enumerate": "parallel"})
            for t in pyro.markov(range(T)):
                z_t = pyro.sample(f"z_{t}", dist.Categorical(pi[z_prev]), infer={"enumerate": "parallel"})
                pyro.sample(f"obs_{t}", dist.Normal(loc[z_t], scale[z_t]).to_event(1), obs=obs[t])
                z_prev = z_t

        def guide(obs: Any) -> None:
            v_a = pyro.param("v_a", torch.ones(K - 1), constraint=dist.constraints.positive)
            v_b = pyro.param("v_b", torch.ones(K - 1), constraint=dist.constraints.positive)
            pyro.sample("v", dist.Beta(v_a, v_b).to_event(1))

            for k in range(K):
                q_pi = pyro.param(
                    f"q_pi_{k}",
                    torch.ones(K),
                    constraint=dist.constraints.positive,
                )
                pyro.sample(f"pi_{k}", dist.Dirichlet(q_pi))

            loc_mu = pyro.param("loc_mu", torch.zeros(K, dim))
            loc_sigma = pyro.param(
                "loc_sigma",
                torch.ones(K, dim),
                constraint=dist.constraints.positive,
            )
            scale_mu = pyro.param("scale_mu", torch.zeros(K, dim))
            scale_sigma = pyro.param(
                "scale_sigma",
                torch.ones(K, dim),
                constraint=dist.constraints.positive,
            )

            with pyro.plate("states", K):
                pyro.sample("loc", dist.Normal(loc_mu, loc_sigma).to_event(1))
                pyro.sample("scale", dist.LogNormal(scale_mu, scale_sigma).to_event(1))

        svi = SVI(model, guide, Adam({"lr": self.lr}), loss=TraceEnum_ELBO(max_plate_nesting=1))
        for _ in range(self.num_steps):
            svi.step(x)

        q_pi_rows = []
        for k in range(K):
            q_pi = pyro.param(f"q_pi_{k}").detach().cpu().numpy()
            q_pi_rows.append(q_pi / np.sum(q_pi))
        transition = np.vstack(q_pi_rows)

        v_a = pyro.param("v_a").detach()
        v_b = pyro.param("v_b").detach()
        v_mean = v_a / (v_a + v_b)
        beta = self._stick_breaking_numpy(v_mean.cpu().numpy())

        loc = pyro.param("loc_mu").detach().cpu().numpy()
        scale = np.exp(pyro.param("scale_mu").detach().cpu().numpy())

        state_sequence = self._viterbi_decode(observations, transition, loc, scale, beta)

        state_counts = np.bincount(state_sequence, minlength=K)
        active = state_counts > 0
        n_active = int(np.sum(active))
        if n_active == 0:
            n_active = 1
            active[0] = True

        active_idx = np.where(active)[0]
        remap = {old: new for new, old in enumerate(active_idx)}
        remapped_states = np.array([remap[s] for s in state_sequence], dtype=int)

        transition_active = transition[np.ix_(active_idx, active_idx)]
        transition_active = transition_active / np.clip(transition_active.sum(axis=1, keepdims=True), 1e-12, None)

        weights_active = state_counts[active_idx].astype(float)
        weights_active /= np.sum(weights_active)

        return HDPHMMResult(
            optimal_num_states=n_active,
            state_sequence=remapped_states,
            transition_matrix=transition_active,
            state_weights=weights_active,
            backend="pyro_hdp_hmm",
        )

    @staticmethod
    def _stick_breaking_numpy(v: np.ndarray) -> np.ndarray:
        parts: list[float] = []
        prod = 1.0
        for val in v:
            parts.append(float(val * prod))
            prod *= float(1.0 - val)
        parts.append(prod)
        beta = np.asarray(parts, dtype=float)
        beta /= np.sum(beta)
        return beta

    @staticmethod
    def _estimate_transition_matrix(state_sequence: np.ndarray, n_states: int, pseudocount: float = 1.0) -> np.ndarray:
        counts = np.full((n_states, n_states), pseudocount, dtype=float)
        if state_sequence.size > 1:
            for prev_state, next_state in zip(state_sequence[:-1], state_sequence[1:]):
                counts[int(prev_state), int(next_state)] += 1.0
        counts /= np.sum(counts, axis=1, keepdims=True)
        return counts

    @staticmethod
    def _viterbi_decode(
        obs: np.ndarray,
        transition: np.ndarray,
        loc: np.ndarray,
        scale: np.ndarray,
        init_prob: np.ndarray,
    ) -> np.ndarray:
        T = obs.shape[0]
        K = transition.shape[0]

        scale_safe = np.clip(scale, 1e-6, None)
        log_trans = np.log(np.clip(transition, 1e-12, None))
        log_init = np.log(np.clip(init_prob, 1e-12, None))

        log_emit = np.empty((T, K), dtype=float)
        for k in range(K):
            diff = (obs - loc[k]) / scale_safe[k]
            log_det = np.sum(np.log(scale_safe[k]))
            quad = 0.5 * np.sum(diff * diff, axis=1)
            log_emit[:, k] = -0.5 * obs.shape[1] * np.log(2.0 * np.pi) - log_det - quad

        dp = np.full((T, K), -np.inf, dtype=float)
        ptr = np.zeros((T, K), dtype=int)

        dp[0] = log_init + log_emit[0]
        for t in range(1, T):
            scores = dp[t - 1][:, None] + log_trans
            ptr[t] = np.argmax(scores, axis=0)
            dp[t] = scores[ptr[t], np.arange(K)] + log_emit[t]

        states = np.zeros(T, dtype=int)
        states[-1] = int(np.argmax(dp[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = ptr[t + 1, states[t + 1]]
        return states


__all__ = ["HDPHMM_Analyzer", "HDPHMMResult"]
