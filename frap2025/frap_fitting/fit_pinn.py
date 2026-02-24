# pyright: reportMissingImports=false, reportGeneralTypeIssues=false, reportRedeclaration=false
"""Physics-Informed Neural Network solver for FRAP image stacks."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from typing import Any, cast
import warnings

import numpy as np
import pandas as pd

_HAS_TORCH = importlib.util.find_spec("torch") is not None

if _HAS_TORCH:
    _torch: Any = importlib.import_module("torch")
    _nn: Any = importlib.import_module("torch.nn")
else:
    _torch = cast(Any, None)
    _nn = cast(Any, None)


@dataclass(frozen=True)
class FRAPPINNFitResult:
    D: float
    k_on: float
    k_off: float
    final_loss: float
    data_loss: float
    pde_loss: float
    backend: str
    epochs: int

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "D": float(self.D),
            "k_on": float(self.k_on),
            "k_off": float(self.k_off),
            "final_loss": float(self.final_loss),
            "data_loss": float(self.data_loss),
            "pde_loss": float(self.pde_loss),
            "backend": self.backend,
            "epochs": int(self.epochs),
        }


if _HAS_TORCH and _torch is not None and _nn is not None:

    class FRAP_PINN(_nn.Module):
        """PINN network mapping ``(x,y,t) -> I`` and learning ``D``, ``k_on``, ``k_off``.

        Notes
        -----
        The PDE residual used in training is

        ``dI/dt - D * (d2I/dx2 + d2I/dy2) + k_on * I_unbleached - k_off * I = 0``.
        """

        def __init__(self, hidden_dim: int = 64, depth: int = 4) -> None:
            super().__init__()
            layers: list[Any] = [_nn.Linear(3, hidden_dim), _nn.Tanh()]
            for _ in range(depth - 1):
                layers.extend([_nn.Linear(hidden_dim, hidden_dim), _nn.Tanh()])
            layers.append(_nn.Linear(hidden_dim, 1))
            self.network = _nn.Sequential(*layers)

            self.log_D = _nn.Parameter(_torch.tensor(-3.0))
            self.log_k_on = _nn.Parameter(_torch.tensor(-1.0))
            self.log_k_off = _nn.Parameter(_torch.tensor(-1.0))

        @property
        def D(self):
            return _torch.exp(self.log_D)

        @property
        def k_on(self):
            return _torch.exp(self.log_k_on)

        @property
        def k_off(self):
            return _torch.exp(self.log_k_off)

        def forward(self, coords):
            return self.network(coords)

else:

    class FRAP_PINN:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("FRAP_PINN requires torch. Install optional dependency group [advanced].")


def _prepare_pinn_data(image_stack: np.ndarray, mask: np.ndarray, bleach_roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.asarray(image_stack, dtype=float)
    if stack.ndim != 3:
        raise ValueError("image_stack must have shape (T, H, W).")

    roi_mask = np.asarray(mask, dtype=bool)
    bleach_mask = np.asarray(bleach_roi, dtype=bool)
    if roi_mask.shape != stack.shape[1:] or bleach_mask.shape != stack.shape[1:]:
        raise ValueError("mask and bleach_roi must match image spatial shape (H, W).")

    yy, xx = np.where(roi_mask)
    if yy.size == 0:
        raise ValueError("mask contains no active pixels.")

    t_size, height, width = stack.shape
    x_norm = xx / max(width - 1, 1)
    y_norm = yy / max(height - 1, 1)

    unbleached_mask = roi_mask & (~bleach_mask)
    if not np.any(unbleached_mask):
        unbleached_mask = roi_mask

    coords_blocks: list[np.ndarray] = []
    intensity_blocks: list[np.ndarray] = []
    unbleached_blocks: list[np.ndarray] = []

    for t in range(t_size):
        t_norm = np.full_like(x_norm, t / max(t_size - 1, 1), dtype=float)
        coords_blocks.append(np.column_stack([x_norm, y_norm, t_norm]))
        intensity_values = stack[t, yy, xx]
        intensity_blocks.append(intensity_values)
        i_unbleached_t = float(np.nanmean(stack[t, unbleached_mask]))
        unbleached_blocks.append(np.full(intensity_values.shape, i_unbleached_t, dtype=float))

    coords = np.vstack(coords_blocks)
    intensities = np.concatenate(intensity_blocks)
    i_unbleached = np.concatenate(unbleached_blocks)

    valid = np.isfinite(coords).all(axis=1) & np.isfinite(intensities) & np.isfinite(i_unbleached)
    return coords[valid], intensities[valid], i_unbleached[valid]


def _fit_frap_classical_fallback(image_stack: np.ndarray, mask: np.ndarray, bleach_roi: np.ndarray) -> dict[str, float | str | int]:
    from frap2025.core.frap_fitting import fit_reaction_diffusion

    stack = np.asarray(image_stack, dtype=float)
    roi_mask = np.asarray(mask, dtype=bool)
    bleach_mask = np.asarray(bleach_roi, dtype=bool)

    target = roi_mask & bleach_mask
    if not np.any(target):
        target = roi_mask

    trace = np.nanmean(stack[:, target], axis=1)
    t = np.arange(stack.shape[0], dtype=float)
    fit = fit_reaction_diffusion(t, trace)
    component = fit.components[0] if fit.components else {}

    return FRAPPINNFitResult(
        D=float(component.get("k_D_s", np.nan)),
        k_on=float(component.get("k_on_s", np.nan)),
        k_off=float(component.get("k_off_s", np.nan)),
        final_loss=np.nan,
        data_loss=np.nan,
        pde_loss=np.nan,
        backend="classical_reaction_diffusion_fallback",
        epochs=0,
    ).to_dict()


def fit_frap_pinn(
    image_stack: np.ndarray | pd.DataFrame,
    mask: np.ndarray,
    bleach_roi: np.ndarray,
    *,
    epochs: int = 500,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    depth: int = 4,
    pde_weight: float = 1.0,
    data_weight: float = 1.0,
    device: str | None = None,
) -> dict[str, float | str | int]:
    """Fit FRAP dynamics with a PINN or a classical fallback.

    Uses the composite loss

    ``L = data_weight * MSE(I_pred, I_obs) + pde_weight * MSE(residual, 0)``.
    """
    if isinstance(image_stack, pd.DataFrame):
        required_cols = {"t", "y", "x", "intensity"}
        if not required_cols.issubset(image_stack.columns):
            raise ValueError("DataFrame image_stack must include columns: t, y, x, intensity")

        t_idx = image_stack["t"].to_numpy(dtype=int)
        y_idx = image_stack["y"].to_numpy(dtype=int)
        x_idx = image_stack["x"].to_numpy(dtype=int)
        values = image_stack["intensity"].to_numpy(dtype=float)

        dense = np.full((int(t_idx.max()) + 1, int(y_idx.max()) + 1, int(x_idx.max()) + 1), np.nan, dtype=float)
        dense[t_idx, y_idx, x_idx] = values
        stack_array = dense
    else:
        stack_array = np.asarray(image_stack, dtype=float)

    if not _HAS_TORCH or _torch is None:
        warnings.warn(
            "torch is unavailable. Falling back to classical reaction-diffusion FRAP fitting.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _fit_frap_classical_fallback(stack_array, mask, bleach_roi)

    coords_np, intensity_np, unbleached_np = _prepare_pinn_data(stack_array, mask, bleach_roi)

    dev = _torch.device(device) if device is not None else _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    coords = _torch.tensor(coords_np, dtype=_torch.float32, device=dev, requires_grad=True)
    target = _torch.tensor(intensity_np.reshape(-1, 1), dtype=_torch.float32, device=dev)
    i_unbleached = _torch.tensor(unbleached_np.reshape(-1, 1), dtype=_torch.float32, device=dev)

    model: Any = FRAP_PINN(hidden_dim=hidden_dim, depth=depth)
    model = model.to(dev)
    optimizer = _torch.optim.Adam(model.parameters(), lr=learning_rate)

    final_total = np.nan
    final_data = np.nan
    final_pde = np.nan

    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)

        pred = model(coords)
        data_loss = _torch.mean((pred - target) ** 2)

        grad_pred = _torch.autograd.grad(
            outputs=pred,
            inputs=coords,
            grad_outputs=_torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
        )[0]

        dI_dx = grad_pred[:, 0:1]
        dI_dy = grad_pred[:, 1:2]
        dI_dt = grad_pred[:, 2:3]

        d2I_dx2 = _torch.autograd.grad(
            outputs=dI_dx,
            inputs=coords,
            grad_outputs=_torch.ones_like(dI_dx),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]

        d2I_dy2 = _torch.autograd.grad(
            outputs=dI_dy,
            inputs=coords,
            grad_outputs=_torch.ones_like(dI_dy),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]

        laplacian = d2I_dx2 + d2I_dy2
        residual = dI_dt - model.D * laplacian + model.k_on * i_unbleached - model.k_off * pred
        pde_loss = _torch.mean(residual**2)

        total_loss = data_weight * data_loss + pde_weight * pde_loss
        total_loss.backward()
        optimizer.step()

        final_total = float(total_loss.detach().cpu().item())
        final_data = float(data_loss.detach().cpu().item())
        final_pde = float(pde_loss.detach().cpu().item())

    return FRAPPINNFitResult(
        D=float(model.D.detach().cpu().item()),
        k_on=float(model.k_on.detach().cpu().item()),
        k_off=float(model.k_off.detach().cpu().item()),
        final_loss=final_total,
        data_loss=final_data,
        pde_loss=final_pde,
        backend="torch_pinn",
        epochs=int(epochs),
    ).to_dict()


__all__ = ["FRAP_PINN", "FRAPPINNFitResult", "fit_frap_pinn"]
