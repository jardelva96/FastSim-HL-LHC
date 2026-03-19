"""Reconstruction quality metrics for fast-simulation evaluation.

All metrics operate on batched tensors of shape ``(batch, n_nodes, 2)``
where channel 0 is normalised energy and channel 1 is normalised time.
"""

from __future__ import annotations

import torch

from .data import denormalize_energy, denormalize_time


def reconstruction_tensors(
    recon: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-event reconstruction quality tensors.

    Returns
    -------
    mse : Tensor (batch,)
        Mean squared error across all nodes and features.
    rel_energy_error : Tensor (batch,)
        Relative total-energy error ``(pred - true) / true``.
    energy_mae : Tensor (batch,)
        Mean absolute error of per-node energy deposits (GeV).
    time_mae : Tensor (batch,)
        Mean absolute error of per-node timing (ns).
    """
    mse = torch.mean((recon - target) ** 2, dim=(1, 2))

    true_energy_nodes = denormalize_energy(target[..., 0])
    pred_energy_nodes = denormalize_energy(recon[..., 0])
    true_energy = true_energy_nodes.sum(dim=1)
    pred_energy = pred_energy_nodes.sum(dim=1)
    rel_energy_error = (pred_energy - true_energy) / true_energy.clamp_min(1e-6)
    energy_mae = torch.mean(torch.abs(pred_energy_nodes - true_energy_nodes), dim=1)

    true_time = denormalize_time(target[..., 1])
    pred_time = denormalize_time(recon[..., 1])
    time_mae = torch.mean(torch.abs(pred_time - true_time), dim=1)
    return mse, rel_energy_error, energy_mae, time_mae


def aggregate_reconstruction_metrics(
    mse: torch.Tensor,
    rel_energy_error: torch.Tensor,
    energy_mae: torch.Tensor,
    time_mae: torch.Tensor,
) -> dict[str, float]:
    """Aggregate per-event tensors into scalar summary metrics.

    Returns a dictionary suitable for JSON serialisation.
    """
    abs_rel = torch.abs(rel_energy_error)
    return {
        "mse_mean": float(mse.mean().item()),
        "mse_std": float(mse.std(unbiased=False).item()),
        "energy_bias_mean": float(rel_energy_error.mean().item()),
        "energy_resolution_rms": float(torch.sqrt(torch.mean(rel_energy_error**2)).item()),
        "energy_abs_bias_p95": float(torch.quantile(abs_rel, 0.95).item()),
        "energy_mae_mean": float(energy_mae.mean().item()),
        "time_mae_mean_ns": float(time_mae.mean().item()),
    }
