"""Physics validation and analysis utilities.

Generates detailed physics reports including closure studies, longitudinal
layer profiles, pileup-dependent resolution curves, 2-D energy/time maps
and conditioned event sampling.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import (
    SimulationConfig,
    SyntheticShowerDataset,
    denormalize_energy,
    denormalize_time,
    node_coordinates,
)
from .evaluate import build_model_from_checkpoint, load_checkpoint, resolve_geometry_from_config
from .metrics import aggregate_reconstruction_metrics, reconstruction_tensors
from .model import forward_model, sample_from_model


def _default_pileup_bins() -> list[float]:
    """Standard pileup bin edges for resolution profiling."""
    return [0.0, 40.0, 80.0, 120.0, 160.0, 200.0]


def _to_profile_table(
    pileup: torch.Tensor,
    rel_energy_error: torch.Tensor,
    bins: Sequence[float],
) -> list[dict[str, float | str]]:
    """Bin relative energy errors by pileup level."""
    profile: list[dict[str, float | str]] = []
    for left, right in zip(bins[:-1], bins[1:], strict=True):
        mask = (pileup >= left) & (pileup < right)
        if mask.sum().item() == 0:
            profile.append(
                {
                    "bin": f"[{left:.0f},{right:.0f})",
                    "count": 0,
                    "bias": 0.0,
                    "resolution_rms": 0.0,
                    "abs_bias_p90": 0.0,
                }
            )
            continue
        bin_err = rel_energy_error[mask]
        profile.append(
            {
                "bin": f"[{left:.0f},{right:.0f})",
                "count": int(mask.sum().item()),
                "bias": float(bin_err.mean().item()),
                "resolution_rms": float(torch.sqrt(torch.mean(bin_err**2)).item()),
                "abs_bias_p90": float(torch.quantile(torch.abs(bin_err), 0.90).item()),
            }
        )
    return profile


@torch.no_grad()
def evaluate_physics_report(
    checkpoint_path: Path | str,
    num_events: int = 1200,
    batch_size: int = 128,
    seed: int = 99,
    device: str = "cpu",
    n_layers_override: int = 0,
    cells_override: int = 0,
    pileup_bins: Sequence[float] | None = None,
) -> dict[str, object]:
    """Run a comprehensive physics validation and return the full report.

    The report includes global reconstruction metrics, energy closure,
    longitudinal layer profiles, pileup-resolved resolution curves and
    averaged 2-D energy/time maps.
    """
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    ckpt_config = dict(checkpoint.get("config", {}))
    geometry = resolve_geometry_from_config(
        ckpt_config,
        n_layers_override=n_layers_override,
        cells_override=cells_override,
    )
    model, model_config = build_model_from_checkpoint(
        checkpoint=checkpoint,
        n_nodes=geometry.n_nodes,
        device=device,
    )
    dataset = SyntheticShowerDataset(num_events=num_events, seed=seed, geometry=geometry)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sim = SimulationConfig()
    pileup_scale = sim.pileup_max

    device_obj = torch.device(device)
    mse_values: list[torch.Tensor] = []
    rel_energy_errors: list[torch.Tensor] = []
    energy_mae_values: list[torch.Tensor] = []
    time_mae_values: list[torch.Tensor] = []
    pileup_values: list[torch.Tensor] = []
    closure_values: list[torch.Tensor] = []

    layer_true_sum = torch.zeros(geometry.n_layers, dtype=torch.float32)
    layer_pred_sum = torch.zeros(geometry.n_layers, dtype=torch.float32)
    layer_abs_err_sum = torch.zeros(geometry.n_layers, dtype=torch.float32)

    true_energy_map_sum = torch.zeros(
        (geometry.n_layers, geometry.cells_per_layer),
        dtype=torch.float32,
    )
    pred_energy_map_sum = torch.zeros(
        (geometry.n_layers, geometry.cells_per_layer),
        dtype=torch.float32,
    )
    true_time_map_sum = torch.zeros(
        (geometry.n_layers, geometry.cells_per_layer),
        dtype=torch.float32,
    )
    pred_time_map_sum = torch.zeros(
        (geometry.n_layers, geometry.cells_per_layer),
        dtype=torch.float32,
    )
    total_events = 0

    for batch in loader:
        batch = {k: v.to(device_obj) for k, v in batch.items()}
        recon, _, _ = forward_model(
            model=model,
            model_type=str(model_config["model_type"]),
            coords=batch["coords"],
            target=batch["target"],
            cond=batch["cond"],
            adj=batch["adj"],
        )

        mse, rel_error, energy_mae, time_mae = reconstruction_tensors(
            recon=recon,
            target=batch["target"],
        )
        mse_values.append(mse.cpu())
        rel_energy_errors.append(rel_error.cpu())
        energy_mae_values.append(energy_mae.cpu())
        time_mae_values.append(time_mae.cpu())
        pileup_values.append((batch["cond"][:, 1] * pileup_scale).detach().cpu())

        true_energy_nodes = denormalize_energy(batch["target"][..., 0]).detach().cpu()
        pred_energy_nodes = denormalize_energy(recon[..., 0]).detach().cpu()
        true_time_nodes = denormalize_time(batch["target"][..., 1]).detach().cpu()
        pred_time_nodes = denormalize_time(recon[..., 1]).detach().cpu()

        bsz = true_energy_nodes.size(0)
        true_energy_map = true_energy_nodes.reshape(
            bsz,
            geometry.n_layers,
            geometry.cells_per_layer,
        )
        pred_energy_map = pred_energy_nodes.reshape(
            bsz,
            geometry.n_layers,
            geometry.cells_per_layer,
        )
        true_time_map = true_time_nodes.reshape(bsz, geometry.n_layers, geometry.cells_per_layer)
        pred_time_map = pred_time_nodes.reshape(bsz, geometry.n_layers, geometry.cells_per_layer)

        layer_true = true_energy_map.sum(dim=2)
        layer_pred = pred_energy_map.sum(dim=2)
        layer_abs = torch.abs(layer_pred - layer_true)
        layer_true_sum += layer_true.sum(dim=0)
        layer_pred_sum += layer_pred.sum(dim=0)
        layer_abs_err_sum += layer_abs.sum(dim=0)

        true_energy_map_sum += true_energy_map.sum(dim=0)
        pred_energy_map_sum += pred_energy_map.sum(dim=0)
        true_time_map_sum += true_time_map.sum(dim=0)
        pred_time_map_sum += pred_time_map.sum(dim=0)
        total_events += bsz

        closure_values.append(
            (
                pred_energy_nodes.sum(dim=1)
                / true_energy_nodes.sum(dim=1).clamp_min(1e-6)
            ).cpu()
        )

    mse_all = torch.cat(mse_values)
    rel_all = torch.cat(rel_energy_errors)
    energy_mae_all = torch.cat(energy_mae_values)
    time_mae_all = torch.cat(time_mae_values)
    pileup_all = torch.cat(pileup_values)
    closure_all = torch.cat(closure_values)

    global_metrics = aggregate_reconstruction_metrics(
        mse=mse_all,
        rel_energy_error=rel_all,
        energy_mae=energy_mae_all,
        time_mae=time_mae_all,
    )
    closure = {
        "closure_mean": float(closure_all.mean().item()),
        "closure_std": float(closure_all.std(unbiased=False).item()),
        "closure_p95_abs_dev": float(torch.quantile(torch.abs(closure_all - 1.0), 0.95).item()),
    }

    layer_profile = []
    for layer in range(geometry.n_layers):
        layer_profile.append(
            {
                "layer": layer,
                "true_energy_mean": float(layer_true_sum[layer].item() / max(total_events, 1)),
                "pred_energy_mean": float(layer_pred_sum[layer].item() / max(total_events, 1)),
                "abs_error_mean": float(layer_abs_err_sum[layer].item() / max(total_events, 1)),
            }
        )

    bins = list(pileup_bins) if pileup_bins is not None else _default_pileup_bins()
    pileup_profile = _to_profile_table(pileup_all, rel_all, bins=bins)

    true_energy_map_mean = true_energy_map_sum / max(total_events, 1)
    pred_energy_map_mean = pred_energy_map_sum / max(total_events, 1)
    abs_energy_err_map = torch.abs(pred_energy_map_mean - true_energy_map_mean)
    true_time_map_mean = true_time_map_sum / max(total_events, 1)
    pred_time_map_mean = pred_time_map_sum / max(total_events, 1)

    return {
        "global_metrics": global_metrics,
        "closure": closure,
        "layer_profile": layer_profile,
        "pileup_profile": pileup_profile,
        "true_energy_map_mean": true_energy_map_mean.tolist(),
        "pred_energy_map_mean": pred_energy_map_mean.tolist(),
        "abs_energy_error_map_mean": abs_energy_err_map.tolist(),
        "true_time_map_mean": true_time_map_mean.tolist(),
        "pred_time_map_mean": pred_time_map_mean.tolist(),
        "model_type": str(model_config["model_type"]),
        "events": int(total_events),
        "checkpoint_path": str(Path(checkpoint_path)),
    }


@torch.no_grad()
def generate_conditioned_samples(
    checkpoint_path: Path | str,
    beam_energy: float,
    pileup: float,
    num_samples: int = 6,
    seed: int = 13,
    device: str = "cpu",
    n_layers_override: int = 0,
    cells_override: int = 0,
) -> dict[str, object]:
    """Generate synthetic events conditioned on beam energy and pileup.

    Returns mean and standard-deviation maps across samples for both
    energy and time.
    """
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    ckpt_config = dict(checkpoint.get("config", {}))
    geometry = resolve_geometry_from_config(
        ckpt_config,
        n_layers_override=n_layers_override,
        cells_override=cells_override,
    )
    model, model_config = build_model_from_checkpoint(
        checkpoint=checkpoint,
        n_nodes=geometry.n_nodes,
        device=device,
    )
    sim = SimulationConfig()
    beam_norm = max(min(beam_energy / sim.beam_energy_max, 1.0), 0.0)
    pileup_norm = max(min(pileup / sim.pileup_max, 1.0), 0.0)

    device_obj = torch.device(device)
    coords = node_coordinates(geometry).unsqueeze(0).repeat(num_samples, 1, 1).to(device_obj)
    cond = torch.tensor(
        [[beam_norm, pileup_norm]] * num_samples,
        dtype=torch.float32,
        device=device_obj,
    )
    generated = sample_from_model(
        model=model,
        model_type=str(model_config["model_type"]),
        coords=coords,
        cond=cond,
        seed=seed,
    )

    energy_map = denormalize_energy(generated[..., 0]).detach().cpu()
    time_map = denormalize_time(generated[..., 1]).detach().cpu()

    energy_map = energy_map.reshape(num_samples, geometry.n_layers, geometry.cells_per_layer)
    time_map = time_map.reshape(num_samples, geometry.n_layers, geometry.cells_per_layer)

    return {
        "beam_energy": float(beam_energy),
        "pileup": float(pileup),
        "model_type": str(model_config["model_type"]),
        "energy_mean_map": energy_map.mean(dim=0).tolist(),
        "energy_std_map": energy_map.std(dim=0, unbiased=False).tolist(),
        "time_mean_map": time_map.mean(dim=0).tolist(),
        "time_std_map": time_map.std(dim=0, unbiased=False).tolist(),
        "n_samples": int(num_samples),
        "checkpoint_path": str(Path(checkpoint_path)),
    }
