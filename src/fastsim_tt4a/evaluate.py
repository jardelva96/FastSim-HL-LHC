from __future__ import annotations

import argparse
from pathlib import Path
import json

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import DetectorGeometry, SyntheticShowerDataset
from .metrics import aggregate_reconstruction_metrics, reconstruction_tensors
from .model import MODEL_GRAPH_CVAE, SUPPORTED_MODELS, build_model, forward_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avaliacao de checkpoint do CVAE em grafo.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-events", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--n-layers", type=int, default=0)
    parser.add_argument("--cells-per-layer", type=int, default=0)
    return parser.parse_args()


def load_checkpoint(ckpt_path: Path | str, map_location: str = "cpu") -> dict[str, object]:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint nao encontrado: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint invalido: chave model_state_dict ausente.")
    return checkpoint


def resolve_geometry_from_config(
    config: dict[str, object],
    n_layers_override: int = 0,
    cells_override: int = 0,
) -> DetectorGeometry:
    n_layers = int(config.get("n_layers", 6))
    cells_per_layer = int(config.get("cells_per_layer", 16))
    if n_layers_override > 0:
        n_layers = n_layers_override
    if cells_override > 0:
        cells_per_layer = cells_override
    return DetectorGeometry(n_layers=n_layers, cells_per_layer=cells_per_layer)


def build_model_from_checkpoint(
    checkpoint: dict[str, object],
    n_nodes: int,
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, object]]:
    config = dict(checkpoint.get("config", {}))
    model_type = str(config.get("model_type", MODEL_GRAPH_CVAE))
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"model_type no checkpoint nao suportado: {model_type}")

    model = build_model(
        model_type=model_type,
        hidden_dim=int(config.get("hidden_dim", 96)),
        latent_dim=int(config.get("latent_dim", 16)),
        n_nodes=n_nodes,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitado, mas nao ha GPU disponivel.")
    model = model.to(torch.device(device))
    model.eval()
    config["model_type"] = model_type
    return model, config


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path | str,
    num_events: int = 1000,
    batch_size: int = 128,
    seed: int = 99,
    device: str = "cpu",
    n_layers_override: int = 0,
    cells_override: int = 0,
) -> dict[str, float]:
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    ckpt_config = dict(ckpt.get("config", {}))
    geometry = resolve_geometry_from_config(
        ckpt_config,
        n_layers_override=n_layers_override,
        cells_override=cells_override,
    )
    model, config = build_model_from_checkpoint(
        checkpoint=ckpt,
        n_nodes=geometry.n_nodes,
        device=device,
    )
    dataset = SyntheticShowerDataset(num_events=num_events, seed=seed, geometry=geometry)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device_obj = torch.device(device)
    mse_values = []
    rel_energy_errors = []
    energy_mae_values = []
    time_mae_values = []

    for batch in loader:
        batch = {k: v.to(device_obj) for k, v in batch.items()}
        recon, _, _ = forward_model(
            model=model,
            model_type=str(config["model_type"]),
            coords=batch["coords"],
            target=batch["target"],
            cond=batch["cond"],
            adj=batch["adj"],
        )
        mse, rel_error, energy_mae, time_mae = reconstruction_tensors(
            recon=recon, target=batch["target"]
        )
        mse_values.append(mse)
        rel_energy_errors.append(rel_error)
        energy_mae_values.append(energy_mae)
        time_mae_values.append(time_mae)

    metrics = aggregate_reconstruction_metrics(
        mse=torch.cat(mse_values),
        rel_energy_error=torch.cat(rel_energy_errors),
        energy_mae=torch.cat(energy_mae_values),
        time_mae=torch.cat(time_mae_values),
    )
    return metrics


@torch.no_grad()
def main() -> None:
    args = parse_args()
    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        num_events=args.num_events,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        n_layers_override=args.n_layers,
        cells_override=args.cells_per_layer,
    )

    print(json.dumps(metrics, indent=2))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)


if __name__ == "__main__":
    main()
