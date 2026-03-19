"""Training pipeline for fast-simulation models.

Provides a fully configurable training loop with early stopping, learning
rate scheduling, gradient clipping and reproducible seeding.  Training
artefacts (checkpoint, history, config, summary) are written to the output
directory for later evaluation and analysis.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .data import DetectorGeometry, SyntheticShowerDataset
from .model import SUPPORTED_MODELS, build_model, forward_model, model_loss

ProgressCallback = Callable[[int, int, dict[str, float]], None]


@dataclass(frozen=True)
class TrainingConfig:
    """All hyper-parameters and runtime options for a training run.

    This dataclass is serialised to ``train_config.json`` alongside
    the model checkpoint so that every experiment is fully reproducible.
    """

    num_events: int = 5000
    epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    hidden_dim: int = 96
    latent_dim: int = 16
    beta: float = 1e-3
    seed: int = 7
    out_dir: str = "artifacts"
    val_split: float = 0.2
    patience: int = 6
    min_delta: float = 1e-4
    grad_clip: float = 1.0
    device: str = "auto"
    num_workers: int = 0
    n_layers: int = 6
    cells_per_layer: int = 16
    model_type: str = "graph_cvae"


def set_seed(seed: int) -> None:
    """Set random seeds across all libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Treino de CVAE em grafo para simulacao rapida.")
    parser.add_argument("--num-events", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=str, default="artifacts")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--cells-per-layer", type=int, default=16)
    parser.add_argument("--model-type", type=str, default="graph_cvae", choices=SUPPORTED_MODELS)
    return parser.parse_args()


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move all tensors in a batch dictionary to *device*."""
    return {k: v.to(device) for k, v in batch.items()}


def resolve_device(device_name: str) -> torch.device:
    """Resolve a device name string into a :class:`torch.device`."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitado, mas nao ha GPU disponivel.")
    return torch.device(device_name)


def train_epoch(
    model: nn.Module,
    model_type: str,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    beta: float,
    device: torch.device,
    grad_clip: float,
) -> tuple[float, float, float]:
    """Run one training epoch and return average (loss, recon, kl)."""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad()
        recon, mu, logvar = forward_model(
            model=model,
            model_type=model_type,
            coords=batch["coords"],
            target=batch["target"],
            cond=batch["cond"],
            adj=batch["adj"],
        )
        loss, recon_loss, kl = model_loss(
            model_type=model_type,
            recon=recon,
            target=batch["target"],
            mu=mu,
            logvar=logvar,
            beta=beta,
        )
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl.item()

    steps = max(len(loader), 1)
    return total_loss / steps, total_recon / steps, total_kl / steps


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    model_type: str,
    loader: DataLoader,
    beta: float,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run one validation epoch and return average (loss, recon, kl)."""
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for batch in loader:
        batch = move_batch(batch, device)
        recon, mu, logvar = forward_model(
            model=model,
            model_type=model_type,
            coords=batch["coords"],
            target=batch["target"],
            cond=batch["cond"],
            adj=batch["adj"],
        )
        loss, recon_loss, kl = model_loss(
            model_type=model_type,
            recon=recon,
            target=batch["target"],
            mu=mu,
            logvar=logvar,
            beta=beta,
        )
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl.item()

    steps = max(len(loader), 1)
    return total_loss / steps, total_recon / steps, total_kl / steps


def run_training(
    config: TrainingConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    """Execute a full training run described by *config*.

    Returns a dictionary with paths to all generated artefacts and the
    training summary.
    """
    set_seed(config.seed)

    if not (0.0 < config.val_split < 1.0):
        raise ValueError("val_split deve estar entre 0 e 1.")
    if config.num_events < 10:
        raise ValueError("num_events deve ser >= 10 para treino/validacao confiavel.")
    if config.model_type not in SUPPORTED_MODELS:
        raise ValueError(f"model_type invalido: {config.model_type}")

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geometry = DetectorGeometry(
        n_layers=config.n_layers,
        cells_per_layer=config.cells_per_layer,
    )
    dataset = SyntheticShowerDataset(
        num_events=config.num_events,
        seed=config.seed,
        geometry=geometry,
    )
    train_size = int(len(dataset) * (1.0 - config.val_split))
    train_size = min(max(train_size, 1), len(dataset) - 1)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.seed)
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    device = resolve_device(config.device)
    model = build_model(
        model_type=config.model_type,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        n_nodes=dataset.geometry.n_nodes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    history = []
    best_val = float("inf")
    best_path = out_dir / "model.pt"
    best_epoch = 0
    bad_epochs = 0

    for epoch in tqdm(range(1, config.epochs + 1), desc="epochs"):
        tr_loss, tr_recon, tr_kl = train_epoch(
            model=model,
            model_type=config.model_type,
            loader=train_loader,
            optimizer=optimizer,
            beta=config.beta,
            device=device,
            grad_clip=config.grad_clip,
        )
        va_loss, va_recon, va_kl = eval_epoch(
            model=model,
            model_type=config.model_type,
            loader=val_loader,
            beta=config.beta,
            device=device,
        )
        lr = float(optimizer.param_groups[0]["lr"])
        epoch_info = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_recon": tr_recon,
            "train_kl": tr_kl,
            "val_loss": va_loss,
            "val_recon": va_recon,
            "val_kl": va_kl,
            "lr": lr,
        }
        history.append(epoch_info)
        print(
            f"epoch={epoch:02d} train={tr_loss:.5f} val={va_loss:.5f} "
            f"(recon={va_recon:.5f}, kl={va_kl:.5f}, lr={lr:.2e})"
        )
        if progress_callback is not None:
            progress_callback(epoch, config.epochs, epoch_info)

        improved = va_loss < (best_val - config.min_delta)
        if improved:
            best_val = va_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "hidden_dim": config.hidden_dim,
                        "latent_dim": config.latent_dim,
                        "beta": config.beta,
                        "seed": config.seed,
                        "n_layers": config.n_layers,
                        "cells_per_layer": config.cells_per_layer,
                        "model_type": config.model_type,
                    },
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                print(
                    f"early stopping no epoch {epoch} (sem melhora por {config.patience} epocas)."
                )
                break
        scheduler.step(va_loss)

    history_path = out_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    config_path = out_dir / "train_config.json"
    with config_path.open("w", encoding="utf-8") as fp:
        json.dump(asdict(config), fp, indent=2)

    summary = {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "checkpoint_path": str(best_path),
        "device": str(device),
        "model_type": config.model_type,
        "params": int(sum(p.numel() for p in model.parameters())),
    }
    summary_path = out_dir / "train_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"checkpoint salvo em: {best_path}")
    print(f"melhor val_loss: {best_val:.6f}")
    return {
        "history": history,
        "summary": summary,
        "history_path": str(history_path),
        "summary_path": str(summary_path),
        "checkpoint_path": str(best_path),
    }


def main() -> None:
    """CLI entry-point for ``fastsim-train``."""
    args = parse_args()
    config = TrainingConfig(
        num_events=args.num_events,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        beta=args.beta,
        seed=args.seed,
        out_dir=args.out_dir,
        val_split=args.val_split,
        patience=args.patience,
        min_delta=args.min_delta,
        grad_clip=args.grad_clip,
        device=args.device,
        num_workers=args.num_workers,
        n_layers=args.n_layers,
        cells_per_layer=args.cells_per_layer,
        model_type=args.model_type,
    )
    run_training(config)


if __name__ == "__main__":
    main()
