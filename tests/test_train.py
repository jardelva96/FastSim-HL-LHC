"""End-to-end training pipeline tests."""

import json
from pathlib import Path

import torch

from fastsim_tt4a.train import TrainingConfig, resolve_device, run_training, set_seed


def test_run_training_end_to_end(tmp_path: Path) -> None:
    """A short training run produces all expected artefacts."""
    config = TrainingConfig(
        num_events=40,
        epochs=3,
        batch_size=8,
        hidden_dim=16,
        latent_dim=4,
        seed=42,
        out_dir=str(tmp_path),
        model_type="graph_cvae",
        n_layers=3,
        cells_per_layer=8,
    )
    result = run_training(config)

    # Check artefact files exist.
    assert Path(result["checkpoint_path"]).exists()
    assert Path(result["history_path"]).exists()
    assert Path(result["summary_path"]).exists()
    assert (tmp_path / "train_config.json").exists()

    # Summary should contain expected keys.
    summary = result["summary"]
    assert "best_val_loss" in summary
    assert "best_epoch" in summary
    assert "params" in summary
    assert summary["model_type"] == "graph_cvae"

    # History should have one entry per epoch (or fewer with early stopping).
    history = result["history"]
    assert 1 <= len(history) <= 3

    # Checkpoint should be loadable.
    ckpt = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "config" in ckpt


def test_run_training_mlp(tmp_path: Path) -> None:
    """MLP baseline trains and produces valid artefacts."""
    config = TrainingConfig(
        num_events=30,
        epochs=2,
        batch_size=10,
        hidden_dim=16,
        latent_dim=4,
        seed=7,
        out_dir=str(tmp_path),
        model_type="mlp_ae",
        n_layers=3,
        cells_per_layer=8,
    )
    result = run_training(config)
    assert Path(result["checkpoint_path"]).exists()
    summary = result["summary"]
    assert summary["model_type"] == "mlp_ae"
    assert summary["params"] > 0


def test_set_seed_reproducibility() -> None:
    """set_seed produces deterministic random values."""
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.allclose(a, b)


def test_resolve_device_cpu() -> None:
    """resolve_device always returns cpu when asked."""
    device = resolve_device("cpu")
    assert device == torch.device("cpu")


def test_resolve_device_auto() -> None:
    """resolve_device('auto') returns a valid device."""
    device = resolve_device("auto")
    assert device.type in ("cpu", "cuda")


def test_training_config_serialises(tmp_path: Path) -> None:
    """TrainingConfig should round-trip through JSON."""
    config = TrainingConfig(
        num_events=50,
        epochs=2,
        batch_size=8,
        hidden_dim=16,
        latent_dim=4,
        seed=1,
        out_dir=str(tmp_path),
    )
    run_training(config)
    loaded = json.loads((tmp_path / "train_config.json").read_text(encoding="utf-8"))
    assert loaded["num_events"] == 50
    assert loaded["model_type"] == "graph_cvae"
