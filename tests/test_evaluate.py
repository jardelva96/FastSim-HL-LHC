"""Evaluation and checkpoint loading tests."""

from pathlib import Path

import pytest

from fastsim_tt4a.data import DetectorGeometry
from fastsim_tt4a.evaluate import (
    build_model_from_checkpoint,
    evaluate_checkpoint,
    load_checkpoint,
    resolve_geometry_from_config,
)
from fastsim_tt4a.train import TrainingConfig, run_training


@pytest.fixture()
def trained_checkpoint(tmp_path: Path) -> Path:
    """Train a tiny model and return the checkpoint path."""
    config = TrainingConfig(
        num_events=30,
        epochs=2,
        batch_size=10,
        hidden_dim=16,
        latent_dim=4,
        seed=7,
        out_dir=str(tmp_path),
        n_layers=3,
        cells_per_layer=8,
    )
    result = run_training(config)
    return Path(result["checkpoint_path"])


def test_load_checkpoint(trained_checkpoint: Path) -> None:
    """load_checkpoint returns a dict with model_state_dict."""
    ckpt = load_checkpoint(trained_checkpoint)
    assert "model_state_dict" in ckpt
    assert "config" in ckpt


def test_load_checkpoint_missing_file() -> None:
    """load_checkpoint raises FileNotFoundError for missing paths."""
    with pytest.raises(FileNotFoundError):
        load_checkpoint("non_existent_model.pt")


def test_resolve_geometry_from_config() -> None:
    """Geometry resolution respects overrides."""
    config = {"n_layers": 6, "cells_per_layer": 16}
    geo = resolve_geometry_from_config(config)
    assert geo.n_layers == 6
    assert geo.cells_per_layer == 16

    geo_override = resolve_geometry_from_config(config, n_layers_override=4, cells_override=10)
    assert geo_override.n_layers == 4
    assert geo_override.cells_per_layer == 10


def test_build_model_from_checkpoint(trained_checkpoint: Path) -> None:
    """Model loaded from checkpoint is in eval mode."""
    ckpt = load_checkpoint(trained_checkpoint)
    geometry = DetectorGeometry(n_layers=3, cells_per_layer=8)
    model, config = build_model_from_checkpoint(ckpt, n_nodes=geometry.n_nodes)
    assert not model.training
    assert "model_type" in config


def test_evaluate_checkpoint_metrics(trained_checkpoint: Path) -> None:
    """evaluate_checkpoint returns all expected metric keys."""
    metrics = evaluate_checkpoint(
        checkpoint_path=trained_checkpoint,
        num_events=20,
        batch_size=10,
        seed=99,
    )
    expected_keys = {
        "mse_mean",
        "mse_std",
        "energy_bias_mean",
        "energy_resolution_rms",
        "energy_abs_bias_p95",
        "energy_mae_mean",
        "time_mae_mean_ns",
    }
    assert set(metrics.keys()) == expected_keys
    for v in metrics.values():
        assert isinstance(v, float)
