"""Physics analysis and conditioned generation tests."""

from pathlib import Path

import pytest

from fastsim_tt4a.analysis import evaluate_physics_report, generate_conditioned_samples
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


def test_physics_report_structure(trained_checkpoint: Path) -> None:
    """evaluate_physics_report returns a complete report dict."""
    report = evaluate_physics_report(
        checkpoint_path=trained_checkpoint,
        num_events=20,
        batch_size=10,
        seed=99,
    )
    assert "global_metrics" in report
    assert "closure" in report
    assert "layer_profile" in report
    assert "pileup_profile" in report
    assert "true_energy_map_mean" in report
    assert "pred_energy_map_mean" in report
    assert "model_type" in report
    assert report["events"] == 20

    # Closure should be close to 1.0 on average.
    closure = report["closure"]
    assert "closure_mean" in closure
    assert "closure_std" in closure

    # Layer profile should have one entry per layer.
    layer_profile = report["layer_profile"]
    assert len(layer_profile) == 3  # n_layers=3 in fixture


def test_conditioned_generation(trained_checkpoint: Path) -> None:
    """generate_conditioned_samples returns valid energy/time maps."""
    result = generate_conditioned_samples(
        checkpoint_path=trained_checkpoint,
        beam_energy=100.0,
        pileup=50.0,
        num_samples=4,
        seed=42,
    )
    assert result["beam_energy"] == 100.0
    assert result["pileup"] == 50.0
    assert result["n_samples"] == 4
    assert "energy_mean_map" in result
    assert "energy_std_map" in result
    assert "time_mean_map" in result
    assert "time_std_map" in result

    # Maps should have shape (n_layers, cells_per_layer).
    energy_map = result["energy_mean_map"]
    assert len(energy_map) == 3  # n_layers=3
    assert len(energy_map[0]) == 8  # cells_per_layer=8
