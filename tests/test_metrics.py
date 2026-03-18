import torch

from fastsim_tt4a.metrics import aggregate_reconstruction_metrics, reconstruction_tensors


def test_metrics_shapes_and_keys() -> None:
    recon = torch.tensor(
        [
            [[0.10, 0.00], [0.15, 0.05]],
            [[0.12, -0.02], [0.11, 0.02]],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [[0.09, 0.01], [0.16, 0.03]],
            [[0.11, -0.01], [0.10, 0.01]],
        ],
        dtype=torch.float32,
    )

    mse, rel_energy_error, energy_mae, time_mae = reconstruction_tensors(recon, target)
    metrics = aggregate_reconstruction_metrics(mse, rel_energy_error, energy_mae, time_mae)

    assert mse.shape == (2,)
    assert rel_energy_error.shape == (2,)
    assert energy_mae.shape == (2,)
    assert time_mae.shape == (2,)
    assert set(metrics) == {
        "mse_mean",
        "mse_std",
        "energy_bias_mean",
        "energy_resolution_rms",
        "energy_abs_bias_p95",
        "energy_mae_mean",
        "time_mae_mean_ns",
    }

