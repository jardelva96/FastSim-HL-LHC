import torch

from fastsim_tt4a.data import (
    DetectorGeometry,
    SyntheticShowerDataset,
    build_grid_adjacency,
    denormalize_energy,
    normalize_energy,
)


def test_adjacency_is_row_normalized() -> None:
    adj = build_grid_adjacency(DetectorGeometry(n_layers=4, cells_per_layer=8))
    row_sum = adj.sum(dim=-1)
    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-6)


def test_dataset_item_shapes() -> None:
    ds = SyntheticShowerDataset(num_events=12, seed=123)
    sample = ds[0]
    n_nodes = ds.geometry.n_nodes
    assert sample["coords"].shape == (n_nodes, 3)
    assert sample["adj"].shape == (n_nodes, n_nodes)
    assert sample["cond"].shape == (2,)
    assert sample["target"].shape == (n_nodes, 2)
    assert torch.all(sample["target"][..., 0] >= 0.0)


def test_geometry_override_changes_node_count() -> None:
    geometry = DetectorGeometry(n_layers=5, cells_per_layer=10)
    ds = SyntheticShowerDataset(num_events=20, seed=11, geometry=geometry)
    assert ds.geometry.n_nodes == 50
    assert ds[0]["target"].shape == (50, 2)


def test_energy_normalization_roundtrip() -> None:
    original = torch.tensor([0.0, 2.5, 10.0, 30.0], dtype=torch.float32)
    back = denormalize_energy(normalize_energy(original))
    assert torch.allclose(back, original, atol=1e-5)
