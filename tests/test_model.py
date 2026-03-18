import torch
from torch.utils.data import DataLoader

from fastsim_tt4a.data import SyntheticShowerDataset
from fastsim_tt4a.model import (
    MODEL_GRAPH_CVAE,
    MODEL_MLP_AE,
    GraphCVAE,
    build_model,
    cvae_loss,
    forward_model,
    model_loss,
    sample_from_model,
)


def test_forward_shapes_and_loss_finite() -> None:
    ds = SyntheticShowerDataset(num_events=16, seed=7)
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))

    model = GraphCVAE(hidden_dim=32, latent_dim=8)
    recon, mu, logvar = model(
        coords=batch["coords"],
        target=batch["target"],
        cond=batch["cond"],
        adj=batch["adj"],
    )
    loss, recon_loss, kl = cvae_loss(recon, batch["target"], mu, logvar)

    assert recon.shape == batch["target"].shape
    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)
    assert torch.isfinite(loss)
    assert torch.isfinite(recon_loss)
    assert torch.isfinite(kl)


def test_mlp_forward_and_loss() -> None:
    ds = SyntheticShowerDataset(num_events=10, seed=4)
    batch = next(iter(DataLoader(ds, batch_size=2, shuffle=False)))
    model = build_model(
        model_type=MODEL_MLP_AE,
        hidden_dim=32,
        latent_dim=8,
        n_nodes=ds.geometry.n_nodes,
    )
    recon, mu, logvar = forward_model(
        model=model,
        model_type=MODEL_MLP_AE,
        coords=batch["coords"],
        target=batch["target"],
        cond=batch["cond"],
        adj=batch["adj"],
    )
    loss, recon_loss, kl = model_loss(
        model_type=MODEL_MLP_AE,
        recon=recon,
        target=batch["target"],
        mu=mu,
        logvar=logvar,
        beta=1e-3,
    )

    assert recon.shape == batch["target"].shape
    assert mu.shape == (2, 1)
    assert logvar.shape == (2, 1)
    assert torch.isfinite(loss)
    assert torch.isfinite(recon_loss)
    assert kl.item() == 0.0


def test_build_graph_model_from_factory() -> None:
    ds = SyntheticShowerDataset(num_events=8, seed=2)
    model = build_model(
        model_type=MODEL_GRAPH_CVAE,
        hidden_dim=32,
        latent_dim=8,
        n_nodes=ds.geometry.n_nodes,
    )
    assert isinstance(model, GraphCVAE)


def test_sampling_from_models_is_finite() -> None:
    ds = SyntheticShowerDataset(num_events=6, seed=5)
    batch = next(iter(DataLoader(ds, batch_size=3, shuffle=False)))

    graph_model = build_model(
        model_type=MODEL_GRAPH_CVAE,
        hidden_dim=32,
        latent_dim=8,
        n_nodes=ds.geometry.n_nodes,
    )
    graph_sample = sample_from_model(
        model=graph_model,
        model_type=MODEL_GRAPH_CVAE,
        coords=batch["coords"],
        cond=batch["cond"],
        seed=123,
    )
    assert graph_sample.shape == batch["target"].shape
    assert torch.isfinite(graph_sample).all()

    mlp_model = build_model(
        model_type=MODEL_MLP_AE,
        hidden_dim=32,
        latent_dim=8,
        n_nodes=ds.geometry.n_nodes,
    )
    mlp_sample = sample_from_model(
        model=mlp_model,
        model_type=MODEL_MLP_AE,
        coords=batch["coords"],
        cond=batch["cond"],
        seed=123,
    )
    assert mlp_sample.shape == batch["target"].shape
    assert torch.isfinite(mlp_sample).all()
