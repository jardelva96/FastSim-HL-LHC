from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .data import compose_encoder_input


MODEL_GRAPH_CVAE = "graph_cvae"
MODEL_MLP_AE = "mlp_ae"
SUPPORTED_MODELS = (MODEL_GRAPH_CVAE, MODEL_MLP_AE)


class GraphConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_features, out_features)
        self.lin_neigh = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        neigh = torch.matmul(adj, x)
        out = self.lin_self(x) + self.lin_neigh(neigh)
        return F.gelu(self.norm(out))


class GraphCVAE(nn.Module):
    def __init__(
        self,
        coord_dim: int = 3,
        target_dim: int = 2,
        cond_dim: int = 2,
        hidden_dim: int = 96,
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        encoder_in_dim = coord_dim + target_dim
        self.coord_dim = coord_dim
        self.target_dim = target_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        self.enc1 = GraphConvBlock(encoder_in_dim, hidden_dim)
        self.enc2 = GraphConvBlock(hidden_dim, hidden_dim)
        self.to_mu = nn.Linear(hidden_dim + cond_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim + cond_dim, latent_dim)

        decoder_in = latent_dim + cond_dim + coord_dim
        self.dec = nn.Sequential(
            nn.Linear(decoder_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def encode(
        self,
        coords: torch.Tensor,
        target: torch.Tensor,
        cond: torch.Tensor,
        adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = compose_encoder_input(coords, target)
        h = self.enc1(x, adj)
        h = self.enc2(h, adj)
        pooled = h.mean(dim=1)
        enc = torch.cat([pooled, cond], dim=-1)
        return self.to_mu(enc), self.to_logvar(enc)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, coords: torch.Tensor, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        bsz, n_nodes, _ = coords.shape
        zc = torch.cat([z, cond], dim=-1).unsqueeze(1).expand(bsz, n_nodes, -1)
        dec_in = torch.cat([zc, coords], dim=-1)
        raw = self.dec(dec_in)

        energy = F.softplus(raw[..., :1])
        time = raw[..., 1:2]
        return torch.cat([energy, time], dim=-1)

    def forward(
        self,
        coords: torch.Tensor,
        target: torch.Tensor,
        cond: torch.Tensor,
        adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(coords=coords, target=target, cond=cond, adj=adj)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(coords=coords, cond=cond, z=z)
        return recon, mu, logvar


def cvae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.mse_loss(recon, target)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - torch.exp(logvar))
    total = recon_loss + beta * kl
    return total, recon_loss.detach(), kl.detach()


class MLPConditionalAutoencoder(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        coord_dim: int = 3,
        target_dim: int = 2,
        cond_dim: int = 2,
        hidden_dim: int = 192,
        latent_dim: int = 24,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.coord_dim = coord_dim
        self.target_dim = target_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        enc_in = (n_nodes * coord_dim) + cond_dim
        self.enc = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        dec_in = latent_dim + cond_dim
        self.dec = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_nodes * target_dim),
        )

    def decode_latent(self, cond: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        raw = self.dec(torch.cat([latent, cond], dim=-1))
        raw = raw.reshape(latent.size(0), self.n_nodes, self.target_dim)

        energy = F.softplus(raw[..., :1])
        time = raw[..., 1:2]
        return torch.cat([energy, time], dim=-1)

    def forward(self, coords: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        bsz = coords.size(0)
        flat_coords = coords.reshape(bsz, -1)
        latent = self.enc(torch.cat([flat_coords, cond], dim=-1))
        return self.decode_latent(cond=cond, latent=latent)


def build_model(
    model_type: str,
    hidden_dim: int,
    latent_dim: int,
    n_nodes: int,
) -> nn.Module:
    if model_type == MODEL_GRAPH_CVAE:
        return GraphCVAE(hidden_dim=hidden_dim, latent_dim=latent_dim)
    if model_type == MODEL_MLP_AE:
        return MLPConditionalAutoencoder(
            n_nodes=n_nodes,
            hidden_dim=hidden_dim * 2,
            latent_dim=latent_dim,
        )
    raise ValueError(f"modelo nao suportado: {model_type}")


def forward_model(
    model: nn.Module,
    model_type: str,
    coords: torch.Tensor,
    target: torch.Tensor,
    cond: torch.Tensor,
    adj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if model_type == MODEL_GRAPH_CVAE:
        recon, mu, logvar = model(  # type: ignore[arg-type]
            coords=coords,
            target=target,
            cond=cond,
            adj=adj,
        )
        return recon, mu, logvar
    if model_type == MODEL_MLP_AE:
        recon = model(coords=coords, cond=cond)  # type: ignore[arg-type]
        zero = torch.zeros((coords.size(0), 1), dtype=coords.dtype, device=coords.device)
        return recon, zero, zero
    raise ValueError(f"modelo nao suportado: {model_type}")


def model_loss(
    model_type: str,
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if model_type == MODEL_GRAPH_CVAE:
        return cvae_loss(recon=recon, target=target, mu=mu, logvar=logvar, beta=beta)
    if model_type == MODEL_MLP_AE:
        recon_loss = F.mse_loss(recon, target)
        zero = torch.tensor(0.0, dtype=recon.dtype, device=recon.device)
        return recon_loss, recon_loss.detach(), zero
    raise ValueError(f"modelo nao suportado: {model_type}")


def sample_from_model(
    model: nn.Module,
    model_type: str,
    coords: torch.Tensor,
    cond: torch.Tensor,
    seed: int | None = None,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if model_type == MODEL_GRAPH_CVAE:
        if not isinstance(model, GraphCVAE):
            raise TypeError("model esperado: GraphCVAE")
        latent = torch.randn(
            (coords.size(0), model.latent_dim),
            dtype=coords.dtype,
            device=coords.device,
        )
        return model.decode(coords=coords, cond=cond, z=latent)

    if model_type == MODEL_MLP_AE:
        if not isinstance(model, MLPConditionalAutoencoder):
            raise TypeError("model esperado: MLPConditionalAutoencoder")
        latent = torch.randn(
            (coords.size(0), model.latent_dim),
            dtype=coords.dtype,
            device=coords.device,
        )
        return model.decode_latent(cond=cond, latent=latent)

    raise ValueError(f"modelo nao suportado: {model_type}")
