"""Neural network architectures for fast calorimeter simulation.

Two conditional generative models are provided:

* **GraphCVAE** -- Graph-convolutional conditional variational autoencoder
  that exploits the spatial structure of the detector via message passing.
* **MLPConditionalAutoencoder** -- Simpler MLP baseline that flattens the
  graph topology for comparison purposes.

A factory function :func:`build_model` and dispatch helpers
(:func:`forward_model`, :func:`model_loss`, :func:`sample_from_model`)
allow switching between architectures with a single string flag.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .data import compose_encoder_input

MODEL_GRAPH_CVAE = "graph_cvae"
MODEL_MLP_AE = "mlp_ae"
SUPPORTED_MODELS = (MODEL_GRAPH_CVAE, MODEL_MLP_AE)


class GraphConvBlock(nn.Module):
    """Single graph-convolution layer with layer-norm and GELU activation.

    Implements a simplified message-passing step where each node aggregates
    features from its neighbours (via the adjacency matrix) and from itself
    through separate linear projections.
    """

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
    """Graph-convolutional Conditional Variational Autoencoder.

    The encoder applies two :class:`GraphConvBlock` layers followed by
    global mean-pooling.  The pooled representation, concatenated with
    the condition vector, is projected into a variational latent space
    (mu, logvar).  The decoder broadcasts the latent + condition back to
    every node and reconstructs per-cell energy and time targets via an
    MLP with dropout regularisation.

    Parameters
    ----------
    coord_dim : int
        Dimensionality of node coordinates (default 3).
    target_dim : int
        Number of target features per node (default 2: energy, time).
    cond_dim : int
        Dimensionality of the condition vector (default 2: beam energy, pileup).
    hidden_dim : int
        Width of hidden layers.
    latent_dim : int
        Dimensionality of the VAE latent space.
    dropout : float
        Dropout probability applied in the decoder (default 0.08).
    """

    def __init__(
        self,
        coord_dim: int = 3,
        target_dim: int = 2,
        cond_dim: int = 2,
        hidden_dim: int = 96,
        latent_dim: int = 16,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        encoder_in_dim = coord_dim + target_dim
        self.coord_dim = coord_dim
        self.target_dim = target_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        # Encoder -- two graph convolution blocks.
        self.enc1 = GraphConvBlock(encoder_in_dim, hidden_dim)
        self.enc2 = GraphConvBlock(hidden_dim, hidden_dim)
        self.to_mu = nn.Linear(hidden_dim + cond_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim + cond_dim, latent_dim)

        # Decoder -- MLP with dropout for regularisation.
        decoder_in = latent_dim + cond_dim + coord_dim
        self.dec = nn.Sequential(
            nn.Linear(decoder_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialisation for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(
        self,
        coords: torch.Tensor,
        target: torch.Tensor,
        cond: torch.Tensor,
        adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode graph nodes into a global latent distribution."""
        x = compose_encoder_input(coords, target)
        h = self.enc1(x, adj)
        h = self.enc2(h, adj)
        pooled = h.mean(dim=1)
        enc = torch.cat([pooled, cond], dim=-1)
        return self.to_mu(enc), self.to_logvar(enc)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from the latent distribution using the reparameterisation trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, coords: torch.Tensor, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector back to per-node energy and time predictions."""
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
        """Full forward pass: encode -> sample -> decode."""
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
    """Compute beta-VAE loss (reconstruction MSE + beta * KL divergence)."""
    recon_loss = F.mse_loss(recon, target)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - torch.exp(logvar))
    total = recon_loss + beta * kl
    return total, recon_loss.detach(), kl.detach()


class MLPConditionalAutoencoder(nn.Module):
    """Flat MLP conditional autoencoder (baseline).

    Flattens node coordinates, concatenates with the condition vector,
    and reconstructs per-node targets through symmetric encoder/decoder
    MLPs.  Used as a baseline to highlight the benefit of exploiting
    graph structure in :class:`GraphCVAE`.

    Parameters
    ----------
    n_nodes : int
        Number of detector nodes.
    coord_dim, target_dim, cond_dim : int
        Feature dimensions (see :class:`GraphCVAE`).
    hidden_dim : int
        Width of hidden layers (doubled relative to GraphCVAE default).
    latent_dim : int
        Bottleneck dimensionality.
    dropout : float
        Dropout probability (default 0.08).
    """

    def __init__(
        self,
        n_nodes: int,
        coord_dim: int = 3,
        target_dim: int = 2,
        cond_dim: int = 2,
        hidden_dim: int = 192,
        latent_dim: int = 24,
        dropout: float = 0.08,
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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        dec_in = latent_dim + cond_dim
        self.dec = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_nodes * target_dim),
        )

    def decode_latent(self, cond: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to per-node predictions."""
        raw = self.dec(torch.cat([latent, cond], dim=-1))
        raw = raw.reshape(latent.size(0), self.n_nodes, self.target_dim)

        energy = F.softplus(raw[..., :1])
        time = raw[..., 1:2]
        return torch.cat([energy, time], dim=-1)

    def forward(self, coords: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Encode coordinates + condition and reconstruct targets."""
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
    """Instantiate a model by type string.

    Raises :class:`ValueError` for unsupported *model_type*.
    """
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
    """Dispatch a forward pass to the correct model interface.

    Returns ``(recon, mu, logvar)`` for all model types.  For models
    without a variational bottleneck *mu* and *logvar* are zero tensors.
    """
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
    """Compute the appropriate loss for a given *model_type*."""
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
    """Draw unconditional samples from the generative model's latent space."""
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
