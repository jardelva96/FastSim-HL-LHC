"""Detector geometry, synthetic shower generation and normalisation utilities.

This module provides the building blocks for creating synthetic calorimeter
datasets used to train and evaluate fast-simulation models.  It defines
a configurable detector geometry, physics-motivated simulation parameters,
graph adjacency construction and energy/time normalisation transforms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DetectorGeometry:
    """Simplified cylindrical calorimeter geometry.

    Parameters
    ----------
    n_layers : int
        Number of radial detector layers (must be >= 1).
    cells_per_layer : int
        Azimuthal granularity per layer (must be >= 2).
    """

    n_layers: int = 6
    cells_per_layer: int = 16

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError(f"n_layers deve ser >= 1, recebido {self.n_layers}")
        if self.cells_per_layer < 2:
            raise ValueError(
                f"cells_per_layer deve ser >= 2, recebido {self.cells_per_layer}"
            )

    @property
    def n_nodes(self) -> int:
        """Total number of read-out cells in the detector."""
        return self.n_layers * self.cells_per_layer


@dataclass(frozen=True)
class SimulationConfig:
    """Physics parameters that control synthetic shower generation.

    All energy values are in GeV and time values in nanoseconds.
    """

    beam_energy_min: float = 30.0
    beam_energy_max: float = 300.0
    pileup_mean: float = 60.0
    pileup_std: float = 20.0
    pileup_max: float = 200.0
    core_sigma_base: float = 0.35
    core_sigma_pileup_scale: float = 500.0
    layer_attenuation: float = 2.4
    energy_noise_base: float = 0.03
    energy_noise_pileup_scale: float = 4000.0
    time_offset: float = 25.0
    time_layer_step: float = 2.0
    time_noise_base: float = 0.5
    time_noise_pileup_scale: float = 120.0


ENERGY_LOG_SCALE = 6.0
TIME_OFFSET = 25.0
TIME_SCALE = 20.0

# Module-level cache for adjacency matrices keyed by (n_layers, cells_per_layer).
_adj_cache: dict[tuple[int, int], torch.Tensor] = {}


def build_grid_adjacency(geometry: DetectorGeometry) -> torch.Tensor:
    """Build a row-normalised adjacency matrix for the detector graph.

    Each cell is connected to its azimuthal neighbours (with periodic
    boundary conditions) and to vertically adjacent cells in neighbouring
    layers.  Self-loops are added before normalisation.

    Results are cached so that repeated calls with the same geometry
    do not recompute the matrix.
    """
    key = (geometry.n_layers, geometry.cells_per_layer)
    if key in _adj_cache:
        return _adj_cache[key].clone()

    n_nodes = geometry.n_nodes
    adj = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)

    for layer in range(geometry.n_layers):
        for cell in range(geometry.cells_per_layer):
            idx = layer * geometry.cells_per_layer + cell
            left_cell = (cell - 1) % geometry.cells_per_layer
            right_cell = (cell + 1) % geometry.cells_per_layer

            left_idx = layer * geometry.cells_per_layer + left_cell
            right_idx = layer * geometry.cells_per_layer + right_cell
            adj[idx, left_idx] = 1.0
            adj[idx, right_idx] = 1.0

            if layer > 0:
                prev_idx = (layer - 1) * geometry.cells_per_layer + cell
                adj[idx, prev_idx] = 1.0
            if layer < geometry.n_layers - 1:
                next_idx = (layer + 1) * geometry.cells_per_layer + cell
                adj[idx, next_idx] = 1.0

    adj = adj + torch.eye(n_nodes, dtype=torch.float32)
    deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
    result = adj / deg

    _adj_cache[key] = result.clone()
    return result


def node_coordinates(geometry: DetectorGeometry) -> torch.Tensor:
    """Return ``(n_nodes, 3)`` coordinate tensor ``[layer_norm, sin(phi), cos(phi)]``."""
    coords = []
    for layer in range(geometry.n_layers):
        layer_norm = float(layer) / max(geometry.n_layers - 1, 1)
        for cell in range(geometry.cells_per_layer):
            angle = (2.0 * math.pi * cell) / geometry.cells_per_layer
            coords.append([layer_norm, math.sin(angle), math.cos(angle)])
    return torch.tensor(coords, dtype=torch.float32)


def compose_encoder_input(coords: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Concatenate node coordinates with target features along the last axis."""
    return torch.cat([coords, target], dim=-1)


def normalize_energy(energy: torch.Tensor) -> torch.Tensor:
    """Apply log1p normalisation to energy deposits (GeV -> normalised)."""
    return torch.log1p(energy.clamp_min(0.0)) / ENERGY_LOG_SCALE


def denormalize_energy(energy_norm: torch.Tensor) -> torch.Tensor:
    """Invert log1p normalisation and clamp to non-negative values."""
    return torch.expm1(energy_norm * ENERGY_LOG_SCALE).clamp_min(0.0)


def normalize_time(time: torch.Tensor) -> torch.Tensor:
    """Centre and scale time measurements (ns -> normalised)."""
    return (time - TIME_OFFSET) / TIME_SCALE


def denormalize_time(time_norm: torch.Tensor) -> torch.Tensor:
    """Invert time normalisation (normalised -> ns)."""
    return (time_norm * TIME_SCALE) + TIME_OFFSET


class SyntheticShowerDataset(Dataset):
    """Vectorised generator for synthetic calorimeter shower events.

    Each event consists of energy deposits and timing measurements across
    a configurable cylindrical detector, conditioned on beam energy and
    pileup level.

    Parameters
    ----------
    num_events : int
        Number of events to generate.
    seed : int
        Random seed for reproducibility.
    geometry : DetectorGeometry | None
        Detector layout; defaults to ``DetectorGeometry()``.
    simulation : SimulationConfig | None
        Physics parameters; defaults to ``SimulationConfig()``.
    """

    def __init__(
        self,
        num_events: int = 5000,
        seed: int = 7,
        geometry: DetectorGeometry | None = None,
        simulation: SimulationConfig | None = None,
    ) -> None:
        super().__init__()
        self.geometry = geometry or DetectorGeometry()
        self.simulation = simulation or SimulationConfig()
        self.adj = build_grid_adjacency(self.geometry)
        self.coords = node_coordinates(self.geometry)
        self.conditions, self.targets = self._generate_events(num_events=num_events, seed=seed)

    def _generate_events(self, num_events: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate *num_events* synthetic shower events (vectorised)."""
        g = torch.Generator().manual_seed(seed)
        n_nodes = self.geometry.n_nodes
        layer_pos = self.coords[:, 0] * max(self.geometry.n_layers - 1, 1)
        phi = torch.atan2(self.coords[:, 1], self.coords[:, 2])
        sim = self.simulation

        beam_energy = torch.empty(num_events).uniform_(
            sim.beam_energy_min, sim.beam_energy_max, generator=g
        )
        pileup = torch.normal(
            sim.pileup_mean,
            sim.pileup_std,
            size=(num_events,),
            generator=g,
        ).clamp(0.0, sim.pileup_max)
        phi0 = torch.empty(num_events).uniform_(-math.pi, math.pi, generator=g)

        angular_distance = torch.remainder(
            phi.unsqueeze(0) - phi0.unsqueeze(1) + math.pi,
            2.0 * math.pi,
        )
        angular_distance = angular_distance - math.pi

        sigma = sim.core_sigma_base + (pileup / sim.core_sigma_pileup_scale)
        core = torch.exp(-0.5 * (angular_distance / sigma.unsqueeze(1)) ** 2)
        layer_weight = torch.exp(-layer_pos / sim.layer_attenuation).unsqueeze(0)
        energy = (layer_weight * core).clamp_min(1e-8)
        energy = energy / energy.sum(dim=1, keepdim=True).clamp_min(1e-8)
        energy = energy * beam_energy.unsqueeze(1)

        noise_sigma = sim.energy_noise_base + (
            pileup / sim.energy_noise_pileup_scale
        )
        noise = torch.normal(0.0, 1.0, size=(num_events, n_nodes), generator=g)
        energy = (energy * (1.0 + noise * noise_sigma.unsqueeze(1))).clamp_min(0.0)

        time_sigma = sim.time_noise_base + (pileup / sim.time_noise_pileup_scale)
        time_noise = torch.normal(0.0, 1.0, size=(num_events, n_nodes), generator=g)
        time = sim.time_offset + (sim.time_layer_step * layer_pos.unsqueeze(0))
        time = time + (time_noise * time_sigma.unsqueeze(1))

        energy_norm = normalize_energy(energy)
        time_norm = normalize_time(time)

        conditions = torch.stack(
            [
                beam_energy / sim.beam_energy_max,
                pileup / sim.pileup_max,
            ],
            dim=-1,
        ).to(dtype=torch.float32)
        targets = torch.stack([energy_norm, time_norm], dim=-1).to(dtype=torch.float32)
        return conditions, targets

    def __len__(self) -> int:
        return self.targets.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "coords": self.coords,
            "adj": self.adj,
            "cond": self.conditions[idx],
            "target": self.targets[idx],
        }
