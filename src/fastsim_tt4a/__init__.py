"""FastSim TT-IV-A -- Fast simulation portfolio for HEP research.

This package provides a complete pipeline for training and evaluating
graph-convolutional generative models for calorimeter shower simulation,
including synthetic data generation, model training, physics validation
and an interactive Streamlit dashboard.
"""

from .data import DetectorGeometry, SimulationConfig, SyntheticShowerDataset
from .evaluate import evaluate_checkpoint, load_checkpoint
from .metrics import aggregate_reconstruction_metrics, reconstruction_tensors
from .model import GraphCVAE, MLPConditionalAutoencoder, build_model
from .train import TrainingConfig, run_training

__all__ = [
    "DetectorGeometry",
    "SimulationConfig",
    "SyntheticShowerDataset",
    "GraphCVAE",
    "MLPConditionalAutoencoder",
    "build_model",
    "aggregate_reconstruction_metrics",
    "reconstruction_tensors",
    "TrainingConfig",
    "run_training",
    "evaluate_checkpoint",
    "load_checkpoint",
]
