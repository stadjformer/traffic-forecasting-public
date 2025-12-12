"""Utilities for STGformer model: training, inference, and HuggingFace Hub integration.

This module wraps the EXTERNAL STGformer implementation from the STGformer package.
For the internal implementation, see utils/stgformer.py.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from external_api import load_model as stgformer_load
from external_api import predict as stgformer_predict
from external_api import save_model as stgformer_save
from external_api import train_external

from utils.config import validate_dataset_name
from utils.dataset import TrafficDataset
from utils.hub import fetch_model_from_hub, push_model_to_hub


def train_model(
    dataset_name: str,
    pytorch_datasets: Dict[str, TrafficDataset],
    adjacency: np.ndarray,
    epochs: int = 200,  # Default from STGformer
    batch_size: int = 64,  # Default from STGformer
    save_dir: Optional[Path] = None,
    verbose: bool = False,
    device: Optional[str] = None,  # Allow overriding device (e.g., for testing)
) -> tuple:
    """
    Train STGformer model.

    Args:
        dataset_name: Dataset name (e.g., "METR-LA", "PEMS-BAY")
        pytorch_datasets: Dict[str, TrafficDataset] with 'train', 'val', 'test' splits
        adjacency: Adjacency matrix (num_nodes, num_nodes)
        epochs: Number of training epochs (default: 200)
        batch_size: Batch size for training (default: 64)
        save_dir: Directory to save model (optional)
        verbose: Print training progress
        device: Device to use (None = auto-detect, or specify 'cpu', 'cuda', 'mps')

    Returns:
        tuple: (model, scaler, data_normalized) where:
            - model is trained STGformer
            - scaler is StandardScaler
            - data_normalized is always False (external API doesn't use normalized imputation)
    """
    dataset_name = validate_dataset_name(dataset_name)

    # Get dataset properties
    train_ds = pytorch_datasets["train"]
    num_nodes = train_ds.num_nodes
    seq_len = train_ds.seq_len
    horizon = train_ds.horizon

    # STGformer expects input_dim=2 with [value, time_of_day]
    # Our datasets already have this format
    if train_ds.input_dim != 2:
        raise ValueError(
            f"STGformer requires input_dim=2 [value, time_of_day], got {train_ds.input_dim}"
        )

    # Prepare data in STGformer format
    # STGformer expects: (samples, in_steps, num_nodes, input_dim)
    # X has 2 features [speed, time_of_day], Y only needs 1 feature [speed]
    data = {
        "x_train": pytorch_datasets["train"].x.numpy(),
        "y_train": pytorch_datasets["train"].y.numpy()[..., 0:1],  # Only speed
        "x_val": pytorch_datasets["val"].x.numpy(),
        "y_val": pytorch_datasets["val"].y.numpy()[..., 0:1],  # Only speed
        "x_test": pytorch_datasets["test"].x.numpy(),
        "y_test": pytorch_datasets["test"].y.numpy()[..., 0:1],  # Only speed
        "adj_mx": adjacency,
    }

    # Auto-detect best available device (unless overridden)
    # Note: STGFormer has in-place operation issues with MPS, so we skip it
    if device is None:
        device = _get_best_device_stgformer()

    # Create training config
    # Set dow_embedding_dim=0 since we don't have day-of-week feature
    # Use tod_embedding_dim=24 for time-of-day embeddings
    config = {
        "num_nodes": num_nodes,
        "in_steps": seq_len,
        "out_steps": horizon,
        "input_dim": 2,  # [value, time_of_day]
        "output_dim": 1,  # Single output feature (speed)
        "steps_per_day": 288,  # 5-min intervals: 24 * 60 / 5 = 288
        "input_embedding_dim": 24,
        "tod_embedding_dim": 24,  # Enable time-of-day embeddings
        "dow_embedding_dim": 0,  # Disable day-of-week (not available)
        "adaptive_embedding_dim": 80,
        "num_heads": 4,
        "num_layers": 3,
        "dropout": 0.1,
        "dropout_a": 0.3,
        "kernel_size": [1],
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "weight_decay": 0.0003,
        "milestones": [20, 30],
        "lr_decay_rate": 0.1,
        "early_stop": 10,
        "clip_grad": 0,
        "device": device,
        "verbose": 1 if verbose else 999999,  # High number to suppress output
    }

    if verbose:
        print(f"Training STGformer on {dataset_name}...")
        print(f"  Nodes: {num_nodes}")
        print(f"  Input sequence length: {seq_len}")
        print(f"  Output sequence length: {horizon}")
        print("  Input features: 2 [value, time_of_day]")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {device}")

    # Train using STGformer API
    model, scaler = train_external(data=data, config=config)

    # Save model if directory provided
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        stgformer_save(model, config, scaler, str(save_dir))

        if verbose:
            print(f"Model saved to: {save_dir}")

    # Return False for data_normalized since external API doesn't support normalized imputation
    return model, scaler, False


def get_stgformer_predictions(
    model,
    scaler,
    pytorch_dataset: TrafficDataset,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Get predictions from STGformer model.

    Args:
        model: Trained STGformer model
        scaler: StandardScaler used during training
        pytorch_dataset: TrafficDataset to predict on
        batch_size: Batch size for inference (default: 64)

    Returns:
        Predictions array of shape (samples, horizon, nodes, 1)
    """
    # Get data
    x_data = pytorch_dataset.x.numpy()

    # Get predictions using STGformer's predict (handles batching)
    predictions = stgformer_predict(model, x_data, scaler, batch_size=batch_size)

    # Ensure shape is (samples, horizon, nodes, 1)
    if predictions.ndim == 3:
        predictions = np.expand_dims(predictions, axis=-1)

    return predictions


def get_predictions_hub(
    dataset_name: str,
    dataset: Dict[str, TrafficDataset],
    split: str = "test",
    batch_size: int = 64,
    force_download: bool = False,
) -> np.ndarray:
    """
    Get predictions from STGformer model loaded from HuggingFace Hub.

    This function matches the interface used by DCRNN, MTGNN, and GWNET for baseline calculations.

    Args:
        dataset_name: Dataset name (e.g., "METR-LA", "PEMS-BAY")
        dataset: Dict of TrafficDataset splits
        split: Which split to predict on (default: "test")
        batch_size: Batch size for inference (default: 64)
        force_download: Force re-download of model from Hub

    Returns:
        Predictions array of shape (samples, horizon, nodes, 1)
    """
    # Load model from Hub
    model, scaler, _ = load_from_hub(dataset_name, force_download=force_download)

    # Get the dataset split
    pytorch_dataset = dataset[split]

    # Get predictions using the existing function
    return get_stgformer_predictions(
        model, scaler, pytorch_dataset, batch_size=batch_size
    )


def load_model(
    dataset_name: str,
    checkpoint_dir: Path,
    device: Optional[str] = None,
    verbose: bool = False,
) -> tuple:
    """
    Load trained STGformer model from checkpoint.

    Args:
        dataset_name: Dataset name (e.g., "METR-LA")
        checkpoint_dir: Directory containing model.safetensors, config.json, metadata.json
        device: Device to load model on (None = auto-detect)
        verbose: Print verbose output

    Returns:
        tuple: (model, scaler) where model is loaded STGformer and scaler is StandardScaler
    """
    if device is None:
        device = _get_best_device_stgformer()

    model_path = checkpoint_dir / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if verbose:
        print(f"Loading STGformer model from {checkpoint_dir}...")

    # Use the STGformer package's load_model function
    model, scaler = stgformer_load(str(checkpoint_dir), device=device)

    if verbose:
        print(f"Model loaded successfully on {device}")

    # Return False for data_normalized since loaded models assume unnormalized test data
    return model, scaler, False


def load_from_hub(
    dataset_name: str,
    force_download: bool = False,
    device: Optional[str] = None,
    verbose: bool = False,
) -> tuple:
    """
    Load trained STGformer model from HuggingFace Hub.

    Args:
        dataset_name: Dataset name (e.g., "METR-LA")
        force_download: Force re-download of model
        device: Device to load model on (None = auto-detect)
        verbose: Print verbose output

    Returns:
        tuple: (model, scaler, data_normalized) where:
            - model is loaded STGformer
            - scaler is StandardScaler
            - data_normalized is always False (loaded models assume unnormalized test data)
    """
    checkpoint_dir = fetch_model_from_hub(
        model_type="STGFORMER",
        dataset_name=dataset_name,
        force_download=force_download,
        verbose=verbose,
    )
    return load_model(dataset_name, checkpoint_dir, device, verbose)


def push_to_hub(
    checkpoint_dir: Path,
    repo_id: str,
    dataset_name: str,
    metrics: Optional[Dict[str, float]] = None,
    commit_message: Optional[str] = None,
    private: bool = False,
) -> str:
    """
    Push a checkpoint to HuggingFace Hub.

    Args:
        checkpoint_dir: Path to local checkpoint directory
        repo_id: HuggingFace repo ID (e.g., "username/STGformer_METR-LA")
        dataset_name: Name of the dataset used for training
        metrics: Optional dict of evaluation metrics
        commit_message: Optional custom commit message
        private: Whether to create a private repository

    Returns:
        URL to the uploaded model on HuggingFace Hub
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Create additional metadata file (STGformer already creates metadata.json and config.json)
    # Add our custom hub metadata
    hub_metadata = {
        "dataset": dataset_name,
        "upload_date": "custom",  # Placeholder, push_model_to_hub will add actual timestamp
        "metrics": metrics or {},
        "framework": "PyTorch",
        "model_type": "STGformer",
    }

    hub_metadata_path = checkpoint_dir / "hub_metadata.json"
    with open(hub_metadata_path, "w") as f:
        json.dump(hub_metadata, f, indent=2)

    # Use shared push function
    return push_model_to_hub(
        checkpoint_dir=checkpoint_dir,
        repo_id=repo_id,
        model_type="STGFORMER",
        dataset_name=dataset_name,
        metrics=metrics,
        commit_message=commit_message,
        private=private,
    )


def _get_best_device_stgformer() -> str:
    """Get the best available device for STGFormer training/inference.

    Note: MPS has in-place operation issues with STGFormer, so we skip it and
    fall back to CPU on Apple Silicon. Use device='mps' explicitly in train_model()
    if you want to try MPS anyway.

    Returns:
        Device string: "cuda" or "cpu" (never "mps")
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    # Skip MPS due to in-place operation compatibility issues with external STGFormer
    return "cpu"
