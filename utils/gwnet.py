"""Utilities for Graph-WaveNet model: training, inference, and HuggingFace Hub integration."""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from gwnet import predict as gwnet_predict
from gwnet import save_model as gwnet_save
from gwnet import train_external

from utils.config import validate_dataset_name
from utils.dataset import TrafficDataset
from utils.hub import fetch_model_from_hub, get_best_device, push_model_to_hub


def train_model(
    dataset_name: str,
    pytorch_datasets: Dict[str, TrafficDataset],
    adjacency: np.ndarray,
    epochs: int = 100,  # Default from gwnet
    batch_size: int = 64,  # Default from gwnet
    save_dir: Optional[Path] = None,
    verbose: bool = False,
) -> dict:
    """
    Train Graph-WaveNet model.

    Args:
        dataset_name: Dataset name (e.g., "METR-LA", "PEMS-BAY")
        pytorch_datasets: Dict[str, TrafficDataset] with 'train', 'val', 'test' splits
        adjacency: Adjacency matrix (num_nodes, num_nodes)
        epochs: Number of training epochs (default: 100)
        batch_size: Batch size for training (default: 64)
        save_dir: Directory to save model (optional)
        verbose: Print training progress

    Returns:
        dict: Trained model result containing 'model', 'scaler', 'config'
    """
    dataset_name = validate_dataset_name(dataset_name)

    # Get dataset properties
    train_ds = pytorch_datasets["train"]
    num_nodes = train_ds.num_nodes
    seq_len = train_ds.seq_len
    horizon = train_ds.horizon
    input_dim = train_ds.input_dim

    # Prepare data in gwnet format
    data = {
        "x_train": pytorch_datasets["train"].x.numpy(),
        "y_train": pytorch_datasets["train"].y.numpy(),
        "x_val": pytorch_datasets["val"].x.numpy(),
        "y_val": pytorch_datasets["val"].y.numpy(),
        "x_test": pytorch_datasets["test"].x.numpy(),
        "y_test": pytorch_datasets["test"].y.numpy(),
        "adj_mx": adjacency,
    }

    # Auto-detect best available device
    device = get_best_device()

    # Create training config
    config = {
        "num_nodes": num_nodes,
        "seq_length": seq_len,
        "horizon": horizon,
        "input_dim": input_dim,
        "output_dim": 1,  # Single output feature
        "nhid": 32,  # Hidden dim (gwnet default)
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "dropout": 0.3,
        "device": device,
        "gcn_bool": True,
        "addaptadj": True,
        "adjtype": "doubletransition",
    }

    if verbose:
        print(f"Training Graph-WaveNet on {dataset_name}...")
        print(f"  Nodes: {num_nodes}")
        print(f"  Input sequence length: {seq_len}")
        print(f"  Output sequence length: {horizon}")
        print(f"  Input features: {input_dim}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {device}")

    # Train using gwnet API
    model_result = train_external(data=data, config=config, verbose=verbose)

    # Save model if directory provided
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        gwnet_save(model_result, str(save_dir))

        if verbose:
            print(f"Model saved to: {save_dir}")

    return model_result


def get_gwnet_predictions(
    model_result: dict,
    pytorch_dataset: TrafficDataset,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Get predictions from Graph-WaveNet model.

    Args:
        model_result: Trained model dict from train_model() or load_model()
                     Contains 'model', 'scaler', 'config'
        pytorch_dataset: TrafficDataset to predict on
        batch_size: Batch size for inference (default: 64)

    Returns:
        Predictions array of shape (samples, horizon, nodes, 1)
    """
    # Get data
    x_data = pytorch_dataset.x.numpy()

    # Get predictions using gwnet's predict (handles batching internally)
    predictions = gwnet_predict(model_result, x_data, batch_size=batch_size)

    # Check if predictions need transposing
    # gwnet might return (samples, 1, nodes, horizon) instead of (samples, horizon, nodes, 1)
    if predictions.shape[1] == 1 and predictions.shape[3] > 1:
        # Transpose from (samples, 1, nodes, horizon) to (samples, horizon, nodes, 1)
        predictions = predictions.transpose(0, 3, 2, 1)

    # predictions shape: (samples, horizon, nodes, 1)
    # Ensure last dimension is 1
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
    Get predictions from Graph-WaveNet model loaded from HuggingFace Hub.

    This function matches the interface used by DCRNN and MTGNN for baseline calculations.

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
    model_result = load_from_hub(dataset_name, force_download=force_download)

    # Get the dataset split
    pytorch_dataset = dataset[split]

    # Get predictions using the existing function
    return get_gwnet_predictions(model_result, pytorch_dataset, batch_size=batch_size)


def load_model(
    dataset_name: str,
    checkpoint_dir: Path,
    device: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """
    Load trained Graph-WaveNet model from checkpoint.

    Args:
        dataset_name: Dataset name (e.g., "METR-LA")
        checkpoint_dir: Directory containing model.pth and config.json
        device: Device to load model on (None = auto-detect)
        verbose: Print verbose output

    Returns:
        dict: Loaded model result containing 'model', 'scaler', 'config'
    """
    if device is None:
        device = get_best_device()

    model_path = checkpoint_dir / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if verbose:
        print(f"Loading Graph-WaveNet model from {checkpoint_dir}...")

    # Use the gwnet package's load_model function
    from gwnet import load_model as gwnet_load_model

    result = gwnet_load_model(str(checkpoint_dir), device=device)

    if verbose:
        print(f"Model loaded successfully on {device}")

    return result


def load_from_hub(
    dataset_name: str,
    force_download: bool = False,
    device: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """
    Load trained Graph-WaveNet model from HuggingFace Hub.

    Args:
        dataset_name: Dataset name (e.g., "METR-LA")
        force_download: Force re-download of model
        device: Device to load model on (None = auto-detect)
        verbose: Print verbose output

    Returns:
        dict: Loaded model result containing 'model', 'scaler', 'config'
    """
    checkpoint_dir = fetch_model_from_hub(
        model_type="GWNET",
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
        repo_id: HuggingFace repo ID (e.g., "username/GraphWaveNet_METR-LA")
        dataset_name: Name of the dataset used for training
        metrics: Optional dict of evaluation metrics
        commit_message: Optional custom commit message
        private: Whether to create a private repository

    Returns:
        URL to the uploaded model on HuggingFace Hub
    """
    return push_model_to_hub(
        checkpoint_dir=Path(checkpoint_dir),
        repo_id=repo_id,
        model_type="GWNET",
        dataset_name=dataset_name,
        metrics=metrics,
        commit_message=commit_message,
        private=private,
    )
