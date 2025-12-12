"""Utilities for MTGNN model: training, inference, and HuggingFace Hub integration."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from mtgnn import MTGNNModel, train_injected
from mtgnn.train_multi_step import TrainingConfig

from utils.config import validate_dataset_name
from utils.dataset import TrafficDataset
from utils.hub import fetch_model_from_hub, get_best_device, push_model_to_hub


def train_model(
    dataset_name: str,
    pytorch_datasets: Dict[str, TrafficDataset],
    adjacency: np.ndarray,
    epochs: int = 100,  # MTGNN default
    batch_size: int = 64,  # MTGNN default
    val_batch_size: Optional[int] = None,  # Default: batch_size // 2
    save_dir: Optional[Path] = None,
    verbose: bool = False,
) -> MTGNNModel:
    """
    Train MTGNN model using the new train_injected API.

    Uses MTGNN default hyperparameters (epochs=100, batch_size=64, etc.)

    Args:
        dataset_name: Dataset name (e.g., "METR-LA", "PEMS-BAY")
        pytorch_datasets: Dict[str, TrafficDataset] with 'train', 'val', 'test' splits
        adjacency: Adjacency matrix (num_nodes, num_nodes)
        epochs: Number of training epochs (default: 100, from MTGNN)
        batch_size: Batch size for training (default: 64, from MTGNN)
        val_batch_size: Batch size for validation (default: batch_size // 2 to reduce memory)
        save_dir: Directory to save model (optional)
        verbose: Print training progress

    Returns:
        MTGNNModel: Trained model wrapper with metrics
    """
    dataset_name = validate_dataset_name(dataset_name)

    # Get dataset properties
    train_ds = pytorch_datasets["train"]
    num_nodes = train_ds.num_nodes
    seq_in_len = train_ds.seq_len
    seq_out_len = train_ds.horizon
    in_dim = train_ds.input_dim

    # Prepare data in MTGNN format
    # MTGNN expects: (samples, seq_len, num_nodes, features)
    data = {
        "x_train": pytorch_datasets["train"].x.numpy(),
        "y_train": pytorch_datasets["train"].y.numpy(),
        "x_val": pytorch_datasets["val"].x.numpy(),
        "y_val": pytorch_datasets["val"].y.numpy(),
        "x_test": pytorch_datasets["test"].x.numpy(),
        "y_test": pytorch_datasets["test"].y.numpy(),
    }

    # Create training config
    # subgraph_size must be <= num_nodes to avoid topk error
    subgraph_size = min(20, num_nodes)

    # Use save_dir for checkpoints if provided, otherwise use default
    checkpoint_save_path = str(save_dir) + "/" if save_dir else "./save/"

    # Auto-detect best available device
    device = get_best_device()

    # Set validation batch size (1/8th of training batch size by default to reduce memory)
    if val_batch_size is None:
        val_batch_size = max(2, batch_size // 8)  # Minimum 2 to avoid shape issues

    # All parameters use MTGNN defaults from TrainingConfig except:
    # - Dataset-specific: num_nodes, seq_in_len, seq_out_len, in_dim
    # - Runtime: device (auto-detected), save (uses save_dir), print_every (based on verbose)
    # - Safety: subgraph_size (capped at num_nodes to avoid topk errors)
    # - Memory: val_batch_size (reduced to prevent OOM)
    # - Configurable: epochs, batch_size (keep as parameters with MTGNN defaults)
    config = TrainingConfig(
        num_nodes=num_nodes,
        seq_in_len=seq_in_len,
        seq_out_len=seq_out_len,
        in_dim=in_dim,
        subgraph_size=subgraph_size,
        epochs=epochs,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        device=device,
        print_every=50 if verbose else 999999,
        save=checkpoint_save_path,
    )

    if verbose:
        print(f"Training MTGNN on {dataset_name}...")
        print(f"  Nodes: {num_nodes}")
        print(f"  Input sequence length: {seq_in_len}")
        print(f"  Output sequence length: {seq_out_len}")
        print(f"  Input features: {in_dim}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Validation batch size: {val_batch_size}")
        print(f"  Device: {device}")

    # Train using new API (now with val_batch_size support)
    model = train_injected(config, data, adjacency)

    # Save model if directory provided
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "model.pth"
        model.save_model(str(model_path))

        # Also save config as JSON
        config_path = save_dir / "config.json"
        with open(config_path, "w") as f:
            config_dict = asdict(config)
            # Add dataset info
            config_dict["dataset_name"] = dataset_name
            config_dict["num_nodes"] = num_nodes
            json.dump(config_dict, f, indent=2)

        # Clean up training checkpoint files (exp*.pth)
        # These are redundant after final model is saved
        import glob

        for checkpoint_file in glob.glob(str(save_dir / "exp*.pth")):
            Path(checkpoint_file).unlink()
            if verbose:
                print(f"Cleaned up checkpoint: {checkpoint_file}")

        if verbose:
            print(f"Model saved to: {save_dir}")

    return model


def get_mtgnn_predictions(
    model: MTGNNModel,
    pytorch_dataset: TrafficDataset,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Get predictions from MTGNN model.

    Args:
        model: Pre-loaded MTGNNModel instance
        pytorch_dataset: TrafficDataset to predict on
        batch_size: Batch size for inference (default: 64)

    Returns:
        Predictions array of shape (samples, horizon, nodes, 1)
    """
    # Get data in MTGNN format: (batch, features, nodes, seq_len)
    x_data = pytorch_dataset.x.numpy()

    # Transpose to MTGNN format
    # From: (samples, seq_len, nodes, features)
    # To: (samples, features, nodes, seq_len)
    x_mtgnn = x_data.transpose(0, 3, 2, 1)

    # Get predictions in batches to avoid OOM
    predictions = []
    for i in range(0, len(x_mtgnn), batch_size):
        batch = x_mtgnn[i : i + batch_size]
        pred = model.predict(batch)
        predictions.append(pred)
    predictions = np.concatenate(predictions, axis=0)

    # predictions shape: (samples, horizon, nodes, features)
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
    Get predictions from MTGNN model loaded from HuggingFace Hub.

    This function matches the interface used by DCRNN for baseline calculations.

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
    model = load_from_hub(dataset_name, force_download=force_download)

    # Get the dataset split
    pytorch_dataset = dataset[split]

    # Get predictions using the existing function
    return get_mtgnn_predictions(model, pytorch_dataset, batch_size=batch_size)


def load_model(
    dataset_name: str,
    checkpoint_dir: Path,
    device: Optional[str] = None,
    verbose: bool = False,
) -> MTGNNModel:
    """
    Load trained MTGNN model from checkpoint.

    Args:
        dataset_name: Dataset name (e.g., "METR-LA")
        checkpoint_dir: Directory containing model.pth and config.json
        device: Device to load model on (None = auto-detect)
        verbose: Print verbose output

    Returns:
        MTGNNModel: Loaded model wrapper
    """
    if device is None:
        device = get_best_device()

    model_path = checkpoint_dir / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if verbose:
        print(f"Loading MTGNN model from {checkpoint_dir}...")

    # Use the MTGNN package's load_model method
    model = MTGNNModel.load_model(str(model_path), device=device)

    if verbose:
        print(f"Model loaded successfully on {device}")
        if model.learned_adj is not None:
            print(
                f"  - Loaded learned adjacency matrix of shape {model.learned_adj.shape}"
            )

    return model


def load_from_hub(
    dataset_name: str,
    force_download: bool = False,
    device: Optional[str] = None,
    verbose: bool = False,
) -> MTGNNModel:
    """
    Load trained MTGNN model from HuggingFace Hub.

    Args:
        dataset_name: Dataset name (e.g., "METR-LA")
        force_download: Force re-download of model
        device: Device to load model on (None = auto-detect)
        verbose: Print verbose output

    Returns:
        MTGNNModel: Loaded model wrapper
    """
    checkpoint_dir = fetch_model_from_hub(
        model_type="MTGNN",
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
        repo_id: HuggingFace repo ID (e.g., "username/MTGNN_METR-LA")
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
        model_type="MTGNN",
        dataset_name=dataset_name,
        metrics=metrics,
        commit_message=commit_message,
        private=private,
    )
