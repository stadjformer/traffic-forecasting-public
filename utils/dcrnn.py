"""Utilities for DCRNN model: loading, training, prediction extraction, and HuggingFace Hub integration."""

from pathlib import Path
from typing import Dict, Optional

import dcrnn_pytorch
import numpy as np
import yaml
from dcrnn_pytorch.lib.utils import DataLoader as DCRNNDataLoader
from dcrnn_pytorch.lib.utils import StandardScaler
from dcrnn_pytorch.model.pytorch.dcrnn_supervisor import DCRNNSupervisor

import utils.io
from utils.config import DCRNN_CONFIGS, validate_dataset_name
from utils.dataset import TrafficDataset
from utils.hub import fetch_model_from_hub, push_model_to_hub


def get_supervisor_and_data(
    dataset_name: str, force_download: bool = False, verbose: bool = False
):
    dataset_name = validate_dataset_name(dataset_name)

    if dataset_name not in DCRNN_CONFIGS.keys():
        raise ValueError(
            f"dataset_name={dataset_name} is not supported (supported values: {list(DCRNN_CONFIGS.keys())})"
        )

    config_path = dcrnn_pytorch.get_config_path(DCRNN_CONFIGS[dataset_name])
    verbose and print("Loading DCRNN config...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    verbose and print("Loading dataset in PyTorch format...")
    pytorch_dataset = utils.io.get_dataset_torch(
        dataset_name=dataset_name, force_download=force_download, verbose=verbose
    )
    verbose and print("Transforming dataset to DCRNN format...")
    dcrnn_data = utils.dcrnn.prepare_dcrnn_data(
        pytorch_dataset, batch_size=config["data"]["batch_size"]
    )
    adj_mx = pytorch_dataset["train"].adj_mx.numpy()

    verbose and print("Instantiating DCRNNSupervisor...")
    return DCRNNSupervisor(
        adj_mx=adj_mx, data_override=dcrnn_data, **config
    ), dcrnn_data


def prepare_dcrnn_data(pytorch_datasets, batch_size=64):
    """
    Convert PyTorch TrafficDatasets to DCRNN-compatible format.

    Applies StandardScaler normalization and wraps data in DCRNN's custom DataLoader.

    Args:
        pytorch_datasets: Dict[str, TrafficDataset] with 'train', 'val', 'test' splits
        batch_size: Batch size for DataLoaders

    Returns:
        Dict containing:
            - train_loader: DCRNN DataLoader for training
            - val_loader: DCRNN DataLoader for validation
            - test_loader: DCRNN DataLoader for testing
            - scaler: StandardScaler fitted on training data (NOTE: Scaling was already applied to data,
            this is returned for convenience / completeness only!!!)
    """
    # Extract numpy arrays from datasets
    data_arrays = {}
    for split_name in ["train", "val", "test"]:
        dataset = pytorch_datasets[split_name]
        # Convert back to numpy for DCRNN
        data_arrays[f"x_{split_name}"] = dataset.x.numpy()
        data_arrays[f"y_{split_name}"] = dataset.y.numpy()

    # Create scaler from training data (first feature only, typically speed/flow)
    scaler = StandardScaler(
        mean=data_arrays["x_train"][..., 0].mean(),
        std=data_arrays["x_train"][..., 0].std(),
    )

    # Scale data (first feature only)
    for split in ["train", "val", "test"]:
        data_arrays[f"x_{split}"][..., 0] = scaler.transform(
            data_arrays[f"x_{split}"][..., 0]
        )
        data_arrays[f"y_{split}"][..., 0] = scaler.transform(
            data_arrays[f"y_{split}"][..., 0]
        )

    # Create DCRNN data loaders
    data = {
        "train_loader": DCRNNDataLoader(
            data_arrays["x_train"], data_arrays["y_train"], batch_size, shuffle=True
        ),
        "val_loader": DCRNNDataLoader(
            data_arrays["x_val"], data_arrays["y_val"], batch_size, shuffle=False
        ),
        "test_loader": DCRNNDataLoader(
            data_arrays["x_test"], data_arrays["y_test"], batch_size, shuffle=False
        ),
        "scaler": scaler,
    }

    return data


def push_to_hub(
    checkpoint_path: str,
    repo_id: str,
    dataset_name: str,
    metrics: Optional[Dict[str, float]] = None,
    commit_message: Optional[str] = None,
    private: bool = False,
) -> str:
    """
    Push a checkpoint to HuggingFace Hub.

    Args:
        checkpoint_path: Path to local checkpoint directory (contains model.tar, scaler, adj_mx, etc.)
        repo_id: HuggingFace repo ID (e.g., "username/dcrnn-metr-la")
        dataset_name: Name of the dataset used for training (e.g., "METR-LA")
        metrics: Optional dict of evaluation metrics (e.g., {"test_mae": 3.45})
        commit_message: Optional custom commit message
        private: Whether to create a private repository

    Returns:
        URL to the uploaded model on HuggingFace Hub
    """
    return push_model_to_hub(
        checkpoint_dir=Path(checkpoint_path),
        repo_id=repo_id,
        model_type="DCRNN",
        dataset_name=dataset_name,
        metrics=metrics,
        commit_message=commit_message,
        private=private,
    )


def get_trained_model_dcrnn(dataset_name, force_download=False, verbose=False):
    """Load trained DCRNN model from HuggingFace Hub.

    Args:
        dataset_name: Dataset name (e.g., "METR-LA")
        force_download: Force re-download even if cached
        verbose: Print loading progress

    Returns:
        Loaded DCRNNSupervisor instance
    """
    model_dir = fetch_model_from_hub(
        model_type="DCRNN",
        dataset_name=dataset_name,
        force_download=force_download,
        verbose=verbose,
    )

    supervisor, _ = get_supervisor_and_data(
        dataset_name=dataset_name,
        force_download=force_download,
        verbose=verbose,
    )
    verbose and print("Loading DCRNN supervisor state from local dir...")
    supervisor.load_from_dir(model_dir)

    return supervisor


def extract_predictions_from_supervisor(
    vals: Dict, dataset: Dict[str, TrafficDataset], split: str = "test"
) -> tuple[np.ndarray, np.ndarray]:
    y_pred_list = vals["prediction"]
    y_true_list = vals["truth"]

    y_pred_stacked = np.stack(y_pred_list, axis=0)
    y_true_stacked = np.stack(y_true_list, axis=0)

    if y_pred_stacked.ndim == 3:
        y_pred = np.transpose(y_pred_stacked, (1, 0, 2))
        y_true = np.transpose(y_true_stacked, (1, 0, 2))
        y_pred = np.expand_dims(y_pred, axis=-1)
        y_true = np.expand_dims(y_true, axis=-1)
    elif y_pred_stacked.ndim == 4:
        y_pred = np.transpose(y_pred_stacked, (1, 0, 2, 3))
        y_true = np.transpose(y_true_stacked, (1, 0, 2, 3))
    else:
        raise ValueError(
            f"Unexpected shape: y_pred_stacked.shape={y_pred_stacked.shape}, "
            f"expected 3 or 4 dimensions"
        )

    ds_split = dataset[split]

    # Trim DataLoader padding: DCRNN's DataLoader pads the last batch by repeating
    # the final sample to make the total divisible by batch_size
    y_pred = y_pred[: len(ds_split)]
    y_true = y_true[: len(ds_split)]

    assert y_pred.shape == y_true.shape, (
        f"Shape mismatch: y_pred.shape={y_pred.shape}, y_true.shape={y_true.shape}"
    )
    assert y_pred.shape[0] == len(ds_split), (
        f"Batch size mismatch: y_pred.shape[0]={y_pred.shape[0]}, "
        f"len(ds_split)={len(ds_split)}"
    )
    assert y_pred.shape[1] == ds_split.horizon, (
        f"Horizon mismatch: y_pred.shape[1]={y_pred.shape[1]}, "
        f"ds_split.horizon={ds_split.horizon}"
    )
    assert y_pred.shape[2] == ds_split.num_nodes, (
        f"Num nodes mismatch: y_pred.shape[2]={y_pred.shape[2]}, "
        f"ds_split.num_nodes={ds_split.num_nodes}"
    )

    return y_pred, y_true


def get_dcrnn_predictions(
    dataset_name: str,
    dataset: Dict[str, TrafficDataset],
    split: str = "test",
    validate_ground_truth: bool = True,
) -> np.ndarray:
    supervisor = get_trained_model_dcrnn(dataset_name)
    mean_loss, vals = supervisor.evaluate(split)
    y_pred, y_true_from_model = extract_predictions_from_supervisor(
        vals, dataset, split
    )

    if validate_ground_truth:
        y_true_from_dataset = dataset[split].y.numpy()[..., 0:1]
        assert np.allclose(y_true_from_model, y_true_from_dataset), (
            "Ground truth from model doesn't match dataset ground truth"
        )

    return y_pred
