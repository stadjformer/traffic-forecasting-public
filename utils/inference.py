"""Model inference utilities with caching support."""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from utils.io import get_dataset_hf
from utils.stgformer import load_from_hub


def get_model_predictions_cached(
    dataset_name: str,
    hf_repo_prefix: str = "STGFORMER_BS200",
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
    sample_indices: Optional[tuple[int, int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Get model predictions with caching.

    Args:
        dataset_name: "METR-LA" or "PEMS-BAY"
        hf_repo_prefix: Model repo prefix on HuggingFace
        cache_dir: Directory to cache predictions (default: outputs/predictions)
        force_recompute: Force recomputation even if cache exists
        sample_indices: Optional (start, end) indices to only compute for a subset

    Returns:
        Tuple of (predictions, ground_truth), both shape [num_samples, horizon, num_nodes, 1]
    """
    if cache_dir is None:
        cache_dir = Path("outputs/predictions")

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Include sample indices in cache key if specified
    if sample_indices:
        start, end = sample_indices
        cache_file = (
            cache_dir
            / f"{dataset_name.lower()}_{hf_repo_prefix.lower()}_predictions_{start}_{end}.pkl"
        )
    else:
        cache_file = (
            cache_dir
            / f"{dataset_name.lower()}_{hf_repo_prefix.lower()}_predictions.pkl"
        )

    if cache_file.exists() and not force_recompute:
        print(f"Loading cached predictions from {cache_file}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data["predictions"], data["ground_truth"]

    print(f"Computing predictions for {dataset_name} using {hf_repo_prefix}...")
    if sample_indices:
        print(f"  Only computing samples {sample_indices[0]} to {sample_indices[1]}")

    # Load model and scaler
    from torch.utils.data import Subset

    from utils.dataset import hf_to_pytorch
    from utils.io import get_graph_metadata

    model, scaler = load_from_hub(
        dataset_name=dataset_name,
        hf_repo_prefix=hf_repo_prefix,
        device="cpu",
        verbose=False,
    )

    # Check if model expects day-of-week features
    add_dow = hasattr(model, "dow_embedding") and model.dow_embedding is not None

    # Load and convert dataset
    hf_dataset = get_dataset_hf(dataset_name)
    adj_mx, _, _ = get_graph_metadata(dataset_name)
    dataset_dict = hf_to_pytorch(hf_dataset, adj_mx=adj_mx, add_dow=add_dow)
    test_dataset = dataset_dict["test"]

    # If sample_indices specified, only use that subset
    if sample_indices:
        start, end = sample_indices
        test_dataset = Subset(test_dataset, range(start, end))

    # Get predictions using the model directly
    from utils.stgformer import get_predictions

    predictions = get_predictions(
        model=model,
        scaler=scaler,
        pytorch_dataset=test_dataset,
        batch_size=64,
        device="cpu",
        data_already_normalized=False,
    )

    # Extract ground truth from PyTorch test dataset
    ground_truth_list = []
    for i in range(len(test_dataset)):
        _, y = test_dataset[i]
        ground_truth_list.append(y.numpy())

    ground_truth = np.array(ground_truth_list)  # [samples, horizon, nodes, features]

    # Cache results
    with open(cache_file, "wb") as f:
        pickle.dump(
            {
                "predictions": predictions,
                "ground_truth": ground_truth,
                "dataset_name": dataset_name,
                "hf_repo_prefix": hf_repo_prefix,
            },
            f,
        )
    print(f"✓ Cached predictions to {cache_file}")

    return predictions, ground_truth


def get_learned_adjacency_matrix(
    dataset_name: str,
    hf_repo_prefix: str = "STGFORMER_BS200",
) -> np.ndarray:
    """Load the learned adjacency matrix from a trained model.

    Args:
        dataset_name: "METR-LA" or "PEMS-BAY"
        hf_repo_prefix: Model repo prefix on HuggingFace

    Returns:
        Adjacency matrix of shape [num_nodes, num_nodes]
    """
    print(f"Loading model {hf_repo_prefix} for {dataset_name}...")
    model, _ = load_from_hub(
        dataset_name=dataset_name,
        hf_repo_prefix=hf_repo_prefix,
        device="cpu",
        verbose=False,
    )

    # Extract learned adjacency matrix using the model's graph construction logic
    # First check for old graph_constructor interface (for backward compatibility)
    if hasattr(model, "graph_constructor") and hasattr(
        model.graph_constructor, "get_adj"
    ):
        adj_np = model.graph_constructor.get_adj().cpu().numpy()
    elif hasattr(model, "graph_mode"):
        # Handle graph_mode-based models
        from stgformer.enums import GraphMode

        # Check if graph is disabled (GraphMode.NONE)
        if model.graph_mode == GraphMode.NONE:
            # Return identity matrix - no graph learning in this mode
            num_nodes = model.num_nodes
            adj_np = np.eye(num_nodes)
        elif (
            hasattr(model, "adaptive_embedding")
            and model.adaptive_embedding is not None
        ):
            # Compute adjacency from adaptive embeddings, applying the same
            # transformations used during forward passes (including sparsification)
            with np.errstate(all="ignore"):  # Ignore overflow warnings
                import torch

                import stgformer.graph_utils

                model.eval()
                with torch.no_grad():
                    # Create dummy temporal embeddings to pass through graph construction
                    # Shape: [in_steps, num_nodes, adaptive_embedding_dim]
                    in_steps = model.in_steps

                    # Use the actual adaptive embeddings expanded over time
                    emb1 = model.adaptive_embedding[0]  # [num_nodes, adaptive_dim]

                    # Expand to temporal format
                    embeddings = emb1.unsqueeze(0).expand(
                        in_steps, -1, -1
                    )  # [in_steps, num_nodes, adaptive_dim]

                    # Use the model's complete graph construction pipeline
                    # This applies ALL transformations: learned graph + sparsification
                    adj = stgformer.graph_utils.construct_adaptive_graph(
                        model.graph_mode,
                        embeddings,
                        model.pooling,
                        model.geo_adj if hasattr(model, "geo_adj") else None,
                        model.lambda_hybrid if hasattr(model, "lambda_hybrid") else 0.5,
                        model.sparsity_k if hasattr(model, "sparsity_k") else None,
                        model.num_nodes if hasattr(model, "num_nodes") else None,
                    )

                    # Squeeze out any extra dimensions
                    # The result might be [1, num_nodes, num_nodes] or [num_nodes, num_nodes]
                    while adj.ndim == 3 and adj.shape[0] == 1:
                        adj = adj.squeeze(0)

                    # If still 3D (temporal pooling kept multiple timesteps), take mean
                    if adj.ndim == 3:
                        adj = adj.mean(dim=0)

                    adj_np = adj.cpu().numpy()
        else:
            raise ValueError(
                f"Model with graph_mode={model.graph_mode} does not have adaptive embeddings"
            )
    else:
        raise ValueError(f"Model {hf_repo_prefix} does not have learnable graph")

    return adj_np
