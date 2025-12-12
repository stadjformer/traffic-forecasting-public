"""Embedding utilities for STGFormer.

This module contains utility functions for initializing embeddings in the STGFormer
architecture. These functions were extracted from the STGFormer class to reduce
code bloat in __init__ and improve maintainability.

IMPORTANT: These are pure utility functions that return tensors, NOT nn.Modules.
This ensures backward compatibility with existing checkpoints (no state_dict key changes).
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from stgformer.model import GraphMode


def create_adaptive_embedding(
    graph_mode: "GraphMode",
    num_nodes: int,
    in_steps: int,
    adaptive_embedding_dim: int,
    geo_adj: "torch.Tensor | None" = None,
) -> torch.nn.Parameter:
    """Create adaptive embedding with optional spectral initialization.

    Args:
        graph_mode: Graph construction mode (determines initialization strategy)
        num_nodes: Number of nodes in the graph
        in_steps: Number of input time steps
        adaptive_embedding_dim: Dimension of adaptive embeddings
        geo_adj: Pre-computed geographic adjacency (required for SPECTRAL_INIT)

    Returns:
        Parameter tensor of shape [in_steps, num_nodes, adaptive_embedding_dim]

    Raises:
        ValueError: If graph_mode is SPECTRAL_INIT but geo_adj is None
    """
    # Import here to avoid circular dependency
    from stgformer.enums import GraphMode

    if graph_mode == GraphMode.SPECTRAL_INIT:
        if geo_adj is None:
            raise ValueError("geo_adj required for SPECTRAL_INIT mode")

        # Import here to avoid circular dependency
        from stgformer.temporal_processing import compute_laplacian

        # Initialize embeddings from Laplacian eigenvectors
        # Uses spectral embedding: eigenvectors of normalized Laplacian
        L = compute_laplacian(geo_adj, normalized=True)
        eigenvalues, eigenvectors = torch.linalg.eigh(L)

        # Take smallest eigenvectors (skip first which is constant for connected graphs)
        # Small eigenvalues = smooth/low-frequency = captures cluster structure
        effective_dim = min(adaptive_embedding_dim, num_nodes - 1)
        # Skip index 0 (constant eigenvector), take indices 1 to effective_dim+1
        init_emb = eigenvectors[:, 1 : effective_dim + 1]

        # Pad if needed
        if effective_dim < adaptive_embedding_dim:
            padding = torch.zeros(num_nodes, adaptive_embedding_dim - effective_dim)
            init_emb = torch.cat([init_emb, padding], dim=-1)

        # Normalize to match xavier_uniform scale for stable training
        fan_in = adaptive_embedding_dim
        target_std = (2.0 / fan_in) ** 0.5
        current_std = init_emb.std()
        if current_std > 0:
            init_emb = init_emb / current_std * target_std

        # Expand to [in_steps, num_nodes, adaptive_embedding_dim]
        return torch.nn.Parameter(
            init_emb.unsqueeze(0).expand(in_steps, -1, -1).clone()
        )
    else:
        # Random initialization (LEARNED, GEOGRAPHIC, HYBRID modes)
        return torch.nn.init.xavier_uniform_(
            torch.nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
        )
