"""Graph construction utilities for STGFormer.

This module contains utility functions for constructing and manipulating graphs
in the STGFormer architecture. These functions were extracted from the STGFormer
class to reduce code bloat and improve maintainability.

IMPORTANT: These are pure utility functions, NOT nn.Modules. This ensures backward
compatibility with existing checkpoints (no state_dict key changes).
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from stgformer.model import GraphMode


def construct_adaptive_graph(
    graph_mode: "GraphMode",
    embeddings: "torch.Tensor | None",
    pooling: torch.nn.Module,
    geo_adj: "torch.Tensor | None" = None,
    lambda_hybrid: float = 0.5,
    sparsity_k: "int | None" = None,
    num_nodes: "int | None" = None,
) -> torch.Tensor:
    """
    Construct graph based on graph_mode setting.

    IMPORTANT: In LEARNED mode (default), the geographic graph is NEVER used
    for propagation. The model learns graph structure purely from embeddings.

    Graph Modes:
    - LEARNED: Graph = E @ E^T (fully learned, random init)
    - SPECTRAL_INIT: Graph = E @ E^T (learned, but E initialized from eigenvectors)
    - GEOGRAPHIC: Graph = actual sensor network adjacency (fixed)
    - HYBRID: Graph = λ·geo + (1-λ)·learned (blended)
    - NONE: Graph = Identity matrix (disables graph propagation entirely)

    Args:
        graph_mode: Graph construction mode
        embeddings: [in_steps, num_nodes, adaptive_embedding_dim] (can be None for NONE mode)
        pooling: Pooling module for temporal aggregation
        geo_adj: Pre-computed geographic adjacency matrix (required for GEOGRAPHIC/HYBRID)
        lambda_hybrid: Weight for hybrid mode (geo_adj weight)
        sparsity_k: Top-k sparsification (None = dense)
        num_nodes: Number of nodes (required for NONE mode when embeddings is None)

    Returns:
        graph: [num_nodes, num_nodes] - Probabilistic adjacency matrix
    """
    # Import here to avoid circular dependency
    from stgformer.model import GraphMode

    if graph_mode == GraphMode.NONE:
        # Return identity matrix - disables graph propagation
        # Each node only sees itself, no message passing between nodes
        if num_nodes is None:
            raise ValueError("num_nodes must be provided when graph_mode=NONE")
        return torch.eye(num_nodes)

    if graph_mode == GraphMode.GEOGRAPHIC:
        # Use pre-computed geographic adjacency only (no learning)
        return normalize_graph(geo_adj)

    elif graph_mode in (GraphMode.LEARNED, GraphMode.SPECTRAL_INIT):
        # LEARNED: learn graph from random-initialized embeddings
        # SPECTRAL_INIT: learn graph from Laplacian-initialized embeddings
        # Note: In both cases, the graph CAN deviate from geographic structure
        learned_graph = compute_learned_graph(embeddings, pooling)
        if sparsity_k is not None:
            learned_graph = sparsify_graph(learned_graph, sparsity_k)
        return learned_graph

    elif graph_mode == GraphMode.HYBRID:
        # Combine geographic and learned graphs
        learned_graph = compute_learned_graph(embeddings, pooling)
        if sparsity_k is not None:
            learned_graph = sparsify_graph(learned_graph, sparsity_k)
        geo_normalized = normalize_graph(geo_adj)
        # Hybrid: lambda * geo + (1-lambda) * learned
        return lambda_hybrid * geo_normalized + (1 - lambda_hybrid) * learned_graph

    else:
        raise ValueError(f"Unknown graph_mode: {graph_mode}")


def compute_learned_graph(
    embeddings: torch.Tensor, pooling: torch.nn.Module
) -> torch.Tensor:
    """
    Original STGFormer adaptive graph learning.

    Computes graph structure from learnable node embeddings:
        Graph[i,j] = softmax(E[i] · E[j])

    This is FULLY LEARNED - the geographic sensor network is NOT used here.
    The model learns which nodes should be "connected" based on the task.

    Why this matters:
    - Powerful: Can learn any graph structure
    - Risk: Starts from scratch, may not converge to meaningful structure
    - That's why we added SPECTRAL_INIT (initialize E from Laplacian eigenvectors)

    Args:
        embeddings: [in_steps, num_nodes, adaptive_embedding_dim] - learnable!
        pooling: Pooling module for temporal aggregation

    Returns:
        graph: [num_nodes, num_nodes] normalized adjacency (row-stochastic)
    """
    # Compute similarity: E @ E^T → [in_steps, num_nodes, num_nodes]
    # High dot product = nodes have similar embeddings = stronger "edge"
    graph = torch.matmul(embeddings, embeddings.transpose(1, 2))

    # Pool across time dimension → [1, num_nodes, num_nodes]
    graph = pooling(graph.transpose(0, 2)).transpose(0, 2)

    # ReLU + Softmax → probabilistic adjacency (rows sum to 1)
    graph = torch.nn.functional.softmax(torch.nn.functional.relu(graph), dim=-1)

    return graph


def normalize_graph(adj: torch.Tensor) -> torch.Tensor:
    """
    Row-normalize adjacency matrix to probability distribution.

    Args:
        adj: [num_nodes, num_nodes] adjacency matrix

    Returns:
        Normalized adjacency with rows summing to 1
    """
    # Handle both 2D and 3D (time-varying) graphs
    if adj.ndim == 2:
        row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        return adj / row_sum
    else:
        row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        return adj / row_sum


def sparsify_graph(adj: torch.Tensor, k: int) -> torch.Tensor:
    """
    Keep only top-k connections per node.

    Args:
        adj: [num_nodes, num_nodes] or [1, num_nodes, num_nodes] adjacency
        k: Number of neighbors to keep per node

    Returns:
        Sparsified and re-normalized adjacency
    """
    # Get the shape for proper handling
    squeeze = False
    if adj.ndim == 3 and adj.shape[0] == 1:
        adj = adj.squeeze(0)
        squeeze = True

    # Get top-k indices per row
    _, top_k_indices = torch.topk(adj, k=min(k, adj.shape[-1]), dim=-1)

    # Create sparse mask
    mask = torch.zeros_like(adj)
    mask.scatter_(-1, top_k_indices, 1.0)

    # Apply mask and re-normalize
    sparse_adj = adj * mask
    sparse_adj = normalize_graph(sparse_adj)

    if squeeze:
        sparse_adj = sparse_adj.unsqueeze(0)

    return sparse_adj
