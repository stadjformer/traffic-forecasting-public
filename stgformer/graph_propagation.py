"""Graph propagation module for STGFormer.

This module contains the GraphPropagate class, which implements the local
spatial processing branch of the STGFormer hybrid architecture.
"""

import torch

from stgformer.enums import PropagationMode


class GraphPropagate(torch.nn.Module):
    """
    Graph convolution branch for LOCAL pattern capture.

    This is the GCN/GraphSAGE-style component of the hybrid architecture.
    It provides local spatial inductive bias through neighborhood aggregation.

    This is message passing via matrix multiplication:
        x_new[n] = Σ_m graph[n,m] * x[m]  (aggregate neighbors' features)

    Supports two propagation modes:
    - POWER: Simple matrix powers A^k (original STGFormer)
        - A⁰x = x (self), A¹x (1-hop), A²x (2-hop), ...
        - Problem: A^k can explode/vanish if eigenvalues ≠ 1
    - CHEBYSHEV: Chebyshev polynomials T_k(L̃) for spectral filtering
        - T₀(L̃)x = x, T₁(L̃)x, T₂(L̃)x, ... (different graph frequency bands)
        - Eigenvalues stay in [-1, 1] → numerically stable
        - Each T_k captures different "frequency" of graph structure

    Note: The graph used here may be LEARNED (E @ E^T), not the original
    geographic adjacency. See GraphMode for options.
    """

    def __init__(
        self,
        k_hops: int,
        p_dropout: float = 0.2,
        propagation_mode: PropagationMode = PropagationMode.POWER,
        chebyshev_polynomials: list[torch.Tensor] | None = None,
    ):
        super().__init__()
        if k_hops < 1:
            raise ValueError(
                f"'k_hops' parameter must be a positive integer, got '{k_hops}'"
            )
        self.k_hops = k_hops
        self.dropout = torch.nn.Dropout(p_dropout)
        self.propagation_mode = propagation_mode

        # Pre-computed Chebyshev polynomial matrices (registered as buffers)
        if propagation_mode == PropagationMode.CHEBYSHEV:
            if chebyshev_polynomials is None:
                raise ValueError(
                    "chebyshev_polynomials must be provided for CHEBYSHEV propagation mode"
                )
            for i, T_k in enumerate(chebyshev_polynomials):
                self.register_buffer(f"cheb_T{i}", T_k)
            self._num_cheb = len(chebyshev_polynomials)

    def forward(self, x, graph):
        """
        K-hop propagation using either power or Chebyshev mode.

        Returns list of k_hops representations: [x⁰, x¹, ..., x^(k_hops-1)]

        Args:
            x: [batch, time, nodes, features]
            graph: [nodes, nodes] or [1, nodes, nodes] - used only for POWER mode
                   Note: This may be a LEARNED graph (E @ E^T), not geographic!

        Returns:
            List of k_hops tensors, each with same shape as input x.
        """
        if self.propagation_mode == PropagationMode.CHEBYSHEV:
            return self._chebyshev_propagate(x)
        else:
            return self._power_propagate(x, graph)

    def _power_propagate(self, x, graph):
        """
        Original STGFormer: simple matrix powers A^k.

        Computes [x, Ax, A²x, ...] where A is the (possibly learned) adjacency.
        This is message passing: each A @ x aggregates neighbor features.
        """
        x_k = x
        x_list = [x]  # x⁰ = x (self, 0-hop)

        # Determine if graph is 2D or 3D
        if graph.ndim == 2:
            einsum_str = "nm,btmf->btnf"
        elif graph.ndim == 3:
            einsum_str = "thi,btij->bthj"
        else:
            raise ValueError(f"Graph must be 2D or 3D, got {graph.ndim}D")

        for _ in range(1, self.k_hops):
            # Message passing: aggregate features from neighbors
            # x_k[n] = Σ_m graph[n,m] * x_k[m]
            x_k = torch.einsum(einsum_str, graph, x_k)
            x_list.append(self.dropout(x_k))

        return x_list

    def _chebyshev_propagate(self, x):
        """
        Chebyshev polynomial filtering: T_k(L̃) @ x.

        Uses pre-computed Chebyshev polynomials of the scaled Laplacian.
        Each T_k captures a different "frequency band" of the graph:
        - T₀ = Identity (DC/average)
        - T₁ = L̃ (local differences)
        - T₂, T₃, ... = higher frequency variations

        Note: Chebyshev mode uses FIXED geographic graph (computed at init),
        not the learned graph. This provides stable spectral filtering.
        """
        x_list = []

        for k in range(self.k_hops):
            T_k = getattr(self, f"cheb_T{k}")
            # T_k: [nodes, nodes], x: [batch, time, nodes, features]
            # This is still message passing, just with T_k instead of A^k
            # Use contiguous() to ensure correct memory layout for torch.compile
            x_k = torch.einsum("nm,btmf->btnf", T_k.contiguous(), x.contiguous())
            if k > 0:
                x_k = self.dropout(x_k)
            x_list.append(x_k)

        return x_list
