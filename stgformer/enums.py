"""Enums for STGFormer configuration options.

This module defines the various modes and options for configuring STGFormer models.
"""

from enum import Enum


class GraphMode(Enum):
    """Graph structure modes for STGFormer.

    LEARNED: Adaptive graph from learned embeddings (default, original STGFormer)
    SPECTRAL_INIT: Learn adaptive graph, initialized from Laplacian eigenvectors
    GEOGRAPHIC: Use pre-computed geographic adjacency only
    HYBRID: Combine geographic and learned adjacency with lambda weighting
    NONE: No graph propagation - disables the graph branch entirely (identity matrix)
    """

    LEARNED = "learned"
    SPECTRAL_INIT = "spectral_init"
    GEOGRAPHIC = "geographic"
    HYBRID = "hybrid"
    NONE = "none"


class PropagationMode(Enum):
    """Propagation modes for graph convolution.

    POWER: Simple matrix powers A^k (original STGFormer)
    CHEBYSHEV: Chebyshev polynomials T_k(L_scaled) for spectral filtering
    """

    POWER = "power"
    CHEBYSHEV = "chebyshev"


class TemporalMode(Enum):
    """Temporal processing modes for STGFormer.

    TRANSFORMER: Standard attention-based temporal processing (default)
    MAMBA: Mamba SSM-based temporal processing
    TCN: Temporal Convolutional Network (causal dilated convolutions)
    DEPTHWISE: Depthwise separable convolution (fast, efficient for short sequences)
    MLP: Simple MLP processing (fastest, good for very short sequences)
    NONE: No temporal processing - skip temporal branch entirely
    """

    TRANSFORMER = "transformer"
    MAMBA = "mamba"
    TCN = "tcn"
    DEPTHWISE = "depthwise"
    MLP = "mlp"
    NONE = "none"
