"""Temporal processing layers for STGFormer.

This module contains all temporal processing implementations including:
- Transformer-based attention
- Mamba SSM
- Temporal Convolutional Networks (TCN)
- Depthwise separable convolutions
- MLP-based processing

Also includes helper functions for Laplacian computation and Chebyshev polynomials.
"""

from typing import TYPE_CHECKING

import torch

# Conditional import for Mamba (CUDA-only)
MAMBA_AVAILABLE = False
try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError:
    pass

if TYPE_CHECKING:
    from mamba_ssm import Mamba


def compute_laplacian(adj: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    """Compute graph Laplacian from adjacency matrix.

    Args:
        adj: [num_nodes, num_nodes] adjacency matrix (will be symmetrized)
        normalized: If True, compute symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
                   If False, compute unnormalized Laplacian L = D - A

    Returns:
        [num_nodes, num_nodes] Laplacian matrix (symmetric, PSD)
    """
    # Symmetrize adjacency matrix
    adj = (adj + adj.T) / 2.0

    # Remove self-loops for Laplacian computation
    adj = adj.clone()
    adj.fill_diagonal_(0)

    # Degree matrix
    d = adj.sum(dim=-1)

    if normalized:
        # Symmetric normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        d_inv_sqrt = torch.zeros_like(d)
        mask = d > 0
        d_inv_sqrt[mask] = d[mask].pow(-0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        laplacian = (
            torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
            - d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        )
    else:
        # Unnormalized Laplacian: L = D - A
        laplacian = torch.diag(d) - adj

    return laplacian


def compute_scaled_laplacian(
    adj: torch.Tensor, lambda_max: float | None = None
) -> torch.Tensor:
    """Compute scaled Laplacian for Chebyshev polynomials.

    Scales eigenvalues to [-1, 1] range: L_scaled = (2/lambda_max) * L - I

    Args:
        adj: [num_nodes, num_nodes] adjacency matrix
        lambda_max: Maximum eigenvalue. If None, computed from eigendecomposition.

    Returns:
        [num_nodes, num_nodes] scaled Laplacian with eigenvalues in [-1, 1]
    """
    L = compute_laplacian(adj, normalized=True)

    if lambda_max is None:
        # Compute largest eigenvalue
        eigenvalues = torch.linalg.eigvalsh(L)
        lambda_max = eigenvalues.max().item()

    # Scale to [-1, 1]: L_scaled = (2/lambda_max) * L - I
    num_nodes = adj.shape[0]
    L_scaled = (2.0 / lambda_max) * L - torch.eye(
        num_nodes, device=adj.device, dtype=adj.dtype
    )

    return L_scaled


def compute_chebyshev_polynomials(L_scaled: torch.Tensor, K: int) -> list[torch.Tensor]:
    """Compute Chebyshev polynomial basis matrices T_0, T_1, ..., T_{K-1}.

    Uses recurrence: T_0(x) = 1, T_1(x) = x, T_k(x) = 2x*T_{k-1}(x) - T_{k-2}(x)

    Args:
        L_scaled: [num_nodes, num_nodes] scaled Laplacian (eigenvalues in [-1, 1])
        K: Number of Chebyshev polynomials to compute

    Returns:
        List of K matrices [T_0(L), T_1(L), ..., T_{K-1}(L)]
    """
    num_nodes = L_scaled.shape[0]
    device, dtype = L_scaled.device, L_scaled.dtype

    # T_0 = I (identity)
    T = [torch.eye(num_nodes, device=device, dtype=dtype)]

    if K > 1:
        # T_1 = L_scaled
        T.append(L_scaled.clone())

    # Recurrence: T_k = 2 * L_scaled * T_{k-1} - T_{k-2}
    for k in range(2, K):
        T_k = 2.0 * L_scaled @ T[k - 1] - T[k - 2]
        T.append(T_k)

    return T


class AttentionLayer(torch.nn.Module):
    """
    Dual-branch attention layer combining spatial and temporal attention.

    The temporal processing can be swapped for different implementations
    (standard attention, fast attention, Mamba, etc.) by overriding
    _compute_temporal_branch() in subclasses.
    """

    def __init__(self, model_dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # QKV projection
        self.qkv_proj = torch.nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)
        # output projection
        self.out_proj = torch.nn.Linear(2 * model_dim, model_dim)

    def forward(self, x):
        """
        Args:
            x: Input [batch, time, nodes, features]
        Returns:
            Attention output with same shape
        """

        # Generate Q, K, V from same input
        query, key, value = self.qkv_proj(x).chunk(3, -1)

        # Spatial attention (over nodes dimension)
        att_spatial = self._compute_spatial_branch(x, query, key, value)

        # Temporal attention (over time dimension) - can be overridden
        att_temporal = self._compute_temporal_branch(x, query, key, value)

        # Combine and project
        return self.out_proj(torch.concat([att_spatial, att_temporal], -1))

    def _compute_spatial_branch(self, x, query, key, value):
        """Spatial attention over nodes. Override for custom spatial processing."""
        return self.attention(
            x, *self._compute_qkv_spatial(query=query, key=key, value=value)
        )

    def _compute_temporal_branch(self, x, query, key, value):
        """
        Temporal attention over time.

        Override this method to swap in alternative temporal processors (e.g., Mamba).

        To implement Mamba:
        1. Create MambaAttentionLayer(FastAttentionLayer) subclass
        2. Override this method to use Mamba SSM instead of attention
        3. Input x: [batch, time, nodes, features]
        4. Process each node's time series through Mamba
        5. Return same shape: [batch, time, nodes, features]

        Example:
            def _compute_temporal_branch(self, x, query, key, value):
                B, T, N, D = x.shape
                x_temporal = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
                out = self.mamba(x_temporal)  # Mamba processes sequences
                return out.reshape(B, N, T, D).permute(0, 2, 1, 3)
        """
        return self.attention(
            x.transpose(1, 2),
            *self._compute_qkv_temporal(query=query, key=key, value=value),
        ).transpose(1, 2)

    def attention(self, x, q, k, v):
        """Apply attention with unflatten using x's shape."""
        return torch.unflatten(
            self._compute_attention(q=q, k=k, v=v), 0, x.shape[0:2]
        ).flatten(start_dim=3)

    def _compute_attention(self, q, k, v):
        """Core attention computation. Override in FastAttentionLayer for linear attention."""
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        return (
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
            .transpose(-3, -2)
            .flatten(start_dim=-2)
        )

    def _compute_qkv_spatial(self, query, key, value):
        return self._compute_qkv(query=query, key=key, value=value)

    def _compute_qkv_temporal(self, query, key, value):
        return self._compute_qkv(
            query=query.transpose(1, 2),
            key=key.transpose(1, 2),
            value=value.transpose(1, 2),
        )

    def _compute_qkv(self, query, key, value):
        q = torch.stack(torch.split(query, self.head_dim, dim=-1), dim=-2).flatten(
            start_dim=0, end_dim=1
        )
        k = torch.stack(torch.split(key, self.head_dim, dim=-1), dim=-2).flatten(
            start_dim=0, end_dim=1
        )
        v = torch.stack(torch.split(value, self.head_dim, dim=-1), dim=-2).flatten(
            start_dim=0, end_dim=1
        )

        return q, k, v


class FastAttentionLayer(AttentionLayer):
    """
    Fast attention with linear complexity O(N+T).
    """

    def __init__(self, model_dim, num_heads=8, qkv_bias=False):
        super().__init__(model_dim=model_dim, num_heads=num_heads, qkv_bias=qkv_bias)

    def _compute_attention(self, q, k, v):
        # Normalize queries and keys (with eps for numerical stability with torch.compile)
        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-6)
        N = q.shape[1]

        # Compute numerator: Q(K^TV) + bias
        kv = torch.einsum("blhm,blhd->bhmd", k, v)
        attention_num = torch.einsum("bnhm,bhmd->bnhd", q, kv)  # [N, H, D]
        attention_num += N * v

        # Compute denominator with bias
        k_sum = torch.einsum("blhm,l->bhm", k, torch.ones([k.shape[1]]).to(k.device))
        attention_normalizer = torch.einsum("bnhm,bhm->bnh", q, k_sum)  # [N, H]

        # Normalize attention
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape)
        )  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        return attention_num / attention_normalizer  # [N, H, D]


class MambaAttentionLayer(FastAttentionLayer):
    """
    Mamba-based attention layer replacing temporal attention with Mamba SSM.

    This layer uses:
    - Standard fast attention for the SPATIAL branch (over nodes)
    - Mamba SSM for the TEMPORAL branch (over time steps)

    Mamba provides:
    - O(T) complexity instead of O(T²) for temporal processing
    - Native sequential modeling with hidden state
    - Better inductive bias for causal/temporal patterns

    Requirements:
    - CUDA-enabled GPU (mamba-ssm is CUDA-only)
    - mamba-ssm >= 2.0.0 installed

    Architecture:
        Input [B, T, N, D]
              │
        ┌─────┴─────┐
        │           │
    Spatial      Temporal
    (Fast Att)   (Mamba SSM)
        │           │
        └─────┬─────┘
              │
        Concatenate → out_proj → [B, T, N, D]

    Args:
        model_dim: Input/output feature dimension
        num_heads: Number of attention heads for spatial branch
        qkv_bias: Whether to use bias in QKV projections
        d_state: Mamba SSM state dimension (default: 16)
        d_conv: Mamba convolution kernel size (default: 4)
        expand: Mamba expansion factor (default: 2)
    """

    def __init__(
        self,
        model_dim,
        num_heads=8,
        qkv_bias=False,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for MambaAttentionLayer but not installed. "
                "Install with: pip install mamba-ssm causal-conv1d. "
                "Note: mamba-ssm requires CUDA-enabled GPU."
            )

        super().__init__(model_dim=model_dim, num_heads=num_heads, qkv_bias=qkv_bias)

        # Mamba block for temporal processing
        # Input: [batch * nodes, time, features]
        # Mamba internally expands features by `expand` factor
        self.mamba = Mamba(
            d_model=model_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Store config for serialization
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

    def _compute_temporal_branch(self, x, query, key, value):
        """
        Process temporal dimension with Mamba SSM instead of attention.

        Mamba processes each node's time series independently, treating
        spatial locations as a batch dimension. This captures temporal
        patterns (trends, periodicity, causality) without quadratic
        complexity in sequence length.

        Args:
            x: Input tensor [batch, time, nodes, features]
            query, key, value: Not used (kept for interface compatibility)

        Returns:
            Temporal features [batch, time, nodes, features]
        """
        B, T, N, D = x.shape

        # Reshape: treat each node's time series as a separate sequence
        # [B, T, N, D] -> [B*N, T, D]
        x_temporal = x.permute(0, 2, 1, 3).reshape(B * N, T, D)

        # Process through Mamba
        # Mamba expects [batch, seq_len, features] and outputs same shape
        out = self.mamba(x_temporal)

        # Reshape back: [B*N, T, D] -> [B, N, T, D] -> [B, T, N, D]
        out = out.reshape(B, N, T, D).permute(0, 2, 1, 3)

        return out


class TCNTemporalLayer(FastAttentionLayer):
    """
    Hybrid layer with spatial attention + TCN temporal processing.

    This layer uses:
    - Standard fast attention for the SPATIAL branch (over nodes)
    - TCN for the TEMPORAL branch (over time steps)

    TCN provides:
    - O(T) complexity with parallel computation
    - Causal processing (no future leakage)
    - Exponentially growing receptive field via dilations
    - Better inductive bias for local temporal patterns

    Architecture:
        Input [B, T, N, D]
              │
        ┌─────┴─────┐
        │           │
    Spatial      Temporal
    (Attention)  (TCN)
        │           │
        └─────┬─────┘
              │
        Concatenate → out_proj → [B, T, N, D]

    Args:
        model_dim: Input/output feature dimension
        num_heads: Number of attention heads for spatial branch
        qkv_bias: Whether to use bias in QKV projections
        num_layers: Number of TCN layers (default: 3)
        kernel_size: Conv kernel size (default: 3)
        dilation_base: Base for exponential dilation (default: 2, gives 1,2,4,...)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        model_dim,
        num_heads=8,
        qkv_bias=False,
        num_layers=3,
        kernel_size=3,
        dilation_base=2,
        dropout=0.1,
    ):
        super().__init__(model_dim=model_dim, num_heads=num_heads, qkv_bias=qkv_bias)

        # Build TCN blocks
        self.tcn_layers = torch.nn.ModuleList()

        for i in range(num_layers):
            dilation = dilation_base**i
            # Causal padding: pad on left only to prevent future leakage
            padding = (kernel_size - 1) * dilation

            self.tcn_layers.append(
                torch.nn.Sequential(
                    # Conv1d expects [batch, channels, length]
                    # We'll process [B*N, D, T]
                    torch.nn.Conv1d(
                        in_channels=model_dim,
                        out_channels=model_dim,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=padding,  # Left padding for causality
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                )
            )

        # Store config for serialization
        self.num_tcn_layers = num_layers
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.tcn_dropout = dropout

    def _compute_temporal_branch(self, x, query, key, value):
        """
        Process temporal dimension with TCN instead of attention.

        TCN processes each node's time series independently with causal
        convolutions, capturing local temporal patterns with exponentially
        growing receptive field.

        Args:
            x: Input tensor [batch, time, nodes, features]
            query, key, value: Not used (kept for interface compatibility)

        Returns:
            Temporal features [batch, time, nodes, features]
        """
        B, T, N, D = x.shape

        # Reshape: treat each node's time series as a separate sequence
        # [B, T, N, D] -> [B*N, D, T] for Conv1d (expects channels before length)
        x_temporal = x.permute(0, 2, 3, 1).reshape(B * N, D, T)

        # Process through TCN layers
        out = x_temporal
        for tcn_layer in self.tcn_layers:
            residual = out
            out = tcn_layer(out)

            # Remove extra padding from causal conv (right side)
            # After padding and conv, we may have extra timesteps on the right
            if out.shape[-1] > T:
                out = out[..., :T]

            # Residual connection
            out = out + residual

        # Reshape back: [B*N, D, T] -> [B, N, D, T] -> [B, T, N, D]
        out = out.reshape(B, N, D, T).permute(0, 3, 1, 2)

        return out


class DepthwiseTemporalLayer(FastAttentionLayer):
    """
    Hybrid layer with spatial attention + depthwise separable convolution for temporal processing.

    This layer uses:
    - Standard fast attention for the SPATIAL branch (over nodes)
    - Depthwise separable convolution for the TEMPORAL branch (over time steps)

    Depthwise separable conv provides:
    - O(T) complexity with efficient parallel computation
    - ~10x fewer parameters than standard convolution (groups=model_dim)
    - Single-pass processing (vs TCN's multi-layer approach)
    - Excellent for capturing local temporal patterns efficiently

    Depthwise separable convolution splits the operation into:
    1. Depthwise: Each channel processed independently (captures temporal patterns per feature)
    2. Pointwise: 1x1 conv to mix information across channels

    This is inspired by MobileNet and is highly efficient for short sequences.

    Architecture:
        Input [B, T, N, D]
              │
        ┌─────┴─────┐
        │           │
    Spatial      Temporal
    (Attention)  (Depthwise Conv)
        │           │
        └─────┬─────┘
              │
        Concatenate → out_proj → [B, T, N, D]

    Args:
        model_dim: Input/output feature dimension
        num_heads: Number of attention heads for spatial branch
        qkv_bias: Whether to use bias in QKV projections
        kernel_size: Convolution kernel size (default: 3)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        model_dim,
        num_heads=8,
        qkv_bias=False,
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__(model_dim=model_dim, num_heads=num_heads, qkv_bias=qkv_bias)

        # Depthwise convolution: each channel processed separately
        # groups=model_dim means each of the D channels gets its own filter
        padding = kernel_size // 2  # Same padding to preserve sequence length
        self.depthwise = torch.nn.Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=model_dim,  # ← Key: depthwise (one filter per channel)
        )

        # Pointwise convolution: 1x1 conv to mix channels
        self.pointwise = torch.nn.Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=1,
        )

        self.dropout = torch.nn.Dropout(dropout)

        # Store config for serialization
        self.depthwise_kernel_size = kernel_size
        self.depthwise_dropout = dropout

    def _compute_temporal_branch(self, x, query, key, value):
        """
        Process temporal dimension with depthwise separable convolution.

        This is much more efficient than standard convolution:
        - Standard conv: kernel_size × in_channels × out_channels parameters
        - Depthwise sep: kernel_size × in_channels + in_channels × out_channels parameters
        - Reduction: ~9x fewer parameters for kernel_size=3, model_dim=96

        Args:
            x: Input tensor [batch, time, nodes, features]
            query, key, value: Not used (kept for interface compatibility)

        Returns:
            Temporal features [batch, time, nodes, features]
        """
        B, T, N, D = x.shape

        # Reshape: treat each node's time series as a separate sequence
        # [B, T, N, D] -> [B*N, D, T] for Conv1d (expects channels before length)
        x_temporal = x.permute(0, 2, 3, 1).reshape(B * N, D, T)

        # Apply depthwise separable convolution
        # Step 1: Depthwise - each channel independently
        out = self.depthwise(x_temporal)
        out = torch.nn.functional.relu(out)

        # Step 2: Pointwise - mix channels
        out = self.pointwise(out)
        out = self.dropout(out)

        # Reshape back: [B*N, D, T] -> [B, N, D, T] -> [B, T, N, D]
        out = out.reshape(B, N, D, T).permute(0, 3, 1, 2)

        return out


class MLPTemporalLayer(FastAttentionLayer):
    """
    Hybrid layer with spatial attention + simple MLP for temporal processing.

    This layer uses:
    - Standard fast attention for the SPATIAL branch (over nodes)
    - Simple MLP for the TEMPORAL branch (processes each timestep independently)

    MLP processing provides:
    - No reshape operations (fastest)
    - Minimal parameters
    - No explicit temporal modeling (useful baseline to test if temporal attention is needed)
    - Relies on spatial attention to capture spatiotemporal patterns

    This is the simplest possible temporal processing, making it a good baseline
    to test whether complex temporal modeling (Transformer/Mamba/TCN) is actually
    necessary, or if the spatial attention branch does most of the work.

    Architecture:
        Input [B, T, N, D]
              │
        ┌─────┴─────┐
        │           │
    Spatial      Temporal
    (Attention)  (MLP)
        │           │
        └─────┬─────┘
              │
        Concatenate → out_proj → [B, T, N, D]

    Args:
        model_dim: Input/output feature dimension
        num_heads: Number of attention heads for spatial branch
        qkv_bias: Whether to use bias in QKV projections
        hidden_dim: Hidden dimension for MLP (default: model_dim, no expansion)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        model_dim,
        num_heads=8,
        qkv_bias=False,
        hidden_dim=None,
        dropout=0.1,
    ):
        super().__init__(model_dim=model_dim, num_heads=num_heads, qkv_bias=qkv_bias)

        if hidden_dim is None:
            hidden_dim = model_dim

        # Simple 2-layer MLP
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(model_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, model_dim),
            torch.nn.Dropout(dropout),
        )

        # Store config for serialization
        self.mlp_hidden_dim = hidden_dim
        self.mlp_dropout = dropout

    def _compute_temporal_branch(self, x, query, key, value):
        """
        Process temporal dimension with simple MLP.

        This applies the same MLP to each timestep independently,
        making it extremely fast (no reshape, no attention computation).

        The hypothesis: For very short sequences (T=12), the spatial attention
        might capture most of the important spatiotemporal patterns, making
        complex temporal modeling unnecessary.

        Args:
            x: Input tensor [batch, time, nodes, features]
            query, key, value: Not used (kept for interface compatibility)

        Returns:
            Temporal features [batch, time, nodes, features]
        """
        # [B, T, N, D] - process directly without reshape
        # MLP is applied independently to each (batch, time, node) position
        return self.mlp(x)
