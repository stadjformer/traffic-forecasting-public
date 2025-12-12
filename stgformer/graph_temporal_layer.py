"""Graph-temporal hybrid layer for STGFormer.

This module contains the GraphTemporalLayer class, which implements the core
hybrid architecture combining local graph propagation with global temporal processing.
"""

import torch

from stgformer.enums import PropagationMode, TemporalMode
from stgformer.graph_propagation import GraphPropagate
from stgformer.temporal_processing import (
    DepthwiseTemporalLayer,
    FastAttentionLayer,
    MambaAttentionLayer,
    MLPTemporalLayer,
    TCNTemporalLayer,
)


class GraphTemporalLayer(torch.nn.Module):
    """
    Hybrid layer: The core STGFormer innovation.

    Note: Called 'SelfAttentionLayer' in the original STGFormer repository.

    This layer combines the two-branch architecture:
    Branch 1: GraphPropagate (local spatial patterns via GCN-style message passing)
    Branch 2: FastAttentionLayer (global spatio-temporal patterns via Transformer)

    Information flow:
    - GraphPropagate: Only sees k-hop neighbors (graph-aware, local)
    - Attention: Sees ALL nodes globally (graph-agnostic, full attention)

    This is the key design of STGFormer as a graph transformer:
    - Graph structure injects inductive bias (neighbors matter more)
    - Attention still allows any node to influence any other

    Compare to:
    - Pure GNN (GCN): Only sees neighbors, no global view
    - Pure Transformer: Sees everything, no structural bias
    - STGFormer: Both! Graph-biased local + attention-based global

    Args:
        order: Number of hops for graph propagation (default=2 means self + 1-hop neighbors).
               Controls receptive field, number of attention layers, and computational cost.
        temporal_mode: How to process the temporal dimension (TRANSFORMER or MAMBA).
        mamba_d_state: Mamba SSM state dimension (only used if temporal_mode=MAMBA).
        mamba_d_conv: Mamba convolution kernel size (only used if temporal_mode=MAMBA).
        mamba_expand: Mamba expansion factor (only used if temporal_mode=MAMBA).
    """

    def __init__(
        self,
        model_dim,
        num_heads=8,
        mlp_ratio=2,
        p_dropout=0.1,
        order=2,
        propagation_mode: PropagationMode = PropagationMode.POWER,
        chebyshev_polynomials: list[torch.Tensor] | None = None,
        use_zero_init: bool = True,
        temporal_mode: TemporalMode = TemporalMode.TRANSFORMER,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        tcn_num_layers: int = 3,
        tcn_kernel_size: int = 3,
        tcn_dilation_base: int = 2,
        tcn_dropout: float = 0.1,
        depthwise_kernel_size: int = 3,
        mlp_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.order = order  # Number of hops (k=2 means [x, Ax] or [T₀x, T₁x])
        self.temporal_mode = temporal_mode

        # Branch 1: Graph propagation (LOCAL - respects connectivity)
        # Produces k representations: [x, Ax, A²x, ...] or [T₀x, T₁x, T₂x, ...]
        self.locals = GraphPropagate(
            k_hops=self.order,
            p_dropout=p_dropout,
            propagation_mode=propagation_mode,
            chebyshev_polynomials=chebyshev_polynomials,
        )

        # Branch 2: Attention layers (GLOBAL - sees everything)
        # One attention layer per hop - each processes the k-hop propagated features
        # Note: Attention has NO graph masking - every node attends to every node
        if temporal_mode == TemporalMode.NONE:
            # No temporal processing - skip attention layers entirely
            self.attn = None
        elif temporal_mode == TemporalMode.MAMBA:
            # Use MambaAttentionLayer for temporal processing
            self.attn = torch.nn.ModuleList(
                [
                    MambaAttentionLayer(
                        model_dim=model_dim,
                        num_heads=num_heads,
                        qkv_bias=True,
                        d_state=mamba_d_state,
                        d_conv=mamba_d_conv,
                        expand=mamba_expand,
                    )
                    for _ in range(self.order)
                ]
            )
        elif temporal_mode == TemporalMode.TCN:
            # Use TCNTemporalLayer for temporal processing
            self.attn = torch.nn.ModuleList(
                [
                    TCNTemporalLayer(
                        model_dim=model_dim,
                        num_heads=num_heads,
                        qkv_bias=True,
                        num_layers=tcn_num_layers,
                        kernel_size=tcn_kernel_size,
                        dilation_base=tcn_dilation_base,
                        dropout=tcn_dropout,
                    )
                    for _ in range(self.order)
                ]
            )
        elif temporal_mode == TemporalMode.DEPTHWISE:
            # Use DepthwiseTemporalLayer for temporal processing
            self.attn = torch.nn.ModuleList(
                [
                    DepthwiseTemporalLayer(
                        model_dim=model_dim,
                        num_heads=num_heads,
                        qkv_bias=True,
                        kernel_size=depthwise_kernel_size,
                        dropout=p_dropout,
                    )
                    for _ in range(self.order)
                ]
            )
        elif temporal_mode == TemporalMode.MLP:
            # Use MLPTemporalLayer for temporal processing
            self.attn = torch.nn.ModuleList(
                [
                    MLPTemporalLayer(
                        model_dim=model_dim,
                        num_heads=num_heads,
                        qkv_bias=True,
                        hidden_dim=mlp_hidden_dim,
                        dropout=p_dropout,
                    )
                    for _ in range(self.order)
                ]
            )
        else:
            # Default: FastAttentionLayer for temporal processing
            self.attn = torch.nn.ModuleList(
                [
                    FastAttentionLayer(
                        model_dim=model_dim, num_heads=num_heads, qkv_bias=True
                    )
                    for _ in range(self.order)
                ]
            )
        # Projection weights for combining orders
        self.order_proj = torch.nn.ModuleList(
            [torch.nn.Linear(model_dim, model_dim) for _ in range(self.order)]
        )
        # Zero initialize to match external STGFormer, or xavier for potentially better training
        if use_zero_init:
            for proj in self.order_proj:
                torch.nn.init.constant_(proj.weight, 0)
                torch.nn.init.constant_(proj.bias, 0)
        else:
            for proj in self.order_proj:
                torch.nn.init.xavier_uniform_(proj.weight)
                torch.nn.init.zeros_(proj.bias)

        # MLP and layer norms
        hidden_features = int(model_dim * mlp_ratio)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=model_dim, out_features=hidden_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(p_dropout),
            torch.nn.Linear(hidden_features, model_dim),
            torch.nn.Dropout(p_dropout),
        )
        self.ln1 = torch.nn.LayerNorm(model_dim)
        self.ln2 = torch.nn.LayerNorm(model_dim)

        self.dropout = torch.nn.Dropout(p_dropout)

    def forward(self, x, graph):
        """
        Combined local + global processing.

        The hybrid insight: Node i's representation combines:
        - local_info: what neighbors say (graph-aware via propagation)
        - global_info: what ALL nodes say (graph-agnostic attention)
        """

        # Branch 1: Graph propagation (LOCAL - only neighbors via A @ x)
        # Returns [x, Ax, A²x, ...] for power mode, or [T₀x, T₁x, ...] for Chebyshev
        x_local = self.locals(x, graph)

        # Branch 2: Attention (GLOBAL - full attention, no graph masking)
        # Each attention layer processes k-hop features but sees ALL nodes
        if self.temporal_mode == TemporalMode.NONE:
            # No temporal processing - only use graph propagation
            # Sum up the graph-propagated features (local information only)
            x_global = x
            for i in range(self.order):
                x_global = x_global + self.order_proj[i](x_local[i])
        else:
            scale = 1
            att_prev = x
            x_global = x
            for i in range(self.order):
                # Attention over ALL nodes (not masked by graph!)
                att_outputs = self.attn[i](x_local[i])
                # Combine with projections and scaling
                x_global = x_global + att_outputs * self.order_proj[i](att_prev) * scale
                att_prev = att_outputs
                scale = (
                    10 ** -(i + 2)
                )  # generalization of scale = [1, 0.01, 0.001] from the original implementation

        x = self.ln1(x + self.dropout(x_global))
        x = self.ln2(x + self.dropout(self.mlp(x)))

        return x
