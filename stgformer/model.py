"""STGFormer: Hybrid Graph Transformer Implementation

STGFormer is a HYBRID GRAPH TRANSFORMER that combines:
- Transformer attention for global spatio-temporal patterns
- Graph convolution for local spatial structure

ARCHITECTURE OVERVIEW:
STGFormer is primarily a Graph Transformer but uses a two-branch design:

Branch 1 (Local): Graph Convolution
- Captures local spatial patterns through K-hop neighborhoods
- Traditional GCN-style message passing
- Efficient for high-interaction local patterns

Branch 2 (Global): Transformer Attention
- Captures global spatio-temporal dependencies
- Linear complexity attention (key innovation)
- Efficient for sparse long-range interactions

The two branches are combined with learned weights in each layer.

Core components to implement:

1. FastAttentionLayer - The transformer innovation
   - Linear attention O(N+T) instead of O(N²)
   - Unified spatio-temporal processing
   - Decomposed inner products: Q(K^TV) instead of (QK^T)V

2. GraphPropagate - The graph convolution component
   - K-hop neighborhood aggregation (like GCN/GraphSAGE)
   - Provides local spatial inductive bias
   - NOT the main architecture - just one branch

3. GraphTemporalLayer - Hybrid layer combining both branches
   - GraphPropagate for local patterns
   - FastAttentionLayer for global patterns
   - Learned combination with residual connections
   - Note: Called 'SelfAttentionLayer' in original STGFormer repo

4. STGFormer - Main hybrid model
   - Adaptive graph learning (learns graph structure)
   - Stack of hybrid GraphTemporalLayer blocks
   - Transformer-style but with graph inductive bias

5. MambaAttentionLayer - Alternative temporal processor
   - Replaces attention with Mamba SSM for temporal branch
   - O(T) complexity, native sequential modeling
   - CUDA-only: requires NVIDIA GPU
"""

import torch

from stgformer import embedding_utils, graph_utils
from stgformer.enums import GraphMode, PropagationMode, TemporalMode
from stgformer.graph_temporal_layer import GraphTemporalLayer
from stgformer.temporal_processing import (
    MAMBA_AVAILABLE,
    compute_chebyshev_polynomials,
    compute_scaled_laplacian,
)


class TemporalKernelPooling(torch.nn.Module):
    """Temporal pooling helper that averages along the last dimension with stride 1."""

    def __init__(self, kernel_size: int = 1):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = kernel_size
        self.pool = (
            torch.nn.AvgPool1d(kernel_size=kernel_size, stride=1)
            if kernel_size > 1
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool is None or self.kernel_size == 1:
            return x
        *leading_dims, length = x.shape
        if self.kernel_size > length:
            raise ValueError(
                f"kernel_size ({self.kernel_size}) cannot exceed length ({length})"
            )
        flat = x.reshape(-1, 1, length)
        pooled = self.pool(flat)
        new_len = pooled.shape[-1]
        return pooled.reshape(*leading_dims, new_len)


class STGFormer(torch.nn.Module):
    """
    STGFormer: Hybrid Graph Transformer for spatio-temporal forecasting.

    ARCHITECTURE: Transformer with graph inductive bias
    - PRIMARY: Transformer-style attention for global spatio-temporal modeling
    - SECONDARY: Graph convolution for local spatial structure
    - INNOVATION: Linear attention + adaptive graph learning

    Architecture flow:
    1. Input embeddings (traffic values + temporal features)
    2. Adaptive graph construction (learns graph structure from data!)
    3. Multi-scale temporal convolutions for temporal patterns
    4. Stack of hybrid GraphTemporalLayer blocks (graph + transformer)
    5. MLP encoder and output projection

    Key differences from pure transformers:
    - Uses graph structure for spatial inductive bias
    - Learns adaptive graph from embeddings (not fixed adjacency)
    - Combines local (GCN) and global (attention) processing

    Graph Mode Options:
    - "learned": Adaptive graph from learned embeddings (default, original STGFormer)
    - "spectral_init": Learn adaptive graph, initialized from Laplacian eigenvectors
    - "geographic": Use pre-computed geographic adjacency only
    - "hybrid": Combine geographic and learned adjacency with lambda weighting

    Propagation Mode Options:
    - "power": Simple matrix powers A^k (original STGFormer)
    - "chebyshev": Chebyshev polynomials T_k(L_scaled) for spectral filtering
    """

    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        input_dim=1,  # Number of raw value features (speed, volume, etc.)
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=0,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        dropout_a=0.3,  # Dropout for adaptive embeddings (separate from attention dropout)
        steps_per_day=288,
        mlp_ratio=4,
        use_mixed_proj=True,
        # Graph structure parameters
        graph_mode: GraphMode = GraphMode.LEARNED,
        geo_adj: "torch.Tensor | None" = None,  # Pre-computed geographic adjacency
        lambda_hybrid: float = 0.5,  # Weight for hybrid mode (geo_adj weight)
        sparsity_k: "int | None" = None,  # Top-k sparsification (None = dense)
        # Propagation parameters
        propagation_mode: PropagationMode = PropagationMode.POWER,
        # Temporal processing parameters
        temporal_mode: TemporalMode = TemporalMode.TRANSFORMER,
        mamba_d_state: int = 16,  # Mamba SSM state dimension
        mamba_d_conv: int = 4,  # Mamba convolution kernel size
        mamba_expand: int = 2,  # Mamba expansion factor
        tcn_num_layers: int = 3,  # TCN number of layers
        tcn_kernel_size: int = 3,  # TCN convolution kernel size
        tcn_dilation_base: int = 2,  # TCN dilation base (exponential growth)
        tcn_dropout: float = 0.1,  # TCN dropout rate
        depthwise_kernel_size: int = 3,  # Depthwise conv kernel size
        mlp_hidden_dim: "int | None" = None,  # MLP hidden dimension (None = model_dim)
        # Initialization parameters
        use_zero_init: bool = True,  # Zero init order_proj weights (matches external STGFormer)
        pre_attn_kernel_size: int = 1,  # Temporal kernel applied before attention/graph pooling
        # Prediction head configuration
        prediction_head_layers: int = 1,  # 1 = linear (default), 2+ = MLP
    ):
        super().__init__()

        if (
            graph_mode
            in (GraphMode.GEOGRAPHIC, GraphMode.HYBRID, GraphMode.SPECTRAL_INIT)
            and geo_adj is None
        ):
            raise ValueError(f"geo_adj must be provided when graph_mode={graph_mode}")

        if propagation_mode == PropagationMode.CHEBYSHEV and geo_adj is None:
            raise ValueError("geo_adj must be provided for CHEBYSHEV propagation mode")

        if graph_mode == GraphMode.NONE and adaptive_embedding_dim > 0:
            raise ValueError(
                "adaptive_embedding_dim must be 0 when graph_mode=NONE "
                "(adaptive embeddings are not used without graph propagation)"
            )

        if temporal_mode == TemporalMode.MAMBA and not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for TemporalMode.MAMBA but not installed. "
                "Install with: pip install mamba-ssm causal-conv1d. "
                "Note: mamba-ssm requires CUDA-enabled GPU."
            )

        # Store configuration
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        # Graph structure configuration
        self.graph_mode = graph_mode
        self.lambda_hybrid = lambda_hybrid
        self.sparsity_k = sparsity_k
        self.propagation_mode = propagation_mode

        # Temporal processing configuration
        self.temporal_mode = temporal_mode
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.tcn_num_layers = tcn_num_layers
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_dilation_base = tcn_dilation_base
        self.tcn_dropout = tcn_dropout
        self.depthwise_kernel_size = depthwise_kernel_size
        self.mlp_hidden_dim = mlp_hidden_dim

        if pre_attn_kernel_size <= 0:
            raise ValueError("pre_attn_kernel_size must be positive")
        if pre_attn_kernel_size > in_steps:
            raise ValueError("pre_attn_kernel_size cannot exceed in_steps")
        if not use_mixed_proj and pre_attn_kernel_size != 1:
            raise ValueError(
                "pre_attn_kernel_size > 1 requires use_mixed_proj=True to keep temporal dimensions aligned"
            )
        self.pre_attn_kernel_size = pre_attn_kernel_size

        # Initialization configuration
        self.use_zero_init = use_zero_init

        # Register geographic adjacency as buffer (not a parameter, but moves with model)
        if geo_adj is not None:
            self.register_buffer("geo_adj", geo_adj)
        else:
            self.geo_adj = None

        # Compute Chebyshev polynomials if needed (from geographic Laplacian)
        chebyshev_polynomials = None
        if propagation_mode == PropagationMode.CHEBYSHEV:
            L_scaled = compute_scaled_laplacian(geo_adj)
            chebyshev_polynomials = compute_chebyshev_polynomials(
                L_scaled, K=2
            )  # order=2

        # Calculate total model dimension from embedding components
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )

        # Input projection (raw value features only, tod/dow separate)
        self.input_proj = torch.nn.Linear(input_dim, input_embedding_dim)

        # Temporal embeddings
        if tod_embedding_dim > 0:
            self.tod_embedding = torch.nn.Embedding(steps_per_day, tod_embedding_dim)
        else:
            self.tod_embedding = None

        if dow_embedding_dim > 0:
            self.dow_embedding = torch.nn.Embedding(7, dow_embedding_dim)
        else:
            self.dow_embedding = None

        # Adaptive embeddings for graph learning (uses utility function)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = embedding_utils.create_adaptive_embedding(
                graph_mode, num_nodes, in_steps, adaptive_embedding_dim, geo_adj
            )
        else:
            self.adaptive_embedding = None

        # Dropout for adaptive embeddings (uses dropout_a, separate from attention dropout)
        self.dropout_a = torch.nn.Dropout(dropout_a)

        # Pooling for graph construction (temporal averaging along adaptive embeddings)
        self.pooling = TemporalKernelPooling(kernel_size=self.pre_attn_kernel_size)

        # Hybrid attention layers (spatial + temporal attention combined)
        self.attn = torch.nn.ModuleList(
            [
                GraphTemporalLayer(
                    model_dim=self.model_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    p_dropout=dropout,
                    order=2,
                    propagation_mode=propagation_mode,
                    chebyshev_polynomials=chebyshev_polynomials,
                    use_zero_init=use_zero_init,
                    temporal_mode=temporal_mode,
                    mamba_d_state=mamba_d_state,
                    mamba_d_conv=mamba_d_conv,
                    mamba_expand=mamba_expand,
                    tcn_num_layers=tcn_num_layers,
                    tcn_kernel_size=tcn_kernel_size,
                    tcn_dilation_base=tcn_dilation_base,
                    tcn_dropout=tcn_dropout,
                    depthwise_kernel_size=depthwise_kernel_size,
                    mlp_hidden_dim=mlp_hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Determine temporal length after optional projection
        if use_mixed_proj:
            self.temporal_out_steps = in_steps - (self.pre_attn_kernel_size - 1)
        else:
            self.temporal_out_steps = in_steps

        # Temporal convolution (if using mixed projection)
        if use_mixed_proj:
            self.temporal_proj = torch.nn.Conv2d(
                self.model_dim,
                self.model_dim,
                kernel_size=(1, self.pre_attn_kernel_size),
                stride=1,
                padding=0,
            )
        else:
            self.temporal_proj = None

        # Encoder projection (compresses temporal dimension)
        self.encoder_proj = torch.nn.Linear(
            self.temporal_out_steps * self.model_dim, self.model_dim
        )

        # MLP encoder layers
        self.encoder = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.model_dim, self.model_dim * mlp_ratio),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(self.model_dim * mlp_ratio, self.model_dim),
                    torch.nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection (prediction head)
        self.prediction_head_layers = prediction_head_layers
        if prediction_head_layers <= 1:
            # Single linear layer (original STGFormer)
            self.output_proj = torch.nn.Linear(self.model_dim, out_steps * output_dim)
        else:
            # MLP prediction head
            layers = []
            for i in range(prediction_head_layers - 1):
                layers.extend([
                    torch.nn.Linear(self.model_dim, self.model_dim),
                    torch.nn.ReLU(),
                ])
            layers.append(torch.nn.Linear(self.model_dim, out_steps * output_dim))
            self.output_proj = torch.nn.Sequential(*layers)

    def _embed_and_encode(self, x):
        """
        Shared embedding and encoding logic used by both forward() and
        get_spatial_representation().

        Args:
            x: Input [batch, in_steps, num_nodes, total_features]

        Returns:
            Spatial representation [batch, num_nodes, model_dim]
        """
        batch_size = x.shape[0]

        # Extract temporal features from FIXED positions (matching external STGFormer)
        # tod is always at index 1, dow at index 2 (if present)
        if self.tod_embedding is not None:
            tod = x[..., 1]  # Fixed index, matches external

        if self.dow_embedding is not None:
            dow = x[..., 2]  # Fixed index, matches external

        # Slice to get input features and project
        x = x[..., : self.input_dim]
        x = self.input_proj(x)

        # Build features list: [input_proj, tod_emb, dow_emb, adaptive_emb]
        features = [x]

        if self.tod_embedding is not None:
            tod_emb = self.tod_embedding((tod * self.steps_per_day).long())
            features.append(tod_emb)

        if self.dow_embedding is not None:
            dow_emb = self.dow_embedding(dow.long())
            features.append(dow_emb)

        if self.adaptive_embedding is not None:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(self.dropout_a(adp_emb))

        # Concatenate all embeddings to model_dim
        x = torch.cat(features, dim=-1)

        # Temporal convolution (if using mixed projection)
        if self.temporal_proj is not None:
            x = self.temporal_proj(x.transpose(1, 3)).transpose(1, 3)

        # Construct adaptive graph and process through attention layers
        graph = self._construct_adaptive_graph(self.adaptive_embedding)
        for attn in self.attn:
            x = attn(x, graph)

        # Compress temporal dimension and process through encoder MLPs
        x = self.encoder_proj(x.transpose(1, 2).flatten(-2))
        for layer in self.encoder:
            x = x + layer(x)

        return x  # [batch, num_nodes, model_dim]

    def forward(self, x):
        """
        Args:
            x: Input [batch, in_steps, num_nodes, total_features]
               Format matches external STGFormer: [value, tod, dow (optional)]
               With input_dim=2: both value and tod go through input_proj,
               and tod (at index 1) is also used for tod_embedding.

        Returns:
            Predictions [batch, out_steps, num_nodes, output_dim]
        """
        batch_size = x.shape[0]

        # Get spatial representation
        x = self._embed_and_encode(x)

        # Project to output and reshape
        x = self.output_proj(x)
        x = x.view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
        x = x.transpose(1, 2)

        return x

    def get_spatial_representation(self, x):
        """
        Extract spatial representation without the forecasting head.

        This method is used for pretraining tasks like masked node prediction,
        where we want the encoded spatial representation per node without
        projecting to the output timesteps.

        Args:
            x: Input [batch, in_steps, num_nodes, total_features]

        Returns:
            Spatial representation [batch, num_nodes, model_dim]
            - Temporal dimension has been compressed
            - Contains learned spatial features per node
            - Suitable for downstream tasks like imputation
        """
        return self._embed_and_encode(x)

    def _construct_adaptive_graph(self, embeddings):
        """
        Construct graph based on graph_mode setting (delegates to utility function).

        See graph_utils.construct_adaptive_graph() for full documentation.

        Args:
            embeddings: [in_steps, num_nodes, adaptive_embedding_dim]

        Returns:
            graph: [num_nodes, num_nodes] - Probabilistic adjacency matrix
        """
        return graph_utils.construct_adaptive_graph(
            self.graph_mode,
            embeddings,
            self.pooling,
            self.geo_adj,
            self.lambda_hybrid,
            self.sparsity_k,
            self.num_nodes,
        )
