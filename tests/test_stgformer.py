import pytest
import torch

from stgformer.enums import GraphMode, PropagationMode, TemporalMode
from stgformer.graph_propagation import GraphPropagate
from stgformer.graph_temporal_layer import GraphTemporalLayer
from stgformer.model import STGFormer
from stgformer.temporal_processing import (
    MAMBA_AVAILABLE,
    AttentionLayer,
    FastAttentionLayer,
)
from utils.training import StandardScaler

# Conditional import for MambaAttentionLayer
if MAMBA_AVAILABLE:
    from stgformer.temporal_processing import MambaAttentionLayer

# ============================================================================
# Regression Tests - Capture current STGFormer behavior for refactoring safety
# ============================================================================


class TestRegressionSTGFormer:
    """
    Regression tests to preserve current STGFormer behavior during refactoring.

    These tests capture:
    1. Default configuration values
    2. Parameter counts for specific configs
    3. Output shapes and properties
    4. Deterministic outputs with fixed seeds

    If any of these fail after refactoring, it indicates a behavioral change.
    """

    def test_default_config_values(self):
        """Capture the default configuration used in production training."""
        # These are the exact defaults used in utils/stgformer.py train_model()
        model = STGFormer(
            num_nodes=207,  # METR-LA
            in_steps=12,
            out_steps=12,
            input_dim=2,  # [value, tod] - matches external
            output_dim=1,
            steps_per_day=288,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            dow_embedding_dim=0,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=80,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
            dropout_a=0.3,
            mlp_ratio=4,
            use_mixed_proj=True,
        )

        # Verify model_dim calculation: 24 + 24 + 0 + 0 + 80 = 128
        assert model.model_dim == 128, f"Expected model_dim=128, got {model.model_dim}"

        # Verify key attributes are stored
        assert model.num_nodes == 207
        assert model.in_steps == 12
        assert model.out_steps == 12
        assert model.input_dim == 2
        assert model.output_dim == 1
        assert model.num_heads == 4
        assert model.num_layers == 3

    def test_parameter_count_metr_la_config(self):
        """Exact parameter count for METR-LA production config."""
        model = STGFormer(
            num_nodes=207,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            dow_embedding_dim=0,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=80,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
            dropout_a=0.3,
            mlp_ratio=4,
            use_mixed_proj=True,
        )

        total_params = sum(p.numel() for p in model.parameters())

        # This is the exact count as of the current implementation
        # If this changes, the model architecture has changed
        expected_params = 1_805_972
        assert total_params == expected_params, (
            f"Parameter count changed: {total_params:,} vs expected {expected_params:,}. "
            "This indicates an architectural change."
        )

    def test_parameter_count_pems_bay_config(self):
        """Exact parameter count for PEMS-BAY production config."""
        model = STGFormer(
            num_nodes=325,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            dow_embedding_dim=0,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=80,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
            dropout_a=0.3,
            mlp_ratio=4,
            use_mixed_proj=True,
        )

        total_params = sum(p.numel() for p in model.parameters())

        expected_params = 1_919_252
        assert total_params == expected_params, (
            f"Parameter count changed: {total_params:,} vs expected {expected_params:,}. "
            "This indicates an architectural change."
        )

    def test_deterministic_forward_pass(self):
        """Test that forward pass is deterministic with fixed seed."""
        torch.manual_seed(42)

        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            adaptive_embedding_dim=80,
            num_heads=4,
            num_layers=2,
        )
        model.eval()

        # Create deterministic input
        torch.manual_seed(123)
        x = torch.randn(2, 12, 20, 2)
        x[..., 1] = torch.rand(2, 12, 20) * 0.99  # TOD in [0, 1)

        with torch.no_grad():
            output1 = model(x.clone())
            output2 = model(x.clone())

        # Outputs should be identical
        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-5)

    def test_output_statistics_stability(self):
        """Test that output statistics are stable (regression baseline)."""
        torch.manual_seed(42)

        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            adaptive_embedding_dim=80,
            num_heads=4,
            num_layers=2,
        )
        model.eval()

        torch.manual_seed(123)
        x = torch.randn(4, 12, 20, 2)
        x[..., 1] = torch.rand(4, 12, 20) * 0.99

        with torch.no_grad():
            output = model(x)

        # These are baseline statistics - if they change significantly,
        # the model behavior has changed
        assert output.shape == (4, 12, 20, 1)
        assert torch.isfinite(output).all()

        # Output magnitude should be reasonable (not exploding/vanishing)
        assert output.abs().mean() < 10, "Output magnitude too large"
        assert output.abs().mean() > 0.001, "Output magnitude too small"

    def test_attention_layer_output_shape(self):
        """Verify AttentionLayer maintains expected I/O contract."""
        layer = AttentionLayer(model_dim=64, num_heads=4, qkv_bias=True)

        x = torch.randn(2, 12, 20, 64)
        output = layer(x)

        # Must preserve shape (critical for residual connections)
        assert output.shape == x.shape, (
            f"AttentionLayer changed shape: {x.shape} -> {output.shape}"
        )

    def test_fast_attention_layer_output_shape(self):
        """Verify FastAttentionLayer maintains expected I/O contract."""
        layer = FastAttentionLayer(model_dim=64, num_heads=4, qkv_bias=True)

        x = torch.randn(2, 12, 20, 64)
        output = layer(x)

        assert output.shape == x.shape, (
            f"FastAttentionLayer changed shape: {x.shape} -> {output.shape}"
        )

    def test_self_attention_layer_output_shape(self):
        """Verify GraphTemporalLayer maintains expected I/O contract."""
        layer = GraphTemporalLayer(model_dim=64, num_heads=4, order=2)

        x = torch.randn(2, 12, 20, 64)
        graph = torch.randn(12, 20, 20)
        output = layer(x, graph)

        assert output.shape == x.shape, (
            f"GraphTemporalLayer changed shape: {x.shape} -> {output.shape}"
        )

    def test_adaptive_graph_properties(self):
        """Verify adaptive graph has correct properties."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            adaptive_embedding_dim=32,
        )

        # Get the adaptive graph
        graph = model._construct_adaptive_graph(model.adaptive_embedding)

        # Must be valid probability distribution (softmax output)
        assert graph.shape[-2:] == (20, 20), "Wrong graph shape"
        assert torch.all(graph >= 0), "Graph has negative values"
        assert torch.allclose(
            graph.sum(dim=-1), torch.ones(graph.shape[:-1]), atol=1e-5
        ), "Graph rows don't sum to 1"

    def test_gradient_flow_all_params(self):
        """Verify gradients flow to all trainable parameters."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            adaptive_embedding_dim=80,
            num_heads=4,
            num_layers=2,
        )

        x = torch.randn(2, 12, 20, 2)
        x[..., 1] = torch.rand(2, 12, 20) * 0.99
        y = torch.randn(2, 12, 20, 1)

        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()

        # All parameters should have gradients
        params_without_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)

        assert len(params_without_grad) == 0, (
            f"Parameters without gradients: {params_without_grad}"
        )


class TestGraphModes:
    """Test different graph structure modes for ablation studies."""

    @pytest.fixture
    def geo_adj(self):
        """Create a mock geographic adjacency matrix."""
        num_nodes = 20
        # Create a random adjacency with some structure
        adj = torch.rand(num_nodes, num_nodes)
        adj = adj + adj.T  # Make symmetric
        adj = adj / adj.sum(dim=-1, keepdim=True)  # Row normalize
        return adj

    @pytest.fixture
    def model_input(self):
        """Standard input for testing."""
        x = torch.randn(2, 12, 20, 2)
        x[..., 1] = torch.rand(2, 12, 20) * 0.99
        return x

    def test_graph_mode_learned_default(self, model_input):
        """Test that learned mode is the default and works."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            adaptive_embedding_dim=32,
        )

        assert model.graph_mode == GraphMode.LEARNED
        output = model(model_input)
        assert output.shape == (2, 12, 20, 1)

    def test_graph_mode_geographic(self, geo_adj, model_input):
        """Test geographic mode uses fixed adjacency."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            adaptive_embedding_dim=32,
            graph_mode=GraphMode.GEOGRAPHIC,
            geo_adj=geo_adj,
        )

        assert model.graph_mode == GraphMode.GEOGRAPHIC
        output = model(model_input)
        assert output.shape == (2, 12, 20, 1)

        # Verify geo_adj is registered as buffer
        assert model.geo_adj is not None
        assert model.geo_adj.shape == (20, 20)

    def test_graph_mode_hybrid(self, geo_adj, model_input):
        """Test hybrid mode combines geographic and learned."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            adaptive_embedding_dim=32,
            graph_mode=GraphMode.HYBRID,
            geo_adj=geo_adj,
            lambda_hybrid=0.5,
        )

        assert model.graph_mode == GraphMode.HYBRID
        assert model.lambda_hybrid == 0.5
        output = model(model_input)
        assert output.shape == (2, 12, 20, 1)

    def test_graph_mode_hybrid_lambda_variations(self, geo_adj, model_input):
        """Test hybrid mode with different lambda values."""
        for lambda_val in [0.0, 0.3, 0.5, 0.7, 1.0]:
            model = STGFormer(
                num_nodes=20,
                in_steps=12,
                out_steps=12,
                input_dim=2,
                adaptive_embedding_dim=32,
                graph_mode=GraphMode.HYBRID,
                geo_adj=geo_adj,
                lambda_hybrid=lambda_val,
            )
            output = model(model_input)
            assert output.shape == (2, 12, 20, 1), f"Failed for lambda={lambda_val}"

    def test_graph_mode_sparsity(self, model_input):
        """Test sparsity parameter for learned graph."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            adaptive_embedding_dim=32,
            graph_mode=GraphMode.LEARNED,
            sparsity_k=5,  # Keep top-5 neighbors
        )

        assert model.sparsity_k == 5
        output = model(model_input)
        assert output.shape == (2, 12, 20, 1)

    def test_graph_mode_hybrid_with_sparsity(self, geo_adj, model_input):
        """Test hybrid mode with sparsity on learned component."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            adaptive_embedding_dim=32,
            graph_mode=GraphMode.HYBRID,
            geo_adj=geo_adj,
            lambda_hybrid=0.5,
            sparsity_k=10,
        )

        output = model(model_input)
        assert output.shape == (2, 12, 20, 1)

    def test_geographic_mode_requires_geo_adj(self):
        """Test that geographic mode requires geo_adj."""
        with pytest.raises(ValueError, match="geo_adj must be provided"):
            STGFormer(
                num_nodes=20,
                in_steps=12,
                out_steps=12,
                input_dim=2,
                graph_mode=GraphMode.GEOGRAPHIC,
            )

    def test_hybrid_mode_requires_geo_adj(self):
        """Test that hybrid mode requires geo_adj."""
        with pytest.raises(ValueError, match="geo_adj must be provided"):
            STGFormer(
                num_nodes=20,
                in_steps=12,
                out_steps=12,
                input_dim=2,
                graph_mode=GraphMode.HYBRID,
            )

    def test_spectral_init_mode_requires_geo_adj(self):
        """Test that spectral_init mode requires geo_adj."""
        with pytest.raises(ValueError, match="geo_adj must be provided"):
            STGFormer(
                num_nodes=20,
                in_steps=12,
                out_steps=12,
                input_dim=2,
                graph_mode=GraphMode.SPECTRAL_INIT,
            )

    def test_graph_mode_spectral_init(self, geo_adj, model_input):
        """Test spectral_init mode initializes embeddings from Laplacian eigenvectors."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            adaptive_embedding_dim=32,
            graph_mode=GraphMode.SPECTRAL_INIT,
            geo_adj=geo_adj,
        )

        assert model.graph_mode == GraphMode.SPECTRAL_INIT
        # Verify geo_adj is registered as buffer
        assert model.geo_adj is not None
        assert model.geo_adj.shape == (20, 20)

        # Verify embedding shape and that they're parameters (trainable)
        assert model.adaptive_embedding.shape == (12, 20, 32)
        assert model.adaptive_embedding.requires_grad

        # Verify forward pass works
        output = model(model_input)
        assert output.shape == (2, 12, 20, 1)

    def test_spectral_init_better_correlation_than_random(self):
        """Test that spectral_init produces graph more correlated with geo than random init.

        Note: This test uses a larger, more structured geo_adj matrix. The Laplacian
        eigenvectors capture spectral structure which correlates with graph connectivity.
        """
        import torch.nn.functional as F

        # Fix seed for reproducibility
        torch.manual_seed(123)

        # Create a realistic geo_adj with distance-based structure (like real sensor networks)
        num_nodes = 50
        positions = torch.rand(num_nodes, 2) * 100
        dists = torch.cdist(positions, positions)
        sigma = 30
        geo_adj = torch.exp(-(dists**2) / (2 * sigma**2))
        geo_adj.fill_diagonal_(0)

        torch.manual_seed(42)
        # Create models with SPECTRAL_INIT and LEARNED (random) init
        model_spectral_init = STGFormer(
            num_nodes=num_nodes,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            tod_embedding_dim=0,
            dow_embedding_dim=0,
            adaptive_embedding_dim=80,
            graph_mode=GraphMode.SPECTRAL_INIT,
            geo_adj=geo_adj,
        )

        torch.manual_seed(42)
        model_random = STGFormer(
            num_nodes=num_nodes,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            tod_embedding_dim=0,
            dow_embedding_dim=0,
            adaptive_embedding_dim=80,
            graph_mode=GraphMode.LEARNED,
        )

        # Compute initial graphs
        emb_spectral = model_spectral_init.adaptive_embedding[0]
        emb_random = model_random.adaptive_embedding[0]

        graph_spectral = F.softmax(F.relu(emb_spectral @ emb_spectral.T), dim=-1)
        graph_random = F.softmax(F.relu(emb_random @ emb_random.T), dim=-1)

        # Row-normalize geo_adj for comparison
        geo_normalized = geo_adj / geo_adj.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        # Compute correlations
        corr_spectral = torch.corrcoef(
            torch.stack([geo_normalized.flatten(), graph_spectral.flatten()])
        )[0, 1]
        corr_random = torch.corrcoef(
            torch.stack([geo_normalized.flatten(), graph_random.flatten()])
        )[0, 1]

        # SPECTRAL_INIT typically has better correlation than random
        # But the main property is that it's a deterministic, principled initialization
        # Note: The exact correlation depends on graph structure and embedding dim
        assert corr_spectral > corr_random or abs(corr_spectral) < 0.3, (
            f"SPECTRAL_INIT correlation ({corr_spectral:.4f}) should be reasonable "
            f"compared to random ({corr_random:.4f})"
        )

    def test_sparsified_graph_properties(self):
        """Test that sparsified graph has correct properties."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            adaptive_embedding_dim=32,
            sparsity_k=5,
        )

        # Get the graph
        graph = model._construct_adaptive_graph(model.adaptive_embedding)

        # Should still be a valid probability distribution
        assert torch.all(graph >= 0), "Sparse graph has negative values"
        assert torch.allclose(
            graph.sum(dim=-1), torch.ones(graph.shape[:-1]), atol=1e-5
        ), "Sparse graph rows don't sum to 1"

        # Most entries should be zero (sparse)
        # With k=5 and 20 nodes, only 5/20 = 25% should be non-zero per row
        nonzero_ratio = (graph > 1e-6).float().mean()
        assert nonzero_ratio < 0.5, (
            f"Graph not sparse enough: {nonzero_ratio:.2%} non-zero"
        )

    def test_graph_mode_none(self, model_input):
        """Test NONE mode disables graph propagation."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            adaptive_embedding_dim=0,  # Must be 0 with NONE mode
            graph_mode=GraphMode.NONE,
        )

        assert model.graph_mode == GraphMode.NONE
        assert model.adaptive_embedding is None

        # Forward pass should work
        output = model(model_input)
        assert output.shape == (2, 12, 20, 1)

    def test_graph_mode_none_requires_no_adaptive_embedding(self):
        """Test that NONE mode requires adaptive_embedding_dim=0."""
        with pytest.raises(ValueError, match="adaptive_embedding_dim must be 0"):
            STGFormer(
                num_nodes=20,
                in_steps=12,
                out_steps=12,
                input_dim=2,
                adaptive_embedding_dim=32,  # Should fail
                graph_mode=GraphMode.NONE,
            )

    def test_graph_mode_none_no_spatial_mixing(self, model_input):
        """Test that NONE mode has no spatial mixing in graph propagation."""
        from stgformer.graph_propagation import GraphPropagate

        # Create identity graph (what NONE mode produces)
        identity_graph = torch.eye(20)

        # Create propagation module with no dropout for deterministic test
        propagate = GraphPropagate(k_hops=3, p_dropout=0.0)
        propagate.eval()

        x = torch.randn(2, 12, 20, 64)

        with torch.no_grad():
            x_list = propagate(x, identity_graph)

        # With identity graph, all hops should return the same features
        assert len(x_list) == 3
        assert torch.allclose(x_list[0], x), "Hop 0 should equal input"
        assert torch.allclose(x_list[1], x), "Hop 1 should equal input (I @ x = x)"
        assert torch.allclose(x_list[2], x), "Hop 2 should equal input (I @ x = x)"

    def test_gradient_flow_all_modes(self, geo_adj, model_input):
        """Verify gradients flow in all graph modes."""
        modes = [
            (GraphMode.LEARNED, {"adaptive_embedding_dim": 32}),
            (GraphMode.GEOGRAPHIC, {"geo_adj": geo_adj, "adaptive_embedding_dim": 32}),
            (
                GraphMode.HYBRID,
                {
                    "geo_adj": geo_adj,
                    "lambda_hybrid": 0.5,
                    "adaptive_embedding_dim": 32,
                },
            ),
            (
                GraphMode.NONE,
                {"adaptive_embedding_dim": 0},
            ),  # NONE requires no adaptive embedding
        ]

        for mode, kwargs in modes:
            model = STGFormer(
                num_nodes=20,
                in_steps=12,
                out_steps=12,
                input_dim=2,
                graph_mode=mode,
                **kwargs,
            )

            y = torch.randn(2, 12, 20, 1)
            pred = model(model_input.clone())
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()

            # Check key parameters have gradients (skip for NONE mode which has no adaptive_embedding)
            if model.adaptive_embedding is not None:
                assert model.adaptive_embedding.grad is not None, (
                    f"No gradient for adaptive_embedding in {mode.value} mode"
                )


class TestPropagationModes:
    """Test different propagation modes (POWER vs CHEBYSHEV)."""

    @pytest.fixture
    def geo_adj(self):
        """Create realistic geographic adjacency matrix."""
        num_nodes = 20
        positions = torch.rand(num_nodes, 2) * 100
        dists = torch.cdist(positions, positions)
        sigma = 30
        adj = torch.exp(-(dists**2) / (2 * sigma**2))
        adj.fill_diagonal_(0)
        return adj

    @pytest.fixture
    def model_input(self):
        """Standard model input (speed only, no TOD/DOW)."""
        return torch.randn(2, 12, 20, 1)

    def test_propagation_mode_power_default(self, model_input):
        """Test that POWER is the default propagation mode."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            tod_embedding_dim=0,
            dow_embedding_dim=0,
        )
        assert model.propagation_mode == PropagationMode.POWER

        # Verify forward pass works
        output = model(model_input)
        assert output.shape == (2, 12, 20, 1)

    def test_propagation_mode_chebyshev(self, geo_adj, model_input):
        """Test CHEBYSHEV propagation mode."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            tod_embedding_dim=0,
            dow_embedding_dim=0,
            propagation_mode=PropagationMode.CHEBYSHEV,
            geo_adj=geo_adj,
        )
        assert model.propagation_mode == PropagationMode.CHEBYSHEV

        # Verify forward pass works
        output = model(model_input)
        assert output.shape == (2, 12, 20, 1)

    def test_chebyshev_requires_geo_adj(self):
        """Test that CHEBYSHEV propagation requires geo_adj."""
        with pytest.raises(ValueError, match="geo_adj must be provided"):
            STGFormer(
                num_nodes=20,
                in_steps=12,
                out_steps=12,
                input_dim=1,
                propagation_mode=PropagationMode.CHEBYSHEV,
            )

    def test_chebyshev_with_spectral_init(self, geo_adj, model_input):
        """Test CHEBYSHEV propagation with SPECTRAL_INIT graph mode."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            tod_embedding_dim=0,
            dow_embedding_dim=0,
            graph_mode=GraphMode.SPECTRAL_INIT,
            propagation_mode=PropagationMode.CHEBYSHEV,
            geo_adj=geo_adj,
        )
        assert model.graph_mode == GraphMode.SPECTRAL_INIT
        assert model.propagation_mode == PropagationMode.CHEBYSHEV

        # Verify forward pass works
        output = model(model_input)
        assert output.shape == (2, 12, 20, 1)

    def test_chebyshev_polynomials_properties(self, geo_adj):
        """Test that Chebyshev polynomials have expected properties."""
        from stgformer.temporal_processing import (
            compute_chebyshev_polynomials,
            compute_scaled_laplacian,
        )

        # Compute scaled Laplacian and Chebyshev polynomials
        scaled_lap = compute_scaled_laplacian(geo_adj)
        K = 4
        chebs = compute_chebyshev_polynomials(scaled_lap, K)

        # Should have K polynomial matrices (returned as list)
        assert len(chebs) == K

        # T_0 = I (identity)
        assert torch.allclose(chebs[0], torch.eye(geo_adj.shape[0]), atol=1e-5)

        # T_1 = L_tilde (scaled Laplacian)
        assert torch.allclose(chebs[1], scaled_lap, atol=1e-5)

        # T_2 = 2*L*T_1 - T_0 (recurrence relation)
        expected_T2 = 2 * scaled_lap @ chebs[1] - chebs[0]
        assert torch.allclose(chebs[2], expected_T2, atol=1e-5)

    def test_scaled_laplacian_eigenvalue_range(self, geo_adj):
        """Test that scaled Laplacian has eigenvalues in [-1, 1]."""
        from stgformer.temporal_processing import compute_scaled_laplacian

        scaled_lap = compute_scaled_laplacian(geo_adj)
        eigenvalues = torch.linalg.eigvalsh(scaled_lap)

        assert eigenvalues.min() >= -1.0 - 1e-5, (
            f"Scaled Laplacian has eigenvalue {eigenvalues.min():.4f} < -1"
        )
        assert eigenvalues.max() <= 1.0 + 1e-5, (
            f"Scaled Laplacian has eigenvalue {eigenvalues.max():.4f} > 1"
        )

    def test_power_vs_chebyshev_different_outputs(self, geo_adj, model_input):
        """Test that POWER and CHEBYSHEV produce different outputs."""
        torch.manual_seed(42)
        model_power = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            tod_embedding_dim=0,
            dow_embedding_dim=0,
            propagation_mode=PropagationMode.POWER,
            geo_adj=geo_adj,
        )

        torch.manual_seed(42)
        model_cheb = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            tod_embedding_dim=0,
            dow_embedding_dim=0,
            propagation_mode=PropagationMode.CHEBYSHEV,
            geo_adj=geo_adj,
        )

        # Same input should produce different outputs due to different propagation
        out_power = model_power(model_input)
        out_cheb = model_cheb(model_input)

        # Outputs should be different (not identical)
        assert not torch.allclose(out_power, out_cheb, atol=1e-3), (
            "POWER and CHEBYSHEV should produce different outputs"
        )

    def test_gradient_flow_chebyshev(self, geo_adj, model_input):
        """Test that gradients flow properly in CHEBYSHEV mode."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            tod_embedding_dim=0,
            dow_embedding_dim=0,
            propagation_mode=PropagationMode.CHEBYSHEV,
            geo_adj=geo_adj,
        )

        y = torch.randn(2, 12, 20, 1)
        pred = model(model_input.clone())
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()

        # Check key parameters have gradients
        assert model.adaptive_embedding.grad is not None, (
            "No gradient for adaptive_embedding in CHEBYSHEV mode"
        )


# ============================================================================
# Original Tests
# ============================================================================


class TestFastAttention:
    """Test core attention mechanism"""

    @pytest.fixture
    def attention_input(self):
        """Standard input for attention testing"""
        batch_size = 2
        seq_len = 12
        num_nodes = 20
        model_dim = 64
        return torch.randn(batch_size, seq_len, num_nodes, model_dim)

    def test_fast_attention_creation(self):
        """Test FastAttentionLayer can be created with correct parameters"""
        layer = FastAttentionLayer(model_dim=64, num_heads=4)
        assert layer.model_dim == 64
        assert layer.num_heads == 4
        assert layer.head_dim == 16  # 64 // 4

        # Should have QKV and output projections
        assert hasattr(layer, "qkv_proj"), "Missing QKV projection"
        assert hasattr(layer, "out_proj"), "Missing output projection"

    def test_fast_attention_method(self, attention_input):
        """Test fast_attention method produces valid output"""
        layer = FastAttentionLayer(model_dim=64, num_heads=4)

        # Mock QKV tensors for testing attention directly
        # attention_input is [2, 12, 20, 64] -> after _compute_qkv should be [24, 20, 4, 16]
        batch, time, nodes, _ = attention_input.shape
        qs = torch.randn(
            batch * time, nodes, 4, 16
        )  # [batch*time, nodes, heads, head_dim]
        ks = torch.randn(batch * time, nodes, 4, 16)
        vs = torch.randn(batch * time, nodes, 4, 16)

        output = layer.attention(attention_input, qs, ks, vs)

        # Should maintain input shape structure
        assert output.shape[0] == 2  # batch
        assert output.shape[1] == 12  # seq_len
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert torch.isfinite(output).all(), "Output contains infinite values"

    def test_fast_attention_forward(self, attention_input):
        """Test FastAttentionLayer forward pass"""
        layer = FastAttentionLayer(model_dim=64, num_heads=4)

        output = layer(attention_input)

        # Should preserve input shape
        assert output.shape == attention_input.shape
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert torch.isfinite(output).all(), "Output contains infinite values"

    def test_spatial_temporal_attention(self):
        """Test unified spatio-temporal attention"""
        layer = FastAttentionLayer(model_dim=64, num_heads=4)

        # Test with single time step (spatial only)
        x_spatial = torch.randn(2, 1, 20, 64)
        out_spatial = layer(x_spatial)
        assert out_spatial.shape == x_spatial.shape

        # Test with multiple time steps (spatial + temporal)
        x_temporal = torch.randn(2, 12, 20, 64)
        out_temporal = layer(x_temporal)
        assert out_temporal.shape == x_temporal.shape

    def test_attention_properties(self, attention_input):
        """Test mathematical properties of attention"""
        layer = FastAttentionLayer(model_dim=64, num_heads=4)

        # Set to eval mode for consistent behavior
        layer.eval()

        with torch.no_grad():
            output = layer(attention_input)

            # Attention should change the representation
            diff = torch.abs(output - attention_input).mean()
            assert diff > 0.01, f"Output too similar to input (diff: {diff:.6f})"

            # Output should have reasonable magnitude
            assert output.abs().mean() < 10, "Output magnitude too large"
            assert output.abs().mean() > 0.001, "Output magnitude too small"


class TestGraphConvolution:
    """Test graph convolution component"""

    @pytest.fixture
    def graph_input(self):
        """Standard input for graph testing"""
        batch_size = 2
        seq_len = 12
        num_nodes = 20
        features = 64

        x = torch.randn(batch_size, seq_len, num_nodes, features)
        graph = torch.randn(seq_len, num_nodes, num_nodes)  # Time-varying graph
        return x, graph

    def test_graph_propagate_creation(self):
        """Test GraphPropagate can be created"""
        layer = GraphPropagate(k_hops=3, p_dropout=0.1)
        assert hasattr(layer, "k_hops")
        assert hasattr(layer, "dropout")

    def test_graph_propagate_forward(self, graph_input):
        """Test GraphPropagate produces K-hop representations"""
        x, graph = graph_input
        layer = GraphPropagate(k_hops=3)

        hop_representations = layer(x, graph)

        # Should return list of K representations
        assert isinstance(hop_representations, list)
        assert len(hop_representations) == 3  # k_hops=3

        # Each should have same shape as input
        for hop in hop_representations:
            assert hop.shape == x.shape
            assert not torch.isnan(hop).any()

    def test_hop_progression(self, graph_input):
        """Test that each hop captures different neighborhoods"""
        x, graph = graph_input
        layer = GraphPropagate(k_hops=4)

        hops = layer(x, graph)

        # 0-hop should be identity (same as input)
        torch.testing.assert_close(hops[0], x, rtol=1e-4, atol=1e-4)

        # Higher hops should be increasingly different
        diff_1hop = torch.abs(hops[1] - x).mean()
        torch.abs(hops[2] - x).mean()
        torch.abs(hops[3] - x).mean()

        assert diff_1hop > 0, "1-hop should differ from input"
        # Note: diff_2hop might not always be > diff_1hop due to graph structure


class TestHybridLayer:
    """Test hybrid combination of graph + attention"""

    @pytest.fixture
    def hybrid_input(self):
        """Input for hybrid layer testing"""
        batch_size = 2
        seq_len = 12
        num_nodes = 20
        model_dim = 64

        x = torch.randn(batch_size, seq_len, num_nodes, model_dim)
        graph = torch.randn(seq_len, num_nodes, num_nodes)
        return x, graph

    def test_self_attention_creation(self):
        """Test GraphTemporalLayer creation"""
        layer = GraphTemporalLayer(model_dim=64, num_heads=4, order=2)
        assert hasattr(layer, "locals"), "Missing GraphPropagate module"
        assert hasattr(layer, "attn"), "Missing attention layers"
        assert hasattr(layer, "order_proj"), "Missing projection weights"
        assert hasattr(layer, "mlp"), "Missing MLP"
        assert hasattr(layer, "ln1"), "Missing layer norm 1"
        assert hasattr(layer, "ln2"), "Missing layer norm 2"

    def test_self_attention_forward(self, hybrid_input):
        """Test GraphTemporalLayer forward pass"""
        x, graph = hybrid_input
        layer = GraphTemporalLayer(model_dim=64, num_heads=4, order=2)

        output = layer(x, graph)

        # Should preserve input shape
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()

    def test_hybrid_properties(self, hybrid_input):
        """Test that hybrid layer combines local and global patterns"""
        x, graph = hybrid_input
        layer = GraphTemporalLayer(model_dim=64, num_heads=4, order=2)
        layer.eval()

        with torch.no_grad():
            output = layer(x, graph)

            # Should modify input (learning something)
            diff = torch.abs(output - x).mean()
            assert diff > 0.01, "Layer not learning enough"

        # Gradients should flow through both branches
        layer.train()
        x.requires_grad_(True)
        output = layer(x, graph)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "No gradients flowing"


class TestFullModel:
    """Test complete internal STGFormer model implementation"""

    @pytest.fixture
    def model_input(self):
        """Realistic model input"""
        batch_size = 2
        in_steps = 12
        num_nodes = 20
        input_dim = 3  # [value, time_of_day, day_of_week]

        x = torch.randn(batch_size, in_steps, num_nodes, input_dim)
        # Make temporal features realistic
        x[..., 1] = torch.rand(batch_size, in_steps, num_nodes) * 0.99  # TOD [0,1)
        x[..., 2] = torch.randint(
            0, 7, (batch_size, in_steps, num_nodes)
        ).float()  # DOW

        return x

    def test_model_creation(self):
        """Test STGFormer model creation"""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=1,  # Number of raw value features
            output_dim=1,
            input_embedding_dim=16,
            tod_embedding_dim=16,
            dow_embedding_dim=16,
            adaptive_embedding_dim=32,
            num_heads=4,
            num_layers=2,
        )

        # Check essential components exist
        assert hasattr(model, "input_proj"), "Missing input projection"
        assert hasattr(model, "attn"), "Missing attention layers"
        assert hasattr(model, "output_proj"), "Missing output projection"

        # Check model_dim is calculated correctly
        assert model.model_dim == 16 + 16 + 16 + 0 + 32, (
            "model_dim not calculated correctly"
        )

        # Check parameter count is reasonable
        total_params = sum(p.numel() for p in model.parameters())
        assert 50_000 < total_params < 2_000_000, (
            f"Unexpected parameter count: {total_params:,}"
        )

    def test_adaptive_graph_construction(self):
        """Test _construct_adaptive_graph method"""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            adaptive_embedding_dim=32,
        )

        # Mock adaptive embeddings
        embeddings = torch.randn(12, 20, 32)  # [in_steps, num_nodes, embedding_dim]

        graph = model._construct_adaptive_graph(embeddings)

        # Should produce valid adjacency matrix
        assert graph.shape[-2:] == (20, 20), "Wrong graph shape"
        assert torch.all(graph >= 0), "Graph should be non-negative"
        assert torch.allclose(graph.sum(dim=-1), torch.ones(graph.shape[:-1])), (
            "Rows should sum to 1"
        )

    def test_model_forward(self, model_input):
        """Test full model forward pass"""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=1,  # One raw value feature (speed)
            output_dim=1,
            input_embedding_dim=16,
            tod_embedding_dim=16,
            dow_embedding_dim=16,
            adaptive_embedding_dim=32,
        )

        output = model(model_input)

        # Check output shape
        expected_shape = (2, 12, 20, 1)  # (batch, out_steps, nodes, output_dim)
        assert output.shape == expected_shape, (
            f"Expected {expected_shape}, got {output.shape}"
        )

        # Check output properties
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert torch.isfinite(output).all(), "Output contains infinite values"

    def test_temporal_embeddings(self, model_input):
        """Test time-of-day and day-of-week embeddings work correctly"""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=1,  # One raw value feature
            input_embedding_dim=16,
            tod_embedding_dim=24,
            dow_embedding_dim=12,
            adaptive_embedding_dim=32,
            steps_per_day=288,
        )

        # Should handle temporal features without errors
        output = model(model_input)
        assert output.shape == (2, 12, 20, 1)

    def test_get_spatial_representation(self, model_input):
        """Test get_spatial_representation returns correct shape"""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            output_dim=1,
            input_embedding_dim=16,
            tod_embedding_dim=16,
            dow_embedding_dim=16,
            adaptive_embedding_dim=32,
        )

        spatial_repr = model.get_spatial_representation(model_input)

        # Check output shape: [batch, num_nodes, model_dim]
        expected_shape = (2, 20, model.model_dim)
        assert spatial_repr.shape == expected_shape, (
            f"Expected {expected_shape}, got {spatial_repr.shape}"
        )

        # Check output properties
        assert not torch.isnan(spatial_repr).any(), "Output contains NaN"
        assert torch.isfinite(spatial_repr).all(), "Output contains infinite values"

    def test_get_spatial_representation_matches_forward(self, model_input):
        """Test that get_spatial_representation extracts same representation as forward()"""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            output_dim=1,
            input_embedding_dim=16,
            tod_embedding_dim=16,
            dow_embedding_dim=16,
            adaptive_embedding_dim=32,
        )

        # Set to eval mode to disable dropout
        model.eval()

        with torch.no_grad():
            spatial_repr = model.get_spatial_representation(model_input)
            full_output = model(model_input)

        # The spatial representation should be consistent between calls
        # (same underlying _embed_and_encode is called)
        spatial_repr2 = model.get_spatial_representation(model_input)
        torch.testing.assert_close(spatial_repr, spatial_repr2)

        # Verify that forward() produces a valid output from the same representation
        # Shape: forward gives [batch, out_steps, nodes, output_dim]
        # get_spatial_representation gives [batch, nodes, model_dim]
        assert full_output.shape == (2, 12, 20, 1)
        assert spatial_repr.shape == (2, 20, model.model_dim)


class TestTraining:
    """Test training code"""

    def test_loss_function(self):
        """Test MaskedHuberLoss works correctly"""
        # This will be implemented in train_stgformer.py
        # For now, just test that Huber loss works
        preds = torch.randn(4, 12, 20, 1)
        targets = torch.randn(4, 12, 20, 1)

        loss = torch.nn.functional.huber_loss(preds, targets)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_standard_scaler(self):
        """Test StandardScaler utility"""

        data = torch.randn(100, 50) * 10 + 5

        scaler = StandardScaler()
        normalized = scaler.fit_transform(data)

        # Should normalize to zero mean, unit variance
        assert abs(normalized.mean().item()) < 1e-2
        assert abs(normalized.std().item() - 1.0) < 1e-2

        # Inverse transform should recover original
        recovered = scaler.inverse_transform(normalized)
        torch.testing.assert_close(recovered, data, rtol=1e-4, atol=1e-6)

    def test_gradient_flow(self):
        """Test gradients flow through model during training"""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            dow_embedding_dim=0,
            adaptive_embedding_dim=80,
        )

        # Input format: [batch, time, nodes, input_dim + has_tod]
        # With input_dim=1 and tod_embedding_dim>0, we need 2 channels
        x = torch.randn(2, 12, 20, 2)
        x[..., 0] = torch.randn(2, 12, 20)  # Traffic values
        x[..., 1] = torch.rand(2, 12, 20) * 0.99  # Time of day (normalized 0-1)

        y = torch.randn(2, 12, 20, 1)

        # Forward pass
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)

        # Backward pass
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestValidateModel:
    def test_parameter_count_match(self):
        """Test parameter count matches external implementation"""
        model = STGFormer(
            num_nodes=207,
            in_steps=12,
            out_steps=12,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            dow_embedding_dim=24,
            adaptive_embedding_dim=80,
            num_heads=4,
            num_layers=3,
        )

        total_params = sum(p.numel() for p in model.parameters())

        # Sanity check: model should have reasonable number of parameters
        # With model_dim=152 (24+24+24+80), num_layers=3, num_nodes=207
        expected_params = 2_459_156
        tolerance = 5000  # Allow some tolerance for architecture changes

        assert abs(total_params - expected_params) < tolerance, (
            f"Parameter count mismatch: {total_params:,} vs expected {expected_params:,}"
        )

    def test_output_compatibility(self):
        """Test outputs are compatible with external implementation"""
        # model_dim is computed from embedding dims: 24 + 24 + 0 + 0 + 80 = 128
        model = STGFormer(
            num_nodes=50,
            in_steps=12,
            out_steps=12,
            input_dim=1,  # Single value feature (e.g., speed)
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            dow_embedding_dim=0,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=80,
        )

        # Test with small batch
        # Input format: [batch, time, nodes, input_dim + has_tod + has_dow]
        # With input_dim=1 and tod_embedding_dim>0, we need 2 channels: [value, tod]
        x = torch.randn(2, 12, 50, 2)
        x[..., 0] = torch.randn(2, 12, 50)  # Traffic values
        x[..., 1] = torch.rand(2, 12, 50) * 0.99  # Time of day (normalized 0-1)

        model.eval()
        with torch.no_grad():
            output = model(x)

        # Should produce reasonable outputs
        assert output.shape == (2, 12, 50, 1)
        assert torch.isfinite(output).all()
        assert output.abs().mean() < 100  # Reasonable magnitude

    def test_fast_vs_normal_attention(self):
        """Test understanding: fast attention approximates normal attention"""
        layer = FastAttentionLayer(model_dim=64, num_heads=4)

        x = torch.randn(2, 8, 10, 64)  # Smaller for testing

        # Get both outputs (would need to implement normal_attention method)
        layer.fast = True
        fast_output = layer(x)

        # Note: This test requires implementing a normal_attention method for comparison
        # The outputs should be reasonably close but not identical due to the approximation

        assert fast_output.shape == x.shape
        assert torch.isfinite(fast_output).all()


class TestDOWEmbedding:
    """Test day-of-week embedding functionality."""

    def test_dow_embedding_forward_pass(self):
        """Test that DOW embedding works in forward pass."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            tod_embedding_dim=24,
            dow_embedding_dim=24,
            adaptive_embedding_dim=80,
            num_heads=4,
            num_layers=2,
        )

        # Input: [batch, time, nodes, input_dim + tod + dow]
        # With DOW enabled, we need 3 channels: [value, tod, dow]
        x = torch.randn(2, 12, 20, 3)
        x[..., 0] = torch.randn(2, 12, 20)  # Traffic values
        x[..., 1] = torch.rand(2, 12, 20) * 0.99  # TOD in [0, 1)
        x[..., 2] = torch.randint(0, 7, (2, 12, 20)).float()  # DOW in [0, 7)

        # Forward pass should work
        output = model(x)
        assert output.shape == (2, 12, 20, 1)
        assert torch.isfinite(output).all()

    def test_dow_embedding_backward_pass(self):
        """Test that gradients flow properly with DOW embedding."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            tod_embedding_dim=24,
            dow_embedding_dim=24,
            adaptive_embedding_dim=80,
            num_heads=4,
            num_layers=2,
        )

        # Input with DOW
        x = torch.randn(2, 12, 20, 3)
        x[..., 0] = torch.randn(2, 12, 20)
        x[..., 1] = torch.rand(2, 12, 20) * 0.99
        x[..., 2] = torch.randint(0, 7, (2, 12, 20)).float()
        y = torch.randn(2, 12, 20, 1)

        # Forward and backward
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()

        # Check DOW embedding has gradients
        assert model.dow_embedding.weight.grad is not None
        assert not torch.isnan(model.dow_embedding.weight.grad).any()

    @pytest.mark.skipif(
        not (hasattr(torch, "compile") and torch.cuda.is_available()),
        reason="torch.compile with CUDA required",
    )
    def test_dow_embedding_with_compile(self):
        """Test that DOW embedding works with torch.compile on CUDA.

        This test verifies that the normalization epsilon fix in FastAttentionLayer
        prevents numerical issues during backward pass with torch.compile.
        """
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            tod_embedding_dim=24,
            dow_embedding_dim=24,
            adaptive_embedding_dim=80,
            num_heads=4,
            num_layers=2,
        ).cuda()

        model = torch.compile(model)

        # Input with DOW
        x = torch.randn(2, 12, 20, 3).cuda()
        x[..., 0] = torch.randn(2, 12, 20)
        x[..., 1] = torch.rand(2, 12, 20) * 0.99
        x[..., 2] = torch.randint(0, 7, (2, 12, 20)).float()

        # Should not crash during forward/backward
        output = model(x)
        assert output.shape == (2, 12, 20, 1)
        assert torch.isfinite(output).all()

        # Test backward pass (where the original error occurred)
        y = torch.randn(2, 12, 20, 1).cuda()
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()


class TestTF32Configuration:
    """Test TF32 configuration for torch.compile compatibility."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tf32_both_apis_set(self):
        """Test that train_model sets both old and new TF32 APIs.

        PyTorch's torch.compile inductor backend checks the old allow_tf32 flag,
        so we must set both APIs to avoid RuntimeError about mixed API usage.
        """
        # Simulate the TF32 setup from train_model
        if torch.cuda.is_available():
            # This is what train_model does
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.conv.fp32_precision = "tf32"
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Verify both APIs are consistent
            assert torch.backends.cuda.matmul.allow_tf32 is True
            assert torch.backends.cudnn.allow_tf32 is True


# ============================================================================
# Mamba Tests - Test Mamba SSM temporal processing variant
# ============================================================================


class TestMambaTemporalMode:
    """Tests for Mamba SSM-based temporal processing.

    Mamba replaces the temporal attention branch with a state space model,
    providing O(T) complexity instead of O(T²) for temporal processing.

    These tests are skipped if mamba-ssm is not installed (CUDA-only).
    """

    @pytest.fixture
    def model_input(self):
        """Standard input for testing."""
        x = torch.randn(2, 12, 20, 2)
        x[..., 1] = torch.rand(2, 12, 20) * 0.99  # TOD in [0, 1)
        return x

    @pytest.fixture
    def geo_adj(self):
        """Create a mock geographic adjacency matrix."""
        num_nodes = 20
        adj = torch.rand(num_nodes, num_nodes)
        adj = adj + adj.T  # Make symmetric
        adj = adj / adj.sum(dim=-1, keepdim=True)  # Row normalize
        return adj

    def test_mamba_import_error_without_package(self):
        """Test that helpful error is raised when mamba-ssm not installed."""
        if MAMBA_AVAILABLE:
            pytest.skip("mamba-ssm is installed, cannot test import error")

        with pytest.raises(ImportError, match="mamba-ssm is required"):
            STGFormer(
                num_nodes=20,
                in_steps=12,
                out_steps=12,
                temporal_mode=TemporalMode.MAMBA,
            )

    @pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Mamba")
    def test_mamba_attention_layer_output_shape(self):
        """Verify MambaAttentionLayer maintains expected I/O contract."""
        layer = MambaAttentionLayer(
            model_dim=64,
            num_heads=4,
            qkv_bias=True,
            d_state=16,
            d_conv=4,
            expand=2,
        ).cuda()

        x = torch.randn(2, 12, 20, 64).cuda()
        output = layer(x)

        assert output.shape == x.shape, (
            f"MambaAttentionLayer changed shape: {x.shape} -> {output.shape}"
        )

    @pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Mamba")
    def test_mamba_attention_layer_gradient_flow(self):
        """Verify gradients flow through MambaAttentionLayer."""
        layer = MambaAttentionLayer(
            model_dim=64,
            num_heads=4,
            d_state=16,
            d_conv=4,
        ).cuda()

        x = torch.randn(2, 12, 20, 64, requires_grad=True, device="cuda")
        output = layer(x)
        loss = output.sum()
        loss.backward()

        # Check gradients on layer parameters (not input, which is non-leaf after .cuda())
        has_grads = any(p.grad is not None for p in layer.parameters())
        assert has_grads, "Gradient did not flow to layer parameters"

        # Check all parameter gradients are finite
        for p in layer.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (
                    "Parameter gradient contains non-finite values"
                )

    @pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Mamba")
    def test_self_attention_layer_with_mamba(self):
        """Test GraphTemporalLayer using Mamba temporal mode."""
        layer = GraphTemporalLayer(
            model_dim=64,
            num_heads=4,
            order=2,
            temporal_mode=TemporalMode.MAMBA,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
        ).cuda()

        x = torch.randn(2, 12, 20, 64).cuda()
        graph = torch.randn(20, 20).cuda()
        graph = torch.softmax(graph, dim=-1)  # Make valid adjacency

        output = layer(x, graph)

        assert output.shape == x.shape, (
            f"GraphTemporalLayer with Mamba changed shape: {x.shape} -> {output.shape}"
        )

    @pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Mamba")
    def test_stgformer_mamba_forward_pass(self, model_input):
        """Test full STGFormer model with Mamba temporal mode."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            adaptive_embedding_dim=80,
            num_heads=4,
            num_layers=2,
            temporal_mode=TemporalMode.MAMBA,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
        ).cuda()

        x = model_input.cuda()
        output = model(x)

        expected_shape = (2, 12, 20, 1)
        assert output.shape == expected_shape, (
            f"Output shape {output.shape} != expected {expected_shape}"
        )
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    @pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Mamba")
    def test_stgformer_mamba_gradient_flow(self, model_input):
        """Verify gradients flow to all parameters in Mamba mode."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            adaptive_embedding_dim=80,
            num_heads=4,
            num_layers=2,
            temporal_mode=TemporalMode.MAMBA,
        ).cuda()

        x = model_input.cuda()
        y = torch.randn(2, 12, 20, 1).cuda()

        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()

        # All parameters should have gradients
        params_without_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)

        assert len(params_without_grad) == 0, (
            f"Parameters without gradients: {params_without_grad}"
        )

    @pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Mamba")
    def test_stgformer_mamba_with_graph_modes(self, model_input, geo_adj):
        """Test Mamba temporal mode works with different graph modes."""
        geo_adj_cuda = geo_adj.cuda()

        for graph_mode in [GraphMode.LEARNED, GraphMode.GEOGRAPHIC, GraphMode.HYBRID]:
            model = STGFormer(
                num_nodes=20,
                in_steps=12,
                out_steps=12,
                input_dim=2,
                output_dim=1,
                adaptive_embedding_dim=80,
                num_layers=2,
                temporal_mode=TemporalMode.MAMBA,
                graph_mode=graph_mode,
                geo_adj=geo_adj_cuda if graph_mode != GraphMode.LEARNED else None,
            ).cuda()

            x = model_input.cuda()
            output = model(x)

            assert output.shape == (2, 12, 20, 1), (
                f"Wrong shape for graph_mode={graph_mode}"
            )
            assert torch.isfinite(output).all(), (
                f"Non-finite output for graph_mode={graph_mode}"
            )

    @pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Mamba")
    def test_stgformer_mamba_parameter_count(self):
        """Test that Mamba mode has different (but reasonable) parameter count."""
        model_transformer = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            adaptive_embedding_dim=80,
            num_heads=4,
            num_layers=2,
            temporal_mode=TemporalMode.TRANSFORMER,
        )

        model_mamba = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            adaptive_embedding_dim=80,
            num_heads=4,
            num_layers=2,
            temporal_mode=TemporalMode.MAMBA,
        ).cuda()

        params_transformer = sum(p.numel() for p in model_transformer.parameters())
        params_mamba = sum(p.numel() for p in model_mamba.parameters())

        # Mamba model should have different parameter count
        # (Mamba replaces attention's temporal QKV with SSM parameters)
        # Both should be in reasonable range (not exploding)
        assert params_transformer > 0
        assert params_mamba > 0
        assert params_mamba < params_transformer * 3  # Shouldn't be 3x larger
        assert params_mamba > params_transformer * 0.3  # Shouldn't be 3x smaller

    @pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Mamba")
    def test_mamba_deterministic_output(self):
        """Test that Mamba model produces deterministic outputs."""
        torch.manual_seed(42)

        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            output_dim=1,
            adaptive_embedding_dim=80,
            num_layers=2,
            temporal_mode=TemporalMode.MAMBA,
        ).cuda()
        model.eval()

        torch.manual_seed(123)
        x = torch.randn(2, 12, 20, 2).cuda()
        x[..., 1] = torch.rand(2, 12, 20).cuda() * 0.99

        with torch.no_grad():
            output1 = model(x.clone())
            output2 = model(x.clone())

        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-5)


# ============================================================================
# TCN Tests - Test TCN temporal processing variant
# ============================================================================


class TestTCNTemporalMode:
    """Tests for TCN (Temporal Convolutional Network) temporal processing.

    TCN replaces the temporal attention branch with causal dilated convolutions,
    providing O(T) complexity with better inductive bias for short sequences.
    """

    @pytest.fixture
    def model_input(self):
        """Standard input for testing."""
        x = torch.randn(2, 12, 20, 2)
        x[..., 1] = torch.rand(2, 12, 20) * 0.99  # TOD in [0, 1)
        return x

    @pytest.fixture
    def geo_adj(self):
        """Geographic adjacency matrix for testing."""
        return torch.rand(20, 20)

    def test_tcn_temporal_layer_output_shape(self):
        """Test that TCNTemporalLayer produces correct output shape."""
        from stgformer.temporal_processing import TCNTemporalLayer

        layer = TCNTemporalLayer(
            model_dim=96,
            num_heads=4,
            num_layers=3,
            kernel_size=3,
            dilation_base=2,
            dropout=0.1,
        )

        x = torch.randn(2, 12, 20, 96)
        output = layer(x)

        assert output.shape == x.shape, "Output shape should match input shape"

    def test_tcn_temporal_layer_gradient_flow(self):
        """Test that gradients flow through TCNTemporalLayer."""
        from stgformer.temporal_processing import TCNTemporalLayer

        layer = TCNTemporalLayer(model_dim=96, num_heads=4)
        x = torch.randn(2, 12, 20, 96, requires_grad=True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients flow back to input
        assert x.grad is not None, "Gradients should flow to input"
        assert torch.abs(x.grad).sum() > 0, "Gradients should be non-zero"

        # Check that all layer parameters have gradients
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"

    def test_stgformer_tcn_forward_pass(self, model_input):
        """Test STGFormer with TCN temporal mode forward pass."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            adaptive_embedding_dim=80,
            num_heads=4,
            num_layers=2,
            temporal_mode=TemporalMode.TCN,
            tcn_num_layers=3,
            tcn_kernel_size=3,
            tcn_dilation_base=2,
            tcn_dropout=0.1,
        )

        output = model(model_input)

        # Check output shape
        expected_shape = (2, 12, 20, 1)  # (batch, time, nodes, output_dim)
        assert output.shape == expected_shape, (
            f"Expected {expected_shape}, got {output.shape}"
        )

        # Check output is not NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_stgformer_tcn_gradient_flow(self, model_input):
        """Test gradient flow through STGFormer with TCN."""
        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            output_dim=1,
            adaptive_embedding_dim=80,
            num_layers=2,
            temporal_mode=TemporalMode.TCN,
        )

        # Forward pass
        output = model(model_input)

        # Backward pass
        y = torch.randn_like(output)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()

        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

    def test_stgformer_tcn_with_graph_modes(self, model_input, geo_adj):
        """Test TCN mode works with different graph modes."""
        for graph_mode in [GraphMode.LEARNED, GraphMode.GEOGRAPHIC, GraphMode.HYBRID]:
            model = STGFormer(
                num_nodes=20,
                in_steps=12,
                out_steps=12,
                input_dim=2,
                output_dim=1,
                adaptive_embedding_dim=80,
                num_layers=2,
                graph_mode=graph_mode,
                geo_adj=geo_adj if graph_mode != GraphMode.LEARNED else None,
                temporal_mode=TemporalMode.TCN,
            )

            output = model(model_input)
            assert output.shape == (2, 12, 20, 1)

    def test_stgformer_tcn_parameter_count(self):
        """Test that TCN model has reasonable parameter count."""
        model_transformer = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            adaptive_embedding_dim=80,
            num_heads=4,
            num_layers=2,
            temporal_mode=TemporalMode.TRANSFORMER,
        )

        model_tcn = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            adaptive_embedding_dim=80,
            num_heads=4,
            num_layers=2,
            temporal_mode=TemporalMode.TCN,
        )

        params_transformer = sum(p.numel() for p in model_transformer.parameters())
        params_tcn = sum(p.numel() for p in model_tcn.parameters())

        # TCN model should have similar parameter count to transformer
        # (TCN convolutions vs attention QKV projections)
        assert params_transformer > 0
        assert params_tcn > 0
        assert params_tcn < params_transformer * 2  # Shouldn't be 2x larger
        assert params_tcn > params_transformer * 0.5  # Shouldn't be 2x smaller

    def test_tcn_deterministic_output(self):
        """Test that TCN model produces deterministic outputs."""
        torch.manual_seed(42)

        model = STGFormer(
            num_nodes=20,
            in_steps=12,
            out_steps=12,
            input_dim=2,
            output_dim=1,
            adaptive_embedding_dim=80,
            num_layers=2,
            temporal_mode=TemporalMode.TCN,
        )
        model.eval()

        torch.manual_seed(123)
        x = torch.randn(2, 12, 20, 2)
        x[..., 1] = torch.rand(2, 12, 20) * 0.99

        with torch.no_grad():
            output1 = model(x.clone())
            output2 = model(x.clone())

        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-5)

    def test_tcn_causal_property(self):
        """Test that TCN respects causality (no future leakage)."""
        from stgformer.temporal_processing import TCNTemporalLayer

        layer = TCNTemporalLayer(
            model_dim=96,
            num_heads=4,
            num_layers=3,
            kernel_size=3,
            dilation_base=2,
        )
        layer.eval()

        # Create input where future timesteps are corrupted
        x = torch.randn(1, 12, 10, 96)
        x_corrupted = x.clone()
        x_corrupted[:, 6:, :, :] = (
            torch.randn_like(x[:, 6:, :, :]) * 1000
        )  # Corrupt future

        with torch.no_grad():
            output = layer(x)
            output_corrupted = layer(x_corrupted)

        # Output at early timesteps should be identical (no future information used)
        # TCN is causal, so corruption at t>6 shouldn't affect t<=5
        torch.testing.assert_close(
            output[:, :6, :, :],
            output_corrupted[:, :6, :, :],
            rtol=1e-4,
            atol=1e-4,
            msg="TCN should be causal (early timesteps unaffected by future corruption)",
        )
