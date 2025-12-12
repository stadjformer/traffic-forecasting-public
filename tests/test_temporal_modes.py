"""Tests for new temporal processing modes: Depthwise and MLP."""

import pytest
import torch

from stgformer.enums import TemporalMode
from stgformer.model import STGFormer
from stgformer.temporal_processing import DepthwiseTemporalLayer, MLPTemporalLayer


class TestDepthwiseTemporalLayer:
    """Test DepthwiseTemporalLayer implementation."""

    def test_layer_creation(self):
        """Test basic layer instantiation."""
        layer = DepthwiseTemporalLayer(
            model_dim=96, num_heads=4, kernel_size=3, dropout=0.1
        )
        assert layer.model_dim == 96
        assert layer.num_heads == 4
        assert layer.depthwise_kernel_size == 3
        assert layer.depthwise_dropout == 0.1

    def test_layer_forward_shape(self):
        """Test forward pass produces correct output shape."""
        layer = DepthwiseTemporalLayer(model_dim=96, num_heads=4, kernel_size=3)
        x = torch.randn(2, 12, 10, 96)  # [batch, time, nodes, features]

        # Mock query, key, value (not used but required for interface)
        qkv = torch.randn(2, 12, 10, 96 * 3)
        query, key, value = qkv.chunk(3, -1)

        out = layer._compute_temporal_branch(x, query, key, value)

        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_depthwise_parameters(self):
        """Test depthwise conv has correct parameter count."""
        model_dim = 96
        kernel_size = 3
        layer = DepthwiseTemporalLayer(model_dim=model_dim, kernel_size=kernel_size)

        # Depthwise: kernel_size * in_channels (groups=model_dim)
        # Pointwise: in_channels * out_channels (1x1 conv)
        # Expected: k*D + D*D (much less than standard k*D*D)
        depthwise_params = sum(p.numel() for p in layer.depthwise.parameters())
        pointwise_params = sum(p.numel() for p in layer.pointwise.parameters())

        # Depthwise should have ~k*D params
        assert depthwise_params < model_dim * kernel_size * 2  # With bias
        # Pointwise should have D*D params
        assert pointwise_params == model_dim * model_dim + model_dim  # +bias

        # Total should be much less than standard conv (k*D*D)
        total_params = depthwise_params + pointwise_params
        standard_conv_params = kernel_size * model_dim * model_dim
        # Depthwise separable is ~3-4x more efficient than standard conv
        assert total_params < standard_conv_params / 2  # At least 2x reduction

    def test_different_kernel_sizes(self):
        """Test layer works with different kernel sizes."""
        for kernel_size in [1, 3, 5, 7]:
            layer = DepthwiseTemporalLayer(
                model_dim=96, num_heads=4, kernel_size=kernel_size
            )
            x = torch.randn(2, 12, 10, 96)
            qkv = torch.randn(2, 12, 10, 96 * 3)
            query, key, value = qkv.chunk(3, -1)

            out = layer._compute_temporal_branch(x, query, key, value)
            assert out.shape == x.shape

    def test_gradient_flow(self):
        """Test gradients flow through the layer."""
        layer = DepthwiseTemporalLayer(model_dim=96, num_heads=4, kernel_size=3)
        x = torch.randn(2, 12, 10, 96, requires_grad=True)
        qkv = torch.randn(2, 12, 10, 96 * 3)
        query, key, value = qkv.chunk(3, -1)

        out = layer._compute_temporal_branch(x, query, key, value)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


class TestMLPTemporalLayer:
    """Test MLPTemporalLayer implementation."""

    def test_layer_creation(self):
        """Test basic layer instantiation."""
        layer = MLPTemporalLayer(model_dim=96, num_heads=4, hidden_dim=128, dropout=0.1)
        assert layer.model_dim == 96
        assert layer.num_heads == 4
        assert layer.mlp_hidden_dim == 128
        assert layer.mlp_dropout == 0.1

    def test_layer_creation_default_hidden_dim(self):
        """Test layer uses model_dim when hidden_dim is None."""
        layer = MLPTemporalLayer(model_dim=96, num_heads=4, hidden_dim=None)
        assert layer.mlp_hidden_dim == 96

    def test_layer_forward_shape(self):
        """Test forward pass produces correct output shape."""
        layer = MLPTemporalLayer(model_dim=96, num_heads=4)
        x = torch.randn(2, 12, 10, 96)  # [batch, time, nodes, features]

        # Mock query, key, value (not used but required for interface)
        qkv = torch.randn(2, 12, 10, 96 * 3)
        query, key, value = qkv.chunk(3, -1)

        out = layer._compute_temporal_branch(x, query, key, value)

        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_mlp_is_simple(self):
        """Test MLP has minimal parameters compared to attention."""
        model_dim = 96
        layer = MLPTemporalLayer(model_dim=model_dim, hidden_dim=model_dim)

        mlp_params = sum(p.numel() for p in layer.mlp.parameters())

        # MLP should have: D*D + D*D (two linear layers)
        # Plus biases and dropout (no params)
        expected_params = 2 * (model_dim * model_dim + model_dim)
        assert mlp_params == expected_params

    def test_no_reshape_operations(self):
        """Test MLP processes input directly without reshape."""
        layer = MLPTemporalLayer(model_dim=96, num_heads=4)
        x = torch.randn(2, 12, 10, 96)
        qkv = torch.randn(2, 12, 10, 96 * 3)
        query, key, value = qkv.chunk(3, -1)

        # The forward pass should not change tensor layout
        out = layer._compute_temporal_branch(x, query, key, value)
        assert out.is_contiguous()  # No reshape means output is contiguous

    def test_gradient_flow(self):
        """Test gradients flow through the layer."""
        layer = MLPTemporalLayer(model_dim=96, num_heads=4)
        x = torch.randn(2, 12, 10, 96, requires_grad=True)
        qkv = torch.randn(2, 12, 10, 96 * 3)
        query, key, value = qkv.chunk(3, -1)

        out = layer._compute_temporal_branch(x, query, key, value)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


class TestSTGFormerWithNewModes:
    """Test STGFormer integration with new temporal modes."""

    @pytest.fixture
    def model_config(self):
        """Common model configuration for testing."""
        return {
            "num_nodes": 10,
            "in_steps": 12,
            "out_steps": 12,
            "input_dim": 1,
            "output_dim": 1,
            "input_embedding_dim": 24,
            "tod_embedding_dim": 0,  # Disable for simpler testing
            "adaptive_embedding_dim": 80,
            "num_heads": 4,
            "num_layers": 2,  # Reduced for faster tests
        }

    def test_depthwise_model_creation(self, model_config):
        """Test STGFormer creates successfully with DEPTHWISE mode."""
        model = STGFormer(
            **model_config,
            temporal_mode=TemporalMode.DEPTHWISE,
            depthwise_kernel_size=3,
        )
        assert model.temporal_mode == TemporalMode.DEPTHWISE
        assert model.depthwise_kernel_size == 3

    def test_mlp_model_creation(self, model_config):
        """Test STGFormer creates successfully with MLP mode."""
        model = STGFormer(
            **model_config, temporal_mode=TemporalMode.MLP, mlp_hidden_dim=96
        )
        assert model.temporal_mode == TemporalMode.MLP
        assert model.mlp_hidden_dim == 96

    def test_depthwise_forward_pass(self, model_config):
        """Test forward pass with DEPTHWISE mode."""
        model = STGFormer(
            **model_config,
            temporal_mode=TemporalMode.DEPTHWISE,
            depthwise_kernel_size=3,
        )
        x = torch.randn(2, 12, 10, 1)  # [batch, time, nodes, features]
        out = model(x)

        assert out.shape == (2, 12, 10, 1)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_mlp_forward_pass(self, model_config):
        """Test forward pass with MLP mode."""
        model = STGFormer(
            **model_config, temporal_mode=TemporalMode.MLP, mlp_hidden_dim=96
        )
        x = torch.randn(2, 12, 10, 1)
        out = model(x)

        assert out.shape == (2, 12, 10, 1)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_depthwise_backward_pass(self, model_config):
        """Test backward pass with DEPTHWISE mode."""
        model = STGFormer(
            **model_config,
            temporal_mode=TemporalMode.DEPTHWISE,
            depthwise_kernel_size=3,
        )
        x = torch.randn(2, 12, 10, 1)
        y = torch.randn(2, 12, 10, 1)

        out = model(x)
        loss = torch.nn.functional.mse_loss(out, y)
        loss.backward()

        # Check gradients exist and are valid
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_mlp_backward_pass(self, model_config):
        """Test backward pass with MLP mode."""
        model = STGFormer(
            **model_config, temporal_mode=TemporalMode.MLP, mlp_hidden_dim=96
        )
        x = torch.randn(2, 12, 10, 1)
        y = torch.randn(2, 12, 10, 1)

        out = model(x)
        loss = torch.nn.functional.mse_loss(out, y)
        loss.backward()

        # Check gradients exist and are valid
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_parameter_count_comparison(self, model_config):
        """Compare parameter counts across temporal modes."""
        baseline = STGFormer(**model_config, temporal_mode=TemporalMode.TRANSFORMER)
        depthwise = STGFormer(
            **model_config,
            temporal_mode=TemporalMode.DEPTHWISE,
            depthwise_kernel_size=3,
        )
        mlp = STGFormer(**model_config, temporal_mode=TemporalMode.MLP)

        baseline_params = sum(p.numel() for p in baseline.parameters())
        depthwise_params = sum(p.numel() for p in depthwise.parameters())
        mlp_params = sum(p.numel() for p in mlp.parameters())

        # All should have reasonable parameter counts (within 20% of baseline)
        assert 0.8 * baseline_params < depthwise_params < 1.2 * baseline_params
        assert 0.8 * baseline_params < mlp_params < 1.2 * baseline_params

    def test_deterministic_output_depthwise(self, model_config):
        """Test DEPTHWISE mode produces deterministic outputs with fixed seed."""
        torch.manual_seed(42)
        model1 = STGFormer(
            **model_config,
            temporal_mode=TemporalMode.DEPTHWISE,
            depthwise_kernel_size=3,
        )
        torch.manual_seed(123)  # Different seed for input
        x = torch.randn(2, 12, 10, 1)
        out1 = model1(x)

        torch.manual_seed(42)
        model2 = STGFormer(
            **model_config,
            temporal_mode=TemporalMode.DEPTHWISE,
            depthwise_kernel_size=3,
        )
        torch.manual_seed(123)  # Same input seed
        x = torch.randn(2, 12, 10, 1)
        out2 = model2(x)

        assert torch.allclose(out1, out2, rtol=1e-5)

    def test_deterministic_output_mlp(self, model_config):
        """Test MLP mode produces deterministic outputs with fixed seed."""
        torch.manual_seed(42)
        model1 = STGFormer(**model_config, temporal_mode=TemporalMode.MLP)
        torch.manual_seed(123)
        x = torch.randn(2, 12, 10, 1)
        out1 = model1(x)

        torch.manual_seed(42)
        model2 = STGFormer(**model_config, temporal_mode=TemporalMode.MLP)
        torch.manual_seed(123)
        x = torch.randn(2, 12, 10, 1)
        out2 = model2(x)

        assert torch.allclose(out1, out2, rtol=1e-5)

    def test_different_batch_sizes(self, model_config):
        """Test new modes work with different batch sizes."""
        model_dw = STGFormer(
            **model_config,
            temporal_mode=TemporalMode.DEPTHWISE,
            depthwise_kernel_size=3,
        )
        model_mlp = STGFormer(**model_config, temporal_mode=TemporalMode.MLP)

        for batch_size in [1, 4, 16, 64]:
            x = torch.randn(batch_size, 12, 10, 1)

            out_dw = model_dw(x)
            assert out_dw.shape == (batch_size, 12, 10, 1)

            out_mlp = model_mlp(x)
            assert out_mlp.shape == (batch_size, 12, 10, 1)

    def test_different_num_nodes(self, model_config):
        """Test new modes work with different numbers of nodes."""
        for num_nodes in [10, 50, 207, 325]:
            config = model_config.copy()
            config["num_nodes"] = num_nodes

            model_dw = STGFormer(
                **config,
                temporal_mode=TemporalMode.DEPTHWISE,
                depthwise_kernel_size=3,
            )
            model_mlp = STGFormer(**config, temporal_mode=TemporalMode.MLP)

            x = torch.randn(2, 12, num_nodes, 1)

            out_dw = model_dw(x)
            assert out_dw.shape == (2, 12, num_nodes, 1)

            out_mlp = model_mlp(x)
            assert out_mlp.shape == (2, 12, num_nodes, 1)


class TestTemporalModeEnum:
    """Test TemporalMode enum."""

    def test_enum_values(self):
        """Test all temporal modes are defined."""
        assert TemporalMode.TRANSFORMER.value == "transformer"
        assert TemporalMode.MAMBA.value == "mamba"
        assert TemporalMode.TCN.value == "tcn"
        assert TemporalMode.DEPTHWISE.value == "depthwise"
        assert TemporalMode.MLP.value == "mlp"

    def test_enum_from_string(self):
        """Test creating enum from string values."""
        assert TemporalMode("depthwise") == TemporalMode.DEPTHWISE
        assert TemporalMode("mlp") == TemporalMode.MLP

    def test_enum_in_model_config(self):
        """Test using enum vs string in model creation."""
        # Enum should work
        model1 = STGFormer(
            num_nodes=10,
            in_steps=12,
            out_steps=12,
            temporal_mode=TemporalMode.DEPTHWISE,
            depthwise_kernel_size=3,
        )
        # String gets stored as-is in STGFormer.__init__
        model2 = STGFormer(
            num_nodes=10,
            in_steps=12,
            out_steps=12,
            temporal_mode="depthwise",
            depthwise_kernel_size=3,
        )

        # Model1 stores enum, model2 stores string, but value should match
        assert model1.temporal_mode.value == "depthwise"
        assert model2.temporal_mode == "depthwise"
        # Or convert model2's string to enum for comparison
        assert model1.temporal_mode == TemporalMode(model2.temporal_mode)
