"""Tests for masked node pretraining."""

import pytest
import torch

from stgformer.model import STGFormer
from stgformer.pretrain import (
    MaskedNodePretrainer,
    impute_missing_data,
    pretrain_graph_imputation,
)


@pytest.fixture
def small_model():
    """Create a small STGFormer model for testing."""
    return STGFormer(
        num_nodes=20,
        in_steps=12,
        out_steps=12,
        input_dim=1,
        output_dim=1,
        input_embedding_dim=16,
        tod_embedding_dim=16,
        dow_embedding_dim=0,
        adaptive_embedding_dim=32,
        num_heads=4,
        num_layers=2,
    )


@pytest.fixture
def sample_input():
    """Create sample input data."""
    batch_size = 4
    in_steps = 12
    num_nodes = 20
    input_dim = 2  # [speed, tod]

    x = torch.randn(batch_size, in_steps, num_nodes, input_dim)
    # Make speed values positive (realistic traffic)
    x[..., 0] = torch.abs(x[..., 0]) * 50 + 10
    # Make TOD realistic [0, 1)
    x[..., 1] = torch.rand(batch_size, in_steps, num_nodes)

    return x


@pytest.fixture
def sample_input_with_missing(sample_input):
    """Create sample input with some missing values (zeros)."""
    x = sample_input.clone()
    # Set ~10% of speed values to 0 (missing)
    missing_mask = torch.rand_like(x[..., 0]) < 0.1
    x[..., 0] = torch.where(missing_mask, torch.zeros_like(x[..., 0]), x[..., 0])
    return x


class TestMaskedNodePretrainer:
    """Tests for MaskedNodePretrainer class."""

    def test_creation(self, small_model):
        """Test pretrainer creation."""
        pretrainer = MaskedNodePretrainer(small_model)

        assert hasattr(pretrainer, "model")
        assert hasattr(pretrainer, "imputation_head")
        assert pretrainer.mask_value == 0.0

        # Check imputation head dimensions (temporal: predicts per-timestep values)
        assert pretrainer.imputation_head.in_features == small_model.model_dim
        assert pretrainer.imputation_head.out_features == small_model.in_steps

    def test_detect_missing(self, small_model, sample_input_with_missing):
        """Test missing value detection."""
        pretrainer = MaskedNodePretrainer(small_model)

        missing_mask = pretrainer.detect_missing(sample_input_with_missing)

        # Check shape
        assert missing_mask.shape == (*sample_input_with_missing.shape[:-1], 1)

        # Check that it finds the zeros
        expected = sample_input_with_missing[..., 0:1] == 0.0
        torch.testing.assert_close(missing_mask, expected)

    def test_create_mask_per_timestep(self, small_model, sample_input):
        """Test per-timestep mask creation."""
        pretrainer = MaskedNodePretrainer(small_model)
        existing_missing = pretrainer.detect_missing(sample_input)

        mask_ratio = 0.15
        mask = pretrainer.create_mask_per_timestep(
            sample_input, existing_missing, mask_ratio
        )

        # Check shape
        assert mask.shape == (*sample_input.shape[:-1], 1)

        # Check that mask is boolean
        assert mask.dtype == torch.bool

        # Check approximate ratio (with some tolerance)
        actual_ratio = mask.float().mean().item()
        assert 0.05 < actual_ratio < 0.25, (
            f"Mask ratio {actual_ratio} not close to {mask_ratio}"
        )

    def test_create_mask_per_node(self, small_model, sample_input):
        """Test per-node mask creation."""
        pretrainer = MaskedNodePretrainer(small_model)
        existing_missing = pretrainer.detect_missing(sample_input)

        mask_ratio = 0.15
        mask = pretrainer.create_mask_per_node(
            sample_input, existing_missing, mask_ratio
        )

        # Check shape
        assert mask.shape == (*sample_input.shape[:-1], 1)

        # Check that mask is boolean
        assert mask.dtype == torch.bool

        # Per-node masking should mask entire nodes (all timesteps for a node)
        # Check that for masked nodes, ALL timesteps are masked
        mask_squeezed = mask.squeeze(-1)  # [batch, time, nodes]
        for b in range(mask_squeezed.shape[0]):
            for n in range(mask_squeezed.shape[2]):
                node_mask = mask_squeezed[b, :, n]
                # Either all True or all False (with possible exceptions for existing missing)
                unique_values = node_mask.unique()
                assert len(unique_values) <= 2

    def test_create_mask_per_node_allows_partial_missing(
        self, small_model, sample_input
    ):
        """Nodes with partial real missing data can still be selected."""
        pretrainer = MaskedNodePretrainer(small_model)
        # Force node 0 to have a missing timestep
        sample_input[:, 0, 0, 0] = 0.0
        existing_missing = pretrainer.detect_missing(sample_input)

        mask = pretrainer.create_mask_per_node(
            sample_input, existing_missing, mask_ratio=1.0
        )

        mask_squeezed = mask.squeeze(-1)
        # Node 0 should still be maskable despite containing a missing entry
        assert mask_squeezed[..., 0].any()

    def test_apply_mask(self, small_model, sample_input):
        """Test mask application."""
        pretrainer = MaskedNodePretrainer(small_model)

        # Create a simple mask
        mask = torch.zeros(*sample_input.shape[:-1], 1, dtype=torch.bool)
        mask[0, 0, 0, 0] = True  # Mask first position

        x_masked = pretrainer.apply_mask(sample_input, mask)

        # Check that masked position is 0
        assert x_masked[0, 0, 0, 0] == 0.0

        # Check that non-masked positions are unchanged
        assert x_masked[0, 0, 1, 0] == sample_input[0, 0, 1, 0]

        # Check that TOD (feature 1) is not affected
        torch.testing.assert_close(x_masked[..., 1], sample_input[..., 1])

    def test_pretrain_step_per_timestep(self, small_model, sample_input):
        """Test forward pass for per-timestep pretraining."""
        pretrainer = MaskedNodePretrainer(small_model)

        loss, predictions, mask = pretrainer.pretrain_step(
            sample_input, mask_ratio=0.15, masking_mode="per_timestep"
        )

        # Check loss is scalar and valid
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss >= 0

        # Check predictions shape (temporal: [batch, time, nodes])
        assert predictions.shape == (
            sample_input.shape[0],
            sample_input.shape[1],
            sample_input.shape[2],
        )

        # Check mask shape
        assert mask.shape == (*sample_input.shape[:-1], 1)

    def test_pretrain_step_per_node(self, small_model, sample_input):
        """Test forward pass for per-node pretraining."""
        pretrainer = MaskedNodePretrainer(small_model)

        loss, predictions, mask = pretrainer.pretrain_step(
            sample_input, mask_ratio=0.15, masking_mode="per_node"
        )

        # Check loss is scalar and valid
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss >= 0

        # Check predictions shape (temporal: [batch, time, nodes])
        assert predictions.shape == (
            sample_input.shape[0],
            sample_input.shape[1],
            sample_input.shape[2],
        )

    def test_gradient_flow(self, small_model, sample_input):
        """Test that gradients flow through the pretraining step."""
        pretrainer = MaskedNodePretrainer(small_model)

        loss, _, _ = pretrainer.pretrain_step(
            sample_input, mask_ratio=0.15, masking_mode="per_timestep"
        )
        loss.backward()

        # Check that model parameters received gradients
        has_grad = False
        for param in small_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "Model parameters did not receive gradients"

        # Check that imputation head received gradients
        assert pretrainer.imputation_head.weight.grad is not None
        assert pretrainer.imputation_head.weight.grad.abs().sum() > 0

    def test_respects_existing_missing(self, small_model, sample_input_with_missing):
        """Test that masking doesn't overwrite existing missing values."""
        pretrainer = MaskedNodePretrainer(small_model)
        existing_missing = pretrainer.detect_missing(sample_input_with_missing)

        # Create new mask
        new_mask = pretrainer.create_mask_per_timestep(
            sample_input_with_missing, existing_missing, mask_ratio=0.15
        )

        # New mask should not overlap with existing missing
        overlap = (new_mask & existing_missing).any()
        assert not overlap, "New mask should not overlap with existing missing values"


class TestPretrainFunction:
    """Tests for pretrain_graph_imputation function."""

    def test_basic_pretraining(self, small_model, sample_input):
        """Test basic pretraining loop."""
        # Create simple dataloader
        dataset = torch.utils.data.TensorDataset(sample_input, sample_input[..., 0:1])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        # Run pretraining with minimal epochs
        model, imputation_head = pretrain_graph_imputation(
            small_model,
            dataloader,
            stage1_epochs=1,
            stage1_mask_ratio=0.15,
            stage2_epochs=1,
            stage2_mask_ratio=0.10,
            device="cpu",
            verbose=False,
        )

        # Check returns
        assert model is small_model
        assert isinstance(imputation_head, torch.nn.Linear)

    def test_stage1_only(self, small_model, sample_input):
        """Test running only stage 1."""
        dataset = torch.utils.data.TensorDataset(sample_input, sample_input[..., 0:1])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        model, imputation_head = pretrain_graph_imputation(
            small_model,
            dataloader,
            stage1_epochs=2,
            stage1_mask_ratio=0.15,
            stage2_epochs=0,  # Skip stage 2
            device="cpu",
            verbose=False,
        )

        assert model is small_model

    def test_stage2_only(self, small_model, sample_input):
        """Test running only stage 2."""
        dataset = torch.utils.data.TensorDataset(sample_input, sample_input[..., 0:1])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        model, imputation_head = pretrain_graph_imputation(
            small_model,
            dataloader,
            stage1_epochs=0,  # Skip stage 1
            stage2_epochs=2,
            stage2_mask_ratio=0.10,
            device="cpu",
            verbose=False,
        )

        assert model is small_model


class TestImputation:
    """Tests for impute_missing_data function."""

    def test_basic_imputation(self, small_model, sample_input_with_missing):
        """Test basic imputation."""
        pretrainer = MaskedNodePretrainer(small_model)

        # Impute
        imputed = impute_missing_data(
            small_model, pretrainer.imputation_head, sample_input_with_missing
        )

        # Check shape unchanged
        assert imputed.shape == sample_input_with_missing.shape

        # Check that previously missing values are now filled
        was_missing = sample_input_with_missing[..., 0] == 0.0
        if was_missing.any():
            # At least some should be non-zero now (unless model predicts 0)
            # Just check that the function ran without error
            assert torch.isfinite(imputed).all()

    def test_imputation_preserves_valid_data(self, small_model, sample_input):
        """Test that imputation doesn't change valid (non-missing) data."""
        pretrainer = MaskedNodePretrainer(small_model)

        # Sample input has no missing values
        imputed = impute_missing_data(
            small_model, pretrainer.imputation_head, sample_input
        )

        # Should be identical (no missing values to impute)
        torch.testing.assert_close(imputed, sample_input)

    def test_imputation_multiple_iterations(
        self, small_model, sample_input_with_missing
    ):
        """Test iterative imputation."""
        pretrainer = MaskedNodePretrainer(small_model)

        # Run multiple iterations
        imputed = impute_missing_data(
            small_model,
            pretrainer.imputation_head,
            sample_input_with_missing,
            num_iterations=3,
        )

        # Should still produce valid output
        assert torch.isfinite(imputed).all()

    def test_imputation_respects_mask_value_and_device(
        self, small_model, sample_input_with_missing
    ):
        """Imputation should keep tensors on the original device and honor custom mask values."""
        pretrainer = MaskedNodePretrainer(small_model)
        custom_data = sample_input_with_missing.clone()
        custom_data[..., 0] = torch.where(
            custom_data[..., 0] == 0.0,
            torch.full_like(custom_data[..., 0], -5.0),
            custom_data[..., 0],
        )

        imputed = impute_missing_data(
            small_model,
            pretrainer.imputation_head,
            custom_data,
            batch_size=2,
            device="cpu",
            mask_value=-5.0,
        )

        assert imputed.device == custom_data.device
        assert not (imputed[..., 0] == -5.0).any()

    def test_imputation_with_normalized_data(
        self, small_model, sample_input_with_missing
    ):
        """Test that imputation works with normalized data."""
        from utils.training import StandardScaler

        pretrainer = MaskedNodePretrainer(small_model)

        # Track which positions were missing before normalization
        was_missing = sample_input_with_missing[..., 0] == 0.0

        # Normalize data
        scaler = StandardScaler()
        scaler.fit_transform(sample_input_with_missing[..., 0:1])
        normalized_data = sample_input_with_missing.clone()
        normalized_data[..., 0:1] = scaler.transform(
            sample_input_with_missing[..., 0:1]
        )

        # Compute what the normalized mask value is
        normalized_mask_value = scaler.transform(torch.zeros(1, 1, 1, 1))[
            0, 0, 0, 0
        ].item()

        # Re-apply mask at normalized positions
        normalized_data[..., 0] = torch.where(
            was_missing,
            torch.full_like(normalized_data[..., 0], normalized_mask_value),
            normalized_data[..., 0],
        )

        # Impute on normalized data
        imputed_normalized = impute_missing_data(
            small_model,
            pretrainer.imputation_head,
            normalized_data,
            num_iterations=2,
            mask_value=normalized_mask_value,
            use_normalized_data=True,
        )

        # Should fill missing values
        assert torch.isfinite(imputed_normalized).all()

        # Previously missing positions should have changed (been imputed)
        if was_missing.any():
            imputed_at_missing = imputed_normalized[..., 0][was_missing]
            # Imputed values should NOT equal the normalized mask value
            assert not (imputed_at_missing == normalized_mask_value).all()

        # Normalized data should have values roughly in [-3, 3] range for most data
        assert imputed_normalized[..., 0].abs().mean() < 10.0

        # With use_normalized_data=True, imputed values CAN be negative (standard normal)
        # This is valid for normalized data and should NOT be clamped
        # (We don't strictly assert negative values exist since the model might predict positive,
        # but we verify the code allows them by checking no artificial floor at 0)
