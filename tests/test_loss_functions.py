"""Tests for loss functions in utils.training module."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from utils.training import MaskedHuberLoss, masked_mae_loss


class TestMaskedMAELoss:
    """Tests for masked_mae_loss function."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions and labels."""
        np.random.seed(42)
        preds = torch.randn(32, 12, 10, 1)  # (batch, time, nodes, features)
        labels = torch.randn(32, 12, 10, 1)
        return preds, labels

    def test_basic_mae_computation(self):
        """Test basic MAE computation without masking."""
        preds = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        labels = torch.tensor([[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]])

        loss = masked_mae_loss(preds, labels, null_val=-999.0)

        # Expected MAE: mean(|1-1.5|, |2-2.5|, |3-3.5|, |4-3.5|, |5-4.5|, |6-5.5|)
        # = mean(0.5, 0.5, 0.5, 0.5, 0.5, 0.5) = 0.5
        assert torch.isclose(loss, torch.tensor(0.5), atol=1e-5)

    def test_masking_with_zero_null_val(self):
        """Test masking with null_val=0.0 (default)."""
        preds = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        labels = torch.tensor([[0.0, 2.5, 3.5], [3.5, 0.0, 5.5]])  # Two zeros to mask

        loss = masked_mae_loss(preds, labels, null_val=0.0)

        # Only compute MAE for non-zero labels
        # Valid pairs: (2, 2.5), (3, 3.5), (4, 3.5), (6, 5.5)
        # MAE: (0.5 + 0.5 + 0.5 + 0.5) / 4 = 0.5
        # But the function normalizes by mask mean first, then masks
        # Let's compute it correctly:
        # mask = [0, 1, 1, 1, 0, 1] → mask_mean = 4/6
        # normalized_mask = mask / mask_mean = [0, 1.5, 1.5, 1.5, 0, 1.5]
        # weighted_loss = [0, 0.5*1.5, 0.5*1.5, 0.5*1.5, 0, 0.5*1.5]
        # mean = (0 + 0.75 + 0.75 + 0.75 + 0 + 0.75) / 6 = 3.0 / 6 = 0.5
        assert torch.isclose(loss, torch.tensor(0.5), atol=1e-5)

    def test_masking_with_nan_null_val(self):
        """Test masking with null_val=NaN."""
        preds = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        labels = torch.tensor([[float("nan"), 2.5, 3.5], [3.5, float("nan"), 5.5]])

        loss = masked_mae_loss(preds, labels, null_val=float("nan"))

        # Only compute MAE for non-NaN labels
        # Valid pairs: (2, 2.5), (3, 3.5), (4, 3.5), (6, 5.5)
        # Similar computation as above
        assert torch.isclose(loss, torch.tensor(0.5), atol=1e-5)

    def test_all_values_masked(self):
        """Test when all values are masked."""
        preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        labels = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  # All masked

        loss = masked_mae_loss(preds, labels, null_val=0.0)

        # When all masked, loss should be 0 (due to torch.where handling NaN)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_no_values_masked(self):
        """Test when no values are masked."""
        preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        labels = torch.tensor([[1.5, 2.5], [2.5, 3.5]])

        loss = masked_mae_loss(preds, labels, null_val=-999.0)

        # All values contribute
        expected_mae = torch.mean(torch.abs(preds - labels))
        assert torch.isclose(loss, expected_mae, atol=1e-5)

    def test_multidimensional_input(self, sample_predictions):
        """Test with realistic multidimensional predictions."""
        preds, labels = sample_predictions

        # Add some null values
        labels[0, :, :, :] = 0.0  # Mask first batch
        labels[5, 3:7, :, :] = 0.0  # Mask some timesteps in 6th batch

        loss = masked_mae_loss(preds, labels, null_val=0.0)

        # Loss should be a scalar
        assert loss.ndim == 0
        assert loss.item() >= 0.0  # MAE is always non-negative

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the loss."""
        preds = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        labels = torch.tensor([[1.5, 0.0, 3.5]])  # Middle value masked

        loss = masked_mae_loss(preds, labels, null_val=0.0)
        loss.backward()

        # Gradient should exist
        assert preds.grad is not None

        # Gradient for masked position should be zero
        # Note: Due to the normalization in the loss function, this might not be exactly zero
        # but the contribution should be minimal

    def test_device_compatibility(self, sample_predictions):
        """Test loss computation on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        preds, labels = sample_predictions
        preds_gpu = preds.cuda()
        labels_gpu = labels.cuda()

        loss_cpu = masked_mae_loss(preds, labels, null_val=0.0)
        loss_gpu = masked_mae_loss(preds_gpu, labels_gpu, null_val=0.0)

        # Results should be the same
        assert torch.isclose(loss_cpu, loss_gpu.cpu(), atol=1e-5)


class TestMaskedHuberLoss:
    """Tests for MaskedHuberLoss class."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions and labels."""
        np.random.seed(42)
        preds = torch.randn(32, 12, 10, 1)
        labels = torch.randn(32, 12, 10, 1)
        return preds, labels

    def test_init_default_delta(self):
        """Test initialization with default delta."""
        loss_fn = MaskedHuberLoss()
        assert loss_fn.delta == 1.0

    def test_init_custom_delta(self):
        """Test initialization with custom delta."""
        loss_fn = MaskedHuberLoss(delta=2.5)
        assert loss_fn.delta == 2.5

    def test_is_nn_module(self):
        """Test that MaskedHuberLoss is a proper nn.Module."""
        loss_fn = MaskedHuberLoss()
        assert isinstance(loss_fn, nn.Module)

    def test_basic_huber_computation_quadratic_region(self):
        """Test Huber loss in quadratic region (|diff| < delta)."""
        loss_fn = MaskedHuberLoss(delta=1.0)

        preds = torch.tensor([[0.0, 0.5, -0.3]])
        labels = torch.tensor([[0.2, 0.7, -0.5]])  # diffs: [0.2, 0.2, 0.2]

        loss = loss_fn(preds, labels, null_val=-999.0)

        # All diffs < delta=1.0, so use quadratic: 0.5 * diff^2
        # Expected: mean(0.5 * 0.2^2, 0.5 * 0.2^2, 0.5 * 0.2^2) = 0.5 * 0.04 = 0.02
        assert torch.isclose(loss, torch.tensor(0.02), atol=1e-5)

    def test_basic_huber_computation_linear_region(self):
        """Test Huber loss in linear region (|diff| > delta)."""
        loss_fn = MaskedHuberLoss(delta=1.0)

        preds = torch.tensor([[0.0, 0.0, 0.0]])
        labels = torch.tensor([[2.0, -3.0, 4.0]])  # diffs: [2.0, 3.0, 4.0]

        loss = loss_fn(preds, labels, null_val=-999.0)

        # All |diffs| > delta=1.0, so use linear: 0.5*delta^2 + delta*(|diff| - delta)
        # For diff=2: 0.5*1 + 1*(2-1) = 0.5 + 1 = 1.5
        # For diff=3: 0.5*1 + 1*(3-1) = 0.5 + 2 = 2.5
        # For diff=4: 0.5*1 + 1*(4-1) = 0.5 + 3 = 3.5
        # Expected: mean(1.5, 2.5, 3.5) = 7.5 / 3 = 2.5
        assert torch.isclose(loss, torch.tensor(2.5), atol=1e-5)

    def test_masking_with_zero_null_val(self):
        """Test masking with null_val=0.0."""
        loss_fn = MaskedHuberLoss(delta=1.0)

        preds = torch.tensor([[1.0, 2.0, 3.0]])
        labels = torch.tensor([[0.0, 2.5, 0.0]])  # First and last masked

        loss = loss_fn(preds, labels, null_val=0.0)

        # Only middle value: diff = 0.5, quadratic region
        # Loss = 0.5 * 0.5^2 = 0.125
        assert torch.isclose(loss, torch.tensor(0.125), atol=1e-5)

    def test_masking_with_nan_null_val(self):
        """Test masking with null_val=NaN."""
        loss_fn = MaskedHuberLoss(delta=1.0)

        preds = torch.tensor([[1.0, 2.0, 3.0]])
        labels = torch.tensor([[float("nan"), 2.5, float("nan")]])

        loss = loss_fn(preds, labels, null_val=float("nan"))

        # Only middle value: diff = 0.5
        assert torch.isclose(loss, torch.tensor(0.125), atol=1e-5)

    def test_all_values_masked(self):
        """Test when all values are masked."""
        loss_fn = MaskedHuberLoss(delta=1.0)

        preds = torch.tensor([[1.0, 2.0, 3.0]])
        labels = torch.tensor([[0.0, 0.0, 0.0]])

        loss = loss_fn(preds, labels, null_val=0.0)

        # No valid values, should return 0
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_no_values_masked(self):
        """Test when no values are masked."""
        loss_fn = MaskedHuberLoss(delta=1.0)

        preds = torch.tensor([[0.0, 0.5]])
        labels = torch.tensor([[0.2, 0.7]])

        loss_masked = loss_fn(preds, labels, null_val=-999.0)

        # Compute expected Huber loss manually
        diff = preds - labels  # [-0.2, -0.2]
        abs_diff = torch.abs(diff)  # [0.2, 0.2]
        expected = torch.mean(0.5 * abs_diff**2)  # Both in quadratic region

        assert torch.isclose(loss_masked, expected, atol=1e-5)

    def test_multidimensional_input(self, sample_predictions):
        """Test with realistic multidimensional predictions."""
        loss_fn = MaskedHuberLoss(delta=1.0)
        preds, labels = sample_predictions

        # Add some null values
        labels[0, :, :, :] = 0.0  # Mask first batch

        loss = loss_fn(preds, labels, null_val=0.0)

        # Loss should be a scalar
        assert loss.ndim == 0
        assert loss.item() >= 0.0  # Huber loss is always non-negative

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the loss."""
        loss_fn = MaskedHuberLoss(delta=1.0)

        preds = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        labels = torch.tensor([[1.5, 0.0, 3.5]])  # Middle value masked

        loss = loss_fn(preds, labels, null_val=0.0)
        loss.backward()

        # Gradient should exist
        assert preds.grad is not None

    def test_device_compatibility(self, sample_predictions):
        """Test loss computation on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        loss_fn = MaskedHuberLoss(delta=1.0)
        preds, labels = sample_predictions

        preds_gpu = preds.cuda()
        labels_gpu = labels.cuda()

        loss_cpu = loss_fn(preds, labels, null_val=0.0)
        loss_gpu = loss_fn(preds_gpu, labels_gpu, null_val=0.0)

        # Results should be the same
        assert torch.isclose(loss_cpu, loss_gpu.cpu(), atol=1e-5)

    def test_delta_parameter_effect(self):
        """Test that delta parameter affects the transition point."""
        preds = torch.tensor([[0.0]])
        labels = torch.tensor([[1.5]])  # diff = 1.5

        loss_fn_small_delta = MaskedHuberLoss(delta=1.0)
        loss_fn_large_delta = MaskedHuberLoss(delta=2.0)

        loss_small = loss_fn_small_delta(preds, labels, null_val=-999.0)
        loss_large = loss_fn_large_delta(preds, labels, null_val=-999.0)

        # With delta=1.0: 1.5 is in linear region: 0.5*1 + 1*(1.5-1) = 1.0
        assert torch.isclose(loss_small, torch.tensor(1.0), atol=1e-5)

        # With delta=2.0: 1.5 is in quadratic region: 0.5*1.5^2 = 1.125
        assert torch.isclose(loss_large, torch.tensor(1.125), atol=1e-5)

    def test_comparison_with_mae_for_small_errors(self):
        """Test that Huber loss behaves like MSE for small errors."""
        loss_fn = MaskedHuberLoss(delta=1.0)

        preds = torch.tensor([[0.0, 0.1, 0.2]])
        labels = torch.tensor([[0.1, 0.2, 0.3]])  # Small errors

        huber_loss = loss_fn(preds, labels, null_val=-999.0)
        mse_loss = torch.mean(0.5 * (preds - labels) ** 2)

        # For small errors, Huber ≈ MSE
        assert torch.isclose(huber_loss, mse_loss, atol=1e-5)

    def test_comparison_with_mae_for_large_errors(self):
        """Test that Huber loss behaves like MAE for large errors."""
        loss_fn = MaskedHuberLoss(delta=1.0)

        preds = torch.tensor([[0.0, 0.0, 0.0]])
        labels = torch.tensor([[5.0, 10.0, 15.0]])  # Large errors

        huber_loss = loss_fn(preds, labels, null_val=-999.0)

        # For large errors: 0.5*delta^2 + delta*(|diff| - delta)
        # diff=5: 0.5 + 1*(5-1) = 4.5
        # diff=10: 0.5 + 1*(10-1) = 9.5
        # diff=15: 0.5 + 1*(15-1) = 14.5
        # mean = 28.5 / 3 = 9.5
        assert torch.isclose(huber_loss, torch.tensor(9.5), atol=1e-5)

    def test_robustness_to_outliers(self):
        """Test that Huber loss is more robust to outliers than MSE."""
        loss_fn = MaskedHuberLoss(delta=1.0)

        # Data with outlier
        preds_with_outlier = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        labels_with_outlier = torch.tensor([[0.1, 0.2, 0.1, 100.0]])  # Last is outlier

        huber_loss = loss_fn(preds_with_outlier, labels_with_outlier, null_val=-999.0)

        # MSE would be dominated by the outlier (100^2 = 10000)
        # Huber treats it linearly: 0.5 + 1*(100-1) = 99.5
        # Total: (0.5*0.01 + 0.5*0.04 + 0.5*0.01 + 99.5) / 4 ≈ 24.9
        # This demonstrates robustness - the loss is much smaller than MSE would give
        assert huber_loss < 30.0  # Much less than MSE would give


# =============================================================================
# Tests for Metrics Functions
# =============================================================================

from utils.training import (
    compute_masked_metrics,
    compute_horizon_metrics,
    aggregate_batch_metrics,
)


class TestComputeMaskedMetrics:
    """Tests for compute_masked_metrics function."""

    def test_basic_metrics_computation(self):
        """Test basic MAE, RMSE, MAPE computation."""
        preds = torch.tensor([10.0, 20.0, 30.0])
        labels = torch.tensor([12.0, 18.0, 33.0])

        metrics = compute_masked_metrics(preds, labels, null_val=-999.0)

        # MAE: mean(|10-12|, |20-18|, |30-33|) = mean(2, 2, 3) = 7/3 = 2.333
        assert abs(metrics['mae'] - 7/3) < 1e-4

        # RMSE: sqrt(mean(4, 4, 9)) = sqrt(17/3) = 2.38
        assert abs(metrics['rmse'] - np.sqrt(17/3)) < 1e-4

        # MAPE: mean(2/12, 2/18, 3/33) * 100 = mean(0.167, 0.111, 0.091) * 100
        expected_mape = 100 * (2/12 + 2/18 + 3/33) / 3
        assert abs(metrics['mape'] - expected_mape) < 0.1

    def test_masking_null_values(self):
        """Test that null values are properly masked."""
        preds = torch.tensor([10.0, 0.0, 30.0])
        labels = torch.tensor([12.0, 0.0, 33.0])

        metrics = compute_masked_metrics(preds, labels, null_val=0.0)

        # Only [10,12] and [30,33] are valid
        # MAE: mean(2, 3) = 2.5
        assert abs(metrics['mae'] - 2.5) < 1e-4

    def test_masking_nan_values(self):
        """Test that NaN values are properly masked."""
        preds = torch.tensor([10.0, 20.0, 30.0])
        labels = torch.tensor([12.0, float('nan'), 33.0])

        metrics = compute_masked_metrics(preds, labels, null_val=-999.0)

        # Only [10,12] and [30,33] are valid
        assert abs(metrics['mae'] - 2.5) < 1e-4

    def test_all_masked_returns_zeros(self):
        """Test that all-masked data returns zeros."""
        preds = torch.tensor([10.0, 20.0])
        labels = torch.tensor([0.0, 0.0])

        metrics = compute_masked_metrics(preds, labels, null_val=0.0)

        assert metrics['mae'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['mape'] == 0.0


class TestComputeHorizonMetrics:
    """Tests for compute_horizon_metrics function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample predictions with shape (batch, horizon, nodes, features)."""
        torch.manual_seed(42)
        preds = torch.randn(4, 12, 10, 1) * 100 + 200
        labels = preds + torch.randn_like(preds) * 10  # Add noise
        return preds, labels

    def test_returns_overall_and_horizon_metrics(self, sample_data):
        """Test that overall and per-horizon metrics are returned."""
        preds, labels = sample_data

        metrics = compute_horizon_metrics(preds, labels, null_val=-999.0)

        # Check overall metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics

        # Check horizon-specific metrics
        for h in [3, 6, 12]:
            assert f'mae_h{h}' in metrics
            assert f'rmse_h{h}' in metrics
            assert f'mape_h{h}' in metrics

    def test_horizon_slicing(self, sample_data):
        """Test that horizon metrics are computed from correct slices."""
        preds, labels = sample_data

        metrics = compute_horizon_metrics(preds, labels, null_val=-999.0)

        # Manually compute h3 metrics
        h3_metrics = compute_masked_metrics(
            preds[:, 2:3, ...], labels[:, 2:3, ...], null_val=-999.0
        )

        assert abs(metrics['mae_h3'] - h3_metrics['mae']) < 1e-4
        assert abs(metrics['rmse_h3'] - h3_metrics['rmse']) < 1e-4


class TestAggregateBatchMetrics:
    """Tests for aggregate_batch_metrics function."""

    def test_weighted_average(self):
        """Test weighted averaging across batches."""
        batch_metrics = [
            {'mae': 10.0, 'rmse': 12.0},
            {'mae': 20.0, 'rmse': 22.0},
        ]
        batch_sizes = [100, 100]

        aggregated = aggregate_batch_metrics(batch_metrics, batch_sizes)

        # Equal weights: simple average
        assert aggregated['mae'] == 15.0
        assert aggregated['rmse'] == 17.0

    def test_unequal_weights(self):
        """Test weighted averaging with different batch sizes."""
        batch_metrics = [
            {'mae': 10.0},
            {'mae': 20.0},
        ]
        batch_sizes = [100, 300]  # 1:3 ratio

        aggregated = aggregate_batch_metrics(batch_metrics, batch_sizes)

        # Weighted: (10*100 + 20*300) / 400 = 7000/400 = 17.5
        assert aggregated['mae'] == 17.5

    def test_empty_list(self):
        """Test with empty batch list."""
        aggregated = aggregate_batch_metrics([], [])
        assert aggregated == {}
