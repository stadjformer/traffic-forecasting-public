"""Tests for StandardScaler in utils.training module."""

import numpy as np
import pytest
import torch

from utils.training import StandardScaler


class TestStandardScaler:
    """Tests for StandardScaler class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        # Shape: (100 samples, 10 features)
        data = np.random.randn(100, 10).astype(np.float32) * 5 + 10
        return torch.from_numpy(data)

    @pytest.fixture
    def constant_data(self):
        """Create constant data for edge case testing."""
        # All features have constant values
        data = torch.ones(50, 5, dtype=torch.float32) * 7.5
        return data

    @pytest.fixture
    def extreme_data(self):
        """Create data with extreme values."""
        np.random.seed(42)
        data = np.random.randn(30, 5).astype(np.float32) * 1e6
        return torch.from_numpy(data)

    def test_init_empty(self):
        """Test initialization without parameters."""
        scaler = StandardScaler()
        assert scaler.mean is None
        assert scaler.std is None

    def test_init_with_params(self):
        """Test initialization with mean and std."""
        mean = torch.tensor([1.0, 2.0, 3.0])
        std = torch.tensor([0.5, 1.0, 1.5])
        scaler = StandardScaler(mean=mean, std=std)
        assert torch.allclose(scaler.mean, mean)
        assert torch.allclose(scaler.std, std)

    def test_fit_transform_computes_correct_stats(self, sample_data):
        """Test that fit_transform computes correct mean and std."""
        scaler = StandardScaler()
        normalized = scaler.fit_transform(sample_data)

        # Verify mean and std were computed
        assert scaler.mean is not None
        assert scaler.std is not None

        # Verify shapes match input features
        assert scaler.mean.shape == (sample_data.shape[1],)
        assert scaler.std.shape == (sample_data.shape[1],)

        # Verify mean is approximately correct
        expected_mean = sample_data.mean(dim=0)
        assert torch.allclose(scaler.mean, expected_mean, atol=1e-5)

        # Verify std is approximately correct (unbiased=False)
        expected_std = sample_data.std(dim=0, unbiased=False)
        assert torch.allclose(scaler.std, expected_std, atol=1e-5)

        # Verify normalized data has mean~0 and std~1
        assert torch.allclose(
            normalized.mean(dim=0), torch.zeros_like(scaler.mean), atol=1e-4
        )
        assert torch.allclose(
            normalized.std(dim=0, unbiased=False),
            torch.ones_like(scaler.std),
            atol=1e-4,
        )

    def test_fit_transform_with_constant_data(self, constant_data):
        """Test fit_transform with constant values (std=0)."""
        scaler = StandardScaler()
        normalized = scaler.fit_transform(constant_data)

        # Mean should be the constant value
        assert torch.allclose(scaler.mean, torch.ones(5) * 7.5)

        # Std should be clamped to minimum value (1e-6)
        assert torch.allclose(scaler.std, torch.ones(5) * 1e-6)

        # Normalized data should be zero (since all values are the same)
        assert torch.allclose(normalized, torch.zeros_like(constant_data), atol=1e-3)

    def test_transform_without_fit_raises_error(self, sample_data):
        """Test that transform raises error if scaler not fitted."""
        scaler = StandardScaler()
        with pytest.raises(RuntimeError, match="StandardScaler must be fitted"):
            scaler.transform(sample_data)

    def test_transform_after_fit(self, sample_data):
        """Test transform works correctly after fit_transform."""
        scaler = StandardScaler()
        scaler.fit_transform(sample_data[:50])  # Fit on first half

        # Transform second half
        transformed = scaler.transform(sample_data[50:])

        # Verify it uses the fitted mean/std from first half
        expected = (sample_data[50:] - scaler.mean) / scaler.std
        assert torch.allclose(transformed, expected, atol=1e-5)

    def test_inverse_transform_without_fit_raises_error(self, sample_data):
        """Test that inverse_transform raises error if scaler not fitted."""
        scaler = StandardScaler()
        with pytest.raises(RuntimeError, match="StandardScaler must be fitted"):
            scaler.inverse_transform(sample_data)

    def test_inverse_transform_reverses_normalization(self, sample_data):
        """Test that inverse_transform correctly reverses normalization."""
        scaler = StandardScaler()
        normalized = scaler.fit_transform(sample_data)
        reconstructed = scaler.inverse_transform(normalized)

        # Should recover original data
        assert torch.allclose(reconstructed, sample_data, atol=1e-4)

    def test_transform_inverse_transform_round_trip(self, sample_data):
        """Test round-trip: transform → inverse_transform."""
        scaler = StandardScaler()
        scaler.fit_transform(sample_data[:50])  # Fit on subset

        # Transform and inverse transform
        transformed = scaler.transform(sample_data)
        reconstructed = scaler.inverse_transform(transformed)

        # Should recover original data
        assert torch.allclose(reconstructed, sample_data, atol=1e-4)

    def test_device_compatibility_cpu_to_gpu(self, sample_data):
        """Test that scaler works when moving data from CPU to GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        scaler = StandardScaler()
        scaler.fit_transform(sample_data)  # Fit on CPU data

        # Transform GPU data
        gpu_data = sample_data.cuda()
        transformed = scaler.transform(gpu_data)

        # Result should be on GPU
        assert transformed.is_cuda

        # Verify correctness
        expected = (gpu_data - scaler.mean.cuda()) / scaler.std.cuda()
        assert torch.allclose(transformed, expected, atol=1e-5)

    def test_device_compatibility_gpu_to_cpu(self, sample_data):
        """Test that scaler works when moving data from GPU to CPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        scaler = StandardScaler()
        gpu_data = sample_data.cuda()
        scaler.fit_transform(gpu_data)  # Fit on GPU data

        # Transform CPU data
        transformed = scaler.transform(sample_data)

        # Result should be on CPU
        assert not transformed.is_cuda

        # Verify correctness
        expected = (sample_data - scaler.mean.cpu()) / scaler.std.cpu()
        assert torch.allclose(transformed, expected, atol=1e-5)

    def test_dtype_compatibility(self, sample_data):
        """Test that scaler handles different dtypes correctly."""
        scaler = StandardScaler()
        scaler.fit_transform(sample_data)  # Fit with float32

        # Transform with float64
        float64_data = sample_data.double()
        transformed = scaler.transform(float64_data)

        # Result should have same dtype as input
        assert transformed.dtype == torch.float64

        # Verify correctness
        expected = (float64_data - scaler.mean.double()) / scaler.std.double()
        assert torch.allclose(transformed, expected, atol=1e-5)

    def test_extreme_values(self, extreme_data):
        """Test scaler with very large values."""
        scaler = StandardScaler()
        normalized = scaler.fit_transform(extreme_data)

        # Should still normalize correctly
        assert torch.allclose(normalized.mean(dim=0), torch.zeros(5), atol=1)
        assert torch.allclose(
            normalized.std(dim=0, unbiased=False), torch.ones(5), atol=1e-4
        )

        # Round trip should work
        reconstructed = scaler.inverse_transform(normalized)
        assert torch.allclose(reconstructed, extreme_data, rtol=1e-4)

    def test_single_sample(self):
        """Test scaler with single sample (edge case)."""
        data = torch.randn(1, 5)
        scaler = StandardScaler()
        scaler.fit_transform(data)

        # Mean should be the data itself
        assert torch.allclose(scaler.mean, data.squeeze(0))

        # Std should be zero (clamped to 1e-6)
        assert torch.allclose(scaler.std, torch.ones(5) * 1e-6)

    def test_fit_transform_accepts_numpy(self):
        """Test that fit_transform accepts numpy arrays."""
        np_data = np.random.randn(50, 3).astype(np.float32)
        scaler = StandardScaler()
        normalized = scaler.fit_transform(np_data)

        # Should convert to tensor and normalize
        assert isinstance(normalized, torch.Tensor)
        assert normalized.shape == (50, 3)

    def test_transform_accepts_numpy(self, sample_data):
        """Test that transform accepts numpy arrays."""
        scaler = StandardScaler()
        scaler.fit_transform(sample_data)

        # Transform numpy array
        np_data = sample_data.numpy()
        transformed = scaler.transform(np_data)

        # Should convert to tensor and normalize
        assert isinstance(transformed, torch.Tensor)
        expected = scaler.transform(sample_data)
        assert torch.allclose(transformed, expected, atol=1e-5)

    def test_inverse_transform_accepts_numpy(self, sample_data):
        """Test that inverse_transform accepts numpy arrays."""
        scaler = StandardScaler()
        normalized = scaler.fit_transform(sample_data)

        # Inverse transform numpy array
        np_normalized = normalized.numpy()
        reconstructed = scaler.inverse_transform(np_normalized)

        # Should convert to tensor and inverse transform
        assert isinstance(reconstructed, torch.Tensor)
        assert torch.allclose(reconstructed, sample_data, atol=1e-4)

    def test_multidimensional_data(self):
        """Test scaler with 3D data (batch, seq, features)."""
        np.random.seed(42)
        # Shape: (32 batch, 12 seq, 5 features)
        data = torch.randn(32, 12, 5)

        scaler = StandardScaler()

        # Should fail - scaler expects 2D input for mean/std computation
        # (Or test the actual behavior if it supports broadcasting)
        # Based on the implementation, it computes mean(dim=0), so let's test actual behavior

        # Fit on flattened features
        data_2d = data.reshape(-1, 5)
        scaler.fit_transform(data_2d)

        # Transform the 3D data
        # The scaler should broadcast correctly
        transformed_3d = scaler.transform(data)

        # Verify shape preserved
        assert transformed_3d.shape == (32, 12, 5)

    def test_preserves_gradient_tracking(self, sample_data):
        """Test that scaler preserves gradient tracking."""
        sample_data.requires_grad_(True)

        scaler = StandardScaler()
        normalized = scaler.fit_transform(sample_data)

        # Gradient should be preserved
        assert normalized.requires_grad

        # Should be able to backprop through it
        loss = normalized.sum()
        loss.backward()
        assert sample_data.grad is not None

    def test_nan_handling(self):
        """Test that scaler handles NaN values correctly."""
        # Data with NaN values
        data = torch.tensor([1.0, 2.0, float("nan"), 3.0, 4.0, float("nan")])

        scaler = StandardScaler()
        normalized = scaler.fit_transform(data)

        # Mean and std should be computed from non-NaN values only
        assert not torch.isnan(scaler.mean), "Mean should not be NaN"
        assert not torch.isnan(scaler.std), "Std should not be NaN"

        # Expected mean of [1, 2, 3, 4] = 2.5
        assert torch.isclose(scaler.mean, torch.tensor(2.5), atol=1e-4)

        # NaN values should remain NaN in output
        assert torch.isnan(normalized[2])
        assert torch.isnan(normalized[5])

        # Non-NaN values should be normalized correctly
        non_nan_mask = ~torch.isnan(normalized)
        assert non_nan_mask.sum() == 4

    def test_nan_with_mask_value(self):
        """Test that scaler handles both NaN and mask_value correctly."""
        # Data with NaN values and zeros to mask
        data = torch.tensor([0.0, 1.0, 2.0, float("nan"), 3.0, 4.0, 0.0])

        scaler = StandardScaler()
        normalized = scaler.fit_transform(data, mask_value=0.0)

        # Mean should be computed from [1, 2, 3, 4] only (excluding 0s and NaN)
        assert torch.isclose(scaler.mean, torch.tensor(2.5), atol=1e-4)

    def test_all_nan_fallback(self):
        """Test that scaler handles all-NaN data gracefully."""
        data = torch.tensor([float("nan"), float("nan"), float("nan")])

        scaler = StandardScaler()
        normalized = scaler.fit_transform(data)

        # Should use fallback values
        assert scaler.mean == 0.0
        assert scaler.std == 1.0
