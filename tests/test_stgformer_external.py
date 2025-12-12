"""Tests for utils.stgformer_external module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from utils.dataset import TrafficDataset
from utils.stgformer_external import (
    _get_best_device_stgformer,
    get_stgformer_predictions,
    load_model,
    train_model,
)


class TestTrainModel:
    """Tests for train_model function."""

    @pytest.fixture
    def mock_datasets(self):
        """Create minimal mock datasets."""
        np.random.seed(42)
        datasets = {}
        for split in ["train", "val", "test"]:
            # STGformer requires input_dim=2 [value, time_of_day]
            x = np.random.randn(20, 12, 10, 2).astype(np.float32)
            # Ensure time_of_day is in [0, 1)
            x[..., 1] = np.abs(x[..., 1]) % 1.0
            y = np.random.randn(20, 12, 10, 1).astype(np.float32)
            datasets[split] = TrafficDataset(
                x, y, seq_len=12, horizon=12, num_nodes=10, input_dim=2, output_dim=1
            )
        return datasets

    def test_train_model_minimal(self, mock_datasets, tmp_path):
        """Test that train_model runs with minimal epochs."""
        adjacency = np.random.rand(10, 10).astype(np.float32)
        adjacency = (adjacency + adjacency.T) / 2  # symmetric

        # Train for just 1 epoch (force CPU to avoid MPS in-place operation issues)
        model, scaler, _ = train_model(
            dataset_name="METR-LA",
            pytorch_datasets=mock_datasets,
            adjacency=adjacency,
            epochs=1,
            batch_size=8,
            save_dir=tmp_path,
            verbose=False,
            device="cpu",  # Force CPU to avoid MPS compatibility issues in tests
        )

        # Check that model and scaler were returned
        assert model is not None
        assert scaler is not None

        # Check that files were saved (safetensors format)
        assert (tmp_path / "model.safetensors").exists()
        assert (tmp_path / "config.json").exists()
        assert (tmp_path / "metadata.json").exists()

    def test_train_model_wrong_input_dim(self, tmp_path):
        """Test that train_model raises error with wrong input_dim."""
        # Create datasets with input_dim=1 (wrong, should be 2)
        np.random.seed(42)
        datasets = {}
        for split in ["train", "val", "test"]:
            x = np.random.randn(20, 12, 10, 1).astype(np.float32)
            y = np.random.randn(20, 12, 10, 1).astype(np.float32)
            datasets[split] = TrafficDataset(
                x, y, seq_len=12, horizon=12, num_nodes=10, input_dim=1, output_dim=1
            )

        adjacency = np.random.rand(10, 10).astype(np.float32)

        with pytest.raises(ValueError, match="requires input_dim=2"):
            train_model(
                dataset_name="METR-LA",
                pytorch_datasets=datasets,
                adjacency=adjacency,
                epochs=1,
                batch_size=8,
                save_dir=tmp_path,
                verbose=False,
            )


class TestGetSTGformerPredictions:
    """Tests for get_stgformer_predictions function."""

    def test_get_predictions_shape(self):
        """Test that get_stgformer_predictions returns correct shape."""
        # Create mock model and scaler
        mock_model = MagicMock()
        mock_scaler = MagicMock()

        # Mock the stgformer_predict function
        with patch("utils.stgformer_external.stgformer_predict") as mock_predict:
            mock_predict.return_value = np.random.randn(20, 12, 10, 1).astype(
                np.float32
            )

            # Create mock dataset
            np.random.seed(42)
            x = np.random.randn(20, 12, 10, 2).astype(np.float32)
            y = np.random.randn(20, 12, 10, 1).astype(np.float32)
            dataset = TrafficDataset(
                x, y, seq_len=12, horizon=12, num_nodes=10, input_dim=2, output_dim=1
            )

            predictions = get_stgformer_predictions(mock_model, mock_scaler, dataset)

            # Check shape
            assert predictions.shape == (20, 12, 10, 1)

            # Check that stgformer_predict was called
            mock_predict.assert_called_once()


class TestLoadModel:
    """Tests for load_model function."""

    def test_load_model_file_not_found(self, tmp_path):
        """Test that load_model raises error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_model("METR-LA", tmp_path, device="cpu")


class TestGetBestDevice:
    """Tests for _get_best_device_stgformer function."""

    def test_get_best_device_returns_string(self):
        """Test that _get_best_device_stgformer returns a valid device string.

        Note: This function never returns 'mps' due to STGFormer compatibility issues.
        """
        device = _get_best_device_stgformer()
        assert isinstance(device, str)
        # STGFormer-specific: never returns 'mps'
        assert device in ["cpu", "cuda"]
