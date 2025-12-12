"""Tests for utils.gwnet module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from utils.dataset import TrafficDataset
from utils.gwnet import (
    get_gwnet_predictions,
    load_model,
    train_model,
)
from utils.hub import get_best_device


class TestTrainModel:
    """Tests for train_model function."""

    @pytest.fixture
    def mock_datasets(self):
        """Create minimal mock datasets."""
        np.random.seed(42)
        datasets = {}
        for split in ["train", "val", "test"]:
            x = np.random.randn(20, 12, 10, 2).astype(np.float32)
            y = np.random.randn(20, 12, 10, 1).astype(np.float32)
            datasets[split] = TrafficDataset(
                x, y, seq_len=12, horizon=12, num_nodes=10, input_dim=2, output_dim=1
            )
        return datasets

    def test_train_model_minimal(self, mock_datasets, tmp_path):
        """Test that train_model runs with minimal epochs."""
        adjacency = np.random.rand(10, 10).astype(np.float32)
        adjacency = (adjacency + adjacency.T) / 2  # symmetric

        # Train for just 1 epoch (device auto-detected)
        model_result = train_model(
            dataset_name="METR-LA",
            pytorch_datasets=mock_datasets,
            adjacency=adjacency,
            epochs=1,
            batch_size=8,
            save_dir=tmp_path,
            verbose=False,
        )

        # Check that model result dict was returned
        assert model_result is not None
        assert isinstance(model_result, dict)
        assert "model" in model_result
        assert "scaler" in model_result
        assert "config" in model_result

        # Check that files were saved (safetensors format)
        assert (tmp_path / "model.safetensors").exists()
        assert (tmp_path / "config.json").exists()
        assert (tmp_path / "metadata.json").exists()


class TestGetGwnetPredictions:
    """Tests for get_gwnet_predictions function."""

    def test_get_predictions_shape(self):
        """Test that get_gwnet_predictions returns correct shape."""
        # Create mock model result
        mock_model_result = {
            "model": MagicMock(),
            "scaler": MagicMock(),
            "config": {"num_nodes": 10, "seq_length": 12, "horizon": 12},
        }

        # Mock the gwnet_predict function
        with patch("utils.gwnet.gwnet_predict") as mock_predict:
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

            predictions = get_gwnet_predictions(mock_model_result, dataset)

            # Check shape
            assert predictions.shape == (20, 12, 10, 1)

            # Check that gwnet_predict was called
            mock_predict.assert_called_once()


class TestLoadModel:
    """Tests for load_model function."""

    def test_load_model_file_not_found(self, tmp_path):
        """Test that load_model raises error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_model("METR-LA", tmp_path, device="cpu")


class TestGetBestDevice:
    """Tests for get_best_device function."""

    def test_get_best_device_returns_string(self):
        """Test that get_best_device returns a valid device string."""
        device = get_best_device()
        assert isinstance(device, str)
        assert device in ["cpu", "cuda", "mps"]
