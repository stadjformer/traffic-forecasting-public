"""Tests for utils.mtgnn module."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from utils.dataset import TrafficDataset
from utils.hub import get_best_device
from utils.mtgnn import (
    get_mtgnn_predictions,
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
        model = train_model(
            dataset_name="METR-LA",
            pytorch_datasets=mock_datasets,
            adjacency=adjacency,
            epochs=1,
            batch_size=8,
            save_dir=tmp_path,
            verbose=False,
        )

        # Check that model was returned
        assert model is not None
        assert hasattr(model, "metrics")
        assert "vmae" in model.metrics

        # Check that files were saved
        assert (tmp_path / "model.pth").exists()
        assert (tmp_path / "config.json").exists()


class TestGetMtgnnPredictions:
    """Tests for get_mtgnn_predictions function."""

    def test_get_predictions_shape(self):
        """Test that get_mtgnn_predictions returns correct shape."""
        # Create mock model with predict method
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.randn(20, 12, 10, 1).astype(
            np.float32
        )

        # Create mock dataset
        np.random.seed(42)
        x = np.random.randn(20, 12, 10, 2).astype(np.float32)
        y = np.random.randn(20, 12, 10, 1).astype(np.float32)
        dataset = TrafficDataset(
            x, y, seq_len=12, horizon=12, num_nodes=10, input_dim=2, output_dim=1
        )

        predictions = get_mtgnn_predictions(mock_model, dataset)

        # Check shape
        assert predictions.shape == (20, 12, 10, 1)

        # Check that model.predict was called with correctly transposed data
        mock_model.predict.assert_called_once()
        call_args = mock_model.predict.call_args[0][0]
        assert call_args.shape == (20, 2, 10, 12)  # (batch, features, nodes, seq_len)


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
        assert device in ["cpu", "cuda", "mps"]
