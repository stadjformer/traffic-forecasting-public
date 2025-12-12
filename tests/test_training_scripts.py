"""Tests for training scripts to ensure they produce correct output shapes."""

import numpy as np
import pytest

import utils.gwnet
import utils.mtgnn
from utils.dataset import TrafficDataset


@pytest.fixture
def mock_datasets():
    """Create mock datasets for testing."""
    np.random.seed(42)
    num_samples = 20
    seq_len = 12
    horizon = 12
    num_nodes = 10
    input_dim = 2
    output_dim = 2  # speed + time-of-day

    datasets = {}
    for split in ["train", "val", "test"]:
        x = np.random.randn(num_samples, seq_len, num_nodes, input_dim).astype(
            np.float32
        )
        y = np.random.randn(num_samples, horizon, num_nodes, output_dim).astype(
            np.float32
        )

        datasets[split] = TrafficDataset(
            x=x,
            y=y,
            seq_len=seq_len,
            horizon=horizon,
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=output_dim,
        )

    return datasets


class TestMTGNNPredictionShape:
    """Test that MTGNN predictions match expected shape for metrics calculation."""

    def test_mtgnn_prediction_shape_matches_ground_truth(self, mock_datasets, tmp_path):
        """Test that MTGNN predictions have shape compatible with ground truth."""
        adjacency = np.random.rand(10, 10).astype(np.float32)
        adjacency = (adjacency + adjacency.T) / 2  # symmetric

        # Train for just 1 epoch
        model = utils.mtgnn.train_model(
            dataset_name="METR-LA",
            pytorch_datasets=mock_datasets,
            adjacency=adjacency,
            epochs=1,
            batch_size=8,
            save_dir=tmp_path,
            verbose=False,
        )

        # Get predictions
        test_predictions = utils.mtgnn.get_mtgnn_predictions(
            model, mock_datasets["test"]
        )

        # Get ground truth (only speed feature, not time-of-day)
        test_true = mock_datasets["test"].y.numpy()[..., 0:1]

        # Verify shapes match
        assert test_predictions.shape == test_true.shape, (
            f"Prediction shape {test_predictions.shape} != ground truth shape {test_true.shape}"
        )

        # Verify shape is (samples, horizon, nodes, 1)
        assert test_predictions.shape == (20, 12, 10, 1), (
            f"Expected shape (20, 12, 10, 1), got {test_predictions.shape}"
        )


class TestGraphWaveNetPredictionShape:
    """Test that Graph-WaveNet predictions match expected shape for metrics calculation."""

    def test_gwnet_prediction_shape_matches_ground_truth(self, mock_datasets, tmp_path):
        """Test that Graph-WaveNet predictions have shape compatible with ground truth."""
        adjacency = np.random.rand(10, 10).astype(np.float32)
        adjacency = (adjacency + adjacency.T) / 2  # symmetric

        # Train for just 1 epoch
        model_result = utils.gwnet.train_model(
            dataset_name="METR-LA",
            pytorch_datasets=mock_datasets,
            adjacency=adjacency,
            epochs=1,
            batch_size=8,
            save_dir=tmp_path,
            verbose=False,
        )

        # Get predictions
        test_predictions = utils.gwnet.get_gwnet_predictions(
            model_result, mock_datasets["test"]
        )

        # Get ground truth (only speed feature, not time-of-day)
        test_true = mock_datasets["test"].y.numpy()[..., 0:1]

        # Verify shapes match
        assert test_predictions.shape == test_true.shape, (
            f"Prediction shape {test_predictions.shape} != ground truth shape {test_true.shape}"
        )

        # Verify shape is (samples, horizon, nodes, 1)
        assert test_predictions.shape == (20, 12, 10, 1), (
            f"Expected shape (20, 12, 10, 1), got {test_predictions.shape}"
        )
