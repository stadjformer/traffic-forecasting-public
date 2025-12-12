"""Tests for utils.dcrnn module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from utils.dataset import TrafficDataset
from utils.dcrnn import (
    extract_predictions_from_supervisor,
    get_dcrnn_predictions,
    prepare_dcrnn_data,
)


class TestExtractPredictionsFromSupervisor:
    """Tests for extract_predictions_from_supervisor function."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset."""
        np.random.seed(42)
        x = np.random.randn(100, 12, 10, 2).astype(np.float32)
        y = np.random.randn(100, 12, 10, 1).astype(np.float32)
        test_ds = TrafficDataset(
            x, y, seq_len=12, horizon=12, num_nodes=10, input_dim=2, output_dim=1
        )
        return {"test": test_ds}

    def test_3d_output_conversion(self, sample_dataset):
        """Test conversion of 3D supervisor output to 4D."""
        # Simulate DCRNN supervisor output: list of (horizon, batch, nodes)
        horizon = 12
        batch = 100
        nodes = 10

        # Create list of predictions (one per horizon step)
        y_pred_list = [
            np.random.randn(batch, nodes).astype(np.float32) for _ in range(horizon)
        ]
        y_true_list = [
            np.random.randn(batch, nodes).astype(np.float32) for _ in range(horizon)
        ]

        vals = {"prediction": y_pred_list, "truth": y_true_list}

        y_pred, y_true = extract_predictions_from_supervisor(
            vals, sample_dataset, split="test"
        )

        # Should be 4D: (batch, horizon, nodes, 1)
        assert y_pred.shape == (100, 12, 10, 1)
        assert y_true.shape == (100, 12, 10, 1)
        assert y_pred.ndim == 4
        assert y_true.ndim == 4

    def test_4d_output_passthrough(self, sample_dataset):
        """Test that 4D supervisor output is handled correctly."""
        # Some models might already output 4D
        horizon = 12
        batch = 100
        nodes = 10
        features = 1

        y_pred_list = [
            np.random.randn(batch, nodes, features).astype(np.float32)
            for _ in range(horizon)
        ]
        y_true_list = [
            np.random.randn(batch, nodes, features).astype(np.float32)
            for _ in range(horizon)
        ]

        vals = {"prediction": y_pred_list, "truth": y_true_list}

        y_pred, y_true = extract_predictions_from_supervisor(
            vals, sample_dataset, split="test"
        )

        # Should still be 4D: (batch, horizon, nodes, features)
        assert y_pred.shape == (100, 12, 10, 1)
        assert y_true.shape == (100, 12, 10, 1)

    def test_padding_removal(self):
        """Test that DataLoader padding is correctly removed."""
        # Create dataset with size that doesn't divide evenly by batch size
        # 6850 samples (like real METR-LA test set)
        dataset_size = 6850
        horizon = 12
        nodes = 207  # METR-LA
        features = 1

        x = np.random.randn(dataset_size, 12, nodes, 2).astype(np.float32)
        y = np.random.randn(dataset_size, 12, nodes, features).astype(np.float32)
        test_ds = TrafficDataset(
            x,
            y,
            seq_len=12,
            horizon=12,
            num_nodes=nodes,
            input_dim=2,
            output_dim=features,
        )
        dataset = {"test": test_ds}

        # Simulate padded output (batch_size=64: 6850 % 64 = 2, needs 62 padding)
        # Total batches: ceil(6850/64) = 108, total samples: 108 * 64 = 6912
        padded_size = 6912

        # Create predictions with padding (last 62 samples are duplicates)
        y_pred_list = [
            np.random.randn(padded_size, nodes).astype(np.float32)
            for _ in range(horizon)
        ]
        y_true_list = [
            np.random.randn(padded_size, nodes).astype(np.float32)
            for _ in range(horizon)
        ]

        # Simulate padding: repeat last sample
        for t in range(horizon):
            y_pred_list[t][dataset_size:] = y_pred_list[t][dataset_size - 1]
            y_true_list[t][dataset_size:] = y_true_list[t][dataset_size - 1]

        vals = {"prediction": y_pred_list, "truth": y_true_list}

        y_pred, y_true = extract_predictions_from_supervisor(
            vals, dataset, split="test"
        )

        # Should trim to actual dataset size
        assert y_pred.shape[0] == dataset_size
        assert y_true.shape[0] == dataset_size
        assert y_pred.shape == (6850, 12, 207, 1)

    def test_no_padding_when_exact_batch_multiple(self):
        """Test that no trimming occurs when dataset size is exact multiple of batch size."""
        # Create dataset that divides evenly
        dataset_size = 6400  # 6400 % 64 = 0
        horizon = 12
        nodes = 207
        features = 1

        x = np.random.randn(dataset_size, 12, nodes, 2).astype(np.float32)
        y = np.random.randn(dataset_size, 12, nodes, features).astype(np.float32)
        test_ds = TrafficDataset(
            x,
            y,
            seq_len=12,
            horizon=12,
            num_nodes=nodes,
            input_dim=2,
            output_dim=features,
        )
        dataset = {"test": test_ds}

        # No padding needed
        y_pred_list = [
            np.random.randn(dataset_size, nodes).astype(np.float32)
            for _ in range(horizon)
        ]
        y_true_list = [
            np.random.randn(dataset_size, nodes).astype(np.float32)
            for _ in range(horizon)
        ]

        vals = {"prediction": y_pred_list, "truth": y_true_list}

        y_pred, y_true = extract_predictions_from_supervisor(
            vals, dataset, split="test"
        )

        # Should have exact size
        assert y_pred.shape[0] == dataset_size
        assert y_true.shape[0] == dataset_size


class TestGetDCRNNPredictions:
    """Tests for get_dcrnn_predictions function."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset."""
        np.random.seed(42)
        x = np.random.randn(100, 12, 10, 2).astype(np.float32)
        y = np.random.randn(100, 12, 10, 1).astype(np.float32)
        test_ds = TrafficDataset(
            x, y, seq_len=12, horizon=12, num_nodes=10, input_dim=2, output_dim=1
        )
        return {"test": test_ds}

    def test_ground_truth_validation_passes(self, sample_dataset):
        """Test that ground truth validation passes when data matches."""
        y_true_from_dataset = sample_dataset["test"].y.numpy()[..., 0:1]

        # Create predictions and matching ground truth
        y_pred = np.random.randn(100, 12, 10, 1).astype(np.float32)

        # Mock supervisor to return matching ground truth
        mock_supervisor = MagicMock()
        mock_supervisor.evaluate.return_value = (
            0.5,  # mean_loss
            {
                "prediction": [y_pred[:, t, :, 0] for t in range(12)],
                "truth": [y_true_from_dataset[:, t, :, 0] for t in range(12)],
            },
        )

        with patch("utils.dcrnn.get_trained_model_dcrnn", return_value=mock_supervisor):
            # Should not raise
            result = get_dcrnn_predictions(
                "METR-LA", sample_dataset, validate_ground_truth=True
            )

            assert result.shape == (100, 12, 10, 1)

    def test_ground_truth_validation_fails(self, sample_dataset):
        """Test that ground truth validation fails when data doesn't match."""
        # Create mismatched ground truth
        y_pred = np.random.randn(100, 12, 10, 1).astype(np.float32)
        y_true_wrong = np.random.randn(100, 12, 10, 1).astype(
            np.float32
        )  # Different data

        mock_supervisor = MagicMock()
        mock_supervisor.evaluate.return_value = (
            0.5,
            {
                "prediction": [y_pred[:, t, :, 0] for t in range(12)],
                "truth": [y_true_wrong[:, t, :, 0] for t in range(12)],
            },
        )

        with patch("utils.dcrnn.get_trained_model_dcrnn", return_value=mock_supervisor):
            # Should raise assertion error
            with pytest.raises(
                AssertionError, match="Ground truth from model doesn't match"
            ):
                get_dcrnn_predictions(
                    "METR-LA", sample_dataset, validate_ground_truth=True
                )

    def test_can_skip_validation(self, sample_dataset):
        """Test that validation can be skipped."""
        y_pred = np.random.randn(100, 12, 10, 1).astype(np.float32)
        y_true_wrong = np.random.randn(100, 12, 10, 1).astype(np.float32)

        mock_supervisor = MagicMock()
        mock_supervisor.evaluate.return_value = (
            0.5,
            {
                "prediction": [y_pred[:, t, :, 0] for t in range(12)],
                "truth": [y_true_wrong[:, t, :, 0] for t in range(12)],
            },
        )

        with patch("utils.dcrnn.get_trained_model_dcrnn", return_value=mock_supervisor):
            # Should not raise even though ground truth doesn't match
            result = get_dcrnn_predictions(
                "METR-LA", sample_dataset, validate_ground_truth=False
            )

            assert result.shape == (100, 12, 10, 1)

    def test_returns_predictions_only(self, sample_dataset):
        """Test that function returns only predictions, not ground truth."""
        y_pred = np.random.randn(100, 12, 10, 1).astype(np.float32)
        y_true = sample_dataset["test"].y.numpy()[..., 0:1]

        mock_supervisor = MagicMock()
        mock_supervisor.evaluate.return_value = (
            0.5,
            {
                "prediction": [y_pred[:, t, :, 0] for t in range(12)],
                "truth": [y_true[:, t, :, 0] for t in range(12)],
            },
        )

        with patch("utils.dcrnn.get_trained_model_dcrnn", return_value=mock_supervisor):
            result = get_dcrnn_predictions("METR-LA", sample_dataset)

            # Should return only predictions, not a tuple
            assert isinstance(result, np.ndarray)
            assert result.shape == y_pred.shape
            assert np.allclose(result, y_pred)


class TestPrepareDCRNNData:
    """Tests for prepare_dcrnn_data conversion function."""

    @pytest.fixture
    def pytorch_datasets(self):
        """Create sample PyTorch datasets."""
        np.random.seed(42)
        datasets = {}
        for split in ["train", "val", "test"]:
            x = np.random.randn(50, 12, 10, 2).astype(np.float32)
            y = np.random.randn(50, 12, 10, 1).astype(np.float32)
            datasets[split] = TrafficDataset(
                x, y, seq_len=12, horizon=12, num_nodes=10, input_dim=2, output_dim=1
            )
        return datasets

    def test_prepare_dcrnn_data_structure(self, pytorch_datasets):
        """Test that prepare_dcrnn_data returns correct structure."""
        from dcrnn_pytorch.lib.utils import DataLoader as DCRNNDataLoader
        from dcrnn_pytorch.lib.utils import StandardScaler

        dcrnn_data = prepare_dcrnn_data(pytorch_datasets, batch_size=16)

        # Check all required keys are present
        assert "train_loader" in dcrnn_data
        assert "val_loader" in dcrnn_data
        assert "test_loader" in dcrnn_data
        assert "scaler" in dcrnn_data

        # Check types
        assert isinstance(dcrnn_data["train_loader"], DCRNNDataLoader)
        assert isinstance(dcrnn_data["val_loader"], DCRNNDataLoader)
        assert isinstance(dcrnn_data["test_loader"], DCRNNDataLoader)
        assert isinstance(dcrnn_data["scaler"], StandardScaler)

    def test_prepare_dcrnn_data_scaler_properties(self, pytorch_datasets):
        """Test that scaler is correctly fitted on training data."""
        dcrnn_data = prepare_dcrnn_data(pytorch_datasets, batch_size=16)
        scaler = dcrnn_data["scaler"]

        # Scaler should have mean and std attributes
        assert hasattr(scaler, "mean")
        assert hasattr(scaler, "std")
        assert scaler.mean is not None
        assert scaler.std is not None
