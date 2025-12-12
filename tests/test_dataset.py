"""Tests for utils.dataset module."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from utils.dataset import TrafficDataset, hf_to_pytorch


class TestTrafficDataset:
    """Tests for TrafficDataset class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample traffic data."""
        np.random.seed(42)
        batch_size = 100
        seq_len = 12
        horizon = 12
        num_nodes = 10
        input_dim = 2
        output_dim = 1

        x = np.random.randn(batch_size, seq_len, num_nodes, input_dim).astype(
            np.float32
        )
        y = np.random.randn(batch_size, horizon, num_nodes, output_dim).astype(
            np.float32
        )

        return {
            "x": x,
            "y": y,
            "seq_len": seq_len,
            "horizon": horizon,
            "num_nodes": num_nodes,
            "input_dim": input_dim,
            "output_dim": output_dim,
        }

    def test_init_with_numpy(self, sample_data):
        """Test initialization with numpy arrays."""
        dataset = TrafficDataset(
            sample_data["x"],
            sample_data["y"],
            seq_len=sample_data["seq_len"],
            horizon=sample_data["horizon"],
            num_nodes=sample_data["num_nodes"],
            input_dim=sample_data["input_dim"],
            output_dim=sample_data["output_dim"],
        )

        assert len(dataset) == 100
        assert dataset.seq_len == 12
        assert dataset.horizon == 12
        assert dataset.num_nodes == 10
        assert dataset.input_dim == 2
        assert dataset.output_dim == 1
        assert isinstance(dataset.x, torch.Tensor)
        assert isinstance(dataset.y, torch.Tensor)

    def test_init_with_tensors(self, sample_data):
        """Test initialization with torch tensors."""
        x_tensor = torch.from_numpy(sample_data["x"])
        y_tensor = torch.from_numpy(sample_data["y"])

        dataset = TrafficDataset(
            x_tensor,
            y_tensor,
            seq_len=sample_data["seq_len"],
            horizon=sample_data["horizon"],
            num_nodes=sample_data["num_nodes"],
            input_dim=sample_data["input_dim"],
            output_dim=sample_data["output_dim"],
        )

        assert len(dataset) == 100
        assert isinstance(dataset.x, torch.Tensor)
        assert isinstance(dataset.y, torch.Tensor)

    def test_getitem(self, sample_data):
        """Test __getitem__ returns correct shapes."""
        dataset = TrafficDataset(
            sample_data["x"],
            sample_data["y"],
            seq_len=sample_data["seq_len"],
            horizon=sample_data["horizon"],
            num_nodes=sample_data["num_nodes"],
            input_dim=sample_data["input_dim"],
            output_dim=sample_data["output_dim"],
        )

        x, y = dataset[0]
        assert x.shape == (12, 10, 2)  # (seq_len, num_nodes, input_dim)
        assert y.shape == (12, 10, 1)  # (horizon, num_nodes, output_dim)

    def test_dataloader_integration(self, sample_data):
        """Test integration with PyTorch DataLoader."""
        dataset = TrafficDataset(
            sample_data["x"],
            sample_data["y"],
            seq_len=sample_data["seq_len"],
            horizon=sample_data["horizon"],
            num_nodes=sample_data["num_nodes"],
            input_dim=sample_data["input_dim"],
            output_dim=sample_data["output_dim"],
        )

        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        x_batch, y_batch = next(iter(loader))

        assert x_batch.shape == (16, 12, 10, 2)  # (batch, seq_len, nodes, input_dim)
        assert y_batch.shape == (16, 12, 10, 1)  # (batch, horizon, nodes, output_dim)

    def test_shape_validation_seq_len_mismatch(self, sample_data):
        """Test that shape validation catches seq_len mismatch."""
        with pytest.raises(AssertionError, match="x seq_len mismatch"):
            TrafficDataset(
                sample_data["x"],
                sample_data["y"],
                seq_len=20,  # Wrong!
                horizon=sample_data["horizon"],
                num_nodes=sample_data["num_nodes"],
                input_dim=sample_data["input_dim"],
                output_dim=sample_data["output_dim"],
            )

    def test_shape_validation_horizon_mismatch(self, sample_data):
        """Test that shape validation catches horizon mismatch."""
        with pytest.raises(AssertionError, match="y horizon mismatch"):
            TrafficDataset(
                sample_data["x"],
                sample_data["y"],
                seq_len=sample_data["seq_len"],
                horizon=20,  # Wrong!
                num_nodes=sample_data["num_nodes"],
                input_dim=sample_data["input_dim"],
                output_dim=sample_data["output_dim"],
            )

    def test_shape_validation_num_nodes_mismatch(self, sample_data):
        """Test that shape validation catches num_nodes mismatch."""
        with pytest.raises(AssertionError, match="x num_nodes mismatch"):
            TrafficDataset(
                sample_data["x"],
                sample_data["y"],
                seq_len=sample_data["seq_len"],
                horizon=sample_data["horizon"],
                num_nodes=50,  # Wrong!
                input_dim=sample_data["input_dim"],
                output_dim=sample_data["output_dim"],
            )

    def test_shape_validation_batch_size_mismatch(self, sample_data):
        """Test that shape validation catches batch size mismatch."""
        # Create y with different batch size
        y_wrong = sample_data["y"][:50]  # Only 50 samples instead of 100

        with pytest.raises(AssertionError, match="x and y must have same batch size"):
            TrafficDataset(
                sample_data["x"],
                y_wrong,
                seq_len=sample_data["seq_len"],
                horizon=sample_data["horizon"],
                num_nodes=sample_data["num_nodes"],
                input_dim=sample_data["input_dim"],
                output_dim=sample_data["output_dim"],
            )

    def test_repr(self, sample_data):
        """Test __repr__ returns useful string."""
        dataset = TrafficDataset(
            sample_data["x"],
            sample_data["y"],
            seq_len=sample_data["seq_len"],
            horizon=sample_data["horizon"],
            num_nodes=sample_data["num_nodes"],
            input_dim=sample_data["input_dim"],
            output_dim=sample_data["output_dim"],
        )

        repr_str = repr(dataset)
        assert "TrafficDataset" in repr_str
        assert "samples=100" in repr_str
        assert "seq_len=12" in repr_str
        assert "nodes=10" in repr_str

    def test_adjacency_matrix_none(self, sample_data):
        """Test that adjacency matrix is optional and defaults to None."""
        dataset = TrafficDataset(
            sample_data["x"],
            sample_data["y"],
            seq_len=sample_data["seq_len"],
            horizon=sample_data["horizon"],
            num_nodes=sample_data["num_nodes"],
            input_dim=sample_data["input_dim"],
            output_dim=sample_data["output_dim"],
        )
        assert dataset.adj_mx is None

    def test_adjacency_matrix_with_numpy(self, sample_data):
        """Test initialization with numpy adjacency matrix."""
        adj_mx = np.random.rand(10, 10).astype(np.float32)

        dataset = TrafficDataset(
            sample_data["x"],
            sample_data["y"],
            seq_len=sample_data["seq_len"],
            horizon=sample_data["horizon"],
            num_nodes=sample_data["num_nodes"],
            input_dim=sample_data["input_dim"],
            output_dim=sample_data["output_dim"],
            adj_mx=adj_mx,
        )

        assert dataset.adj_mx is not None
        assert isinstance(dataset.adj_mx, torch.Tensor)
        assert dataset.adj_mx.shape == (10, 10)

    def test_adjacency_matrix_with_tensor(self, sample_data):
        """Test initialization with torch tensor adjacency matrix."""
        adj_mx = torch.rand(10, 10)

        dataset = TrafficDataset(
            sample_data["x"],
            sample_data["y"],
            seq_len=sample_data["seq_len"],
            horizon=sample_data["horizon"],
            num_nodes=sample_data["num_nodes"],
            input_dim=sample_data["input_dim"],
            output_dim=sample_data["output_dim"],
            adj_mx=adj_mx,
        )

        assert dataset.adj_mx is not None
        assert isinstance(dataset.adj_mx, torch.Tensor)
        assert dataset.adj_mx.shape == (10, 10)

    def test_adjacency_matrix_shape_validation(self, sample_data):
        """Test that adjacency matrix shape is validated."""
        adj_mx = np.random.rand(5, 5).astype(np.float32)  # Wrong size!

        with pytest.raises(AssertionError, match="adj_mx shape mismatch"):
            TrafficDataset(
                sample_data["x"],
                sample_data["y"],
                seq_len=sample_data["seq_len"],
                horizon=sample_data["horizon"],
                num_nodes=sample_data["num_nodes"],
                input_dim=sample_data["input_dim"],
                output_dim=sample_data["output_dim"],
                adj_mx=adj_mx,
            )

    def test_serialise_deserialise_roundtrip(self, sample_data, tmp_path):
        """Test that serialise/deserialise preserves all data."""
        # Create datasets with and without adjacency matrix
        adj_mx = np.random.rand(10, 10).astype(np.float32)

        train_ds = TrafficDataset(
            sample_data["x"],
            sample_data["y"],
            seq_len=sample_data["seq_len"],
            horizon=sample_data["horizon"],
            num_nodes=sample_data["num_nodes"],
            input_dim=sample_data["input_dim"],
            output_dim=sample_data["output_dim"],
            adj_mx=adj_mx,
        )

        val_ds = TrafficDataset(
            sample_data["x"][:50],  # Smaller validation set
            sample_data["y"][:50],
            seq_len=sample_data["seq_len"],
            horizon=sample_data["horizon"],
            num_nodes=sample_data["num_nodes"],
            input_dim=sample_data["input_dim"],
            output_dim=sample_data["output_dim"],
            adj_mx=adj_mx,
        )

        datasets_dict = {"train": train_ds, "val": val_ds}

        # Serialise to temp file
        cache_file = tmp_path / "test_datasets.pt"
        TrafficDataset.serialise(datasets_dict, cache_file)

        # Verify file was created
        assert cache_file.exists()

        # Deserialise
        loaded_datasets = TrafficDataset.deserialise(cache_file)

        # Check structure
        assert "train" in loaded_datasets
        assert "val" in loaded_datasets
        assert len(loaded_datasets) == 2

        # Check train dataset metadata
        train_loaded = loaded_datasets["train"]
        assert train_loaded.seq_len == train_ds.seq_len
        assert train_loaded.horizon == train_ds.horizon
        assert train_loaded.num_nodes == train_ds.num_nodes
        assert train_loaded.input_dim == train_ds.input_dim
        assert train_loaded.output_dim == train_ds.output_dim
        assert len(train_loaded) == len(train_ds)

        # Check val dataset metadata
        val_loaded = loaded_datasets["val"]
        assert val_loaded.seq_len == val_ds.seq_len
        assert len(val_loaded) == 50

        # Check tensor data is preserved
        assert torch.allclose(train_loaded.x, train_ds.x)
        assert torch.allclose(train_loaded.y, train_ds.y)
        assert torch.allclose(val_loaded.x, val_ds.x)
        assert torch.allclose(val_loaded.y, val_ds.y)

        # Check adjacency matrix is preserved
        assert train_loaded.adj_mx is not None
        assert torch.allclose(train_loaded.adj_mx, train_ds.adj_mx)
        assert torch.allclose(val_loaded.adj_mx, val_ds.adj_mx)

        # Check that we can still use the dataset
        x, y = train_loaded[0]
        assert x.shape == (12, 10, 2)
        assert y.shape == (12, 10, 1)


class TestHFToPyTorch:
    """Tests for hf_to_pytorch conversion function."""

    @pytest.fixture
    def mock_hf_dataset(self):
        """Create a mock HuggingFace dataset structure."""
        import pandas as pd
        from datasets import Dataset

        # Create sample data for 3 nodes, 5 samples each
        # 2 input timesteps (t0, t1), 2 output timesteps (t0, t1)
        # 2 features per timestep (d0, d1)
        base_time = np.datetime64("2024-01-01T00:00")
        data = []
        for node_id in range(3):
            for sample_idx in range(5):
                timestamp = base_time + np.timedelta64(sample_idx * 5, "m")
                row = {
                    "node_id": node_id,
                    "t0_timestamp": str(timestamp.astype("datetime64[m]")),
                    # Input features
                    "x_t-1_d0": np.random.randn(),
                    "x_t-1_d1": -999.0,
                    "x_t+0_d0": np.random.randn(),
                    "x_t+0_d1": -999.0,
                    # Output features
                    "y_t+1_d0": np.random.randn(),
                    "y_t+1_d1": -999.0,
                    "y_t+2_d0": np.random.randn(),
                    "y_t+2_d1": -999.0,
                }
                data.append(row)

        df = pd.DataFrame(data)

        # Create datasets for each split
        from datasets import DatasetDict

        return DatasetDict(
            {
                "train": Dataset.from_pandas(df, preserve_index=False),
                "val": Dataset.from_pandas(df, preserve_index=False),
                "test": Dataset.from_pandas(df, preserve_index=False),
            }
        )

    def test_conversion_shape(self, mock_hf_dataset):
        """Test that conversion produces correct shapes."""
        pytorch_datasets = hf_to_pytorch(mock_hf_dataset)

        # Check all splits are present
        assert "train" in pytorch_datasets
        assert "val" in pytorch_datasets
        assert "test" in pytorch_datasets

        # Check train dataset
        train_ds = pytorch_datasets["train"]
        assert isinstance(train_ds, TrafficDataset)
        assert len(train_ds) == 5  # 5 samples
        assert train_ds.num_nodes == 3
        assert train_ds.seq_len == 2  # t-1, t+0
        assert train_ds.horizon == 2  # t+1, t+2
        assert train_ds.input_dim == 2  # d0, d1
        assert train_ds.output_dim == 2  # d0, d1 for output

    def test_metadata_consistency(self, mock_hf_dataset):
        """Test that metadata is consistent across splits."""
        pytorch_datasets = hf_to_pytorch(mock_hf_dataset)

        train_ds = pytorch_datasets["train"]
        val_ds = pytorch_datasets["val"]
        test_ds = pytorch_datasets["test"]

        # All splits should have same metadata
        assert train_ds.num_nodes == val_ds.num_nodes == test_ds.num_nodes
        assert train_ds.seq_len == val_ds.seq_len == test_ds.seq_len
        assert train_ds.horizon == val_ds.horizon == test_ds.horizon
        assert train_ds.input_dim == val_ds.input_dim == test_ds.input_dim
        assert train_ds.output_dim == val_ds.output_dim == test_ds.output_dim

    def test_tensor_shapes(self, mock_hf_dataset):
        """Test that individual tensors have correct shapes."""
        pytorch_datasets = hf_to_pytorch(mock_hf_dataset)
        train_ds = pytorch_datasets["train"]

        x, y = train_ds[0]
        assert x.shape == (2, 3, 2)  # (seq_len, num_nodes, input_dim)
        assert y.shape == (2, 3, 2)  # (horizon, num_nodes, output_dim)

    def test_dataloader_batching(self, mock_hf_dataset):
        """Test that batching works correctly."""
        pytorch_datasets = hf_to_pytorch(mock_hf_dataset)
        train_ds = pytorch_datasets["train"]

        loader = DataLoader(train_ds, batch_size=2, shuffle=False)
        x_batch, y_batch = next(iter(loader))

        assert x_batch.shape == (2, 2, 3, 2)  # (batch, seq_len, nodes, input_dim)
        assert y_batch.shape == (2, 2, 3, 2)  # (batch, horizon, nodes, output_dim)

    def test_conversion_without_adjacency_matrix(self, mock_hf_dataset):
        """Test conversion without adjacency matrix."""
        pytorch_datasets = hf_to_pytorch(mock_hf_dataset)

        # All datasets should have None adjacency matrix
        for split_name, dataset in pytorch_datasets.items():
            assert dataset.adj_mx is None

    def test_conversion_with_adjacency_matrix(self, mock_hf_dataset):
        """Test conversion with adjacency matrix."""
        adj_mx = np.random.rand(3, 3).astype(np.float32)
        pytorch_datasets = hf_to_pytorch(mock_hf_dataset, adj_mx=adj_mx)

        # All datasets should have the same adjacency matrix
        for split_name, dataset in pytorch_datasets.items():
            assert dataset.adj_mx is not None
            assert isinstance(dataset.adj_mx, torch.Tensor)
            assert dataset.adj_mx.shape == (3, 3)
            # Check it's the same matrix
            assert torch.allclose(dataset.adj_mx, torch.from_numpy(adj_mx))

    def test_conversion_with_wrong_adjacency_matrix_shape(self, mock_hf_dataset):
        """Test conversion fails with wrong adjacency matrix shape."""
        adj_mx = np.random.rand(5, 5).astype(np.float32)  # Wrong size! Should be 3x3

        with pytest.raises(AssertionError, match="adj_mx shape mismatch"):
            hf_to_pytorch(mock_hf_dataset, adj_mx=adj_mx)

    def test_time_features_recomputed_from_timestamps(self, mock_hf_dataset):
        """Ensure d1 values are recomputed from timestamps rather than HF data."""
        pytorch_datasets = hf_to_pytorch(mock_hf_dataset)
        train_ds = pytorch_datasets["train"]

        x = train_ds.x.numpy()  # (samples, seq_len, nodes, features)
        y = train_ds.y.numpy()  # (samples, horizon, nodes, features)
        num_nodes = train_ds.num_nodes

        minutes_per_day = 24 * 60
        x_offsets = np.array([-1, 0], dtype=np.int64)
        y_offsets = np.array([1, 2], dtype=np.int64)

        for sample_idx in range(x.shape[0]):
            base_minutes = (sample_idx * 5) % minutes_per_day
            expected_x = (
                (base_minutes + x_offsets * 5) % minutes_per_day
            ) / minutes_per_day
            expected_y = (
                (base_minutes + y_offsets * 5) % minutes_per_day
            ) / minutes_per_day
            expected_x = np.tile(expected_x[:, None], (1, num_nodes))
            expected_y = np.tile(expected_y[:, None], (1, num_nodes))

            np.testing.assert_allclose(
                x[sample_idx, :, :, 1],  # d1 channel
                expected_x,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                y[sample_idx, :, :, 1],  # d1 channel
                expected_y,
                atol=1e-6,
            )

    def test_regression_small_synthetic_dataset(self):
        """Regression test with deterministic synthetic data."""
        import pandas as pd
        from datasets import Dataset, DatasetDict

        num_nodes = 2
        samples_per_node = 2

        rows = []
        for node_id in range(num_nodes):
            for sample_idx in range(samples_per_node):
                base = node_id * 100 + sample_idx * 10
                rows.append(
                    {
                        "node_id": node_id,
                        "x_t-1_d0": float(base + 1),
                        "x_t+0_d0": float(base + 2),
                        "y_t+1_d0": float(base + 3),
                        "y_t+2_d0": float(base + 4),
                    }
                )

        df = pd.DataFrame(rows)
        hf_dataset = DatasetDict(
            {"train": Dataset.from_pandas(df, preserve_index=False)}
        )

        pytorch_datasets = hf_to_pytorch(hf_dataset)
        train_ds = pytorch_datasets["train"]

        expected_x = np.zeros(
            (samples_per_node, 2, num_nodes, 1), dtype=np.float32
        )  # (samples, seq_len, nodes, features)
        expected_y = np.zeros(
            (samples_per_node, 2, num_nodes, 1), dtype=np.float32
        )  # (samples, horizon, nodes, features)

        for node_id in range(num_nodes):
            for sample_idx in range(samples_per_node):
                base = node_id * 100 + sample_idx * 10
                expected_x[sample_idx, 0, node_id, 0] = base + 1  # x_t-1
                expected_x[sample_idx, 1, node_id, 0] = base + 2  # x_t+0
                expected_y[sample_idx, 0, node_id, 0] = base + 3  # y_t+1
                expected_y[sample_idx, 1, node_id, 0] = base + 4  # y_t+2

        assert torch.allclose(train_ds.x, torch.from_numpy(expected_x))
        assert torch.allclose(train_ds.y, torch.from_numpy(expected_y))

    def test_temporal_ordering_with_negative_indices(self):
        """Test that columns with negative time indices are sorted correctly."""
        import pandas as pd
        from datasets import Dataset, DatasetDict

        # Create dataset with negative time indices (realistic format)
        data = []
        for node_id in range(2):
            for sample_idx in range(3):
                row = {
                    "node_id": node_id,
                    # Input: t-2, t-1, t+0 (past to present)
                    "x_t-2_d0": float(sample_idx * 10 + 1),
                    "x_t-1_d0": float(sample_idx * 10 + 2),
                    "x_t+0_d0": float(sample_idx * 10 + 3),
                    # Output: t+1, t+2 (future)
                    "y_t+1_d0": float(sample_idx * 10 + 4),
                    "y_t+2_d0": float(sample_idx * 10 + 5),
                }
                data.append(row)

        df = pd.DataFrame(data)
        hf_dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(df, preserve_index=False),
            }
        )

        pytorch_datasets = hf_to_pytorch(hf_dataset)
        train_ds = pytorch_datasets["train"]

        # Check dimensions
        assert train_ds.seq_len == 3  # t-2, t-1, t+0
        assert train_ds.horizon == 2  # t+1, t+2
        assert train_ds.input_dim == 1  # Only d0
        assert train_ds.output_dim == 1  # Only d0

        # Check temporal ordering: values should be in chronological order
        # For node 0, sample 0: x should be [1, 2, 3], y should be [4, 5]
        x, y = train_ds[0]

        # x shape: (seq_len=3, num_nodes=2, input_dim=1)
        # Check node 0's values are in chronological order
        assert x[0, 0, 0].item() == 1.0  # t-2
        assert x[1, 0, 0].item() == 2.0  # t-1
        assert x[2, 0, 0].item() == 3.0  # t+0

        # y shape: (horizon=2, num_nodes=2, output_dim=1)
        assert y[0, 0, 0].item() == 4.0  # t+1
        assert y[1, 0, 0].item() == 5.0  # t+2

    def test_validation_split_renamed_to_val(self):
        """Test that HuggingFace 'validation' split is renamed to 'val'."""
        import pandas as pd
        from datasets import Dataset, DatasetDict

        data = []
        for node_id in range(2):
            row = {
                "node_id": node_id,
                "x_t-1_d0": 1.0,
                "x_t+0_d0": 2.0,
                "y_t+1_d0": 3.0,
            }
            data.append(row)

        df = pd.DataFrame(data)
        hf_dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(df, preserve_index=False),
                "validation": Dataset.from_pandas(
                    df, preserve_index=False
                ),  # HF uses "validation"
                "test": Dataset.from_pandas(df, preserve_index=False),
            }
        )

        pytorch_datasets = hf_to_pytorch(hf_dataset)

        # Should have "val" not "validation"
        assert "val" in pytorch_datasets
        assert "validation" not in pytorch_datasets
        assert "train" in pytorch_datasets
        assert "test" in pytorch_datasets

    def test_to_continuous_with_stride_1(self):
        """Test conversion of windowed dataset to continuous time series with stride=1 (X only)."""
        # Create windowed data with stride=1
        # Windows overlap: window[i+1] starts where window[i] is at timestep 1
        batch_size = 100
        seq_len = 12
        horizon = 12
        num_nodes = 10
        input_dim = 2

        # Generate continuous data first (need enough for both X and Y)
        continuous_length = (
            batch_size + seq_len + horizon - 1
        )  # 111 + 12 - 1 = 122 timesteps
        continuous_data = (
            np.arange(continuous_length * num_nodes * input_dim)
            .reshape(continuous_length, num_nodes, input_dim)
            .astype(np.float32)
        )

        # Create overlapping windows with stride=1
        x_windows = []
        y_windows = []
        for i in range(batch_size):
            x_windows.append(continuous_data[i : i + seq_len])
            y_windows.append(
                continuous_data[i + seq_len : i + seq_len + horizon]
            )  # Y follows X

        x = np.stack(x_windows, axis=0)  # [100, 12, 10, 2]
        y = np.stack(y_windows, axis=0)  # [100, 12, 10, 2]

        # Create dataset
        dataset = TrafficDataset(
            x,
            y,
            seq_len=seq_len,
            horizon=horizon,
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=input_dim,
        )

        # Convert to continuous (X only, default behavior)
        reconstructed = dataset.to_continuous(include_targets=False)

        # Check shape (X only: batch_size + seq_len - 1)
        expected_length = batch_size + seq_len - 1  # 111
        assert reconstructed.shape == (expected_length, num_nodes, input_dim)
        assert reconstructed.shape[0] == batch_size + seq_len - 1

        # Check that reconstruction matches original continuous data (X part only)
        np.testing.assert_allclose(
            reconstructed, continuous_data[:expected_length], rtol=1e-5
        )

    def test_to_continuous_stride_validation_fails(self):
        """Test that to_continuous raises error when stride != 1."""
        # Create data with stride=2 (non-overlapping windows)
        batch_size = 50
        seq_len = 12
        num_nodes = 10
        input_dim = 2

        # Generate continuous data
        continuous_length = batch_size * 2 + seq_len
        continuous_data = (
            np.arange(continuous_length * num_nodes * input_dim)
            .reshape(continuous_length, num_nodes, input_dim)
            .astype(np.float32)
        )

        # Create windows with stride=2
        x_windows = []
        y_windows = []
        for i in range(0, batch_size * 2, 2):  # Step by 2!
            x_windows.append(continuous_data[i : i + seq_len])
            y_windows.append(continuous_data[i + 1 : i + seq_len + 1])

        x = np.stack(x_windows, axis=0)
        y = np.stack(y_windows, axis=0)

        dataset = TrafficDataset(
            x,
            y,
            seq_len=seq_len,
            horizon=seq_len,
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=input_dim,
        )

        # Should raise assertion error
        with pytest.raises(
            AssertionError, match="Stride == 1 assumption does not hold"
        ):
            dataset.to_continuous()

    def test_to_continuous_preserves_features(self):
        """Test that to_continuous preserves both features correctly."""
        batch_size = 50
        seq_len = 12
        horizon = 12
        num_nodes = 5
        input_dim = 2

        # Generate continuous data with distinct features (enough for X and Y)
        continuous_length_full = batch_size + seq_len + horizon - 1
        # Feature 0: values 0-N
        # Feature 1: values 1000-1000+N (to distinguish from feature 0)
        continuous_f0 = (
            np.arange(continuous_length_full * num_nodes)
            .reshape(continuous_length_full, num_nodes)
            .astype(np.float32)
        )
        continuous_f1 = (
            np.arange(1000, 1000 + continuous_length_full * num_nodes)
            .reshape(continuous_length_full, num_nodes)
            .astype(np.float32)
        )
        continuous_data = np.stack([continuous_f0, continuous_f1], axis=-1)

        # Create overlapping windows with stride=1
        x_windows = []
        y_windows = []
        for i in range(batch_size):
            x_windows.append(continuous_data[i : i + seq_len])
            y_windows.append(continuous_data[i + seq_len : i + seq_len + horizon])

        x = np.stack(x_windows, axis=0)
        y = np.stack(y_windows, axis=0)

        dataset = TrafficDataset(
            x,
            y,
            seq_len=seq_len,
            horizon=horizon,
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=input_dim,
        )

        # Convert to continuous (X only)
        reconstructed = dataset.to_continuous(include_targets=False)

        # Check both features are preserved (X part only)
        expected_length = batch_size + seq_len - 1
        assert reconstructed.shape == (expected_length, num_nodes, 2)
        np.testing.assert_allclose(
            reconstructed[:, :, 0], continuous_f0[:expected_length], rtol=1e-5
        )
        np.testing.assert_allclose(
            reconstructed[:, :, 1], continuous_f1[:expected_length], rtol=1e-5
        )

    def test_to_continuous_with_targets(self):
        """Test conversion including targets (X + Y from last window)."""
        batch_size = 100
        seq_len = 12
        horizon = 12
        num_nodes = 10
        input_dim = 2

        # Generate continuous data that includes both X and Y regions
        continuous_length = batch_size + seq_len + horizon - 1  # 111 + 12 - 1 = 122
        continuous_data = (
            np.arange(continuous_length * num_nodes * input_dim)
            .reshape(continuous_length, num_nodes, input_dim)
            .astype(np.float32)
        )

        # Create overlapping windows with stride=1
        x_windows = []
        y_windows = []
        for i in range(batch_size):
            x_windows.append(continuous_data[i : i + seq_len])
            y_windows.append(continuous_data[i + seq_len : i + seq_len + horizon])

        x = np.stack(x_windows, axis=0)  # [100, 12, 10, 2]
        y = np.stack(y_windows, axis=0)  # [100, 12, 10, 2]

        # Create dataset
        dataset = TrafficDataset(
            x,
            y,
            seq_len=seq_len,
            horizon=horizon,
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=input_dim,
        )

        # Convert to continuous including targets
        reconstructed = dataset.to_continuous(include_targets=True)

        # Check shape
        assert reconstructed.shape == (continuous_length, num_nodes, input_dim)
        assert reconstructed.shape[0] == batch_size + seq_len + horizon - 1

        # Check that reconstruction matches original continuous data
        np.testing.assert_allclose(reconstructed, continuous_data, rtol=1e-5)

    def test_to_continuous_y_validation_fails(self):
        """Test that to_continuous with targets raises error when Y doesn't follow X."""
        batch_size = 50
        seq_len = 12
        horizon = 12
        num_nodes = 10
        input_dim = 2

        # Create X with stride=1
        continuous_length = batch_size + seq_len - 1
        continuous_data = (
            np.arange(continuous_length * num_nodes * input_dim)
            .reshape(continuous_length, num_nodes, input_dim)
            .astype(np.float32)
        )

        x_windows = []
        for i in range(batch_size):
            x_windows.append(continuous_data[i : i + seq_len])
        x = np.stack(x_windows, axis=0)

        # Create Y that DOESN'T follow X (random data instead)
        y = np.random.randn(batch_size, horizon, num_nodes, input_dim).astype(
            np.float32
        )

        dataset = TrafficDataset(
            x,
            y,
            seq_len=seq_len,
            horizon=horizon,
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=input_dim,
        )

        # Should raise assertion error when trying to include targets
        with pytest.raises(AssertionError, match="Y doesn't follow X sequentially"):
            dataset.to_continuous(include_targets=True)
