"""PyTorch dataset classes for spatiotemporal traffic data."""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TrafficDataset(Dataset):
    """
    PyTorch Dataset for spatiotemporal traffic data.

    Stores data in shape: (num_samples, seq_len/horizon, num_nodes, num_features)
    Where:
    - seq_len/horizon: temporal dimension (time steps)
    - num_nodes: spatial dimension (number of sensors)
    - num_features: feature dimension (speed, time-of-day, etc.)

    Metadata is stored as attributes for easy access when configuring models.

    Args:
        x: Input sequences, shape (num_samples, seq_len, num_nodes, input_dim)
        y: Target sequences, shape (num_samples, horizon, num_nodes, output_dim)
        seq_len: Number of input time steps
        horizon: Number of output time steps
        num_nodes: Number of spatial nodes (sensors)
        input_dim: Number of input features per (node, timestep)
        output_dim: Number of output features per (node, timestep)
        adj_mx: Optional adjacency matrix, shape (num_nodes, num_nodes)
    """

    def __init__(
        self, x, y, seq_len, horizon, num_nodes, input_dim, output_dim, adj_mx=None
    ):
        # Convert to tensors if needed
        self.x = torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x
        self.y = torch.from_numpy(y).float() if isinstance(y, np.ndarray) else y

        # Validate shapes
        assert self.x.shape[0] == self.y.shape[0], "x and y must have same batch size"
        assert self.x.shape[1] == seq_len, (
            f"x seq_len mismatch: {self.x.shape[1]} != {seq_len}"
        )
        assert self.y.shape[1] == horizon, (
            f"y horizon mismatch: {self.y.shape[1]} != {horizon}"
        )
        assert self.x.shape[2] == num_nodes, (
            f"x num_nodes mismatch: {self.x.shape[2]} != {num_nodes}"
        )
        assert self.y.shape[2] == num_nodes, (
            f"y num_nodes mismatch: {self.y.shape[2]} != {num_nodes}"
        )
        assert self.x.shape[3] == input_dim, (
            f"x input_dim mismatch: {self.x.shape[3]} != {input_dim}"
        )
        assert self.y.shape[3] == output_dim, (
            f"y output_dim mismatch: {self.y.shape[3]} != {output_dim}"
        )

        # Store metadata as attributes
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Store adjacency matrix if provided
        if adj_mx is not None:
            adj_mx = (
                torch.from_numpy(adj_mx).float()
                if isinstance(adj_mx, np.ndarray)
                else adj_mx
            )
            assert adj_mx.shape == (num_nodes, num_nodes), (
                f"adj_mx shape mismatch: {adj_mx.shape} != ({num_nodes}, {num_nodes})"
            )
        self.adj_mx = adj_mx

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __repr__(self):
        return (
            f"TrafficDataset(samples={len(self)}, seq_len={self.seq_len}, "
            f"horizon={self.horizon}, nodes={self.num_nodes}, "
            f"input_dim={self.input_dim}, output_dim={self.output_dim})"
        )

    @staticmethod
    def serialise(datasets_dict: Dict[str, "TrafficDataset"], path: Path):
        """Save multiple TrafficDataset splits to a single file."""
        torch.serialization.add_safe_globals([TrafficDataset])
        torch.save({"datasets": datasets_dict}, path)

    @staticmethod
    def deserialise(path: Path) -> Dict[str, "TrafficDataset"]:
        """Load multiple TrafficDataset splits from a file."""
        torch.serialization.add_safe_globals([TrafficDataset])
        return torch.load(path)["datasets"]

    def to_continuous(self, include_targets: bool = False) -> np.ndarray:
        """
        Convert windowed dataset to continuous time series.

        Assumes stride=1 (each window shifts by 1 timestep) and validates this assumption.
        Reconstructs the original continuous time series from overlapping windows by:
        1. Taking all timesteps from the first window's X
        2. Taking only the last (new) timestep from each subsequent window's X
        3. Optionally appending all timesteps from the last window's Y (if include_targets=True)

        Args:
            include_targets: If True, append Y from last window to get full continuous series.
                            Use True for training (when you have labels).
                            Use False for inference (when you only have observations).

        Returns:
            np.ndarray: Continuous time series array
                Shape without targets: (num_samples + seq_len - 1, num_nodes, num_features)
                Shape with targets: (num_samples + seq_len + horizon - 1, num_nodes, num_features)

        Raises:
            AssertionError: If stride != 1 assumption is violated
        """
        # Validate stride=1 assumption for X
        # If stride=1, window[0][timestep=11] == window[1][timestep=10]
        # (last timestep of first window == second-to-last of second window)
        assert torch.allclose(
            self.x[0, -1, 0, 1],  # Window 0, last timestep, sensor 0, time feature
            self.x[
                1, -2, 0, 1
            ],  # Window 1, second-to-last timestep, sensor 0, time feature
        ), "Stride == 1 assumption does not hold for this dataset"

        # Reconstruct continuous series from X
        # Take first window fully, then last timestep of each subsequent window
        continuous = np.concatenate(
            [
                self.x[0, :, :, :].numpy(),  # [seq_len, num_nodes, num_features]
                self.x[
                    1:, -1, :, :
                ].numpy(),  # [num_samples-1, num_nodes, num_features]
            ],
            axis=0,
        )  # Result: [num_samples + seq_len - 1, num_nodes, num_features]

        # Validate X reconstruction
        expected_timesteps = self.x.shape[0] + self.x.shape[1] - 1
        assert continuous.shape[0] == expected_timesteps, (
            f"X reconstruction failed: got {continuous.shape[0]} timesteps, "
            f"expected {expected_timesteps}"
        )

        if include_targets:
            # Validate that Y follows X sequentially
            assert torch.allclose(
                self.y[
                    0, 0, 0, 1
                ],  # Window 0, first Y timestep, sensor 0, time feature
                self.x[
                    1, -1, 0, 1
                ],  # Window 1, last X timestep, sensor 0, time feature
            ), "Y doesn't follow X sequentially (stride assumption violated for Y)"

            # Append Y from last window (future timesteps not in any X)
            continuous = np.concatenate(
                [
                    continuous,
                    self.y[-1, :, :, :].numpy(),  # [horizon, num_nodes, num_features]
                ],
                axis=0,
            )

            # Validate full reconstruction
            expected_timesteps = self.x.shape[0] + self.x.shape[1] + self.y.shape[1] - 1
            assert continuous.shape[0] == expected_timesteps, (
                f"Full reconstruction failed: got {continuous.shape[0]} timesteps, "
                f"expected {expected_timesteps}"
            )

        return continuous


def hf_to_pytorch(hf_dataset, adj_mx=None, verbose=False, add_dow=False):
    """
    Convert HuggingFace DatasetDict to PyTorch TrafficDatasets.

    Args:
        hf_dataset: datasets.DatasetDict with train/val/test splits
            Expected columns: node_id, t0_timestamp, x_t{i}_d{j}, y_t{i}_d{j}
        adj_mx: Optional adjacency matrix, shape (num_nodes, num_nodes)
        verbose: If True, show progress bars
        add_dow: If True, compute day-of-week from t0_timestamp and add as extra feature.
            DOW is stored as raw 0-6 values (Monday=0, Sunday=6) matching STGFormer.

    Returns:
        Dict[str, TrafficDataset]: Dictionary mapping split names to TrafficDataset objects
    """
    pytorch_datasets = {}
    metadata = None
    column_cache = None

    for split_name, split_data in hf_dataset.items():
        # Normalize split name (HF uses "validation", we use "val")
        normalized_split_name = "val" if split_name == "validation" else split_name
        df = split_data.to_pandas()

        # Cache column metadata from the first split and reuse for the rest
        if column_cache is None:
            x_cols = _sorted_time_feature_columns(df.columns, prefix="x_t")
            y_cols = _sorted_time_feature_columns(df.columns, prefix="y_t")
            seq_len, horizon, raw_input_dim, raw_output_dim = _infer_dataset_dimensions(
                x_cols, y_cols
            )
            x_timestep_offsets = _extract_timestep_offsets(x_cols)
            y_timestep_offsets = _extract_timestep_offsets(y_cols)
            x_time_dim_index = _find_feature_index(x_cols, target_dim=1)
            y_time_dim_index = _find_feature_index(y_cols, target_dim=1)

            needs_time_features = (x_time_dim_index is not None) or (
                y_time_dim_index is not None
            )

            column_cache = {
                "x_cols": x_cols,
                "y_cols": y_cols,
                "seq_len": seq_len,
                "horizon": horizon,
                "raw_input_dim": raw_input_dim,
                "raw_output_dim": raw_output_dim,
                "x_time_dim_index": x_time_dim_index,
                "y_time_dim_index": y_time_dim_index,
                "needs_time_features": needs_time_features,
                "x_timestep_offsets": x_timestep_offsets,
                "y_timestep_offsets": y_timestep_offsets,
            }
        else:
            x_cols = column_cache["x_cols"]
            y_cols = column_cache["y_cols"]
            seq_len = column_cache["seq_len"]
            horizon = column_cache["horizon"]
            raw_input_dim = column_cache["raw_input_dim"]
            raw_output_dim = column_cache["raw_output_dim"]
            x_time_dim_index = column_cache["x_time_dim_index"]
            y_time_dim_index = column_cache["y_time_dim_index"]
            needs_time_features = column_cache["needs_time_features"]
            x_timestep_offsets = column_cache["x_timestep_offsets"]
            y_timestep_offsets = column_cache["y_timestep_offsets"]

        # Check if we have timestamp column for time feature computation
        has_timestamp = "t0_timestamp" in df.columns
        if (add_dow or needs_time_features) and not has_timestamp:
            raise ValueError(
                "HF dataset must include 't0_timestamp' to compute temporal features"
            )

        # Extract sequences per node
        x_data, y_data = [], []
        node_groups = df.groupby("node_id", sort=True)
        node_iterator = tqdm(
            node_groups,
            total=df["node_id"].nunique(),
            desc=f"Processing {normalized_split_name} nodes",
            disable=not verbose,
        )
        for node_id, node_df in node_iterator:
            node_df = node_df.sort_index()
            timestamps = (
                node_df["t0_timestamp"].to_numpy()
                if (add_dow or needs_time_features)
                else None
            )

            # Reshape to (samples, timesteps, features)
            x_node = node_df[x_cols].to_numpy().reshape(-1, seq_len, raw_input_dim)
            y_node = node_df[y_cols].to_numpy().reshape(-1, horizon, raw_output_dim)

            # Recompute normalized time-of-day (d1) features from timestamps
            if needs_time_features and timestamps is not None:
                if x_time_dim_index is not None:
                    x_time_features = _compute_normalized_time_of_day(
                        timestamps, x_timestep_offsets
                    )
                    x_node[:, :, x_time_dim_index] = x_time_features
                if y_time_dim_index is not None:
                    y_time_features = _compute_normalized_time_of_day(
                        timestamps, y_timestep_offsets
                    )
                    y_node[:, :, y_time_dim_index] = y_time_features

            # Compute DOW for each timestep if requested
            if add_dow:
                x_dow = _compute_dow_features(timestamps, x_timestep_offsets)
                y_dow = _compute_dow_features(timestamps, y_timestep_offsets)
                # Add DOW as extra feature dimension
                x_node = np.concatenate([x_node, x_dow[..., np.newaxis]], axis=-1)
                y_node = np.concatenate([y_node, y_dow[..., np.newaxis]], axis=-1)

            x_data.append(x_node)
            y_data.append(y_node)

        # Stack along node axis: (samples, seq_len/horizon, num_nodes, features)
        x_data = np.stack(x_data, axis=2)
        y_data = np.stack(y_data, axis=2)

        # Store num_nodes from first split
        if metadata is None:
            num_nodes = x_data.shape[2]
            # Final input/output dim includes DOW if added
            input_dim = raw_input_dim + (1 if add_dow else 0)
            output_dim = raw_output_dim + (1 if add_dow else 0)
            metadata = {
                "num_nodes": num_nodes,
                "input_dim": input_dim,
                "output_dim": output_dim,
            }
        else:
            num_nodes = metadata["num_nodes"]
            input_dim = metadata["input_dim"]
            output_dim = metadata["output_dim"]

        # Create TrafficDataset with metadata
        pytorch_datasets[normalized_split_name] = TrafficDataset(
            x_data,
            y_data,
            seq_len=seq_len,
            horizon=horizon,
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=output_dim,
            adj_mx=adj_mx,
        )

        print(
            f"Loaded {normalized_split_name}: {pytorch_datasets[normalized_split_name]}"
        )

    return pytorch_datasets


def _sorted_time_feature_columns(columns, prefix):
    """Return sorted feature columns for a given prefix."""
    prefix = prefix or ""
    filtered = [col for col in columns if col.startswith(prefix)]
    return sorted(
        filtered,
        key=lambda col: (
            int(col.split("_")[1][1:].replace("+", "")),
            int(col.split("_")[2][1:]),
        ),
    )


def _infer_dataset_dimensions(x_cols, y_cols):
    """Infer seq_len/horizon and feature dims from sorted columns."""
    seq_len = len([col for col in x_cols if col.endswith("_d0")])
    horizon = len([col for col in y_cols if col.endswith("_d0")])

    first_x_timestep = x_cols[0].split("_")[1]
    first_y_timestep = y_cols[0].split("_")[1]
    raw_input_dim = len(
        [col for col in x_cols if col.startswith(f"x_{first_x_timestep}_")]
    )
    raw_output_dim = len(
        [col for col in y_cols if col.startswith(f"y_{first_y_timestep}_")]
    )
    return seq_len, horizon, raw_input_dim, raw_output_dim


def _find_feature_index(columns, target_dim):
    """Return the feature index (0-based) for the requested dimension."""
    if not columns:
        return None

    prefix = columns[0].split("_")[0]
    first_timestep = columns[0].split("_")[1]
    timestep_prefix = f"{prefix}_{first_timestep}_"
    feature_columns = [col for col in columns if col.startswith(timestep_prefix)]
    for idx, col in enumerate(feature_columns):
        if col.endswith(f"d{target_dim}"):
            return idx
    return None


def _extract_timestep_offsets(columns):
    """Extract sorted timestep offsets as integer multiples of 5-minute steps."""
    offsets = sorted({int(col.split("_")[1][1:].replace("+", "")) for col in columns})
    return np.asarray(offsets, dtype=np.int32)


def _compute_dow_features(
    timestamps: np.ndarray, timestep_offsets: np.ndarray
) -> np.ndarray:
    """
    Compute day-of-week for each timestep offset from t0_timestamp.

    Args:
        timestamps: Array of ISO format timestamp strings (t0_timestamp values)
        timestep_offsets: Array of timestep offsets (e.g., [-11, ..., 0])

    Returns:
        np.ndarray: DOW values shape (num_samples, num_timesteps), values 0-6 (Mon-Sun)
    """
    if timestep_offsets is None or len(timestep_offsets) == 0:
        return np.zeros((len(timestamps), 0), dtype=np.float32)

    base_times = np.array(timestamps, dtype="datetime64[m]")
    offsets = np.asarray(timestep_offsets, dtype=np.int64)
    minutes = offsets * 5

    adjusted = base_times[:, None] + minutes[None, :] * np.timedelta64(1, "m")
    dow = (adjusted.astype("datetime64[D]").astype(np.int64) + 3) % 7
    return dow.astype(np.float32)


def _compute_normalized_time_of_day(
    timestamps: np.ndarray, timestep_offsets: np.ndarray
) -> np.ndarray:
    """
    Compute normalized time-of-day (d1) features from timestamps.

    Args:
        timestamps: Array of base timestamps (t0) for each sample.
        timestep_offsets: Array of timestep offsets (multiples of 5 minutes).

    Returns:
        np.ndarray: Normalized time-of-day values in [0, 1), shape (samples, timesteps).
    """
    if timestep_offsets is None or len(timestep_offsets) == 0:
        return np.zeros((len(timestamps), 0), dtype=np.float32)

    base_minutes = np.array(timestamps, dtype="datetime64[m]").astype(np.int64)
    offsets = np.asarray(timestep_offsets, dtype=np.int64)
    total_minutes = base_minutes[:, None] + offsets[None, :] * 5
    minutes_per_day = 24 * 60
    normalized = np.mod(total_minutes, minutes_per_day) / minutes_per_day
    return normalized.astype(np.float32)
