#!/usr/bin/env python3
"""Quick diagnostic script for traffic forecasting datasets.

Usage:
    python scripts/diagnose_dataset.py LARGEST-GLA
    python scripts/diagnose_dataset.py METR-LA
    python scripts/diagnose_dataset.py --parquet data/LARGEST-GLA/train.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def diagnose_parquet(path: str, sample_size: int = 10000):
    """Diagnose a parquet file with random sampling."""
    print(f"\n=== Parquet: {path} ===")

    df = pd.read_parquet(path)
    total_rows = len(df)

    # Random sample
    np.random.seed(42)
    sample_idx = np.random.choice(total_rows, size=min(sample_size, total_rows), replace=False)
    sample = df.iloc[sample_idx]

    print(f"Total rows: {total_rows:,}")
    print(f"Columns: {len(df.columns)}")

    if "node_id" in df.columns:
        print(f"Unique nodes: {df['node_id'].nunique()}")

    # Get x and y columns
    x_cols = [c for c in sample.columns if c.startswith("x_")]
    y_cols = [c for c in sample.columns if c.startswith("y_")]
    print(f"X columns: {len(x_cols)}, Y columns: {len(y_cols)}")

    if x_cols:
        x_vals = sample[x_cols].values
        print(f"\nX values (sample of {len(sample):,}):")
        print(f"  Range: [{np.nanmin(x_vals):.2f}, {np.nanmax(x_vals):.2f}]")
        print(f"  Mean: {np.nanmean(x_vals):.2f}")
        print(f"  NaN: {np.isnan(x_vals).sum()} / {x_vals.size} ({100*np.isnan(x_vals).sum()/x_vals.size:.2f}%)")
        print(f"  Zeros: {(x_vals == 0).sum()} / {x_vals.size} ({100*(x_vals == 0).sum()/x_vals.size:.2f}%)")

    if y_cols:
        y_vals = sample[y_cols].values
        print(f"\nY values (sample of {len(sample):,}):")
        print(f"  Range: [{np.nanmin(y_vals):.2f}, {np.nanmax(y_vals):.2f}]")
        print(f"  Mean: {np.nanmean(y_vals):.2f}")
        print(f"  NaN: {np.isnan(y_vals).sum()} / {y_vals.size} ({100*np.isnan(y_vals).sum()/y_vals.size:.2f}%)")

    del df, sample
    return True


def diagnose_pytorch_cache(path: str, sample_size: int = 1000):
    """Diagnose a PyTorch cached dataset."""
    print(f"\n=== PyTorch Cache: {path} ===")

    from utils.dataset import TrafficDataset

    data = torch.load(path, map_location="cpu", weights_only=False)
    datasets = data["datasets"]

    for split, ds in datasets.items():
        print(f"\n--- {split} ---")
        print(f"x shape: {ds.x.shape}")
        print(f"y shape: {ds.y.shape}")
        print(f"input_dim: {ds.input_dim}, output_dim: {ds.output_dim}")

        # Sample analysis (avoid loading full tensor)
        x_sample = ds.x[:sample_size]
        y_sample = ds.y[:sample_size]

        print(f"\nX sample ({sample_size} samples):")
        print(f"  Range: [{x_sample[~torch.isnan(x_sample)].min():.2f}, {x_sample[~torch.isnan(x_sample)].max():.2f}]")
        print(f"  NaN: {torch.isnan(x_sample).sum().item()} / {x_sample.numel()} ({100*torch.isnan(x_sample).sum().item()/x_sample.numel():.2f}%)")

        # Check NaN distribution per node
        nan_per_node = torch.isnan(x_sample[..., 0]).sum(dim=(0, 1))
        nodes_all_nan = (nan_per_node == sample_size * ds.seq_len).sum().item()
        nodes_no_nan = (nan_per_node == 0).sum().item()
        print(f"\nNode NaN analysis:")
        print(f"  Nodes with 100% NaN: {nodes_all_nan}")
        print(f"  Nodes with 0% NaN: {nodes_no_nan}")
        print(f"  Nodes with some NaN: {ds.num_nodes - nodes_all_nan - nodes_no_nan}")

        # Check if scaler would fail
        print(f"\nScaler compatibility:")
        raw_mean = x_sample[..., 0].mean(dim=0)  # Mean across samples
        nan_in_mean = torch.isnan(raw_mean).any().item()
        print(f"  Would scaler fail (NaN in mean): {nan_in_mean}")

        if ds.adj_mx is not None:
            print(f"\nAdjacency matrix: {ds.adj_mx.shape}")
            print(f"  Range: [{ds.adj_mx.min():.4f}, {ds.adj_mx.max():.4f}]")
            print(f"  NaN count: {torch.isnan(ds.adj_mx).sum().item()}")

    return True


def diagnose_adjacency(path: str):
    """Diagnose adjacency matrix."""
    print(f"\n=== Adjacency Matrix: {path} ===")

    adj = np.load(path)
    print(f"Shape: {adj.shape}")
    print(f"Dtype: {adj.dtype}")
    print(f"Range: [{adj.min():.6f}, {adj.max():.6f}]")
    print(f"NaN count: {np.isnan(adj).sum()}")
    print(f"Non-zero: {(adj != 0).sum()} / {adj.size} ({100*(adj != 0).sum()/adj.size:.2f}%)")
    print(f"Symmetric: {np.allclose(adj, adj.T)}")

    return True


def diagnose_dataset(dataset_name: str):
    """Run full diagnostics for a named dataset."""
    from utils.config import DATA_DIR

    data_dir = DATA_DIR / dataset_name

    if not data_dir.exists():
        print(f"ERROR: Dataset directory not found: {data_dir}")
        return False

    print(f"\n{'='*60}")
    print(f"DATASET DIAGNOSTICS: {dataset_name}")
    print(f"{'='*60}")

    # Check parquet files
    for split in ["train", "val", "test"]:
        parquet_path = data_dir / f"{split}.parquet"
        if parquet_path.exists():
            diagnose_parquet(str(parquet_path))

    # Check adjacency matrix
    adj_path = data_dir / "sensor_graph" / "adj_mx.npy"
    if adj_path.exists():
        diagnose_adjacency(str(adj_path))

    # Check PyTorch cache
    cache_path = data_dir / "pytorch_cache" / "datasets.pt"
    if cache_path.exists():
        diagnose_pytorch_cache(str(cache_path))
    else:
        print(f"\nNo PyTorch cache found at {cache_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Diagnose traffic forecasting datasets")
    parser.add_argument("dataset", nargs="?", help="Dataset name (e.g., LARGEST-GLA, METR-LA)")
    parser.add_argument("--parquet", help="Path to a specific parquet file")
    parser.add_argument("--cache", help="Path to a specific PyTorch cache file")
    parser.add_argument("--adj", help="Path to a specific adjacency matrix")

    args = parser.parse_args()

    if args.parquet:
        diagnose_parquet(args.parquet)
    elif args.cache:
        diagnose_pytorch_cache(args.cache)
    elif args.adj:
        diagnose_adjacency(args.adj)
    elif args.dataset:
        diagnose_dataset(args.dataset)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
