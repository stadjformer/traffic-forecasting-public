"""Benchmark hf_to_pytorch conversion on real datasets."""

from __future__ import annotations

import argparse
import statistics
import time

import utils.dataset
from utils.io import get_dataset_hf, get_graph_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark hf_to_pytorch conversion speed."
    )
    parser.add_argument(
        "--dataset",
        default="PEMS-BAY",
        help="Dataset name (default: PEMS-BAY).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed runs (default: 3).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warm-up runs to ignore (default: 1).",
    )
    parser.add_argument(
        "--no-add-dow",
        action="store_false",
        dest="add_dow",
        help="Disable day-of-week computation (enabled by default).",
    )
    parser.set_defaults(add_dow=True)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose conversion output.",
    )
    return parser.parse_args()


def benchmark(dataset: str, repeats: int, warmup: int, add_dow: bool, verbose: bool):
    dataset_hf = get_dataset_hf(dataset)
    adj_mx, _, _ = get_graph_metadata(dataset)

    print(f"Dataset: {dataset}")
    for split_name, split_data in dataset_hf.items():
        print(f"  {split_name}: {split_data.num_rows:,} samples")
    print(f"add_dow={add_dow}, repeats={repeats}, warmup={warmup}")

    def run_conversion():
        return utils.dataset.hf_to_pytorch(
            dataset_hf, adj_mx=adj_mx, verbose=verbose, add_dow=add_dow
        )

    for i in range(warmup):
        print(f"Warm-up run {i + 1}/{warmup}...")
        _ = run_conversion()

    timings = []
    for i in range(repeats):
        start = time.perf_counter()
        _ = run_conversion()
        duration = time.perf_counter() - start
        timings.append(duration)
        print(f"Run {i + 1}/{repeats}: {duration:.3f}s")

    print("---")
    print(f"Best: {min(timings):.3f}s")
    print(f"Mean: {statistics.mean(timings):.3f}s")
    if len(timings) > 1:
        print(f"Std: {statistics.pstdev(timings):.4f}s")


if __name__ == "__main__":
    args = parse_args()
    benchmark(
        dataset=args.dataset,
        repeats=args.repeats,
        warmup=args.warmup,
        add_dow=getattr(args, "add_dow", True),
        verbose=args.verbose,
    )
