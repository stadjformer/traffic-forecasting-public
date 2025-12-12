"""Benchmark Mamba vs Transformer performance for STGFormer.

This script profiles the performance of Mamba and Transformer temporal processing
to identify bottlenecks and compare efficiency for short sequences.

Requirements:
- CUDA-enabled GPU
- mamba-ssm installed (uv sync --extra cuda)

Usage:
    uv run python scripts/benchmark_mamba.py
    uv run python scripts/benchmark_mamba.py --batch-size 64 --num-nodes 207 --seq-len 12
    uv run python scripts/benchmark_mamba.py --profile  # Detailed profiling
"""

import argparse
import time
from contextlib import contextmanager

import torch
import torch.nn as nn

# Check CUDA availability
if not torch.cuda.is_available():
    print("ERROR: CUDA not available. This benchmark requires a CUDA-enabled GPU.")
    print("Please run on a machine with NVIDIA GPU and CUDA installed.")
    exit(1)

# Try importing Mamba
try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError:
    print("ERROR: mamba-ssm not installed.")
    print("Install with: uv sync --extra cuda")
    exit(1)


@contextmanager
def timer(name):
    """Context manager for timing code blocks."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"{name}: {(end - start) * 1000:.2f} ms")


class TransformerTemporal(nn.Module):
    """Simplified transformer temporal processing (baseline)."""

    def __init__(self, model_dim, num_heads=4):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        """
        Process temporal dimension with standard attention.

        Args:
            x: [batch, time, nodes, features]
        Returns:
            [batch, time, nodes, features]
        """
        B, T, N, D = x.shape

        # Reshape to [B*N, T, D] to process each node independently
        x_temporal = x.permute(0, 2, 1, 3).reshape(B * N, T, D)

        # QKV projection
        qkv = self.qkv(x_temporal)  # [B*N, T, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B * N, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B * N, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * N, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: [B*N, heads, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = torch.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [B*N, heads, T, head_dim]

        # Reshape back
        out = out.transpose(1, 2).reshape(B * N, T, D)
        out = self.out_proj(out)

        # Reshape to [B, T, N, D]
        out = out.reshape(B, N, T, D).permute(0, 2, 1, 3)

        return out


class MambaTemporal(nn.Module):
    """Mamba SSM temporal processing."""

    def __init__(self, model_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.model_dim = model_dim
        self.mamba = Mamba(
            d_model=model_dim, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, x):
        """
        Process temporal dimension with Mamba SSM.

        Args:
            x: [batch, time, nodes, features]
        Returns:
            [batch, time, nodes, features]
        """
        B, T, N, D = x.shape

        # Reshape: [B, T, N, D] -> [B*N, T, D]
        x_temporal = x.permute(0, 2, 1, 3).reshape(B * N, T, D)

        # Process through Mamba
        out = self.mamba(x_temporal)

        # Reshape back: [B*N, T, D] -> [B, T, N, D]
        out = out.reshape(B, N, T, D).permute(0, 2, 1, 3)

        return out


def benchmark_forward_pass(model, x, name, num_warmup=10, num_iterations=100):
    """Benchmark forward pass performance."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {name}")
    print(f"{'=' * 60}")

    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        _ = model(x)

    # Benchmark
    print(f"Running benchmark ({num_iterations} iterations)...")
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_iterations):
        _ = model(x)

    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / num_iterations * 1000  # ms
    print(f"Average time: {avg_time:.2f} ms")

    return avg_time


def profile_components(model, x, name):
    """Profile individual components of the model."""
    print(f"\n{'=' * 60}")
    print(f"Component Profiling: {name}")
    print(f"{'=' * 60}")

    B, T, N, D = x.shape

    # Profile reshaping (first permute + reshape)
    with timer("  1. Permute [B,T,N,D] -> [B,N,T,D]"):
        for _ in range(100):
            x_perm = x.permute(0, 2, 1, 3)

    with timer("  2. Reshape [B,N,T,D] -> [B*N,T,D]"):
        x_perm = x.permute(0, 2, 1, 3)
        for _ in range(100):
            _ = x_perm.reshape(B * N, T, D)

    # Profile core computation
    x_temporal = x.permute(0, 2, 1, 3).reshape(B * N, T, D)

    if isinstance(model, MambaTemporal):
        with timer("  3. Mamba forward pass"):
            for _ in range(100):
                out = model.mamba(x_temporal)
    else:
        with timer("  3. Transformer forward pass"):
            for _ in range(100):
                # QKV + attention
                qkv = model.qkv(x_temporal)
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.view(B * N, T, model.num_heads, model.head_dim).transpose(1, 2)
                k = k.view(B * N, T, model.num_heads, model.head_dim).transpose(1, 2)
                v = v.view(B * N, T, model.num_heads, model.head_dim).transpose(1, 2)
                scores = torch.matmul(q, k.transpose(-2, -1)) / (model.head_dim**0.5)
                attn = torch.softmax(scores, dim=-1)
                out = torch.matmul(attn, v)
                out = out.transpose(1, 2).reshape(B * N, T, D)
                out = model.out_proj(out)

    # Profile reshaping back
    with timer("  4. Reshape [B*N,T,D] -> [B,N,T,D]"):
        out_temp = torch.randn(B * N, T, D, device=x.device)
        for _ in range(100):
            out_reshaped = out_temp.reshape(B, N, T, D)

    with timer("  5. Permute [B,N,T,D] -> [B,T,N,D]"):
        out_reshaped = torch.randn(B, N, T, D, device=x.device)
        for _ in range(100):
            _ = out_reshaped.permute(0, 2, 1, 3)


def memory_usage(model, x, name):
    """Measure memory usage."""
    print(f"\n{'=' * 60}")
    print(f"Memory Usage: {name}")
    print(f"{'=' * 60}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Forward pass
    _ = model(x)
    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    peak = torch.cuda.max_memory_allocated() / 1024**2  # MB

    print(f"Allocated memory: {allocated:.2f} MB")
    print(f"Peak memory: {peak:.2f} MB")

    return allocated, peak


def main():
    parser = argparse.ArgumentParser(description="Benchmark Mamba vs Transformer")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=207,
        help="Number of nodes (METR-LA=207, PEMS-BAY=325)",
    )
    parser.add_argument("--seq-len", type=int, default=12, help="Sequence length")
    parser.add_argument("--model-dim", type=int, default=96, help="Model dimension")
    parser.add_argument("--num-warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument(
        "--num-iterations", type=int, default=100, help="Benchmark iterations"
    )
    parser.add_argument("--profile", action="store_true", help="Run detailed profiling")
    args = parser.parse_args()

    print("=" * 60)
    print("STGFormer: Mamba vs Transformer Benchmark")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()
    print("Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num nodes: {args.num_nodes}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Model dimension: {args.model_dim}")
    print(f"  Effective batch (B*N): {args.batch_size * args.num_nodes}")
    print()

    # Create models
    device = torch.device("cuda")
    transformer = TransformerTemporal(args.model_dim, num_heads=4).to(device)
    mamba = MambaTemporal(args.model_dim, d_state=16, d_conv=4, expand=2).to(device)

    # Create input
    x = torch.randn(
        args.batch_size, args.seq_len, args.num_nodes, args.model_dim, device=device
    )

    # Model parameters
    print("Model Parameters:")
    transformer_params = sum(p.numel() for p in transformer.parameters())
    mamba_params = sum(p.numel() for p in mamba.parameters())
    print(f"  Transformer: {transformer_params:,}")
    print(f"  Mamba: {mamba_params:,}")

    # Benchmark forward pass
    transformer_time = benchmark_forward_pass(
        transformer, x, "Transformer", args.num_warmup, args.num_iterations
    )
    mamba_time = benchmark_forward_pass(
        mamba, x, "Mamba", args.num_warmup, args.num_iterations
    )

    # Memory usage
    transformer_mem, transformer_peak = memory_usage(transformer, x, "Transformer")
    mamba_mem, mamba_peak = memory_usage(mamba, x, "Mamba")

    # Component profiling (if requested)
    if args.profile:
        profile_components(transformer, x, "Transformer")
        profile_components(mamba, x, "Mamba")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<15} {'Time (ms)':<12} {'Memory (MB)':<15} {'Peak (MB)':<12}")
    print("-" * 60)
    print(
        f"{'Transformer':<15} {transformer_time:<12.2f} {transformer_mem:<15.2f} {transformer_peak:<12.2f}"
    )
    print(f"{'Mamba':<15} {mamba_time:<12.2f} {mamba_mem:<15.2f} {mamba_peak:<12.2f}")
    print("-" * 60)

    speedup = transformer_time / mamba_time
    if speedup > 1:
        print(f"Mamba is {speedup:.2f}x FASTER than Transformer")
    else:
        print(f"Mamba is {1 / speedup:.2f}x SLOWER than Transformer")

    mem_ratio = mamba_peak / transformer_peak
    if mem_ratio > 1:
        print(f"Mamba uses {mem_ratio:.2f}x MORE memory than Transformer")
    else:
        print(f"Mamba uses {mem_ratio:.2f}x LESS memory than Transformer")


if __name__ == "__main__":
    main()
