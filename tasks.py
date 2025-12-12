"""Task automation using invoke.

For Mac/non-CUDA systems, use: uvx invoke <task>
This avoids uv dependency resolution issues with CUDA-only packages.
"""

import csv
import os
import sys

# Detect if we're running via uvx (python executable not in local .venv)
# But skip re-exec if we're already being run by uv (to prevent infinite loop)
_is_uvx = (
    ".venv" not in sys.executable
    and "UV_INTERNAL__PARENT_INTERPRETER" not in os.environ
)
if _is_uvx:
    # Re-exec via 'uv run invoke' to use local .venv
    args = ["uv", "run", "invoke"] + sys.argv[1:]
    os.environ["UV_NO_SYNC"] = "1"  # Prevent dependency resolution
    os.execvp("uv", args)

# Now safe to import project dependencies
# ruff: noqa: E402
from invoke import task


def _has_cuda() -> bool:
    """Best-effort check for CUDA availability (for CUDA-only experiments)."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


_CUDA_AVAILABLE = _has_cuda()


def _build_cmd(base_cmd, dry_run=False, push_to_hub=False, verbose=False, dataset=None):
    """Build command string with common flags."""
    cmd = base_cmd
    if dry_run:
        cmd += " --dry-run"
    if push_to_hub:
        cmd += " --push-to-hub"
    if verbose:
        cmd += " --verbose"
    if dataset:
        cmd += f" --dataset {dataset}"
    return cmd


@task
def install_kernel(ctx):
    """Install the project's Python environment as a Jupyter kernel."""
    ctx.run("uv run python -m ipykernel install --user --name=traffic-forecasting")


@task
def lab(ctx):
    """Start Jupyter Lab with the project kernel."""
    ctx.run("uv run jupyter lab --kernel=traffic-forecasting")


@task
def notebook(ctx):
    """Start Jupyter Notebook with the project kernel."""
    ctx.run("uv run jupyter notebook --kernel=traffic-forecasting")


@task
def test(ctx, verbose=False):
    """Run tests with pytest."""
    cmd = "uv run pytest"
    if verbose:
        cmd += " -v"
    ctx.run(cmd)


@task
def compute_baselines(ctx, force=False, to_file=False, highlight=False):
    """Compute baseline metrics for all models and datasets.

    Args:
        force: Recompute metrics even if cached.
        to_file: Save results to markdown file (docs/FINAL_RESULTS_WITH_BASELINES.md).
        highlight: Bold the lowest value per metric when printing/saving tables.
    """
    from pathlib import Path

    from utils.baselines import backup_results as backup_baselines
    from utils.baselines import calculate_baseline_metrics
    from utils.baselines import format_results_table as format_baselines_table

    datasets = ["METR-LA", "PEMS-BAY"]
    models = [
        "AR",
        "VAR",
        "DCRNN",
        "MTGNN",
        "GWNET",
        "STGFORMER",
        "STGFORMER_INTERNAL",
        "STGFORMER_INTERNAL_DOW",
        "STGFORMER_FINAL",  # Final architecture: Cheb+TCN+Xavier+DOW+ExcludeMissing+K16
    ]  # ARIMA, VARIME

    for dataset in datasets:
        print(f"\n{'=' * 60}")
        print(f"Computing metrics for {dataset}")
        print(f"{'=' * 60}\n")

        # Backup once per dataset if forcing recalculation
        if force:
            backup_file = backup_baselines(dataset)
            if backup_file:
                print(f"Backed up to {backup_file.name}")

        for model in models:
            print(f"{model}...", end=" ", flush=True)
            calculate_baseline_metrics(dataset, model, force=force)
            print("done")

        print(f"\n{dataset} Results:")
        print(
            format_baselines_table(
                dataset, model_order=models, highlight_best=highlight
            )
        )

    # Save to file if requested
    if to_file:
        output_path = Path("docs/FINAL_RESULTS_WITH_BASELINES.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Final Results with Baselines",
            "",
            "Comparison of our final model (STGFORMER_FINAL) against baseline models.",
            "",
        ]

        for dataset in datasets:
            lines.append(f"## {dataset}")
            lines.append("")
            table = format_baselines_table(
                dataset, model_order=models, highlight_best=True
            )
            lines.append(table)
            lines.append("")

        output_path.write_text("\n".join(lines))
        print(f"\n✓ Wrote baseline results to {output_path}")


@task
def train_dcrnn(ctx, dry_run=False, push_to_hub=False):
    """Train DCRNN models on both METR-LA and PEMS-BAY datasets and pushes trained models to HF hub.
    Dry mode skips training, only does eval"""
    cmd = _build_cmd(
        "uv run python scripts/train_dcrnn.py", dry_run=dry_run, push_to_hub=push_to_hub
    )
    ctx.run(f"{cmd} METR-LA", pty=True)
    ctx.run(f"{cmd} PEMS-BAY", pty=True)


@task
def train_mtgnn(ctx, dry_run=False, push_to_hub=False, verbose=False):
    """Train MTGNN models on both METR-LA and PEMS-BAY datasets and pushes trained models to HF hub.
    Dry mode skips training, only does eval. Use --verbose to see training progress."""
    cmd = _build_cmd(
        "uv run python scripts/train_mtgnn.py",
        dry_run=dry_run,
        push_to_hub=push_to_hub,
        verbose=verbose,
    )
    ctx.run(f"{cmd} METR-LA", pty=True)
    ctx.run(f"{cmd} PEMS-BAY", pty=True)


@task
def train_gwnet(ctx, dry_run=False, push_to_hub=False, verbose=False):
    """Train Graph-WaveNet models on both METR-LA and PEMS-BAY datasets and pushes trained models to HF hub.
    Dry mode trains for 1 epoch only. Use --verbose to see training progress."""
    cmd = _build_cmd(
        "uv run python scripts/train_gwnet.py",
        dry_run=dry_run,
        push_to_hub=push_to_hub,
        verbose=verbose,
    )
    ctx.run(f"{cmd} METR-LA", pty=True)
    ctx.run(f"{cmd} PEMS-BAY", pty=True)


@task
def train_stgformer_external(ctx, dry_run=False, push_to_hub=False, verbose=False):
    """Train STGformer (external) models on both METR-LA and PEMS-BAY datasets and pushes trained models to HF hub.
    Dry mode trains for 1 epoch only. Use --verbose to see training progress."""
    cmd = _build_cmd(
        "uv run python scripts/train_stgformer_external.py",
        dry_run=dry_run,
        push_to_hub=push_to_hub,
        verbose=verbose,
    )
    ctx.run(f"{cmd} METR-LA", pty=True)
    ctx.run(f"{cmd} PEMS-BAY", pty=True)


@task(name="train-stgformer-internal")
def train_stgformer_internal(
    ctx, dry_run=False, push_to_hub=False, verbose=False, dataset=None
):
    """Train internal STGFormer models using baseline config.

    By default trains on both METR-LA and PEMS-BAY (as specified in config).
    Use --dataset to train on a single dataset.
    Use --dry-run for 1 epoch test. Use --verbose to see training progress."""
    cmd = _build_cmd(
        "uv run python scripts/train_stgformer.py --config configs/stgformer_baseline.yaml",
        dry_run=dry_run,
        push_to_hub=push_to_hub,
        verbose=verbose,
        dataset=dataset,
    )
    ctx.run(cmd, pty=True)


@task(name="train-final")
def train_final_model(
    ctx, dry_run=False, push_to_hub=False, verbose=False, dataset=None
):
    """Train FINAL model (Cheb+TCN+Xavier+DOW+ExcludeMissing+K16) with full 100 epochs.

    This is the best performing architecture across all experiments.
    By default trains on both METR-LA and PEMS-BAY.
    Use --dataset to train on a single dataset.
    Use --dry-run for 1 epoch test. Use --verbose to see training progress."""
    cmd = _build_cmd(
        "uv run python scripts/train_stgformer.py --config configs/stgformer_cheb_tcn_xavier_dow_exclude_missing_k16_full.yaml",
        dry_run=dry_run,
        push_to_hub=push_to_hub,
        verbose=verbose,
        dataset=dataset,
    )
    ctx.run(cmd, pty=True)


# Experiment configs mapping
EXPERIMENTS = {
    "bs200_short": (
        "stgformer_baseline_bs200_short.yaml",
        "Baseline with batch_size=200, 20 epochs, early_stop=5",
    ),
    "bs100_short": (
        "stgformer_baseline_bs100_short.yaml",
        "Baseline with batch_size=100, 20 epochs, early_stop=5 (for comparison with model that can't fit batch_size=200 on our GPU)",
    ),
    "bs200_xavier": (
        "stgformer_baseline_bs200_xavier.yaml",
        "Baseline with batch_size=200, 20 epochs, early_stop=5, xavier init",
    ),
    "bs200_dow": (
        "stgformer_with_dow.yaml",
        "Baseline with batch_size=200, 20 epochs, early_stop=5 and DOW embeddings",
    ),
    "bs200_exclude_missing": (
        "stgformer_baseline_bs200_exclude_missing.yaml",
        "Baseline with batch_size=200, exclude missing values from normalization",
    ),
    "geo": ("stgformer_geographic.yaml", "Geographic (pre-computed) graph"),
    "spectral_init": (
        "stgformer_spectral_init.yaml",
        "Learned graph initialized from Laplacian eigenvectors",
    ),
    "chebyshev": ("stgformer_chebyshev.yaml", "Chebyshev polynomial propagation"),
    "hybrid": ("stgformer_hybrid.yaml", "Hybrid graph (geographic + learned)"),
    "geo_cheb": (
        "stgformer_geographic_chebyshev.yaml",
        "Geographic graph + Chebyshev polynomial propagation",
    ),
    "hybrid_cheb": (
        "stgformer_hybrid_chebyshev.yaml",
        "Hybrid graph + Chebyshev polynomial propagation",
    ),
    "mamba": (
        "stgformer_mamba.yaml",
        "Mamba SSM temporal mode (d_state=16, requires CUDA)",
    ),
    # "mamba2": (
    #     "stgformer_mamba2.yaml",
    #     "Mamba2 SSM temporal mode (d_state=64, requires CUDA)",
    # ), # This is super slow and TCN seems to perform better than vanilla Mamba so likely no point running it
    "mamba_fast": (
        "stgformer_mamba_fast.yaml",
        "Optimized Mamba (d_state=8, expand=1)",
    ),
    "tcn": (
        "stgformer_tcn.yaml",
        "TCN temporal mode (causal dilated convolutions)",
    ),
    "depthwise": (
        "stgformer_depthwise.yaml",
        "Depthwise separable conv temporal mode",
    ),
    "mlp": (
        "stgformer_mlp.yaml",
        "MLP temporal mode",
    ),
    "pretrain": (
        "stgformer_pretrain.yaml",
        "Masked node pretraining (5+5 curriculum) + forecasting fine-tune",
    ),
    "pretrain_impute": (
        "stgformer_pretrain_impute.yaml",
        "Masked node pretraining (5+5 curriculum) with imputation before fine-tuning",
    ),
    "pretrain_stage1only": (
        "stgformer_pretrain_stage1only.yaml",
        "Stage 1 only pretraining (10 epochs per-timestep masking)",
    ),
    "pretrain_stage1only_impute": (
        "stgformer_pretrain_stage1only_impute.yaml",
        "Stage 1 only pretraining with imputation",
    ),
    "pretrain_stage2only": (
        "stgformer_pretrain_stage2only.yaml",
        "Stage 2 only pretraining (10 epochs per-node masking)",
    ),
    "pretrain_stage2only_impute": (
        "stgformer_pretrain_stage2only_impute.yaml",
        "Stage 2 only pretraining with imputation",
    ),
    "pretrain_impute_normalized": (
        "stgformer_pretrain_impute_normalized.yaml",
        "Masked node pretraining (5+5 curriculum) with imputation on NORMALIZED data",
    ),
    "pretrain_impute_normalized_lr": (
        "stgformer_pretrain_impute_normalized_lr0.0003.yaml",
        "Pretrain+impute normalized with lower fine-tuning LR (0.0003)",
    ),
    "pretrain_stage1only_normalized": (
        "stgformer_pretrain_stage1only_normalized.yaml",
        "Stage 1 only pretraining on NORMALIZED data (10 epochs per-timestep masking)",
    ),
    "pretrain_stage1only_normalized_impute": (
        "stgformer_pretrain_stage1only_normalized_impute.yaml",
        "Stage 1 only pretraining on NORMALIZED data with imputation (loads pretrained checkpoint)",
    ),
    "pretrain_stage2only_normalized": (
        "stgformer_pretrain_stage2only_normalized.yaml",
        "Stage 2 only pretraining on NORMALIZED data (10 epochs per-node masking)",
    ),
    "pretrain_stage2only_normalized_impute": (
        "stgformer_pretrain_stage2only_normalized_impute.yaml",
        "Stage 2 only pretraining on NORMALIZED data with imputation (loads pretrained checkpoint)",
    ),
    "cheb_tcn": (
        "stgformer_cheb_tcn.yaml",
        "Chebyshev propagation + TCN temporal mode",
    ),
    "cheb_tcn_exclude_missing": (
        "stgformer_cheb_tcn_exclude_missing.yaml",
        "Chebyshev+TCN excluding missing values from normalization",
    ),
    "cheb_tcn_xavier": (
        "stgformer_cheb_tcn_xavier.yaml",
        "Chebyshev+TCN with Xavier initialization",
    ),
    "cheb_tcn_dow": (
        "stgformer_cheb_tcn_dow.yaml",
        "Chebyshev+TCN with DOW embeddings",
    ),
    "cheb_tcn_xavier_dow": (
        "stgformer_cheb_tcn_xavier_dow.yaml",
        "Chebyshev+TCN with Xavier initialization and DOW embeddings",
    ),
    "cheb_tcn_xavier_dow_exclude_missing": (
        "stgformer_cheb_tcn_xavier_dow_exclude_missing.yaml",
        "Chebyshev+TCN+Xavier+DOW excluding missing values from normalization",
    ),
    "cheb_tcn_xavier_dow_stage1_norm_impute": (
        "stgformer_cheb_tcn_xavier_dow_stage1only_norm_impute.yaml",
        "Chebyshev+TCN+Xavier+DOW with stage1-only normalized pretraining + imputation",
    ),
    "cheb_tcn_xavier_dow_stage2_norm_impute": (
        "stgformer_cheb_tcn_xavier_dow_stage2only_norm_impute.yaml",
        "Chebyshev+TCN+Xavier+DOW with stage2-only normalized pretraining + imputation",
    ),
    "cheb_tcn_xavier_dow_pretrain_norm_impute": (
        "stgformer_cheb_tcn_xavier_dow_pretrain_norm_impute.yaml",
        "Chebyshev+TCN+Xavier+DOW with full normalized pretraining + imputation",
    ),
    "cheb_tcn_xavier_dow_exclude_missing_k8": (
        "stgformer_cheb_tcn_xavier_dow_exclude_missing_k8.yaml",
        "Chebyshev+TCN+Xavier+DOW excluding missing values, sparsity_k=8",
    ),
    "cheb_tcn_xavier_dow_exclude_missing_k16": (
        "stgformer_cheb_tcn_xavier_dow_exclude_missing_k16.yaml",
        "Chebyshev+TCN+Xavier+DOW excluding missing values, sparsity_k=16 [FINAL ARCHITECTURE]",
    ),
    "final_full": (
        "stgformer_cheb_tcn_xavier_dow_exclude_missing_k16_full.yaml",
        "FINAL ARCHITECTURE - Full 100-epoch training (Cheb+TCN+Xavier+DOW+ExcludeMissing+K16)",
    ),
    "baseline_dow": (
        "stgformer_baseline_dow.yaml",
        "Baseline with DOW embeddings, 100 epochs (comparable setup to final model)",
    ),
    "ablation_no_xavier": (
        "stgformer_cheb_tcn_dow_exclude_missing_k16.yaml",
        "w/o Xavier initialization",
    ),
    "ablation_no_dow": (
        "stgformer_cheb_tcn_xavier_exclude_missing_k16.yaml",
        "w/o DOW embeddings",
    ),
    "ablation_no_exclude_missing": (
        "stgformer_cheb_tcn_xavier_dow_k16.yaml",
        "w/o ExcludeMissing normalization",
    ),
    "ablation_no_tcn": (
        "stgformer_cheb_xavier_dow_exclude_missing_k16.yaml",
        "w/o TCN (use Transformer instead)",
    ),
    "ablation_no_cheb": (
        "stgformer_tcn_xavier_dow_exclude_missing_k16.yaml",
        "w/o Chebyshev (use standard graph convolution instead)",
    ),
    "ablation_no_graph": (
        "stgformer_cheb_tcn_xavier_dow_exclude_missing_k16_no_graph.yaml",
        "w/o graph propagation (GraphMode.NONE)",
    ),
    "ablation_no_temporal": (
        "stgformer_cheb_tcn_xavier_dow_exclude_missing_k16_no_temporal.yaml",
        "w/o temporal processing (TemporalMode.NONE)",
    ),
    "test_e2e": (
        "stgformer_test_e2e.yaml",
        "E2E test: pretrain->save->impute->train (1 epoch each, METR-LA only)",
    ),
}

# Experiments that require CUDA-only dependencies (mamba-ssm)
CUDA_ONLY_EXPERIMENTS = {"mamba", "mamba2", "mamba_fast"}

# Named experiment subsets for convenience when computing metrics.
# Extend or modify these lists to fit the workflows you care about.
EXPERIMENT_SUBSETS = {
    "initial": ["bs200_short", "bs200_xavier", "bs200_dow", "bs200_exclude_missing"],
    "spatial": [
        "bs200_short",
        "geo",
        "hybrid",
        "spectral_init",
        "chebyshev",
        "geo_cheb",
        "hybrid_cheb",
    ],
    "temporal": [
        "bs200_short",
        "mlp",
        "tcn",
        "depthwise",
        "mamba_fast",
        "mamba",
    ],
    "pretraining": [
        "bs200_short",
        "pretrain",
        "pretrain_impute",
        "pretrain_impute_normalized",
        "pretrain_impute_normalized_lr",
        "pretrain_stage1only",
        "pretrain_stage1only_impute",
        "pretrain_stage1only_normalized",
        "pretrain_stage1only_normalized_impute",
        "pretrain_stage2only",
        "pretrain_stage2only_impute",
        "pretrain_stage2only_normalized",
        "pretrain_stage2only_normalized_impute",
    ],
    "cheb_tcn_extensions": [
        "cheb_tcn",
        "cheb_tcn_exclude_missing",
        "cheb_tcn_xavier",
        "cheb_tcn_dow",
        "cheb_tcn_xavier_dow",
        "cheb_tcn_xavier_dow_exclude_missing",
        "cheb_tcn_xavier_dow_exclude_missing_k8",
        "cheb_tcn_xavier_dow_exclude_missing_k16",
        "cheb_tcn_xavier_dow_stage1_norm_impute",
        "cheb_tcn_xavier_dow_stage2_norm_impute",
        "cheb_tcn_xavier_dow_pretrain_norm_impute",
    ],
    "ablation": [
        "cheb_tcn_xavier_dow_exclude_missing_k16",  # FULL MODEL (baseline for comparison)
        "cheb_tcn_xavier_dow_exclude_missing",  # w/o Sparsity (K=∞)
        "ablation_no_dow",  # w/o DOW
        "ablation_no_exclude_missing",  # w/o ExcludeMissing
        "ablation_no_tcn",  # w/o TCN (use transformer)
        "ablation_no_cheb",  # w/o Chebyshev (use standard graph conv)
        "ablation_no_graph",  # w/o Graph (GraphMode.NONE)
        "ablation_no_temporal",
    ],
}


# Pending experiments to run
PENDING_EXPERIMENTS = []

# Long running or non-crucial experiments to run overnight
OVERNIGHT_EXPERIMENTS = []


def _list_experiments():
    """Show available experiments."""
    print("Available experiments:")
    for key, (_, desc) in EXPERIMENTS.items():
        print(f"  {key}: {desc}")
    print(
        "\nUsage: invoke train-experiment --name <experiment> [--dry-run] [--dataset X]"
    )


@task
def train_experiment(
    ctx, name="list", dry_run=False, push_to_hub=False, verbose=False, dataset=None
):
    """Train a specific STGFormer experiment.

    Available experiments: bs200_short, dow, geo, spectral_init, chebyshev, hybrid,
    geo_cheb, hybrid_cheb, mamba, mamba_fast, tcn, depthwise, mlp,
    pretrain, pretrain_impute, pretrain_stage1only, pretrain_stage1only_impute,
    pretrain_stage2only, pretrain_stage2only_impute

    Examples:
        invoke train-experiment --name bs200_short
        invoke train-experiment --name depthwise --dry-run
        invoke train-experiment --name hybrid --dataset METR-LA --push-to-hub
    """
    if name == "list":
        _list_experiments()
        return

    if name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{name}'. Available: {', '.join(EXPERIMENTS.keys())}"
        )

    if name in CUDA_ONLY_EXPERIMENTS and not _CUDA_AVAILABLE:
        print(
            f"Skipping experiment '{name}': requires CUDA-enabled GPU but none detected."
        )
        return

    config_file, desc = EXPERIMENTS[name]
    print(f"Running experiment: {name} ({desc})")

    cmd = _build_cmd(
        f"uv run python scripts/train_stgformer.py --config configs/{config_file}",
        dry_run=dry_run,
        push_to_hub=push_to_hub,
        verbose=verbose,
        dataset=dataset,
    )
    ctx.run(cmd, pty=True)


@task
def train_all_experiments(ctx, dry_run=False, push_to_hub=False, verbose=False):
    """Train all STGFormer experiment variants (bs200, DOW, geographic, hybrid, etc.)."""
    _train_experiments_from_list(
        ctx=ctx,
        experiment_list=EXPERIMENTS.keys(),
        dry_run=dry_run,
        push_to_hub=push_to_hub,
        verbose=verbose,
    )


@task(name="train-pending")
def train_pending_experiments(ctx, dry_run=False, push_to_hub=False, verbose=False):
    _train_experiments_from_list(
        ctx=ctx,
        experiment_list=PENDING_EXPERIMENTS,
        dry_run=dry_run,
        push_to_hub=push_to_hub,
        verbose=verbose,
    )


@task(name="train-overnight")
def train_overnight_experiments(ctx, dry_run=False, push_to_hub=False, verbose=False):
    _train_experiments_from_list(
        ctx=ctx,
        experiment_list=PENDING_EXPERIMENTS,
        dry_run=dry_run,
        push_to_hub=push_to_hub,
        verbose=verbose,
    )


def _train_experiments_from_list(
    ctx, experiment_list, dry_run=False, push_to_hub=False, verbose=False
):
    """Train all experiments specified in `experiment_list`.

    Continues to next experiment if one fails. Reports summary at the end.
    Use for overnight/batch runs.
    """
    succeeded = []
    failed = []
    skipped = []

    for name in experiment_list:
        if name not in EXPERIMENTS:
            print(f"\nUnknown experiment '{name}', skipping...")
            failed.append((name, "Unknown experiment"))
            continue

        if name in CUDA_ONLY_EXPERIMENTS and not _CUDA_AVAILABLE:
            msg = "Requires CUDA-enabled GPU; skipping on CPU-only host"
            print(f"\nSkipping experiment '{name}': {msg}")
            skipped.append((name, msg))
            continue

        config_file, desc = EXPERIMENTS[name]
        print(f"\n{'=' * 60}")
        print(f"Running experiment: {name} ({desc})")
        print(f"{'=' * 60}\n")

        try:
            cmd = _build_cmd(
                f"uv run python scripts/train_stgformer.py --config configs/{config_file}",
                dry_run=dry_run,
                push_to_hub=push_to_hub,
                verbose=verbose,
            )
            # Capture output to analyze errors
            result = ctx.run(cmd, pty=False, warn=True, hide=False)
            if result.ok:
                succeeded.append(name)
            else:
                # Analyze actual error from output
                output = (result.stdout or "") + (result.stderr or "")

                if "ImportError" in output and "mamba" in output.lower():
                    error_msg = "Missing CUDA dependencies (run: uv sync --extra cuda)"
                elif "CUDA" in output or "nvcc" in output:
                    error_msg = (
                        "CUDA-related error (check if CUDA toolkit is installed)"
                    )
                elif "ImportError" in output:
                    # Extract the module name if possible
                    import_match = output.split("ImportError")[-1].split("\n")[0]
                    error_msg = f"Import error:{import_match}"
                elif "RuntimeError" in output or "ValueError" in output:
                    # Try to extract the actual error message
                    for line in output.split("\n"):
                        if "Error:" in line or "Exception:" in line:
                            error_msg = line.strip()
                            break
                    else:
                        error_msg = f"Exit code {result.return_code} (see output above)"
                else:
                    error_msg = f"Exit code {result.return_code} (see output above)"

                print(f"\nExperiment '{name}' failed: {error_msg}")
                failed.append((name, error_msg))
        except Exception as e:
            # Fallback for unexpected errors (not command failures)
            error_msg = f"Unexpected error: {type(e).__name__}"
            print(f"\nExperiment '{name}' failed: {error_msg}")
            failed.append((name, error_msg))
            continue

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"Succeeded ({len(succeeded)}): {', '.join(succeeded) if succeeded else 'none'}"
    )
    if failed:
        print(f"FAILED ({len(failed)}):")
        for name, error in failed:
            print(f"   - {name}: {error}")
    if skipped:
        print(f"SKIPPED ({len(skipped)}):")
        for name, reason in skipped:
            print(f"   - {name}: {reason}")


@task
def train_internal_models(ctx, dry_run=False, push_to_hub=False, verbose=False):
    """Train all variants of the models implemented and explored directly in this repo"""
    train_stgformer_internal(
        ctx, dry_run=dry_run, push_to_hub=push_to_hub, verbose=verbose
    )
    train_all_experiments(
        ctx, dry_run=dry_run, push_to_hub=push_to_hub, verbose=verbose
    )


@task
def train_baselines(ctx, dry_run=False, push_to_hub=False, verbose=False):
    """Train all baselines models from scratch. Use --verbose to see training progress."""
    train_dcrnn(ctx, dry_run=dry_run, push_to_hub=push_to_hub)
    train_mtgnn(ctx, dry_run=dry_run, push_to_hub=push_to_hub, verbose=verbose)
    train_gwnet(ctx, dry_run=dry_run, push_to_hub=push_to_hub, verbose=verbose)
    train_stgformer_external(
        ctx, dry_run=dry_run, push_to_hub=push_to_hub, verbose=verbose
    )


@task
def train_all(ctx, dry_run=False, push_to_hub=False, verbose=False):
    """Rerun training of all baselines and models considered in this work"""
    train_baselines(ctx, dry_run=dry_run, push_to_hub=push_to_hub, verbose=verbose)
    train_internal_models(
        ctx, dry_run=dry_run, push_to_hub=push_to_hub, verbose=verbose
    )


@task
def compute_results(ctx, force=False, subset=None, highlight=False, backup=False):
    """Compute metrics for STGFormer internal baseline and all experiments.

    Reads experiment configs from configs/ to get hf_repo_prefix for each experiment.

    Args:
        force: Recompute metrics even if cached.
        backup: Only back up current results files without recomputing metrics.
        subset: Optional experiment subset key defined in EXPERIMENT_SUBSETS.
        highlight: Bold the lowest value per metric when printing tables.
    """
    from pathlib import Path

    import yaml

    from utils.results import (
        backup_results,
        calculate_experiment_metrics,
        format_results_table,
        get_results_file,
    )

    datasets = ["METR-LA", "PEMS-BAY"]

    if subset:
        try:
            experiment_names = EXPERIMENT_SUBSETS[subset]
        except KeyError:
            raise ValueError(
                f"Unknown subset '{subset}'. Available: {', '.join(EXPERIMENT_SUBSETS)}"
            ) from None
    else:
        # Central exclusion list for experiments to skip when computing results without predefined subsets.
        # Update this set to change which experiments are ignored.
        exclusion_list = {"test_e2e"}

        # Default to all experiments minus any excluded ones.
        experiment_names = [
            name for name in EXPERIMENTS.keys() if name not in exclusion_list
        ]

    # Read hf_repo_prefix from each experiment config in EXPERIMENTS (or subset)
    model_prefixes: dict[str, str] = {}
    for exp_name in experiment_names:
        config_file, _ = EXPERIMENTS[exp_name]
        config_path = Path("configs") / config_file
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                prefix = config.get("output", {}).get("hf_repo_prefix")
                if prefix:
                    model_prefixes[prefix] = exp_name
        else:
            print(
                f"Warning: config file missing for experiment '{exp_name}' ({config_path}). Skipping."
            )

    if model_prefixes:
        header = "Experiment descriptions"
        if subset:
            header += f" (subset '{subset}')"
        print(header + ":")
        for model_prefix, exp_name in model_prefixes.items():
            _, desc = EXPERIMENTS.get(exp_name, (None, None))
            if desc:
                print(f"  - {model_prefix}: {desc}")
            else:
                print(f"  - {model_prefix}: <no description>")

    def _result_exists(dataset_name: str, model_name: str) -> bool:
        results_file = get_results_file(dataset_name)
        if not results_file.exists():
            return False
        try:
            with open(results_file, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row.get("model") == model_name.upper():
                        return True
        except Exception:
            return False
        return False

    for dataset in datasets:
        print(f"\n{'=' * 60}")
        print(f"Computing metrics for {dataset}")
        print(f"{'=' * 60}\n")

        if backup or force:
            backup_file = backup_results(dataset)
            if backup_file:
                print(f"Backed up to {backup_file.name}")

        if backup:
            print("Backup-only flag set, skipping metric computation.")
            continue

        for model_prefix, exp_name in model_prefixes.items():
            needs_compute = force or not _result_exists(dataset, model_prefix)
            if (
                needs_compute
                and exp_name in CUDA_ONLY_EXPERIMENTS
                and not _CUDA_AVAILABLE
            ):
                print(
                    f"{model_prefix}... skipped (requires CUDA-enabled GPU to recompute)"
                )
                continue

            print(f"{model_prefix}...", end=" ", flush=True)
            try:
                calculate_experiment_metrics(dataset, model_prefix, force=force)
                print("done")
            except Exception as e:
                print(f"FAILED: {e}")

        print(f"\n{dataset} Results:")
        print(
            format_results_table(
                dataset,
                model_order=list(model_prefixes.keys()),
                highlight_best=highlight,
            )
        )


@task
def compile_experiment_results(
    ctx, subsets=None, output_path="docs/EXPERIMENT_RESULTS.md", top_n=None
):
    """Generate a markdown report of experiment results by subset and dataset.

    Args:
        subsets: Comma-separated list of experiment subsets to include
        output_path: Path to output markdown file
        top_n: If specified, keep only top N models per subset (by avg MAE across horizons).
               Always keeps the first model (baseline). Appends '_TRIMMED' to filename.
    """
    from pathlib import Path

    import pandas as pd
    import yaml

    from utils.results import format_results_table, get_results_file

    datasets = ["METR-LA", "PEMS-BAY"]

    if subsets:
        subset_names = [s.strip() for s in subsets.split(",") if s.strip()]
    else:
        subset_names = list(EXPERIMENT_SUBSETS.keys())

    # Validate subsets
    for name in subset_names:
        if name not in EXPERIMENT_SUBSETS:
            raise ValueError(
                f"Unknown subset '{name}'. Available: {', '.join(EXPERIMENT_SUBSETS)}"
            )

    def _get_model_order(exp_names):
        prefixes = []
        descriptions = {}
        for exp_name in exp_names:
            config_file, desc = EXPERIMENTS.get(exp_name, (None, None))
            if not config_file:
                continue
            config_path = Path("configs") / config_file
            if not config_path.exists():
                print(
                    f"Warning: config file missing for experiment '{exp_name}' ({config_path}). Skipping."
                )
                continue
            with open(config_path) as f:
                config = yaml.safe_load(f)
            prefix = config.get("output", {}).get("hf_repo_prefix")
            if prefix:
                prefixes.append(prefix)
                descriptions[prefix] = desc or exp_name
        return prefixes, descriptions

    def _filter_top_n(model_order, dataset, n):
        """Filter to top N models by average MAE, always keeping the first (baseline)."""
        if not n or len(model_order) <= n:
            return model_order

        results_file = get_results_file(dataset)
        if not results_file.exists():
            print(f"Warning: No results file for {dataset}, cannot filter by top_n")
            return model_order

        df = pd.read_csv(results_file)
        # Calculate average MAE across 3 horizons (column names have spaces)
        df["avg_mae"] = df[["mae_15 min", "mae_30 min", "mae_1 hour"]].mean(axis=1)

        # Always keep first model (baseline)
        baseline = model_order[0]
        rest = model_order[1:]

        # Filter rest by avg MAE
        rest_df = df[df["model"].isin(rest)].copy()
        rest_df = rest_df.sort_values("avg_mae")
        top_rest = rest_df.head(n - 1)["model"].tolist()

        return [baseline] + top_rest

    # Adjust output path if using top_n
    output = Path(output_path)
    if top_n:
        top_n = int(top_n)
        stem = output.stem
        suffix = output.suffix
        output = output.with_name(f"{stem}_TRIMMED{suffix}")

    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Experiment Results",
        "",
    ]

    if top_n:
        lines.insert(
            1,
            f"**Note:** Showing top {top_n} models per subset (by avg MAE across horizons). Baseline always included.",
        )
        lines.insert(2, "")

    for subset_name in subset_names:
        exp_names = EXPERIMENT_SUBSETS[subset_name]
        model_order, desc_map = _get_model_order(exp_names)
        if not model_order:
            continue

        lines.append(f"## {subset_name.replace('_', ' ').title()}")
        lines.append("")
        for dataset in datasets:
            # Filter to top N if requested
            filtered_order = (
                _filter_top_n(model_order, dataset, top_n) if top_n else model_order
            )

            lines.append(f"### {dataset}")
            lines.append("")
            lines.append("**Experiment descriptions**")
            for prefix in filtered_order:
                desc = desc_map.get(prefix, "<no description>")
                lines.append(f"- `{prefix}`: {desc}")
            lines.append("")
            table = format_results_table(
                dataset, model_order=filtered_order, highlight_best=True
            )
            lines.append(table)
            lines.append("")

    output.write_text("\n".join(lines))
    print(f"Wrote experiment results to {output}")


@task
def ablation_study(ctx, subset="ablation", output_path="docs/ABLATION_RESULTS.md"):
    """Generate ablation study results comparing variants against the full model.

    The first experiment in the subset is used as the baseline (full model).
    Results show the delta (difference) from the baseline for each ablation.

    Usage:
        invoke ablation-study
        invoke ablation-study --subset ablation --output-path docs/my_ablation.md
    """
    from pathlib import Path

    import pandas as pd
    import yaml

    from utils.results import get_results_file

    datasets = ["METR-LA", "PEMS-BAY"]

    if subset not in EXPERIMENT_SUBSETS:
        raise ValueError(
            f"Unknown subset '{subset}'. Available: {', '.join(EXPERIMENT_SUBSETS)}"
        )

    exp_names = EXPERIMENT_SUBSETS[subset]
    if len(exp_names) < 2:
        raise ValueError("Ablation study requires at least 2 experiments in subset")

    # Map experiment names to model prefixes and get descriptions
    def _get_model_info(exp_names):
        info = []
        for exp_name in exp_names:
            config_file, desc = EXPERIMENTS.get(exp_name, (None, None))
            if not config_file:
                continue
            config_path = Path("configs") / config_file
            if not config_path.exists():
                print(f"Warning: config missing for '{exp_name}'. Skipping.")
                continue
            with open(config_path) as f:
                config = yaml.safe_load(f)
            prefix = config.get("output", {}).get("hf_repo_prefix")
            if prefix:
                info.append((exp_name, prefix, desc or exp_name))
        return info

    model_info = _get_model_info(exp_names)
    if len(model_info) < 2:
        raise ValueError("Could not find enough valid experiments for ablation study")

    # Default column names based on removing components from full model
    default_column_names = {
        "cheb_tcn_xavier_dow_exclude_missing_k16": "Full Model",
        "cheb_tcn_xavier_dow_exclude_missing": "w/o Sparsity (K=∞)",
        "ablation_no_dow": "w/o DOW",
        "ablation_no_exclude_missing": "w/o ExcludeMissing",
        "ablation_no_tcn": "w/o TCN",
        "ablation_no_cheb": "w/o Cheb",
    }

    baseline_exp, baseline_prefix, baseline_desc = model_info[0]
    ablation_models = model_info[1:]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Ablation Study Results",
        "",
        f"**Baseline (Full Model):** {default_column_names.get(baseline_exp, baseline_desc)}",
        "",
        "This table shows the **delta** (difference) from the baseline for each ablation.",
        "Negative values (green) indicate improvement over baseline, positive values (red) indicate degradation.",
        "",
    ]

    for dataset in datasets:
        lines.append(f"## {dataset}")
        lines.append("")

        # Load results CSV
        results_file = get_results_file(dataset)
        if not results_file.exists():
            lines.append(f"**Error:** Results file not found: {results_file}")
            lines.append("")
            continue

        df = pd.read_csv(results_file)

        # Get baseline results
        baseline_row = df[df["model"] == baseline_prefix.upper()]
        if baseline_row.empty:
            lines.append(f"**Error:** Baseline results not found for {baseline_prefix}")
            lines.append("")
            continue
        baseline_row = baseline_row.iloc[0]

        # Build comparison table
        horizons = [15, 30, 60]
        metrics = ["MAE", "RMSE", "MAPE"]

        # Table header
        header_row = ["T", "Metric", default_column_names.get(baseline_exp, "Baseline")]
        for exp_name, prefix, desc in ablation_models:
            col_name = default_column_names.get(exp_name, desc)
            header_row.append(col_name)

        lines.append("| " + " | ".join(header_row) + " |")
        lines.append(
            "|:"
            + "|:".join(["-" * max(6, len(h)) for h in header_row[:-1]])
            + "|"
            + "-" * max(6, len(header_row[-1]))
            + "|"
        )

        # Table rows
        for horizon in horizons:
            horizon_label = f"{horizon} min" if horizon < 60 else "1 hour"
            col_suffix = f"_{horizon} min" if horizon < 60 else "_1 hour"

            for i, metric in enumerate(metrics):
                col_name = f"{metric.lower()}{col_suffix}"
                baseline_val = baseline_row.get(col_name)
                if pd.isna(baseline_val):
                    continue

                # Format baseline value
                if metric == "MAPE":
                    baseline_str = f"{baseline_val:.3f}%"
                else:
                    baseline_str = f"{baseline_val:.3f}"

                row = []
                if i == 0:
                    row.append(horizon_label)
                else:
                    row.append("")
                row.append(metric)
                row.append(baseline_str)

                # Add delta for each ablation
                for exp_name, prefix, desc in ablation_models:
                    model_row = df[df["model"] == prefix.upper()]
                    if model_row.empty:
                        row.append("N/A")
                        continue

                    val = model_row.iloc[0].get(col_name)
                    if pd.isna(val):
                        row.append("N/A")
                        continue

                    delta = val - baseline_val
                    if metric == "MAPE":
                        delta_str = f"{delta:+.3f}%"
                    else:
                        delta_str = f"{delta:+.3f}"

                    row.append(delta_str)

                lines.append("| " + " | ".join(row) + " |")

            # Spacing between horizons
            if horizon != horizons[-1]:
                lines.append("| | | " + " | ".join([""] * len(ablation_models)) + " |")

        lines.append("")

    output.write_text("\n".join(lines))
    print(f"Wrote ablation study results to {output}")
