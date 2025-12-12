"""Utilities for internal STGFormer model: training, inference, and HuggingFace Hub integration.

This module wraps the INTERNAL STGFormer implementation from stgformer/model.py.
It mirrors the interface of utils/stgformer_external.py for easy comparison.
"""

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from stgformer.model import GraphMode, PropagationMode, STGFormer, TemporalMode
from stgformer.pretrain import impute_missing_data, pretrain_graph_imputation
from utils.config import validate_dataset_name
from utils.config_classes import (
    GraphConfig,
    ModelArchConfig,
    PretrainConfig,
    TemporalConfig,
    TrainingConfig,
)
from utils.dataset import TrafficDataset
from utils.hub import fetch_model_from_hub, get_best_device, push_model_to_hub
from utils.training import MaskedHuberLoss, StandardScaler, masked_mae_loss

# Suppress torch.compile inductor warning (internal to PyTorch, can't be fixed)
warnings.filterwarnings("ignore", message=".*Online softmax is disabled.*")


def train_model(
    dataset_name: str,
    pytorch_datasets: Dict[str, TrafficDataset],
    training_config: Optional[TrainingConfig] = None,
    graph_config: Optional[GraphConfig] = None,
    temporal_config: Optional[TemporalConfig] = None,
    model_arch_config: Optional[ModelArchConfig] = None,
    pretrain_config_obj: Optional[PretrainConfig] = None,
    propagation_mode: Union[str, PropagationMode] = "power",
    save_dir: Optional[Path] = None,
    wandb_run: Optional[object] = None,
    use_imputation: bool = False,
    use_zero_init: bool = True,
    exclude_missing_from_norm: bool = False,
    freeze_encoder_epochs: int = 0,
) -> tuple[STGFormer, StandardScaler, bool]:
    """
    Train internal STGFormer model using configuration objects.

    Args:
        dataset_name: Dataset name (e.g., "METR-LA", "PEMS-BAY")
        pytorch_datasets: Dict[str, TrafficDataset] with 'train', 'val', 'test' splits
        training_config: Training hyperparameters (TrainingConfig), uses defaults if None
        graph_config: Graph structure config (GraphConfig), uses defaults if None
        temporal_config: Temporal processing config (TemporalConfig), uses defaults if None
        model_arch_config: Optional model architecture overrides (ModelArchConfig)
        pretrain_config_obj: Optional pretraining config (PretrainConfig)
        propagation_mode: Propagation mode ("power" or "chebyshev")
        save_dir: Directory to save model (optional)
        wandb_run: Optional wandb Run object for logging
        use_imputation: If True, impute missing values using pretrained imputation head
        use_zero_init: Zero-initialize order_proj weights (default: True)
        freeze_encoder_epochs: Freeze encoder for first N epochs (only train output_proj)

    Returns:
        tuple: (model, scaler, data_normalized) where:
            - model is trained STGFormer
            - scaler is StandardScaler
            - data_normalized is True if test data is already normalized, False otherwise

    Example:
        >>> training_cfg = TrainingConfig(epochs=100, batch_size=64)
        >>> graph_cfg = GraphConfig(graph_mode="hybrid", geo_adj=adj_matrix)
        >>> temporal_cfg = TemporalConfig(temporal_mode="tcn")
        >>> model, scaler, data_normalized = train_model(
        ...     "METR-LA",
        ...     datasets,
        ...     training_config=training_cfg,
        ...     graph_config=graph_cfg,
        ...     temporal_config=temporal_cfg,
        ... )
    """
    # Use defaults if config objects not provided
    if training_config is None:
        training_config = TrainingConfig()
    if graph_config is None:
        graph_config = GraphConfig()
    if temporal_config is None:
        temporal_config = TemporalConfig()

    # Extract values from config objects for use in function body
    epochs = training_config.epochs
    batch_size = training_config.batch_size
    learning_rate = training_config.learning_rate
    weight_decay = training_config.weight_decay
    early_stop = training_config.early_stop
    milestones = training_config.milestones
    lr_decay_rate = training_config.lr_decay_rate
    clip_grad = training_config.clip_grad
    seed = training_config.seed
    verbose = training_config.verbose
    device = training_config.device
    use_torch_compile = training_config.use_torch_compile

    graph_mode = graph_config.graph_mode
    geo_adj = graph_config.geo_adj
    lambda_hybrid = graph_config.lambda_hybrid
    sparsity_k = graph_config.sparsity_k

    temporal_mode = temporal_config.temporal_mode
    mamba_d_state = temporal_config.mamba_d_state
    mamba_d_conv = temporal_config.mamba_d_conv
    mamba_expand = temporal_config.mamba_expand
    tcn_num_layers = temporal_config.tcn_num_layers
    tcn_kernel_size = temporal_config.tcn_kernel_size
    tcn_dilation_base = temporal_config.tcn_dilation_base
    tcn_dropout = temporal_config.tcn_dropout
    depthwise_kernel_size = temporal_config.depthwise_kernel_size
    mlp_hidden_dim = temporal_config.mlp_hidden_dim

    # Convert config objects to dicts for internal use
    model_config = model_arch_config.to_dict() if model_arch_config else None
    pretrain_config = pretrain_config_obj.to_dict() if pretrain_config_obj else None

    dataset_name = validate_dataset_name(dataset_name)

    # Set random seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # For fully deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Match external implementation
    if milestones is None:
        milestones = [20, 30]

    # Get dataset properties
    train_ds = pytorch_datasets["train"]
    num_nodes = train_ds.num_nodes
    seq_len = train_ds.seq_len
    horizon = train_ds.horizon
    dataset_input_dim = train_ds.input_dim

    # Auto-detect device
    if device is None:
        device = get_best_device()

    # Enable TF32 for faster matmuls on Ampere+ GPUs (negligible precision loss)
    # Must set BOTH old and new APIs for torch.compile compatibility:
    # - New API (PyTorch 2.9+): fp32_precision = "tf32"
    # - Old API: allow_tf32 = True (still used internally by torch.compile/inductor)
    if torch.cuda.is_available():
        # New API
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        # Old API (required for torch.compile inductor backend)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Determine if we can use AMP (only on CUDA)
    use_amp = device == "cuda" and torch.cuda.is_available()

    device = torch.device(device)

    # Build model config (defaults + overrides)
    default_model_config = {
        "input_embedding_dim": 24,
        "tod_embedding_dim": 0,  # 0 = disabled, 24 = enabled
        "dow_embedding_dim": 0,  # 0 = disabled, 24 = enabled
        "spatial_embedding_dim": 0,
        "adaptive_embedding_dim": 80,
        "num_heads": 4,
        "num_layers": 3,
        "dropout": 0.1,
        "dropout_a": 0.3,
        "mlp_ratio": 4,
        "use_mixed_proj": True,
        "pre_attn_kernel_size": 1,
        "prediction_head_layers": 1,  # 1 = linear (default), 2+ = MLP
    }
    if model_config:
        default_model_config.update(model_config)
    cfg = default_model_config

    # input_dim is always 1 (speed only goes through input projection)
    # TOD and DOW are handled via separate embedding layers
    input_dim = 1

    # Determine how many features we need from the dataset
    needs_tod = cfg["tod_embedding_dim"] > 0
    needs_dow = cfg["dow_embedding_dim"] > 0
    required_dataset_dim = input_dim + (1 if needs_tod else 0) + (1 if needs_dow else 0)

    if dataset_input_dim < required_dataset_dim:
        raise ValueError(
            f"Dataset has {dataset_input_dim} features but model config requires {required_dataset_dim} "
            f"(input_dim={input_dim}, tod={needs_tod}, dow={needs_dow})"
        )

    if verbose:
        print(f"Training internal STGFormer on {dataset_name}...")
        print(f"  Nodes: {num_nodes}")
        print(f"  Input sequence length: {seq_len}")
        print(f"  Output sequence length: {horizon}")
        print(
            f"  TOD embedding: {cfg['tod_embedding_dim']} ({'enabled' if needs_tod else 'disabled'})"
        )
        print(
            f"  DOW embedding: {cfg['dow_embedding_dim']} ({'enabled' if needs_dow else 'disabled'})"
        )
        print(f"  Graph mode: {graph_mode}")
        print(
            f"  Graph init: {'spectral (from geo_adj)' if graph_mode == 'spectral_init' else 'xavier uniform'}"
        )
        if sparsity_k is not None:
            print(f"  Graph sparsity_k: {sparsity_k}")
        print(f"  Graph propagation: {propagation_mode}")
        print(f"  Temporal mode: {temporal_mode}")
        print(f"  Pre-attention kernel size: {cfg['pre_attn_kernel_size']}")

        # NaN diagnostics for data quality (batch processing on GPU for efficiency)
        train_x = pytorch_datasets["train"].x
        train_y = pytorch_datasets["train"].y
        batch_size_diag = 256
        x_nan_count = 0
        y_nan_count = 0
        # Track which nodes have any non-NaN value
        nodes_have_valid = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        for i in range(0, len(train_x), batch_size_diag):
            x_batch = train_x[i:i + batch_size_diag].to(device)
            y_batch = train_y[i:i + batch_size_diag].to(device)
            x_nan_count += torch.isnan(x_batch).sum().item()
            y_nan_count += torch.isnan(y_batch).sum().item()
            # Update nodes that have at least one valid value
            batch_valid = ~torch.isnan(x_batch[..., 0]).all(dim=(0, 1))
            nodes_have_valid |= batch_valid

        x_total = train_x.numel()
        y_total = train_y.numel()
        nodes_all_nan = (~nodes_have_valid).sum().item()
        print(f"  Data quality:")
        print(f"    X NaN: {x_nan_count:,} / {x_total:,} ({100*x_nan_count/x_total:.2f}%)")
        print(f"    Y NaN: {y_nan_count:,} / {y_total:,} ({100*y_nan_count/y_total:.2f}%)")
        if nodes_all_nan > 0:
            print(f"    Nodes with 100% NaN: {nodes_all_nan} (will be filled with 0)")

        # Detailed initialization info
        init_strategy = []
        if graph_mode == "spectral_init":
            init_strategy.append("graph from Laplacian eigenvectors")
        else:
            init_strategy.append("graph xavier uniform")

        if use_zero_init:
            init_strategy.append("order_proj zero")
        else:
            init_strategy.append("order_proj xavier")

        init_strategy.append("other layers pytorch default")

        print(f"  Initialization: {', '.join(init_strategy)}")
        print(f"  Seed: {seed if seed is not None else 'None (non-deterministic)'}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {device}")
        if use_amp:
            print("  Using: AMP (mixed precision)")

    # Convert graph_mode string to enum (case-insensitive)
    graph_mode_enum = (
        GraphMode(graph_mode.lower()) if isinstance(graph_mode, str) else graph_mode
    )

    # Convert temporal_mode string to enum (case-insensitive)
    temporal_mode_enum = (
        TemporalMode(temporal_mode.lower())
        if isinstance(temporal_mode, str)
        else temporal_mode
    )

    # Convert propagation_mode string to enum (case-insensitive)
    propagation_mode_enum = (
        PropagationMode(propagation_mode.lower())
        if isinstance(propagation_mode, str)
        else propagation_mode
    )

    # Convert geo_adj to tensor if provided
    geo_adj_tensor = None
    if geo_adj is not None:
        geo_adj_tensor = torch.tensor(geo_adj, dtype=torch.float32)

    # Create model
    model = STGFormer(
        num_nodes=num_nodes,
        in_steps=seq_len,
        out_steps=horizon,
        input_dim=input_dim,  # Always 1 (speed only)
        output_dim=1,
        steps_per_day=288,  # 5-min intervals
        input_embedding_dim=cfg["input_embedding_dim"],
        tod_embedding_dim=cfg["tod_embedding_dim"],
        dow_embedding_dim=cfg["dow_embedding_dim"],
        spatial_embedding_dim=cfg["spatial_embedding_dim"],
        adaptive_embedding_dim=cfg["adaptive_embedding_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        dropout_a=cfg["dropout_a"],
        mlp_ratio=cfg["mlp_ratio"],
        use_mixed_proj=cfg["use_mixed_proj"],
        pre_attn_kernel_size=cfg["pre_attn_kernel_size"],
        # Graph structure
        graph_mode=graph_mode_enum,
        geo_adj=geo_adj_tensor,
        lambda_hybrid=lambda_hybrid,
        sparsity_k=sparsity_k,
        # Propagation mode
        propagation_mode=propagation_mode_enum,
        # Temporal processing
        temporal_mode=temporal_mode_enum,
        mamba_d_state=mamba_d_state,
        mamba_d_conv=mamba_d_conv,
        mamba_expand=mamba_expand,
        tcn_num_layers=tcn_num_layers,
        tcn_kernel_size=tcn_kernel_size,
        tcn_dilation_base=tcn_dilation_base,
        tcn_dropout=tcn_dropout,
        depthwise_kernel_size=depthwise_kernel_size,
        mlp_hidden_dim=mlp_hidden_dim,
        # Initialization
        use_zero_init=use_zero_init,
        # Prediction head
        prediction_head_layers=cfg["prediction_head_layers"],
    ).to(device)

    # Count model parameters (always, for W&B logging)
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Model parameters: {total_params:,}")

    # === NORMALIZATION ===
    # Fit scaler once on training data (used for both pretraining and fine-tuning)
    train_x = pytorch_datasets["train"].x[..., 0:1]  # Only speed values
    scaler = StandardScaler()

    # Optionally exclude missing values (0.0) when fitting normalization statistics
    # This ensures mean ≈ 0 and std ≈ 1 for normalized data when missing values are present
    mask_value = 0.0 if exclude_missing_from_norm else None
    scaler.fit_transform(train_x, mask_value=mask_value)

    # === PRETRAINING PHASE ===
    # Pretraining can happen on raw or normalized data (configurable)
    imputation_head = None

    if pretrain_config is not None:
        # Check if we should load existing pretrained model
        if pretrain_config.get("load_from"):
            load_from_prefix = pretrain_config["load_from"]
            if verbose:
                print(f"\n{'=' * 40}")
                print("Loading Pretrained Model")
                print(f"{'=' * 40}")
                print(f"  Loading from: {load_from_prefix}")

            try:
                imputation_head = load_pretrained_from_hub(
                    hf_repo_prefix=load_from_prefix,
                    dataset_name=dataset_name,
                    model=model,
                    force_download=False,
                    device=device,
                    verbose=verbose,
                )
                if verbose:
                    print("Pretrained model loaded successfully")
                    print(f"{'=' * 40}\n")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load pretrained model from {load_from_prefix}: {e}"
                )

        stage1_epochs = pretrain_config.get("stage1_epochs", 5)
        stage2_epochs = pretrain_config.get("stage2_epochs", 5)

        # Only run pretraining if not loading from existing checkpoint
        if not pretrain_config.get("load_from") and (
            stage1_epochs > 0 or stage2_epochs > 0
        ):
            use_normalized_data = pretrain_config.get("use_normalized_data", False)

            # Get data fraction for pretraining (default: use all data)
            data_fraction = pretrain_config.get("pretrain_data_fraction", 1.0)

            if verbose:
                print(f"\n{'=' * 40}")
                print("Pretraining Phase")
                print(f"{'=' * 40}")
                print(
                    f"  Stage 1 (per-timestep): {stage1_epochs} epochs, mask={pretrain_config.get('stage1_mask_ratio', 0.15)}"
                )
                print(
                    f"  Stage 2 (per-node): {stage2_epochs} epochs, mask={pretrain_config.get('stage2_mask_ratio', 0.10)}"
                )
                print(
                    f"  Batch size: {pretrain_config.get('pretrain_batch_size', batch_size)}"
                )
                print(
                    f"  Data: {'normalized' if use_normalized_data else 'unnormalized (raw)'}"
                )
                print(f"  Data fraction: {data_fraction:.0%} of training samples")

            # Prepare data for pretraining (normalized or raw)
            pretrain_train_x = pytorch_datasets["train"].x.clone()
            pretrain_val_x = pytorch_datasets["val"].x.clone()

            # Optionally subsample training data for faster pretraining
            if data_fraction < 1.0:
                num_samples = pretrain_train_x.shape[0]
                num_keep = int(num_samples * data_fraction)
                # Random sample indices (deterministic with seed for reproducibility)
                generator = torch.Generator().manual_seed(42)
                indices = torch.randperm(num_samples, generator=generator)[:num_keep]
                pretrain_train_x = pretrain_train_x[indices]
                if verbose:
                    print(f"  Sampled {num_keep:,} / {num_samples:,} training samples")

            if use_normalized_data:
                # Use the main scaler for normalization
                pretrain_train_x[..., 0:1] = scaler.transform(
                    pretrain_train_x[..., 0:1]
                )
                pretrain_val_x[..., 0:1] = scaler.transform(pretrain_val_x[..., 0:1])

            # Replace NaN with 0 after normalization (NaN values would propagate through model)
            pretrain_train_x = torch.nan_to_num(pretrain_train_x, nan=0.0)
            pretrain_val_x = torch.nan_to_num(pretrain_val_x, nan=0.0)

            # Create dummy targets (zeros) to prevent data leakage
            # Pretraining only uses input x, but TensorDataset requires y
            dummy_train_y = torch.zeros_like(pretrain_train_x[..., 0:1])
            dummy_val_y = torch.zeros_like(pretrain_val_x[..., 0:1])

            pretrain_dataset = TensorDataset(pretrain_train_x, dummy_train_y)
            # Use pretrain_batch_size if specified, otherwise fall back to training batch_size
            pretrain_batch_size = pretrain_config.get("pretrain_batch_size", batch_size)
            pretrain_loader = DataLoader(
                pretrain_dataset,
                batch_size=pretrain_batch_size,
                shuffle=True,
            )

            # Create validation dataloader (for monitoring only, no gradient updates)
            pretrain_val_dataset = TensorDataset(pretrain_val_x, dummy_val_y)
            pretrain_val_loader = DataLoader(
                pretrain_val_dataset,
                batch_size=pretrain_batch_size,
                shuffle=False,  # No shuffling for validation
            )

            # Run pretraining
            model, imputation_head = pretrain_graph_imputation(
                model=model,
                train_loader=pretrain_loader,
                val_loader=pretrain_val_loader,
                stage1_epochs=stage1_epochs,
                stage1_mask_ratio=pretrain_config.get("stage1_mask_ratio", 0.15),
                stage2_epochs=stage2_epochs,
                stage2_mask_ratio=pretrain_config.get("stage2_mask_ratio", 0.10),
                learning_rate=pretrain_config.get("learning_rate", 0.001),
                device=device,
                verbose=verbose,
                wandb_run=wandb_run,
            )

            if verbose:
                print("Pretraining complete.")
                print(f"{'=' * 40}\n")

            # Save pretrained checkpoint if save_to is specified
            if pretrain_config.get("save_to"):
                save_to_prefix = pretrain_config["save_to"]
                if verbose:
                    print("Saving pretrained checkpoint to HuggingFace Hub...")

                # Save to local checkpoint dir first
                pretrain_checkpoint_dir = (
                    save_dir / "pretrained_checkpoint"
                    if save_dir
                    else Path("pretrained_checkpoint")
                )
                save_pretrained_checkpoint(
                    model=model,
                    imputation_head=imputation_head,
                    checkpoint_dir=pretrain_checkpoint_dir,
                    pretrain_config=pretrain_config,
                    dataset_name=dataset_name,
                )

                # Push to HuggingFace Hub
                try:
                    url = push_pretrained_to_hub(
                        checkpoint_dir=pretrain_checkpoint_dir,
                        hf_repo_prefix=save_to_prefix,
                        dataset_name=dataset_name,
                        private=False,
                    )
                    if verbose:
                        print(f"Pretrained checkpoint uploaded to: {url}")
                except Exception as e:
                    print(f"Warning: Failed to upload pretrained checkpoint: {e}")

    # === IMPUTATION PHASE ===
    # If use_imputation and we have a trained imputation head, impute missing values
    # Only x tensors are modified; targets (y) remain untouched to avoid leakage.
    training_data_normalized = (
        False  # Track if training data is already normalized after imputation
    )

    if use_imputation and imputation_head is not None:
        use_normalized_data = pretrain_config.get("use_normalized_data", False)

        # If we're using normalized data, we'll keep it normalized after imputation
        if use_normalized_data:
            training_data_normalized = True

        if verbose:
            print(f"\n{'=' * 40}")
            print("Imputation Phase")
            print(f"{'=' * 40}")
            print(
                f"  Data: {'normalized' if use_normalized_data else 'unnormalized (raw)'}"
            )

        num_iterations = pretrain_config.get("imputation_iterations", 3)
        batch_size = pretrain_config.get("pretrain_batch_size", 100)
        mask_value = 0.0

        def _impute_split(split: str, detailed: bool = False) -> None:
            if split not in pytorch_datasets:
                return

            split_x = pytorch_datasets[split].x.clone()
            total_values = split_x[..., 0].numel()
            missing_before = (split_x[..., 0] == mask_value).sum().item()

            if verbose and detailed:
                missing_pct = (missing_before / max(total_values, 1)) * 100
                print(f"  {split.title()} data shape: {tuple(split_x.shape)}")
                print(
                    f"  Missing values: {missing_before:,} / {total_values:,} ({missing_pct:.2f}%)"
                )
                print("  Imputing with pretrained model...")
                print(f"  Imputation iterations: {num_iterations}")
                print(f"  Batch size: {batch_size}")

            if missing_before == 0:
                if verbose and detailed:
                    print(f"  {split.title()}: no missing values detected.")
                return

            # If using normalized data, normalize before imputation and KEEP normalized
            # (no need to inverse transform since fine-tuning will use normalized data anyway)
            if use_normalized_data:
                # Create mask BEFORE normalization (while missing values are still 0.0)
                missing_mask = split_x[..., 0:1] == mask_value

                # Normalize data
                split_x_norm = split_x.clone()
                split_x_norm[..., 0:1] = scaler.transform(split_x_norm[..., 0:1])

                # After normalization, the mask_value (0.0) becomes (0-mean)/std
                # We need to compute what the normalized mask value is
                normalized_mask_value = scaler.transform(torch.zeros(1, 1, 1, 1))[
                    0, 0, 0, 0
                ].item()

                # Re-apply the mask with the normalized mask value
                split_x_norm[..., 0:1] = torch.where(
                    missing_mask,
                    torch.full_like(split_x_norm[..., 0:1], normalized_mask_value),
                    split_x_norm[..., 0:1],
                )

                # Impute in normalized space and keep normalized
                imputed_x = impute_missing_data(
                    model=model,
                    imputation_head=imputation_head,
                    data=split_x_norm,
                    num_iterations=num_iterations,
                    batch_size=batch_size,
                    device=device,
                    mask_value=normalized_mask_value,
                    use_normalized_data=True,
                )
            else:
                # Impute in raw space (will be normalized later for fine-tuning)
                imputed_x = impute_missing_data(
                    model=model,
                    imputation_head=imputation_head,
                    data=split_x,
                    num_iterations=num_iterations,
                    batch_size=batch_size,
                    device=device,
                    mask_value=mask_value,
                    use_normalized_data=False,
                )

            pytorch_datasets[split].x = imputed_x

            if verbose:
                missing_after = (imputed_x[..., 0] == mask_value).sum().item()
                filled = missing_before - missing_after
                if filled > 0:
                    rate = (filled / missing_before) * 100
                    print(
                        f"  {split.title()} after imputation: {missing_after:,} missing ({rate:.1f}% filled)"
                    )
                else:
                    print(
                        f"  {split.title()} after imputation: {missing_after:,} missing"
                    )

                if detailed and filled > 0:
                    was_missing = split_x[..., 0] == mask_value
                    if was_missing.any():
                        imputed_values = imputed_x[..., 0][was_missing]
                        print("  Imputed value stats:")
                        print(f"    Mean: {imputed_values.mean().item():.2f}")
                        print(f"    Std:  {imputed_values.std().item():.2f}")
                        print(f"    Min:  {imputed_values.min().item():.2f}")
                        print(f"    Max:  {imputed_values.max().item():.2f}")

        # Impute all splits (train, val, test)
        # If use_normalized_data=True, all splits will be kept in normalized form
        _impute_split("train", detailed=True)
        _impute_split("val")
        _impute_split("test")

        if verbose:
            print(f"{'=' * 40}\n")

    # Try to compile model for faster execution (PyTorch 2.0+)
    # Skip compilation for TemporalMode.NONE due to torch.compile stride tracking issues
    # Can also be disabled via config: training.use_torch_compile = false
    if (
        use_torch_compile
        and hasattr(torch, "compile")
        and device.type == "cuda"
        and temporal_mode_enum != TemporalMode.NONE
    ):
        try:
            model = torch.compile(model)
            if verbose:
                print("  Using: torch.compile")
        except Exception:
            pass  # Fall back to eager mode if compile fails
    elif verbose:
        # Report why compilation was skipped
        if not use_torch_compile:
            print("  torch.compile: disabled via config")
        elif temporal_mode_enum == TemporalMode.NONE:
            print("  torch.compile: skipped (TemporalMode.NONE has stride issues)")
        elif device.type != "cuda":
            print("  torch.compile: skipped (only enabled on CUDA)")
        elif not hasattr(torch, "compile"):
            print("  torch.compile: not available (requires PyTorch 2.0+)")

    # Pre-normalize datasets for fine-tuning (scaler already fitted earlier)
    # If imputation was done on normalized data, data is already normalized - skip this step
    if training_data_normalized:
        # Data is already normalized from imputation phase - use directly
        # This applies to train, val, AND test splits (all were imputed)
        train_x_norm = pytorch_datasets["train"].x.clone()
        val_x_norm = pytorch_datasets["val"].x.clone()
    else:
        # Normalize data now (much faster than normalizing each batch)
        train_x_norm = pytorch_datasets["train"].x.clone()
        train_x_norm[..., 0:1] = scaler.transform(train_x_norm[..., 0:1])
        val_x_norm = pytorch_datasets["val"].x.clone()
        val_x_norm[..., 0:1] = scaler.transform(val_x_norm[..., 0:1])

    # Replace NaN values with 0 in normalized data (0 = mean after normalization)
    # This prevents NaN from propagating through the model
    train_x_norm = torch.nan_to_num(train_x_norm, nan=0.0)
    val_x_norm = torch.nan_to_num(val_x_norm, nan=0.0)

    # Create normalized dataset wrappers
    train_dataset = TensorDataset(train_x_norm, pytorch_datasets["train"].y)
    val_dataset = TensorDataset(val_x_norm, pytorch_datasets["val"].y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Setup optimizer, scheduler, and loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=lr_decay_rate
    )
    criterion = MaskedHuberLoss()

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Create GradScaler for AMP
    grad_scaler = torch.amp.GradScaler(enabled=use_amp)

    # Define custom x-axis for training metrics to start from 0
    # This allows training to have its own 0-based epoch counter independent of pretraining
    if wandb_run is not None:
        wandb_run.define_metric("train/step")
        wandb_run.define_metric("train/*", step_metric="train/step")
        wandb_run.define_metric("val/step")
        wandb_run.define_metric("val/*", step_metric="val/step")
        # Batch-level logging uses global step directly (no custom x-axis needed)

    # Log model parameters to W&B config (appears in runs table, not charts)
    if wandb_run is not None:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb_run.config.update(
            {
                "model_params_total": total_params,
                "model_params_trainable": trainable_params,
            },
            allow_val_change=True,
        )

    # Track batches per epoch for global step calculation
    batches_per_epoch = len(train_loader)

    # Freeze encoder for first N epochs if requested (only train output_proj)
    encoder_frozen = False
    if freeze_encoder_epochs > 0:
        # Get the underlying model if compiled
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        # Freeze all parameters except output_proj
        for name, param in base_model.named_parameters():
            if not name.startswith("output_proj"):
                param.requires_grad = False
        encoder_frozen = True
        if verbose:
            trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
            print(f"  Encoder frozen for first {freeze_encoder_epochs} epochs")
            print(f"  Trainable parameters: {trainable:,} (output_proj only)")

    epoch_pbar = tqdm(range(epochs), desc="Training", disable=not verbose, position=0)
    for epoch in epoch_pbar:
        # Unfreeze encoder after freeze_encoder_epochs
        if encoder_frozen and epoch >= freeze_encoder_epochs:
            base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            for param in base_model.parameters():
                param.requires_grad = True
            encoder_frozen = False
            if verbose:
                trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
                print(f"\n  Encoder unfrozen at epoch {epoch}")
                print(f"  Trainable parameters: {trainable:,} (all)")
        # Train (data is pre-normalized)
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            clip_grad=clip_grad,
            use_amp=use_amp,
            grad_scaler=grad_scaler,
            verbose=verbose,
            wandb_run=wandb_run,
            log_interval=100,
            global_step_offset=epoch * batches_per_epoch,
        )

        # Compute train metrics on a subset (full train set would be too slow)
        # Use first N batches from train_loader for metrics
        train_metrics = _evaluate(
            model,
            train_loader,
            device,
            scaler,
            use_amp,
            compute_all_metrics=True,
            max_batches=100,  # Sample ~100 batches for train metrics
        )

        # Validate with full metrics (MAE, RMSE, MAPE for all horizons)
        val_metrics = _evaluate(
            model,
            val_loader,
            device,
            scaler,
            use_amp,
            compute_all_metrics=True,
        )
        val_loss = val_metrics['mae']  # Use overall MAE for early stopping

        scheduler.step()

        # Log to W&B if enabled
        if wandb_run is not None:
            log_dict = {
                "epoch": epoch,
                "train/lr": optimizer.param_groups[0]["lr"],
                # Train overall metrics (sampled)
                "train/loss": train_loss,
                "train/mae": train_metrics['mae'],
                "train/rmse": train_metrics['rmse'],
                "train/mape": train_metrics['mape'],
                # Train horizon metrics
                "train/mae_h3": train_metrics.get('mae_h3', 0),
                "train/rmse_h3": train_metrics.get('rmse_h3', 0),
                "train/mape_h3": train_metrics.get('mape_h3', 0),
                "train/mae_h6": train_metrics.get('mae_h6', 0),
                "train/rmse_h6": train_metrics.get('rmse_h6', 0),
                "train/mape_h6": train_metrics.get('mape_h6', 0),
                "train/mae_h12": train_metrics.get('mae_h12', 0),
                "train/rmse_h12": train_metrics.get('rmse_h12', 0),
                "train/mape_h12": train_metrics.get('mape_h12', 0),
                # Val overall metrics
                "val/mae": val_metrics['mae'],
                "val/rmse": val_metrics['rmse'],
                "val/mape": val_metrics['mape'],
                "val/best_mae": min(best_val_loss, val_loss),
                # Val horizon metrics
                "val/mae_h3": val_metrics.get('mae_h3', 0),
                "val/rmse_h3": val_metrics.get('rmse_h3', 0),
                "val/mape_h3": val_metrics.get('mape_h3', 0),
                "val/mae_h6": val_metrics.get('mae_h6', 0),
                "val/rmse_h6": val_metrics.get('rmse_h6', 0),
                "val/mape_h6": val_metrics.get('mape_h6', 0),
                "val/mae_h12": val_metrics.get('mae_h12', 0),
                "val/rmse_h12": val_metrics.get('rmse_h12', 0),
                "val/mape_h12": val_metrics.get('mape_h12', 0),
            }
            wandb_run.log(log_dict)

        # Update progress bar
        epoch_pbar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_mae=f"{val_loss:.4f}",
            val_rmse=f"{val_metrics['rmse']:.4f}",
            best=f"{best_val_loss:.4f}",
        )

        # Early stopping - reject invalid losses (NaN, 0, or suspiciously low)
        # A val_loss of 0 typically indicates NaN predictions being masked out entirely
        MIN_VALID_LOSS = 1e-6  # Loss below this is suspicious for traffic prediction
        loss_is_valid = (
            not math.isnan(val_loss)
            and not math.isinf(val_loss)
            and val_loss > MIN_VALID_LOSS
        )

        if not loss_is_valid:
            print(
                f"\n  WARNING: Invalid val_loss={val_loss:.6f} at epoch {epoch + 1} "
                f"(NaN/Inf/zero detected, skipping model checkpoint)"
            )
            patience_counter += 1
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1

        if patience_counter >= early_stop:
            epoch_pbar.set_description(f"Early stop (patience={early_stop})")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    # Save model if directory provided
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_model(model, scaler, save_dir, dataset_name)
        if verbose:
            print(f"Model saved to: {save_dir}")

    # Return whether test data is already normalized
    # (True if imputation used normalized data, False otherwise)
    return model, scaler, training_data_normalized


def _train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    scaler,
    clip_grad=5.0,
    use_amp=False,
    grad_scaler=None,
    verbose=False,
    wandb_run=None,
    log_interval=100,
    global_step_offset=0,
):
    """Train for one epoch with AMP support. Data should be pre-normalized.

    Args:
        wandb_run: Optional wandb Run object for batch-level logging
        log_interval: Log to W&B every N batches (default: 100)
        global_step_offset: Offset for global step counter (for multi-epoch runs)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Wrap dataloader with tqdm if verbose
    batch_iter = tqdm(
        dataloader, desc="  Batches", leave=False, disable=not verbose, position=1
    )

    for x, y in batch_iter:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()

        # Forward pass with optional AMP
        with torch.amp.autocast(device.type, enabled=use_amp):
            pred = model(x)
            # Inverse transform predictions and compute loss against original y
            pred_inv = scaler.inverse_transform(pred)
            y_target = y[..., 0:1]  # Only speed for loss
            loss = criterion(pred_inv, y_target, null_val=0.0)

        # NaN detection - catch early and provide diagnostics
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n  NaN/Inf DETECTED at batch {num_batches + 1}!")
            print(f"    pred has NaN: {torch.isnan(pred).any().item()}, "
                  f"Inf: {torch.isinf(pred).any().item()}")
            print(f"    pred_inv has NaN: {torch.isnan(pred_inv).any().item()}, "
                  f"Inf: {torch.isinf(pred_inv).any().item()}")
            print(f"    y_target has NaN: {torch.isnan(y_target).any().item()}")
            print(f"    x has NaN: {torch.isnan(x).any().item()}")
            if torch.isnan(pred).any():
                print(f"    pred stats: min={pred[~torch.isnan(pred)].min().item():.4f}, "
                      f"max={pred[~torch.isnan(pred)].max().item():.4f}")
            # Skip this batch but continue training
            continue

        # Backward pass with gradient scaling for AMP
        if use_amp and grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            if clip_grad > 0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar with current loss
        if verbose:
            batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        # Log to W&B every log_interval batches
        if wandb_run is not None and num_batches % log_interval == 0:
            global_step = global_step_offset + num_batches
            wandb_run.log({"batch/loss": loss.item()}, step=global_step)

    return total_loss / max(num_batches, 1)


def _evaluate(
    model,
    dataloader,
    device,
    scaler,
    use_amp=False,
    compute_all_metrics=False,
    max_batches=None,
):
    """Evaluate model and compute metrics.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run on
        scaler: StandardScaler for inverse transform
        use_amp: Whether to use automatic mixed precision
        compute_all_metrics: If True, compute MAE/RMSE/MAPE for all horizons.
                            If False, only compute overall MAE (faster).
        max_batches: If set, only evaluate on first N batches (for sampling)

    Returns:
        If compute_all_metrics=False: float (MAE loss)
        If compute_all_metrics=True: Dict with all metrics
    """
    from utils.training import compute_horizon_metrics, aggregate_batch_metrics

    model.eval()

    if not compute_all_metrics:
        # Fast path: only compute MAE for early stopping
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)

                with torch.amp.autocast(device.type, enabled=use_amp):
                    pred = model(x)
                    pred_inv = scaler.inverse_transform(pred)
                    y_target = y[..., 0:1]
                    loss = masked_mae_loss(pred_inv, y_target, null_val=0.0)

                total_loss += loss.item()
                num_batches += 1

                if max_batches is not None and num_batches >= max_batches:
                    break

        return total_loss / max(num_batches, 1)

    # Full metrics path: compute MAE, RMSE, MAPE for all horizons
    batch_metrics_list = []
    batch_sizes = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast(device.type, enabled=use_amp):
                pred = model(x)
                pred_inv = scaler.inverse_transform(pred)
                y_target = y[..., 0:1]

            # Compute all metrics for this batch
            batch_metrics = compute_horizon_metrics(
                pred_inv, y_target, null_val=0.0, horizons=[3, 6, 12]
            )
            batch_metrics_list.append(batch_metrics)
            batch_sizes.append(x.shape[0])

            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    # Aggregate across batches
    return aggregate_batch_metrics(batch_metrics_list, batch_sizes)


def save_model(
    model: STGFormer,
    scaler: StandardScaler,
    save_dir: Path,
    dataset_name: str,
) -> None:
    """
    Save STGFormer model checkpoint.

    Saves:
    - model.safetensors: Model weights
    - config.json: Model configuration
    - scaler.json: Scaler mean and std

    Args:
        model: Trained STGFormer model
        scaler: Fitted StandardScaler
        save_dir: Directory to save checkpoint
        dataset_name: Dataset name for metadata
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights (strip "_orig_mod." prefix from torch.compile if present)
    state_dict = {}
    for key, value in model.state_dict().items():
        if key.startswith("_orig_mod."):
            state_dict[key[len("_orig_mod.") :]] = value.cpu().contiguous()
        else:
            state_dict[key] = value.cpu().contiguous()
    save_file(state_dict, save_dir / "model.safetensors")

    # Save model config
    config = {
        "num_nodes": model.num_nodes,
        "in_steps": model.in_steps,
        "out_steps": model.out_steps,
        "input_dim": model.input_dim,
        "output_dim": model.output_dim,
        "steps_per_day": model.steps_per_day,
        "input_embedding_dim": model.input_embedding_dim,
        "tod_embedding_dim": model.tod_embedding_dim,
        "dow_embedding_dim": model.dow_embedding_dim,
        "spatial_embedding_dim": model.spatial_embedding_dim,
        "adaptive_embedding_dim": model.adaptive_embedding_dim,
        "num_heads": model.num_heads,
        "num_layers": model.num_layers,
        "dropout_a": model.dropout_a.p,  # Get dropout probability from module
        "use_mixed_proj": model.use_mixed_proj,
        "pre_attn_kernel_size": model.pre_attn_kernel_size,
        "model_dim": model.model_dim,
        "dataset": dataset_name,
        # Graph structure
        "graph_mode": model.graph_mode.value,
        "lambda_hybrid": model.lambda_hybrid,
        "sparsity_k": model.sparsity_k,
        # Propagation mode
        "propagation_mode": model.propagation_mode.value,
        # Temporal mode
        "temporal_mode": model.temporal_mode.value,
        "mamba_d_state": model.mamba_d_state,
        "mamba_d_conv": model.mamba_d_conv,
        "mamba_expand": model.mamba_expand,
        "tcn_num_layers": model.tcn_num_layers,
        "tcn_kernel_size": model.tcn_kernel_size,
        "tcn_dilation_base": model.tcn_dilation_base,
        "tcn_dropout": model.tcn_dropout,
        # Initialization
        "use_zero_init": model.use_zero_init,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save scaler
    scaler_data = {
        "mean": scaler.mean.tolist() if scaler.mean is not None else None,
        "std": scaler.std.tolist() if scaler.std is not None else None,
    }
    with open(save_dir / "scaler.json", "w") as f:
        json.dump(scaler_data, f, indent=2)


def load_model(
    checkpoint_dir: Path,
    dataset_name: str,
    device: Optional[str] = None,
    verbose: bool = False,
) -> tuple:
    """
    Load trained STGFormer model from checkpoint.

    Args:
        checkpoint_dir: Directory containing model.safetensors, config.json, scaler.json
        dataset_name: Dataset name (e.g., "METR-LA") - needed to load geo_adj if required
        device: Device to load model on (None = auto-detect)
        verbose: Print verbose output

    Returns:
        tuple: (model, scaler)
    """
    checkpoint_dir = Path(checkpoint_dir)

    if device is None:
        device = get_best_device()

    if verbose:
        print(f"Loading internal STGFormer model from {checkpoint_dir}...")

    # Load config
    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)

    # Parse enum values from config with backward-compatible defaults
    graph_mode = GraphMode(config.get("graph_mode", "learned"))
    propagation_mode = PropagationMode(config.get("propagation_mode", "power"))
    temporal_mode = TemporalMode(config.get("temporal_mode", "transformer"))

    # Load geographic adjacency if needed for graph mode or propagation mode
    geo_adj = None
    if (
        graph_mode in (GraphMode.GEOGRAPHIC, GraphMode.SPECTRAL_INIT, GraphMode.HYBRID)
        or propagation_mode == PropagationMode.CHEBYSHEV
    ):
        import utils.io

        geo_adj, _, _ = utils.io.get_graph_metadata(dataset_name)
        geo_adj = torch.from_numpy(geo_adj).float()

    # Create model
    model = STGFormer(
        num_nodes=config["num_nodes"],
        in_steps=config["in_steps"],
        out_steps=config["out_steps"],
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        steps_per_day=config.get("steps_per_day", 288),
        input_embedding_dim=config["input_embedding_dim"],
        tod_embedding_dim=config["tod_embedding_dim"],
        dow_embedding_dim=config["dow_embedding_dim"],
        spatial_embedding_dim=config.get("spatial_embedding_dim", 0),
        adaptive_embedding_dim=config["adaptive_embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout_a=config.get("dropout_a", 0.3),  # Default to 0.3 if not in config
        use_mixed_proj=config.get("use_mixed_proj", True),
        pre_attn_kernel_size=config.get("pre_attn_kernel_size", 1),
        # Graph structure
        graph_mode=graph_mode,
        geo_adj=geo_adj,
        lambda_hybrid=config.get("lambda_hybrid", 0.5),
        sparsity_k=config.get("sparsity_k", None),
        # Propagation mode
        propagation_mode=propagation_mode,
        # Temporal mode
        temporal_mode=temporal_mode,
        mamba_d_state=config.get("mamba_d_state", 16),
        mamba_d_conv=config.get("mamba_d_conv", 4),
        mamba_expand=config.get("mamba_expand", 2),
        tcn_num_layers=config.get("tcn_num_layers", 3),
        tcn_kernel_size=config.get("tcn_kernel_size", 3),
        tcn_dilation_base=config.get("tcn_dilation_base", 2),
        tcn_dropout=config.get("tcn_dropout", 0.1),
        # Initialization (default True for backward compatibility with older checkpoints)
        use_zero_init=config.get("use_zero_init", True),
    )

    # Load weights
    state_dict = load_file(checkpoint_dir / "model.safetensors")

    # Strip "_orig_mod." prefix if model was saved with torch.compile()
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            cleaned_state_dict[key[len("_orig_mod.") :]] = value
        else:
            cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()

    # Load scaler
    with open(checkpoint_dir / "scaler.json") as f:
        scaler_data = json.load(f)

    scaler = StandardScaler(
        mean=torch.tensor(scaler_data["mean"]) if scaler_data["mean"] else None,
        std=torch.tensor(scaler_data["std"]) if scaler_data["std"] else None,
    )

    if verbose:
        print(f"Model loaded successfully on {device}")

    return model, scaler


def load_from_hub(
    dataset_name: str,
    hf_repo_prefix: str,
    force_download: bool = False,
    device: Optional[str] = None,
    verbose: bool = False,
) -> tuple:
    """
    Load trained STGFormer model from HuggingFace Hub.

    Args:
        dataset_name: Dataset name (e.g., "METR-LA")
        hf_repo_prefix: HF repo prefix (required)
        force_download: Force re-download of model
        device: Device to load model on (None = auto-detect)
        verbose: Print verbose output

    Returns:
        tuple: (model, scaler)
    """
    if not hf_repo_prefix:
        raise ValueError("hf_repo_prefix is required")
    checkpoint_dir = fetch_model_from_hub(
        model_type=hf_repo_prefix,
        dataset_name=dataset_name,
        force_download=force_download,
        verbose=verbose,
    )
    return load_model(checkpoint_dir, dataset_name, device, verbose)


def get_predictions(
    model: STGFormer,
    scaler: StandardScaler,
    pytorch_dataset: TrafficDataset,
    batch_size: int = 64,
    device: Optional[str] = None,
    data_already_normalized: bool = False,
) -> np.ndarray:
    """
    Get predictions from STGFormer model.

    Args:
        model: Trained STGFormer model
        scaler: StandardScaler used during training
        pytorch_dataset: TrafficDataset to predict on
        batch_size: Batch size for inference (default: 64)
        device: Device for inference (None = auto-detect)
        data_already_normalized: If True, skip normalization (data is already normalized)

    Returns:
        Predictions array of shape (samples, horizon, nodes, 1)
    """
    if device is None:
        device = get_best_device()

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(pytorch_dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)

            # Normalize input values (unless already normalized)
            if data_already_normalized:
                x_normalized = x
            else:
                x_normalized = x.clone()
                x_normalized[..., 0:1] = scaler.transform(x[..., 0:1])

            # Forward pass
            pred = model(x_normalized)

            # Inverse transform predictions
            pred_inv = scaler.inverse_transform(pred)

            all_predictions.append(pred_inv.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    return predictions


def get_predictions_hub(
    dataset_name: str,
    dataset: Dict[str, TrafficDataset],
    split: str = "test",
    batch_size: int = 64,
    force_download: bool = False,
    *,
    hf_repo_prefix: str,
) -> np.ndarray:
    """
    Get predictions from STGFormer model loaded from HuggingFace Hub.

    This function matches the interface used by other models for baseline calculations.

    Args:
        dataset_name: Dataset name (e.g., "METR-LA", "PEMS-BAY")
        dataset: Dict of TrafficDataset splits
        split: Which split to predict on (default: "test")
        batch_size: Batch size for inference (default: 64)
        force_download: Force re-download of model from Hub
        hf_repo_prefix: HF repo prefix (required)

    Returns:
        Predictions array of shape (samples, horizon, nodes, 1)
    """
    if not hf_repo_prefix:
        raise ValueError(
            "hf_repo_prefix is required to avoid accidentally loading the wrong model"
        )
    model, scaler = load_from_hub(
        dataset_name, hf_repo_prefix=hf_repo_prefix, force_download=force_download
    )
    pytorch_dataset = dataset[split]
    return get_predictions(model, scaler, pytorch_dataset, batch_size=batch_size)


def push_to_hub(
    checkpoint_dir: Path,
    repo_id: str,
    dataset_name: str,
    hf_repo_prefix: str,
    metrics: Optional[Dict[str, float]] = None,
    commit_message: Optional[str] = None,
    private: bool = False,
    description: Optional[str] = None,
) -> str:
    """
    Push a checkpoint to HuggingFace Hub.

    Args:
        checkpoint_dir: Path to local checkpoint directory
        repo_id: HuggingFace repo ID
        dataset_name: Name of the dataset used for training
        hf_repo_prefix: HF repo prefix (required)
        metrics: Optional dict of evaluation metrics
        commit_message: Optional custom commit message
        private: Whether to create a private repository
        description: Optional custom description for model card

    Returns:
        URL to the uploaded model on HuggingFace Hub
    """
    if not hf_repo_prefix:
        raise ValueError(
            "hf_repo_prefix is required to avoid accidentally overwriting existing models"
        )
    checkpoint_dir = Path(checkpoint_dir)

    # Add hub metadata
    hub_metadata = {
        "dataset": dataset_name,
        "metrics": metrics or {},
        "framework": "PyTorch",
        "hf_repo_prefix": hf_repo_prefix,
        "implementation": "internal",
    }

    hub_metadata_path = checkpoint_dir / "hub_metadata.json"
    with open(hub_metadata_path, "w") as f:
        json.dump(hub_metadata, f, indent=2)

    # Use shared push function
    return push_model_to_hub(
        checkpoint_dir=checkpoint_dir,
        repo_id=repo_id,
        model_type=hf_repo_prefix,
        dataset_name=dataset_name,
        metrics=metrics,
        commit_message=commit_message,
        private=private,
        description=description,
    )


def save_pretrained_checkpoint(
    model: STGFormer,
    imputation_head: torch.nn.Module,
    checkpoint_dir: Path,
    pretrain_config: Dict[str, Any],
    dataset_name: str,
) -> None:
    """
    Save pretrained model checkpoint (model + imputation head + config).

    Args:
        model: Pretrained STGFormer model
        imputation_head: Trained imputation head (Linear layer)
        checkpoint_dir: Directory to save checkpoint
        pretrain_config: Pretraining configuration dict
        dataset_name: Dataset name (for metadata)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model state dict
    model_state = model.state_dict()
    # Strip "_orig_mod." prefix if model was compiled
    cleaned_state = {}
    for key, value in model_state.items():
        if key.startswith("_orig_mod."):
            cleaned_state[key[len("_orig_mod.") :]] = value
        else:
            cleaned_state[key] = value

    save_file(cleaned_state, checkpoint_dir / "model.safetensors")

    # Save imputation head state dict
    imputation_state = imputation_head.state_dict()
    save_file(imputation_state, checkpoint_dir / "imputation_head.safetensors")

    # Save pretraining config
    pretrain_metadata = {
        "dataset_name": dataset_name,
        "pretrain_config": pretrain_config,
        "model_dim": model.model_dim,
        "num_nodes": model.num_nodes,
    }

    with open(checkpoint_dir / "pretrain_config.json", "w") as f:
        json.dump(pretrain_metadata, f, indent=2)

    print(f"Pretrained checkpoint saved to {checkpoint_dir}")


def load_pretrained_checkpoint(
    model: STGFormer,
    checkpoint_dir: Path,
    device: Optional[str] = None,
    verbose: bool = False,
) -> torch.nn.Module:
    """
    Load pretrained model checkpoint (model weights + imputation head).

    Args:
        model: STGFormer model instance to load weights into
        checkpoint_dir: Directory containing pretrained checkpoint
        device: Device to load onto (None = auto-detect)
        verbose: Print verbose output

    Returns:
        Loaded imputation head module
    """
    if device is None:
        device = get_best_device()

    device = torch.device(device)
    checkpoint_dir = Path(checkpoint_dir)

    # Load and verify pretrain config
    with open(checkpoint_dir / "pretrain_config.json") as f:
        metadata = json.load(f)

    if verbose:
        print(f"Loading pretrained checkpoint from {checkpoint_dir}")
        print(f"  Dataset: {metadata['dataset_name']}")
        print(f"  Model dim: {metadata['model_dim']}")

    # Verify model compatibility
    if model.model_dim != metadata["model_dim"]:
        raise ValueError(
            f"Model dimension mismatch: expected {metadata['model_dim']}, "
            f"got {model.model_dim}"
        )

    # Load model weights
    model_state = load_file(checkpoint_dir / "model.safetensors")
    # Strip "_orig_mod." prefix if present
    cleaned_state = {}
    for key, value in model_state.items():
        if key.startswith("_orig_mod."):
            cleaned_state[key[len("_orig_mod.") :]] = value
        else:
            cleaned_state[key] = value

    # Check if output_proj architecture matches (linear vs MLP)
    # If not, skip output_proj weights and let the model use its randomly initialized head
    checkpoint_has_linear = "output_proj.weight" in cleaned_state
    model_has_linear = hasattr(model.output_proj, "weight")  # nn.Linear has .weight directly

    if checkpoint_has_linear != model_has_linear:
        # Architecture mismatch - skip output_proj weights
        if verbose:
            print("  Note: Prediction head architecture differs from checkpoint, using random init for head")
        cleaned_state = {k: v for k, v in cleaned_state.items() if not k.startswith("output_proj")}
        model.load_state_dict(cleaned_state, strict=False)
    else:
        model.load_state_dict(cleaned_state)

    model.to(device)

    # Load imputation head (temporal: predicts per-timestep values)
    imputation_head = torch.nn.Linear(model.model_dim, model.in_steps).to(device)
    imputation_state = load_file(checkpoint_dir / "imputation_head.safetensors")

    # Check for checkpoint compatibility (old checkpoints used [model_dim, 1] for mean prediction)
    expected_shape = (model.in_steps, model.model_dim)
    actual_shape = tuple(imputation_state["weight"].shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"Incompatible pretrained checkpoint: imputation head has shape {actual_shape}, "
            f"but expected {expected_shape}. This checkpoint was likely created with an older "
            f"version that used mean-based loss. Please re-train the pretrained model with the "
            f"current code (which uses per-timestep loss for better temporal learning)."
        )

    imputation_head.load_state_dict(imputation_state)

    if verbose:
        print("Pretrained checkpoint loaded successfully")

    return imputation_head


def push_pretrained_to_hub(
    checkpoint_dir: Path,
    hf_repo_prefix: str,
    dataset_name: str,
    private: bool = False,
) -> str:
    """
    Push pretrained checkpoint to HuggingFace Hub.

    Args:
        checkpoint_dir: Path to local checkpoint directory
        hf_repo_prefix: HF repo prefix - either:
            - Full repo ID like "username/stgformer-pretrained" (if contains "/")
            - Model prefix like "STGFORMER_PRETRAINED" (uses HF_USERNAME_UPLOAD from .env)
        dataset_name: Dataset name (appended to create full repo ID)
        private: Whether to create private repository

    Returns:
        URL to the uploaded checkpoint
    """
    from utils.config import HF_USERNAME_UPLOAD

    checkpoint_dir = Path(checkpoint_dir)

    # Construct repo ID based on whether prefix contains "/"
    dataset_suffix = dataset_name.lower().replace("-", "")

    if "/" in hf_repo_prefix:
        # Full repo ID provided (e.g., "witgaw/stgformer-pretrained")
        repo_id = f"{hf_repo_prefix}-{dataset_suffix}"
    else:
        # Model prefix provided (e.g., "STGFORMER_PRETRAINED")
        # Construct: {HF_USERNAME}/{PREFIX}_{DATASET}
        if not HF_USERNAME_UPLOAD:
            raise RuntimeError(
                "HF_USERNAME_FOR_UPLOAD not set in .env_public. "
                "Cannot upload to HuggingFace Hub. Either set HF_USERNAME_FOR_UPLOAD "
                "or provide a full repo ID (e.g., 'username/model-name')."
            )
        prefix_upper = hf_repo_prefix.upper()
        repo_id = f"{HF_USERNAME_UPLOAD}/{prefix_upper}_{dataset_name}"

    # Add hub metadata
    hub_metadata = {
        "dataset": dataset_name,
        "checkpoint_type": "pretrained",
        "framework": "PyTorch",
        "hf_repo_prefix": hf_repo_prefix,
    }

    hub_metadata_path = checkpoint_dir / "hub_metadata.json"
    with open(hub_metadata_path, "w") as f:
        json.dump(hub_metadata, f, indent=2)

    # Use shared push function
    description = (
        f"STGFormer pretrained checkpoint for {dataset_name}. "
        "This checkpoint contains pretrained model weights and imputation head "
        "from masked node pretraining. Use with load_from config option."
    )
    return push_model_to_hub(
        checkpoint_dir=checkpoint_dir,
        repo_id=repo_id,
        model_type="STGFORMER_PRETRAINED",  # Use STGFORMER_* pattern for model card
        dataset_name=dataset_name,
        commit_message="Upload pretrained checkpoint",
        private=private,
        description=description,
    )


def load_pretrained_from_hub(
    hf_repo_prefix: str,
    dataset_name: str,
    model: STGFormer,
    force_download: bool = False,
    device: Optional[str] = None,
    verbose: bool = False,
) -> torch.nn.Module:
    """
    Load pretrained checkpoint from HuggingFace Hub.

    Args:
        hf_repo_prefix: HF repo prefix - either:
            - Full repo ID like "username/stgformer-pretrained" (if contains "/")
            - Model prefix like "STGFORMER_PRETRAINED" (uses HF_USERNAME_DOWNLOAD from .env)
        dataset_name: Dataset name
        model: STGFormer model instance to load weights into
        force_download: Force re-download
        device: Device to load onto
        verbose: Print verbose output

    Returns:
        Loaded imputation head module
    """
    from huggingface_hub import snapshot_download

    from utils.config import HF_USERNAME_DOWNLOAD, MODELS_DIR

    # Construct repo ID based on whether prefix contains "/"
    dataset_suffix = dataset_name.lower().replace("-", "")

    if "/" in hf_repo_prefix:
        # Full repo ID provided (e.g., "witgaw/stgformer-pretrained")
        repo_id = f"{hf_repo_prefix}-{dataset_suffix}"
    else:
        # Model prefix provided (e.g., "STGFORMER_PRETRAINED")
        # Construct: {HF_USERNAME}/{PREFIX}_{DATASET}
        if not HF_USERNAME_DOWNLOAD:
            raise RuntimeError(
                "HF_USERNAME_FOR_MODEL_DOWNLOAD not set in .env_public. "
                "Cannot download from HuggingFace Hub. Either set HF_USERNAME_FOR_MODEL_DOWNLOAD "
                "or provide a full repo ID (e.g., 'username/model-name')."
            )
        prefix_upper = hf_repo_prefix.upper()
        repo_id = f"{HF_USERNAME_DOWNLOAD}/{prefix_upper}_{dataset_name}"

    if verbose:
        print(f"Downloading pretrained checkpoint from {repo_id}...")

    # Download from hub directly (using snapshot_download)
    # Use repo_id-based cache path to avoid mixing up different pretrained models
    # Convert repo_id "username/MODEL_NAME" -> "MODEL_NAME" for the cache path
    repo_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    checkpoint_dir = MODELS_DIR / repo_name / f"model_{dataset_suffix}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if (
        force_download
        or not checkpoint_dir.exists()
        or not any(checkpoint_dir.iterdir())
    ):
        snapshot_download(
            repo_id=repo_id,
            local_dir=checkpoint_dir,
            local_dir_use_symlinks=False,
        )

    # Load checkpoint
    return load_pretrained_checkpoint(
        model=model,
        checkpoint_dir=checkpoint_dir,
        device=device,
        verbose=verbose,
    )
