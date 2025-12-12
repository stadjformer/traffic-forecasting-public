#!/usr/bin/env python3
"""Train internal STGFormer model on traffic forecasting datasets.

Usage:
    python scripts/train_stgformer.py --config configs/stgformer_baseline.yaml
    python scripts/train_stgformer.py --config configs/stgformer_baseline.yaml --dataset METR-LA --dry-run

With pretraining:
    python scripts/train_stgformer.py --config configs/stgformer_pretrain.yaml
"""

import argparse
import time

import utils.baselines
import utils.io
import utils.stgformer
from utils.config import SUPPORTED_DATASETS, load_experiment_config
from utils.config_classes import (
    GraphConfig,
    ModelArchConfig,
    PretrainConfig,
    TemporalConfig,
    TrainingConfig,
)


def _init_wandb(config, dataset_name, dry_run=False):
    """Initialize W&B run if enabled in config.

    Raises an exception if W&B is enabled but user is not logged in.
    This prevents training from blocking on an interactive login prompt.
    """
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        print("W&B: disabled in config")
        return None
    if dry_run:
        print("W&B: disabled (dry run mode)")
        return None

    try:
        import wandb
    except ImportError:
        raise ImportError(
            "W&B logging is enabled but wandb is not installed. "
            "Either install it with 'pip install wandb' or disable W&B in config (wandb.enabled: false)"
        )

    # Check if logged in BEFORE calling init (which would block with a prompt)
    if wandb.api.api_key is None:
        raise RuntimeError(
            "W&B logging is enabled but you are not logged in. "
            "Either run 'wandb login', set WANDB_API_KEY environment variable, "
            "or disable W&B in config (wandb.enabled: false)"
        )

    # Build run name from config
    hf_prefix = config["output"].get("hf_repo_prefix", "STGFORMER")
    run_name = f"{hf_prefix}_{dataset_name}"

    # Flatten config for W&B
    experiment_subset = config.get("experiment_subset", "")

    flat_config = {
        "dataset": dataset_name,
        "description": config.get("description", ""),
        "graph_mode": config["graph"]["mode"],
        "lambda_hybrid": config["graph"].get("lambda_hybrid", 0.5),
        **config.get("model", {}),
        **config.get("training", {}),
    }
    if experiment_subset:
        flat_config["experiment_subset"] = experiment_subset

    # Get description for notes
    description = config.get("description", "")

    run = wandb.init(
        project=wandb_cfg.get("project", "traffic-forecasting"),
        entity=wandb_cfg.get("entity"),
        name=run_name,
        config=flat_config,
        notes=description,
        group=dataset_name,
        job_type="train",
        tags=[dataset_name, hf_prefix],
        reinit=True,
    )

    if experiment_subset:
        run.summary["experiment_subset"] = experiment_subset

    print(f"W&B: initialized run '{run_name}' -> {run.url}")
    return run


def _validate_pretraining_config(config):
    """Validate pretraining configuration.

    Returns:
        pretraining_cfg or None if pretraining is disabled
    """
    pretrain_cfg = config.get("pretraining")
    use_imputation = config.get("use_imputation", False)

    # Validate: use_imputation requires pretraining
    if use_imputation and not pretrain_cfg:
        raise ValueError(
            "use_imputation: true requires a 'pretraining' section in config. "
            "The imputation head is only available after pretraining."
        )

    if not pretrain_cfg:
        return None

    # Validate: can't have both save_to and load_from
    if pretrain_cfg.get("save_to") and pretrain_cfg.get("load_from"):
        raise ValueError(
            "Cannot set both 'pretraining.save_to' and 'pretraining.load_from'. "
            "Either pretrain fresh and save, or load existing pretrained model."
        )

    # Warning: load_from with epoch configs
    if pretrain_cfg.get("load_from"):
        has_epochs = (
            pretrain_cfg.get("stage1_epochs", 0) > 0
            or pretrain_cfg.get("stage2_epochs", 0) > 0
        )
        if has_epochs:
            print(
                "Warning: pretraining.load_from is set, ignoring stage epoch configs "
                "(loading existing pretrained model instead of training fresh)"
            )

    return pretrain_cfg


def main():
    parser = argparse.ArgumentParser(description="Train internal STGFormer model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(SUPPORTED_DATASETS.keys()),
        help="Train single dataset (overrides config datasets list)",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Push to HuggingFace Hub"
    )
    parser.add_argument("--dry-run", action="store_true", help="1 epoch test run")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--load-pretrained",
        type=str,
        help="Load pretrained weights from HF repo prefix (e.g., emelle/STGFormer-pretrain)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--freeze-encoder-epochs",
        type=int,
        default=0,
        help="Freeze encoder for first N epochs (only train prediction head)",
    )
    parser.add_argument(
        "--prediction-head-layers",
        type=int,
        help="Number of layers in prediction head (1=linear, 2+=MLP)",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config, dataset_override=args.dataset)
    train_cfg = config["training"]

    # Override batch size if provided
    if args.batch_size:
        train_cfg["batch_size"] = args.batch_size
        print(f"Overriding batch size: {args.batch_size}")

    # Override prediction head layers if provided
    if args.prediction_head_layers:
        if "model" not in config:
            config["model"] = {}
        config["model"]["prediction_head_layers"] = args.prediction_head_layers
        print(f"Overriding prediction head layers: {args.prediction_head_layers}")

    # Inject --load-pretrained into config if provided
    if args.load_pretrained:
        if "pretraining" not in config:
            config["pretraining"] = {}
        config["pretraining"]["load_from"] = args.load_pretrained
        print(f"Loading pretrained weights from: {args.load_pretrained}")

    # Validate pretraining configuration
    pretrain_cfg = _validate_pretraining_config(config)
    use_imputation = config.get("use_imputation", False)

    # Validate Mamba dependencies if using Mamba temporal mode
    temporal_cfg = config.get("temporal", {})
    if temporal_cfg.get("mode") == "mamba":
        try:
            import mamba_ssm  # noqa: F401
        except ImportError:
            raise ImportError(
                "Mamba temporal mode requires mamba-ssm package, which is CUDA-only.\n"
                "Install with: uv sync --extra cuda\n"
                "Requirements: Linux + NVIDIA GPU + CUDA toolkit with nvcc compiler\n"
                "Not supported on: macOS, CPU-only systems, AMD GPUs"
            )

    # Validate hf_repo_prefix only when pushing to hub
    hf_repo_prefix = config["output"].get("hf_repo_prefix")
    if args.push_to_hub and not hf_repo_prefix:
        raise ValueError(
            "--push-to-hub requires 'output.hf_repo_prefix' in config to avoid "
            "accidentally overwriting existing models. "
            "Example: hf_repo_prefix: STGFORMER_MAMBA"
        )

    # Validate HuggingFace Hub access early (before training) to fail fast
    if args.push_to_hub:
        first_dataset = config["datasets"][0]
        _, repo_id_for_validation, _ = utils.io.get_model_paths(
            hf_repo_prefix, first_dataset, dry_run=args.dry_run
        )
        utils.io.validate_hf_hub_access(repo_id_for_validation, create_if_missing=True)

    for dataset_name in config["datasets"]:
        print(f"\n{'=' * 60}\nTraining on {dataset_name}\n{'=' * 60}")

        # Initialize W&B for this dataset (if enabled)
        wandb_run = _init_wandb(config, dataset_name, dry_run=args.dry_run)

        # Load data (always includes DOW - model config decides if it's used)
        pytorch_datasets = utils.io.get_dataset_torch(
            dataset_name, verbose=args.verbose
        )

        # Load geographic adjacency if needed for graph mode or propagation mode
        graph_cfg = config["graph"]
        prop_cfg = config.get("propagation", {})
        geo_adj = None
        if (
            graph_cfg["mode"] in ("geographic", "spectral_init", "hybrid")
            or prop_cfg.get("mode") == "chebyshev"
        ):
            geo_adj, _, _ = utils.io.get_graph_metadata(dataset_name)

        # Get save path
        model_dir, repo_id, is_private = utils.io.get_model_paths(
            hf_repo_prefix, dataset_name, dry_run=args.dry_run
        )

        # Train
        start = time.time()
        init_cfg = config.get("initialization", {})
        temporal_cfg = config.get("temporal", {})

        # Build config objects
        training_config = TrainingConfig(
            epochs=1 if args.dry_run else train_cfg.get("epochs", 100),
            batch_size=train_cfg.get("batch_size", 64),
            learning_rate=train_cfg.get("learning_rate", 0.001),
            weight_decay=train_cfg.get("weight_decay", 0.0003),
            early_stop=train_cfg.get("early_stop", 10),
            milestones=train_cfg.get("milestones"),
            lr_decay_rate=train_cfg.get("lr_decay_rate", 0.1),
            clip_grad=train_cfg.get("clip_grad", 0.0),
            seed=train_cfg.get("seed", 42),
            verbose=True,
            device=train_cfg.get("device"),  # None = auto-detect
            use_torch_compile=train_cfg.get("use_torch_compile", True),
        )

        graph_config = GraphConfig(
            graph_mode=graph_cfg["mode"],
            geo_adj=geo_adj,
            lambda_hybrid=graph_cfg.get("lambda_hybrid", 0.5),
            sparsity_k=graph_cfg.get("sparsity_k"),
        )

        temporal_config = TemporalConfig(
            temporal_mode=temporal_cfg.get("mode", "transformer"),
            mamba_d_state=temporal_cfg.get("d_state", 16),
            mamba_d_conv=temporal_cfg.get("d_conv", 4),
            mamba_expand=temporal_cfg.get("expand", 2),
            tcn_num_layers=temporal_cfg.get("num_layers", 3),
            tcn_kernel_size=temporal_cfg.get("kernel_size", 3),
            tcn_dilation_base=temporal_cfg.get("dilation_base", 2),
            tcn_dropout=temporal_cfg.get("dropout", 0.1),
            depthwise_kernel_size=temporal_cfg.get("kernel_size", 3),
            mlp_hidden_dim=temporal_cfg.get("hidden_dim"),
        )

        model_arch_config = None
        if config.get("model"):
            model_arch_config = ModelArchConfig(**config["model"])

        pretrain_config_obj = None
        if pretrain_cfg is not None:
            # Pass config dict directly - dataclass handles defaults
            pretrain_config_obj = PretrainConfig(**pretrain_cfg)
            # Override epochs for dry-run mode
            if args.dry_run:
                pretrain_config_obj.stage1_epochs = min(1, pretrain_config_obj.stage1_epochs)
                pretrain_config_obj.stage2_epochs = min(1, pretrain_config_obj.stage2_epochs)

        model, scaler, data_normalized = utils.stgformer.train_model(
            dataset_name=dataset_name,
            pytorch_datasets=pytorch_datasets,
            training_config=training_config,
            graph_config=graph_config,
            temporal_config=temporal_config,
            model_arch_config=model_arch_config,
            pretrain_config_obj=pretrain_config_obj,
            propagation_mode=prop_cfg.get("mode", "power"),
            save_dir=model_dir,
            wandb_run=wandb_run,
            use_imputation=use_imputation,
            use_zero_init=init_cfg.get("use_zero_init", True),
            exclude_missing_from_norm=config.get("exclude_missing_from_norm", False),
            freeze_encoder_epochs=args.freeze_encoder_epochs,
        )
        training_time = time.time() - start
        print(f"\nTraining completed in {training_time:.1f}s")

        # Evaluate
        preds = utils.stgformer.get_predictions(
            model,
            scaler,
            pytorch_datasets["test"],
            data_already_normalized=data_normalized,
        )
        true = pytorch_datasets["test"].y.numpy()[..., 0:1]
        metrics = utils.baselines.calculate_metrics(preds, true, null_vals=0.0)

        print("\nTest Metrics:")
        for h in ["15 min", "30 min", "1 hour"]:
            print(
                f"  {h} - MAE: {metrics[f'mae_{h}']:.3f}, RMSE: {metrics[f'rmse_{h}']:.3f}"
            )

        # Log test metrics to W&B
        if wandb_run is not None:
            # Log as regular metrics (for charts)
            wandb_run.log(
                {
                    "test/mae_15min": metrics["mae_15 min"],
                    "test/mae_30min": metrics["mae_30 min"],
                    "test/mae_1hour": metrics["mae_1 hour"],
                    "test/rmse_15min": metrics["rmse_15 min"],
                    "test/rmse_30min": metrics["rmse_30 min"],
                    "test/rmse_1hour": metrics["rmse_1 hour"],
                    "test/mape_15min": metrics["mape_15 min"],
                    "test/mape_30min": metrics["mape_30 min"],
                    "test/mape_1hour": metrics["mape_1 hour"],
                    "training_time_seconds": training_time,
                }
            )
            # Also set as summary metrics (prominent in runs table)
            wandb_run.summary["test_mae_15min"] = metrics["mae_15 min"]
            wandb_run.summary["test_mae_30min"] = metrics["mae_30 min"]
            wandb_run.summary["test_mae_1hour"] = metrics["mae_1 hour"]
            wandb_run.summary["test_rmse_15min"] = metrics["rmse_15 min"]
            wandb_run.summary["test_rmse_30min"] = metrics["rmse_30 min"]
            wandb_run.summary["test_rmse_1hour"] = metrics["rmse_1 hour"]
            wandb_run.summary["training_time_seconds"] = training_time
            wandb_run.finish()

        # Push to hub (hf_repo_prefix and HF access already validated above)
        if args.push_to_hub:
            url = utils.stgformer.push_to_hub(
                model_dir,
                repo_id,
                dataset_name,
                hf_repo_prefix=hf_repo_prefix,  # type: ignore (validated above)
                private=is_private,
                description=config.get("description"),
            )
            print(f"Pushed to: {url}")

    print("\nDone!")


if __name__ == "__main__":
    main()
