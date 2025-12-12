#!/usr/bin/env python
"""Train MTGNN model on traffic forecasting datasets."""

import argparse
import utils.baselines
from utils.config import SUPPORTED_DATASETS
import utils.dataset
import utils.io
import utils.mtgnn
def train_mtgnn(
    dataset_name: str,
    push_to_hub: bool = False,
    hub_repo_id: str = None,
    dry_run: bool = False,
    verbose: bool = False,
):
    """
    Train MTGNN model on specified dataset.
    Args:
        dataset_name: Dataset name (e.g., "METR-LA", "PEMS-BAY")
        push_to_hub: Whether to push trained model to HuggingFace Hub
        hub_repo_id: HuggingFace repo ID (e.g., "username/MTGNN_METR-LA")
        dry_run: If True, set epochs to 1 for quick testing
        verbose: If True, print training progress every 50 iterations
    print(f"Training MTGNN on {dataset_name}...")
    # Load dataset
    print("Loading dataset...")
    pytorch_datasets = utils.io.get_dataset_torch(
        dataset_name=dataset_name, force_download=False, verbose=verbose
    )
    # Get adjacency matrix
    adjacency = pytorch_datasets["train"].adj_mx.numpy()
    # Set training parameters
    if dry_run:
        print("DRY RUN mode, setting epochs to 1")
        epochs = 1
    else:
        epochs = 100
    # Get model paths (includes _test suffix and privacy settings for dry-run)
    model_dir, auto_repo_id, is_private = utils.io.get_model_paths(
        "MTGNN", dataset_name, dry_run=dry_run
    model_dir.mkdir(parents=True, exist_ok=True)
    # Use provided hub_repo_id or fall back to auto-generated one
    if push_to_hub:
        if hub_repo_id is None:
            hub_repo_id = auto_repo_id
            print(f"Using auto-generated repo ID: {hub_repo_id}")
        # Validate HuggingFace Hub access early
        utils.io.validate_hf_hub_access(hub_repo_id, create_if_missing=True)
    print("Starting training...")
    import time
    start_time = time.time()
    # All hyperparameters use MTGNN defaults except epochs (configurable for dry-run)
    model = utils.mtgnn.train_model(
        dataset_name=dataset_name,
        pytorch_datasets=pytorch_datasets,
        adjacency=adjacency,
        save_dir=model_dir,
        verbose=verbose,
        epochs=epochs,
    end_time = time.time()
    training_time = end_time - start_time
    print(
        f"\nTraining completed in {training_time:.2f} seconds ({training_time / 60:.2f} minutes)"
    print(f"Best validation MAE: {model.metrics['vmae']:.4f}")
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_predictions = utils.mtgnn.get_mtgnn_predictions(
        model, pytorch_datasets["test"]
    test_true = pytorch_datasets["test"].y.numpy()[
        ..., 0:1
    ]  # Only first feature (speed)
    # Calculate test metrics (excluding null values)
    metrics = utils.baselines.calculate_metrics(
        test_predictions, test_true, null_vals=0.0
    print("\nTest Metrics:")
        f"  15 min - MAE: {metrics['mae_15 min']:.4f}, MAPE: {metrics['mape_15 min'] * 100:.4f}%, RMSE: {metrics['rmse_15 min']:.4f}"
        f"  30 min - MAE: {metrics['mae_30 min']:.4f}, MAPE: {metrics['mape_30 min'] * 100:.4f}%, RMSE: {metrics['rmse_30 min']:.4f}"
        f"  1 hour - MAE: {metrics['mae_1 hour']:.4f}, MAPE: {metrics['mape_1 hour'] * 100:.4f}%, RMSE: {metrics['rmse_1 hour']:.4f}"
        print(f"\nPushing model to HuggingFace Hub: {hub_repo_id}")
        if dry_run:
            print(f"  (dry-run mode: private={is_private})")
        url = utils.mtgnn.push_to_hub(
            checkpoint_dir=model_dir,
            repo_id=hub_repo_id,
            dataset_name=dataset_name,
            metrics={
                "Validation MAE": float(model.metrics["vmae"]),
                "Test MAE (15 min)": float(metrics["mae_15 min"]),
                "Test MAPE (15 min)": float(metrics["mape_15 min"]),
                "Test RMSE (15 min)": float(metrics["rmse_15 min"]),
            },
            private=is_private,
        )
        print(f"Model pushed to: {url}")
    return model
def main():
    parser = argparse.ArgumentParser(
        description="Train MTGNN model on traffic forecasting datasets"
    parser.add_argument(
        "dataset",
        type=str,
        choices=list(SUPPORTED_DATASETS.keys()),
        help="Dataset name to train on",
        "--push-to-hub",
        action="store_true",
        help="Push trained model to HuggingFace Hub",
        "--hub-repo-id",
        default=None,
        help="HuggingFace repo ID (e.g., username/MTGNN_METR-LA)",
        "--dry-run",
        help="Quick test run with 1 epoch (skip full training, test data loading only)",
        "--verbose",
        help="Print training progress every 50 iterations (default: off)",
    args = parser.parse_args()
    train_mtgnn(
        dataset_name=args.dataset,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,  # Pass through as-is, let train_mtgnn handle it
        dry_run=args.dry_run,
        verbose=args.verbose,
    print("\nTraining complete!")
if __name__ == "__main__":
    main()
