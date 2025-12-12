#!/usr/bin/env python
"""Train DCRNN model on traffic forecasting datasets."""

import argparse
import time

import utils
import utils.dataset
import utils.dcrnn
from utils.config import SUPPORTED_DATASETS


def train_dcrnn(
    dataset_name: str,
    push_to_hub: bool = False,
    hub_repo_id: str = None,
    dry_run: bool = False,
):
    """
    Train DCRNN model on specified dataset.

    Args:
        dataset_name: Dataset name (e.g., "METR-LA", "PEMS-BAY")
        push_to_hub: Whether to push trained model to HuggingFace Hub
        hub_repo_id: HuggingFace repo ID (e.g., "username/dcrnn-metr-la")
        dry_run: If True, set epochs to 0 for quick testing
        cleanup: If True, remove local checkpoint directory after upload (default: True)

    Returns:
        supervisor: Trained DCRNNSupervisor instance
    """
    print(f"Training DCRNN on {dataset_name}...")

    # Get model paths (includes _test suffix and privacy settings for dry-run)
    model_dir, auto_repo_id, is_private = utils.io.get_model_paths(
        "DCRNN", dataset_name, dry_run=dry_run
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    # Use provided hub_repo_id or fall back to auto-generated one
    if push_to_hub:
        if hub_repo_id is None:
            hub_repo_id = auto_repo_id
            print(f"Using auto-generated repo ID: {hub_repo_id}")

        # Validate HuggingFace Hub access early
        utils.io.validate_hf_hub_access(hub_repo_id, create_if_missing=True)

    supervisor, _ = utils.dcrnn.get_supervisor_and_data(dataset_name=dataset_name)

    if dry_run:
        print("DRY RUN mode, skipping training")
        training_time = 0
    else:
        print("Starting training...")
        start_time = time.time()
        supervisor.train(save_model=0)
        end_time = time.time()
        training_time = end_time - start_time
        print(
            f"Training completed in {training_time:.2f} seconds ({training_time / 60:.2f} minutes)"
        )

    val_score, _ = supervisor.evaluate("val")
    print(f"MAE (val): {val_score}")
    if not dry_run:
        print(f"Total training time: {training_time:.2f}s ({training_time / 60:.2f}m)")

    model_path = supervisor.save_best_to_dir(str(model_dir))
    print(f"Model saved to: {model_path}")

    if push_to_hub:
        print(f"\nPushing model to HuggingFace Hub: {hub_repo_id}")
        if dry_run:
            print(f"  (dry-run mode: private={is_private})")

        url = utils.dcrnn.push_to_hub(
            checkpoint_path=str(model_dir),
            repo_id=hub_repo_id,
            dataset_name=dataset_name,
            metrics={"MAE (val split)": float(val_score)},
            private=is_private,
        )
        print(f"Model pushed to: {url}")

    return


def main():
    parser = argparse.ArgumentParser(
        description="Train DCRNN model on traffic forecasting datasets"
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=list(SUPPORTED_DATASETS.keys()),
        help="Dataset name to train on",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push trained model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID (e.g., username/dcrnn-metr-la)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick test run with 0 epochs (skip training, test data loading only)",
    )

    args = parser.parse_args()

    train_dcrnn(
        dataset_name=args.dataset,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,  # Pass through as-is, let train_dcrnn handle it
        dry_run=args.dry_run,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
