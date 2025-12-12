import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import datasets
import huggingface_hub
import numpy as np
import pandas as pd
from huggingface_hub import HfApi

import utils.dataset
from utils.config import (
    DATA_DIR,
    HF_USERNAME_UPLOAD,
    MODELS_DIR,
    RESULTS_DIR,
    SUPPORTED_DATASETS,
    validate_dataset_name,
)
from utils.dataset import TrafficDataset


def get_dataset_hf(
    dataset_name: str, force_download: bool = False
) -> datasets.DatasetDict:
    """
    Returns the dataset (with splits predefined) in HuggingFace's DatasetDict format
    """
    _fetch_data(dataset_name=dataset_name, force_download=force_download)
    local_dir, _ = _get_local_data_dir_and_hf_uri(dataset_name=dataset_name)

    return datasets.load_dataset(
        "parquet",
        data_files={
            "train": str(local_dir / "train.parquet"),
            "validation": str(local_dir / "val.parquet"),
            "test": str(local_dir / "test.parquet"),
        },
    )


def get_dataset_torch(
    dataset_name: str,
    force_download: bool = False,
    verbose: bool = False,
) -> Dict[str, TrafficDataset]:
    """
    Returns the dataset (with splits predefined) in internal Dict[str, TrafficDataset] format (PyTorch compatible)

    Features: [speed, time_of_day, day_of_week] - DOW computed from timestamps automatically.
    """
    dataset_name = validate_dataset_name(dataset_name)
    cache_dir = DATA_DIR / dataset_name / "pytorch_cache"
    cache_file = cache_dir / "datasets.pt"

    if not force_download and cache_file.exists():
        verbose and print(f"Loading PyTorch datasets from cache: {cache_file}")
        return TrafficDataset.deserialise(cache_file)

    dataset_hf = get_dataset_hf(
        dataset_name=dataset_name, force_download=force_download
    )

    adj_mx, _, _ = get_graph_metadata(dataset_name=dataset_name, force_download=False)
    verbose and print("Converting from HuggingFace format...")
    pytorch_datasets = utils.dataset.hf_to_pytorch(
        dataset_hf, adj_mx=adj_mx, verbose=verbose, add_dow=True
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    verbose and print(f"Saving PyTorch datasets to cache: {cache_file}")
    TrafficDataset.serialise(pytorch_datasets, cache_file)

    return pytorch_datasets


def get_graph_metadata(
    dataset_name: str, force_download=False
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Returns adjacency matrix (as constructed and used by DCRNN project), sensor distances and sensor locations
    """
    _fetch_data(dataset_name=dataset_name, force_download=force_download)

    local_dir, _ = _get_local_data_dir_and_hf_uri(dataset_name=dataset_name)

    graph_dir = local_dir / "sensor_graph"

    adj_mx = np.load(graph_dir / "adj_mx.npy")
    distances = pd.read_csv(graph_dir / "distances.csv")
    sensor_locations = pd.read_csv(graph_dir / "sensor_locations.csv")

    return adj_mx, distances, sensor_locations


def _fetch_data(dataset_name, force_download=False):
    local_dir, hf_uri = _get_local_data_dir_and_hf_uri(dataset_name=dataset_name)

    os.makedirs(local_dir, exist_ok=True)

    # Check if all required files exist (not just README.md)
    required_files = ["train.parquet", "val.parquet", "test.parquet", "README.md"]
    files_exist = all((local_dir / f).exists() for f in required_files)

    if force_download or not files_exist:
        huggingface_hub.snapshot_download(
            repo_id=hf_uri,
            repo_type="dataset",
            local_dir=local_dir,
        )


def _get_local_data_dir_and_hf_uri(dataset_name):
    dataset_name = validate_dataset_name(dataset_name)

    local_dir = DATA_DIR / dataset_name
    hf_uri = SUPPORTED_DATASETS[dataset_name]

    return local_dir, hf_uri


def validate_hf_hub_access(
    repo_id: str, create_if_missing: bool = True, verbose: bool = True
) -> None:
    """
    Validate that HuggingFace Hub can be accessed and user has write permissions.

    This should be called at the start of training when push_to_hub=True to fail early
    rather than after hours of training.

    Args:
        repo_id: Repository ID (e.g., "username/model-name")
        create_if_missing: If True, create the repo if it doesn't exist
        verbose: If True, print validation progress (default: True)

    Raises:
        RuntimeError: If authentication fails or permissions are insufficient
        ValueError: If repo_id format is invalid
    """
    if "/" not in repo_id:
        raise ValueError(
            f"Invalid repo_id format: '{repo_id}'. Expected format: 'username/repo-name'"
        )

    if verbose:
        print("\nValidating HuggingFace Hub access...")

    api = HfApi()

    # Check if user is authenticated
    try:
        user_info = api.whoami()
        username = user_info["name"]
        verbose and print(f"  Authenticated as: {username}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to authenticate with HuggingFace Hub. "
            f"Please run: huggingface-cli login\n"
            f"Error: {e}"
        )

    # Validate repo_id matches authenticated user
    repo_owner = repo_id.split("/")[0]
    if repo_owner != username:
        verbose and print(
            f"  Warning: repo owner '{repo_owner}' doesn't match "
            f"authenticated user '{username}'. Upload may fail if you don't have access."
        )

    # Check if repo exists or try to create it
    try:
        # Try to get repo info (this will fail if repo doesn't exist)
        api.repo_info(repo_id=repo_id, repo_type="model")
        verbose and print(f"  Repository: https://huggingface.co/{repo_id}")
    except huggingface_hub.utils.RepositoryNotFoundError:
        if create_if_missing:
            try:
                api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
                verbose and print(
                    f"  Created repository: https://huggingface.co/{repo_id}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create repository '{repo_id}'. "
                    f"Please check your permissions.\n"
                    f"Error: {e}"
                )
        else:
            raise RuntimeError(
                f"Repository '{repo_id}' does not exist and create_if_missing=False"
            )
    except Exception as e:
        raise RuntimeError(
            f"Failed to access repository '{repo_id}'. "
            f"Please check your permissions and network connection.\n"
            f"Error: {e}"
        )

    # Test write access by uploading a tiny test file
    try:
        import io as builtin_io

        test_content = "# HF write access test\nThis file confirms write permissions."
        test_file = builtin_io.BytesIO(test_content.encode())

        api.upload_file(
            path_or_fileobj=test_file,
            path_in_repo=".hf_write_test",
            repo_id=repo_id,
            repo_type="model",
            commit_message="test: validate write access",
        )

        # Clean up the test file
        api.delete_file(
            path_in_repo=".hf_write_test",
            repo_id=repo_id,
            repo_type="model",
            commit_message="test: cleanup write access test",
        )

        verbose and print("  Write access confirmed - ready for upload\n")
    except Exception as e:
        raise RuntimeError(
            f"Cannot write to repository '{repo_id}'. You may not have write permissions.\n"
            f"Error: {e}"
        )


def get_model_paths(
    model_name: str, dataset_name: str, dry_run: bool = False
) -> tuple[Path, str, bool]:
    """
    Get standardized model directory path, repo ID, and privacy setting.

    For dry-run mode:
    - Adds "_dry_run" suffix to local directory and repo ID
    - Makes repository private

    Args:
        model_name: Model name (e.g., "MTGNN", "GWNET", "DCRNN")
        dataset_name: Dataset name (e.g., "METR-LA", "PEMS-BAY")
        dry_run: If True, use dry-run paths and private repos

    Returns:
        tuple of (local_dir, repo_id, is_private)
        - local_dir: Path to local model directory
        - repo_id: HuggingFace repository ID (format: "username/MODEL_DATASET[_dry_run]")
        - is_private: Whether the repository should be private

    Example:
        >>> get_model_paths("MTGNN", "METR-LA", dry_run=False)
        (Path('models/MTGNN/model_metr-la'), 'username/MTGNN_METR-LA', False)

        >>> get_model_paths("MTGNN", "METR-LA", dry_run=True)
        (Path('models/MTGNN_dry_run/model_metr-la'), 'username/MTGNN_METR-LA_dry_run', True)
    """
    # Normalize names
    model_name_upper = model_name.upper()
    dataset_name_lower = dataset_name.lower()
    dataset_name_upper = dataset_name.upper()

    # Local directory path
    if dry_run:
        # Dry-run: models/MTGNN_dry_run/model_metr-la/
        local_dir = (
            MODELS_DIR / f"{model_name_upper}_dry_run" / f"model_{dataset_name_lower}"
        )
    else:
        # Production: models/MTGNN/model_metr-la/
        local_dir = MODELS_DIR / model_name_upper / f"model_{dataset_name_lower}"

    # Get username from config
    username = HF_USERNAME_UPLOAD
    if not username:
        raise RuntimeError(
            "Cannot construct repo_id: 'HF_USERNAME_FOR_UPLOAD' not set in .env_public"
        )

    # Construct repo ID
    if dry_run:
        # Dry-run: username/MTGNN_METR-LA_dry_run (private)
        repo_id = f"{username}/{model_name_upper}_{dataset_name_upper}_dry_run"
    else:
        # Production: username/MTGNN_METR-LA (public)
        repo_id = f"{username}/{model_name_upper}_{dataset_name_upper}"

    # Dry-run repos are private
    is_private = dry_run

    return local_dir, repo_id, is_private


def backup_results_file(
    dataset_name: str, file_suffix: str = "results"
) -> Optional[Path]:
    """Create a timestamped backup of a results file for a dataset.

    Args:
        dataset_name: Name of the dataset (METR-LA or PEMS-BAY)
        file_suffix: Suffix for the results file (e.g., "results", "baselines")
                    Results file will be {dataset}_{suffix}.csv

    Returns:
        Path to backup file if created, None otherwise
    """
    dataset_name = validate_dataset_name(dataset_name)
    results_file = RESULTS_DIR / f"{dataset_name.lower()}_{file_suffix}.csv"

    if results_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = (
            RESULTS_DIR / f"{dataset_name.lower()}_{file_suffix}.backup_{timestamp}.csv"
        )
        shutil.copy2(results_file, backup_file)
        return backup_file
    return None
