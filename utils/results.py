"""Utilities for computing and managing experiment results (internal STGFormer variants)."""

from typing import Union

import pandas as pd

import utils.io
import utils.stgformer
from utils.baselines import calculate_metrics
from utils.config import RESULTS_DIR, validate_dataset_name
from utils.formatting import format_results_table as format_table
from utils.io import backup_results_file


def get_results_file(dataset_name: str):
    """Get path to results CSV file for a dataset."""
    dataset_name = validate_dataset_name(dataset_name)
    return RESULTS_DIR / f"{dataset_name.lower()}_results.csv"


def calculate_experiment_metrics(
    dataset_name: str,
    model_name: str,
    force: bool = False,
    null_vals: Union[float, list[float]] = 0.0,
    force_download: bool = False,
) -> pd.DataFrame:
    """Calculate metrics for an experiment model and save to results file.

    Args:
        dataset_name: Name of the dataset (METR-LA or PEMS-BAY)
        model_name: HuggingFace repo prefix for the model (e.g., STGFORMER_DOW)
        force: If True, recalculate even if results exist
        null_vals: Values to mask when computing metrics
        force_download: If True, force re-download from HuggingFace Hub

    Returns:
        DataFrame with all results for the dataset
    """
    dataset_name = validate_dataset_name(dataset_name)
    model_name = model_name.upper()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = get_results_file(dataset_name)

    if results_file.exists():
        df = pd.read_csv(results_file)
        existing_row = df[df["model"] == model_name].index
        if not force and len(existing_row) > 0:
            return df
    else:
        df = None
        existing_row = []

    dataset = utils.io.get_dataset_torch(dataset_name)

    # take speed only, time of day dimension not needed for error calcs
    y_true = dataset["test"].y.numpy()[..., 0:1]

    y_pred = utils.stgformer.get_predictions_hub(
        dataset_name,
        dataset,
        force_download=force_download,
        hf_repo_prefix=model_name,
    )

    metrics = calculate_metrics(y_pred, y_true, null_vals)
    metrics["model"] = model_name

    if df is not None and len(existing_row) > 0:
        # Update existing row
        df.loc[existing_row[0]] = metrics
    elif df is not None:
        # Append to existing dataframe
        df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
    else:
        # Create new dataframe
        df = pd.DataFrame([metrics])

    # Reorder columns to put 'model' first
    cols = ["model"] + [col for col in df.columns if col != "model"]
    df = df[cols]

    df.to_csv(results_file, index=False)

    return df


def backup_results(dataset_name: str):
    """Create a timestamped backup of results file for a dataset.

    Args:
        dataset_name: Name of the dataset (METR-LA or PEMS-BAY)

    Returns:
        Path to backup file if created, None otherwise
    """
    return backup_results_file(dataset_name, file_suffix="results")


def format_results_table(
    dataset_name: str,
    model_order: list[str] = None,
    return_dataframe: bool = False,
    decimals: int = 3,
    highlight_best: bool = False,
):
    """Format results as a markdown table.

    Args:
        dataset_name: Name of the dataset (METR-LA or PEMS-BAY)
        model_order: Optional list to specify model ordering
        return_dataframe: If True, return DataFrame instead of markdown string
        decimals: Number of decimal places for formatting

    Returns:
        Markdown string or DataFrame with formatted results
    """
    return format_table(
        dataset_name,
        file_suffix="results",
        model_order=model_order,
        return_dataframe=return_dataframe,
        decimals=decimals,
        highlight_best=highlight_best,
    )
