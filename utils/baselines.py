from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

import utils.dcrnn
import utils.gwnet
import utils.io
import utils.mtgnn
import utils.stgformer
import utils.stgformer_external
from utils.config import METRIC_HORIZONS, RESULTS_DIR, validate_dataset_name
from utils.formatting import format_results_table as format_table
from utils.io import backup_results_file


# =============================================================================
# NaN Handling Utilities (matches utils/training.py pattern)
# =============================================================================

def prepare_predictions_and_mask_np(y_pred, y_true, null_vals):
    """Prepare predictions and compute valid mask for metrics (NumPy version).

    This implements the standard NaN handling policy:
    - y_pred NaN → Replace with 0 (model's fault, should be penalized)
    - y_true NaN → Mask out (missing data, not model's fault)
    - y_true == null_val → Mask out (explicit null marker)

    Args:
        y_pred: Model predictions
        y_true: Ground truth labels
        null_vals: List of values to treat as null/missing in y_true

    Returns:
        Tuple of (cleaned_preds, valid_mask)
    """
    # Replace NaN in predictions with 0 (model's fault if it produces NaN)
    y_pred_clean = np.nan_to_num(y_pred, nan=0.0)

    # Mask NaN in labels (missing data, not model's fault)
    mask = ~np.isnan(y_true)

    # Also mask null_vals in labels
    for null_val in null_vals:
        if not np.isnan(null_val):
            mask &= y_true != null_val

    return y_pred_clean, mask


def _masked_metric(y_true, y_pred, metric_fn, null_vals):
    """Compute metric with proper NaN handling.

    Uses prepare_predictions_and_mask_np() for consistent NaN handling:
    - y_pred NaN → replaced with 0 (penalized)
    - y_true NaN → masked out (not counted)
    """
    y_pred_clean, mask = prepare_predictions_and_mask_np(y_pred, y_true, null_vals)
    if not mask.any():
        return np.nan
    return metric_fn(y_true[mask], y_pred_clean[mask])


def calculate_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    null_vals: Union[float, list[float]] = 0.0,
) -> dict:
    if isinstance(null_vals, (int, float)):
        null_vals = [null_vals]

    results = {}

    for horizon_name, horizon_steps in METRIC_HORIZONS.items():
        y_true_horizon = y_true[:, :horizon_steps, :, :]
        y_pred_horizon = y_pred[:, :horizon_steps, :, :]

        y_true_flat = y_true_horizon.flatten()
        y_pred_flat = y_pred_horizon.flatten()

        mae = _masked_metric(y_true_flat, y_pred_flat, mean_absolute_error, null_vals)
        mape = _masked_metric(
            y_true_flat, y_pred_flat, mean_absolute_percentage_error, null_vals
        )
        mse = _masked_metric(y_true_flat, y_pred_flat, mean_squared_error, null_vals)
        rmse = np.sqrt(mse)

        results[f"mae_{horizon_name}"] = mae
        results[f"mape_{horizon_name}"] = mape
        results[f"rmse_{horizon_name}"] = rmse

    return results


def calculate_baseline_metrics(
    dataset_name: str,
    model_name: str,
    force: bool = False,
    null_vals: Union[float, list[float]] = 0.0,
    force_download: bool = False,
) -> pd.DataFrame:
    dataset_name = validate_dataset_name(dataset_name)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / f"{dataset_name.lower()}_baselines.csv"

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

    if model_name.upper() == "DCRNN":
        y_pred = utils.dcrnn.get_dcrnn_predictions(dataset_name, dataset)
    elif model_name.upper() == "MTGNN":
        y_pred = utils.mtgnn.get_predictions_hub(
            dataset_name, dataset, force_download=force_download
        )
    elif model_name.upper() == "GWNET":
        y_pred = utils.gwnet.get_predictions_hub(
            dataset_name, dataset, force_download=force_download
        )
    elif model_name.upper() == "STGFORMER":
        y_pred = utils.stgformer_external.get_predictions_hub(
            dataset_name, dataset, force_download=force_download
        )
    elif model_name.upper() == "STGFORMER_INTERNAL":
        y_pred = utils.stgformer.get_predictions_hub(
            dataset_name, dataset, force_download=force_download
        )
    elif model_name.upper() == "STGFORMER_INTERNAL_DOW":
        y_pred = utils.stgformer.get_predictions_hub(
            dataset_name,
            dataset,
            force_download=force_download,
            hf_repo_prefix="STGFORMER_INTERNAL_DOW",
        )
    elif model_name.upper() == "STGFORMER_FINAL":
        y_pred = utils.stgformer.get_predictions_hub(
            dataset_name,
            dataset,
            force_download=force_download,
            hf_repo_prefix="STGFORMER_FINAL",
        )
    elif model_name.upper() == "STGFORMER_BS200":
        y_pred = utils.stgformer.get_predictions_hub(
            dataset_name,
            dataset,
            force_download=force_download,
            hf_repo_prefix="STGFORMER_BS200",
        )
    elif model_name.upper() == "STGFORMER_DOW":
        y_pred = utils.stgformer.get_predictions_hub(
            dataset_name,
            dataset,
            force_download=force_download,
            hf_repo_prefix="STGFORMER_TOD",
        )
    elif model_name.upper() == "STGFORMER_GEO":
        y_pred = utils.stgformer.get_predictions_hub(
            dataset_name,
            dataset,
            force_download=force_download,
            hf_repo_prefix="STGFORMER_GEO",
        )
    elif model_name.upper() == "STGFORMER_SPECTRAL_INIT":
        y_pred = utils.stgformer.get_predictions_hub(
            dataset_name,
            dataset,
            force_download=force_download,
            hf_repo_prefix="STGFORMER_SPECTRAL_INIT",
        )
    elif model_name.upper() == "STGFORMER_CHEBYSHEV":
        y_pred = utils.stgformer.get_predictions_hub(
            dataset_name,
            dataset,
            force_download=force_download,
            hf_repo_prefix="STGFORMER_CHEBYSHEV",
        )
    elif model_name.upper() == "STGFORMER_HYBRID":
        y_pred = utils.stgformer.get_predictions_hub(
            dataset_name,
            dataset,
            force_download=force_download,
            hf_repo_prefix="STGFORMER_HYBRID",
        )
    elif model_name.upper() in ("AR", "ARIMA", "VARIMA", "VAR"):
        # Lazy import to avoid darts startup messages
        from utils import classical

        if model_name.upper() == "AR":
            y_pred = classical.get_ar_predictions(
                dataset_name, dataset, split="test", verbose=True
            )
        elif model_name.upper() == "ARIMA":
            y_pred = classical.get_arima_predictions(
                dataset_name, dataset, split="test", verbose=True
            )
        elif model_name.upper() == "VARIMA":
            y_pred = classical.get_varima_predictions(
                dataset_name, dataset, split="test", verbose=True
            )
        else:  # VAR
            y_pred = classical.get_var_predictions(
                dataset_name, dataset, split="test", verbose=True
            )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

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
    """Create a timestamped backup of baselines results file for a dataset.

    Args:
        dataset_name: Name of the dataset (METR-LA or PEMS-BAY)

    Returns:
        Path to backup file if created, None otherwise
    """
    return backup_results_file(dataset_name, file_suffix="baselines")


def format_results_table(
    dataset_name: str,
    model_order: list[str] = None,
    return_dataframe: bool = False,
    decimals: int = 3,
    highlight_best: bool = False,
):
    """Format baselines results as a markdown table.

    Args:
        dataset_name: Name of the dataset (METR-LA or PEMS-BAY)
        model_order: Optional list to specify model ordering
        return_dataframe: If True, return DataFrame instead of markdown string
        decimals: Number of decimal places for formatting
        highlight_best: If True, bold the lowest value per metric column

    Returns:
        Markdown string or DataFrame with formatted results
    """
    return format_table(
        dataset_name,
        file_suffix="baselines",
        model_order=model_order,
        return_dataframe=return_dataframe,
        decimals=decimals,
        highlight_best=highlight_best,
    )
