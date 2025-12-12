"""Shared utilities for formatting experiment results.

This module contains formatting functions for results tables that are
used by both baselines.py and results.py.
"""

from math import isclose
from typing import Optional

import pandas as pd

from utils.config import METRIC_HORIZONS, RESULTS_DIR, validate_dataset_name


def format_results_table(
    dataset_name: str,
    file_suffix: str = "results",
    model_order: Optional[list[str]] = None,
    return_dataframe: bool = False,
    decimals: int = 3,
    highlight_best: bool = False,
):
    """Format results as a markdown table.

    Args:
        dataset_name: Name of the dataset (METR-LA or PEMS-BAY)
        file_suffix: Suffix for the results file (e.g., "results", "baselines")
                    Results file will be {dataset}_{suffix}.csv
        model_order: Optional list to specify model ordering. If provided, only
                    models in this list will be included in the output.
        return_dataframe: If True, return DataFrame instead of markdown string
        decimals: Number of decimal places for formatting
        highlight_best: If True, bold the lowest value per metric column

    Returns:
        Markdown string or DataFrame with formatted results

    Raises:
        FileNotFoundError: If results file doesn't exist
    """
    dataset_name = validate_dataset_name(dataset_name)
    results_file = RESULTS_DIR / f"{dataset_name.lower()}_{file_suffix}.csv"

    if not results_file.exists():
        raise FileNotFoundError(f"No results file found for {dataset_name}")

    df = pd.read_csv(results_file)

    if model_order:
        # Only include models that exist in the results
        available = [m for m in model_order if m in df["model"].values]
        df = df.set_index("model").loc[available].reset_index()

    rows = []
    for i, horizon_name in enumerate(METRIC_HORIZONS.keys()):
        for j, metric_type in enumerate(["MAE", "RMSE", "MAPE"]):
            col_name = f"{metric_type.lower()}_{horizon_name}"
            best_value = None
            if highlight_best:
                metric_series = df[col_name].dropna()
                if not metric_series.empty:
                    best_value = metric_series.min()

            if j == 0:
                row_data = {"T": horizon_name, "Metric": metric_type}
            else:
                row_data = {"T": "", "Metric": metric_type}

            for _, model_row in df.iterrows():
                model_name = model_row["model"]
                value = model_row[col_name]

                if metric_type == "MAPE":
                    formatted_value = f"{value * 100:.{decimals}f}%"
                else:
                    formatted_value = f"{value:.{decimals}f}"

                if (
                    highlight_best
                    and best_value is not None
                    and pd.notna(value)
                    and isclose(value, best_value, rel_tol=0.0, abs_tol=1e-12)
                ):
                    formatted_value = f"**{formatted_value}**"

                row_data[model_name] = formatted_value

            rows.append(row_data)

        # Add empty separator row between horizons
        if i < len(METRIC_HORIZONS) - 1:
            empty_row = {"T": "", "Metric": ""}
            for _, model_row in df.iterrows():
                empty_row[model_row["model"]] = ""
            rows.append(empty_row)

    result_df = pd.DataFrame(rows)

    if return_dataframe:
        return result_df
    else:
        return result_df.to_markdown(index=False)
