"""Tests for results management in utils.results module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from utils.config import METRIC_HORIZONS
from utils.results import (
    backup_results,
    calculate_experiment_metrics,
    format_results_table,
    get_results_file,
)


class TestGetResultsFile:
    """Tests for get_results_file function."""

    def test_get_results_file_metr_la(self):
        """Test getting results file path for METR-LA."""
        result_path = get_results_file("METR-LA")
        assert result_path.name == "metr-la_results.csv"

    def test_get_results_file_pems_bay(self):
        """Test getting results file path for PEMS-BAY."""
        result_path = get_results_file("PEMS-BAY")
        assert result_path.name == "pems-bay_results.csv"

    def test_get_results_file_case_insensitive(self):
        """Test that dataset name is case-insensitive."""
        result1 = get_results_file("metr-la")
        result2 = get_results_file("METR-LA")
        result3 = get_results_file("Metr-La")

        assert result1 == result2 == result3

    def test_get_results_file_invalid_dataset(self):
        """Test that invalid dataset name raises error."""
        with pytest.raises(ValueError, match="is not supported"):
            get_results_file("INVALID_DATASET")


class TestBackupResults:
    """Tests for backup_results function."""

    def test_backup_creates_timestamped_file(self):
        """Test that backup creates a timestamped backup file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mock results file
            results_file = tmpdir / "metr-la_results.csv"
            df = pd.DataFrame({"model": ["TEST"], "mae_15min": [1.5]})
            df.to_csv(results_file, index=False)

            # Patch RESULTS_DIR in io module (where backup_results_file is)
            with patch("utils.io.RESULTS_DIR", tmpdir):
                backup_path = backup_results("METR-LA")

            # Check backup was created
            assert backup_path is not None
            assert backup_path.exists()
            assert "backup_" in backup_path.name
            assert backup_path.name.startswith("metr-la_results.backup_")

            # Check content matches
            backup_df = pd.read_csv(backup_path)
            assert backup_df.equals(df)

    def test_backup_no_file_returns_none(self):
        """Test that backup returns None if results file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Patch RESULTS_DIR in io module (where backup_results_file is)
            with patch("utils.io.RESULTS_DIR", tmpdir):
                backup_path = backup_results("METR-LA")

            assert backup_path is None


class TestCalculateExperimentMetrics:
    """Tests for calculate_experiment_metrics function."""

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset."""
        mock_ds = MagicMock()
        # Create test data: (batch, horizon, num_nodes, features)
        y_data = torch.randn(100, 12, 50, 2)  # 2 features (speed, time-of-day)
        mock_ds["test"].y = y_data
        return mock_ds

    @pytest.fixture
    def mock_predictions(self):
        """Create mock predictions."""
        return np.random.randn(100, 12, 50, 1)

    def test_calculate_metrics_creates_csv(self, mock_dataset, mock_predictions):
        """Test that calculate_experiment_metrics creates results CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            results_file = tmpdir / "metr-la_results.csv"

            with patch("utils.results.RESULTS_DIR", tmpdir):
                with patch("utils.results.get_results_file", return_value=results_file):
                    with patch("utils.io.get_dataset_torch", return_value=mock_dataset):
                        with patch(
                            "utils.stgformer.get_predictions_hub",
                            return_value=mock_predictions,
                        ):
                            df = calculate_experiment_metrics(
                                "METR-LA", "TEST_MODEL", force=True
                            )

            # Check CSV was created
            assert results_file.exists()

            # Check dataframe has model column
            assert "model" in df.columns
            assert df["model"].iloc[0] == "TEST_MODEL"

            # Check dataframe has metric columns
            for horizon in METRIC_HORIZONS.keys():
                assert f"mae_{horizon}" in df.columns
                assert f"rmse_{horizon}" in df.columns
                assert f"mape_{horizon}" in df.columns

    def test_calculate_metrics_updates_existing_row(
        self, mock_dataset, mock_predictions
    ):
        """Test that calculate_experiment_metrics updates existing model row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            results_file = tmpdir / "metr-la_results.csv"

            # Create existing results file
            existing_df = pd.DataFrame(
                {
                    "model": ["TEST_MODEL", "OTHER_MODEL"],
                    "mae_15min": [1.5, 2.0],
                    "rmse_15min": [2.0, 2.5],
                    "mape_15min": [0.05, 0.06],
                }
            )
            results_file.parent.mkdir(parents=True, exist_ok=True)
            existing_df.to_csv(results_file, index=False)

            with patch("utils.results.RESULTS_DIR", tmpdir):
                with patch("utils.results.get_results_file", return_value=results_file):
                    with patch("utils.io.get_dataset_torch", return_value=mock_dataset):
                        with patch(
                            "utils.stgformer.get_predictions_hub",
                            return_value=mock_predictions,
                        ):
                            df = calculate_experiment_metrics(
                                "METR-LA", "TEST_MODEL", force=True
                            )

            # Should still have 2 rows (updated, not appended)
            assert len(df) == 2
            assert "TEST_MODEL" in df["model"].values
            assert "OTHER_MODEL" in df["model"].values

    def test_calculate_metrics_appends_new_model(self, mock_dataset, mock_predictions):
        """Test that calculate_experiment_metrics appends new model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            results_file = tmpdir / "metr-la_results.csv"

            # Create existing results file
            existing_df = pd.DataFrame(
                {
                    "model": ["OTHER_MODEL"],
                    "mae_15min": [2.0],
                    "rmse_15min": [2.5],
                    "mape_15min": [0.06],
                }
            )
            results_file.parent.mkdir(parents=True, exist_ok=True)
            existing_df.to_csv(results_file, index=False)

            with patch("utils.results.RESULTS_DIR", tmpdir):
                with patch("utils.results.get_results_file", return_value=results_file):
                    with patch("utils.io.get_dataset_torch", return_value=mock_dataset):
                        with patch(
                            "utils.stgformer.get_predictions_hub",
                            return_value=mock_predictions,
                        ):
                            df = calculate_experiment_metrics(
                                "METR-LA", "NEW_MODEL", force=True
                            )

            # Should have 2 rows now
            assert len(df) == 2
            assert "OTHER_MODEL" in df["model"].values
            assert "NEW_MODEL" in df["model"].values

    def test_calculate_metrics_skips_if_exists(self, mock_dataset, mock_predictions):
        """Test that calculate_experiment_metrics skips if results exist and force=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            results_file = tmpdir / "metr-la_results.csv"

            # Create existing results file
            existing_df = pd.DataFrame(
                {
                    "model": ["TEST_MODEL"],
                    "mae_15min": [1.5],
                    "rmse_15min": [2.0],
                    "mape_15min": [0.05],
                }
            )
            results_file.parent.mkdir(parents=True, exist_ok=True)
            existing_df.to_csv(results_file, index=False)

            with patch("utils.results.RESULTS_DIR", tmpdir):
                with patch("utils.results.get_results_file", return_value=results_file):
                    with patch(
                        "utils.io.get_dataset_torch", return_value=mock_dataset
                    ) as mock_get_ds:
                        with patch(
                            "utils.stgformer.get_predictions_hub",
                            return_value=mock_predictions,
                        ) as mock_get_pred:
                            df = calculate_experiment_metrics(
                                "METR-LA", "TEST_MODEL", force=False
                            )

            # Should not have called get_dataset or get_predictions
            mock_get_ds.assert_not_called()
            mock_get_pred.assert_not_called()

            # Should return existing dataframe
            assert len(df) == 1
            assert df["model"].iloc[0] == "TEST_MODEL"

    def test_calculate_metrics_model_first_column(self, mock_dataset, mock_predictions):
        """Test that 'model' column is first in results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            results_file = tmpdir / "metr-la_results.csv"

            with patch("utils.results.RESULTS_DIR", tmpdir):
                with patch("utils.results.get_results_file", return_value=results_file):
                    with patch("utils.io.get_dataset_torch", return_value=mock_dataset):
                        with patch(
                            "utils.stgformer.get_predictions_hub",
                            return_value=mock_predictions,
                        ):
                            df = calculate_experiment_metrics(
                                "METR-LA", "TEST_MODEL", force=True
                            )

            # 'model' should be first column
            assert df.columns[0] == "model"


class TestFormatResultsTable:
    """Tests for format_results_table function."""

    @pytest.fixture
    def sample_results_file(self, tmp_path):
        """Create a sample results CSV file."""
        results_file = tmp_path / "metr-la_results.csv"

        # Create sample results with all metric columns
        # Note: horizon names must match METRIC_HORIZONS keys ("15 min" not "15min")
        data = {
            "model": ["MODEL_A", "MODEL_B"],
            "mae_15 min": [1.234, 1.345],
            "rmse_15 min": [2.123, 2.234],
            "mape_15 min": [0.056, 0.067],
            "mae_30 min": [2.234, 2.345],
            "rmse_30 min": [3.123, 3.234],
            "mape_30 min": [0.078, 0.089],
            "mae_1 hour": [3.234, 3.345],
            "rmse_1 hour": [4.123, 4.234],
            "mape_1 hour": [0.090, 0.101],
        }
        df = pd.DataFrame(data)
        df.to_csv(results_file, index=False)
        return results_file

    def test_format_results_table_markdown_output(self, sample_results_file):
        """Test that format_results_table returns markdown string."""
        # Patch RESULTS_DIR in formatting module to use the temp directory
        with patch("utils.formatting.RESULTS_DIR", sample_results_file.parent):
            result = format_results_table("METR-LA", return_dataframe=False)

        # Should be a string
        assert isinstance(result, str)

        # Should contain markdown table elements
        assert "|" in result
        assert "MODEL_A" in result
        assert "MODEL_B" in result

    def test_format_results_table_dataframe_output(self, sample_results_file):
        """Test that format_results_table can return DataFrame."""
        with patch("utils.formatting.RESULTS_DIR", sample_results_file.parent):
            result = format_results_table("METR-LA", return_dataframe=True)

        # Should be a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Should have T and Metric columns
        assert "T" in result.columns
        assert "Metric" in result.columns

        # Should have model columns
        assert "MODEL_A" in result.columns
        assert "MODEL_B" in result.columns

    def test_format_results_table_includes_all_metrics(self, sample_results_file):
        """Test that formatted table includes all metrics."""
        with patch("utils.formatting.RESULTS_DIR", sample_results_file.parent):
            result = format_results_table("METR-LA", return_dataframe=True)

        # Should have rows for each metric type
        metrics = result["Metric"].unique()
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "MAPE" in metrics

    def test_format_results_table_percentage_for_mape(self, sample_results_file):
        """Test that MAPE values are formatted as percentages."""
        with patch("utils.formatting.RESULTS_DIR", sample_results_file.parent):
            result = format_results_table("METR-LA", return_dataframe=True, decimals=2)

        # Find MAPE rows
        mape_rows = result[result["Metric"] == "MAPE"]

        # Check that values contain '%'
        for col in ["MODEL_A", "MODEL_B"]:
            for val in mape_rows[col]:
                if val:  # Skip empty strings
                    assert "%" in val

    def test_format_results_table_custom_decimals(self, sample_results_file):
        """Test that custom decimal precision works."""
        with patch("utils.formatting.RESULTS_DIR", sample_results_file.parent):
            result = format_results_table("METR-LA", return_dataframe=True, decimals=1)

        # Find MAE row
        mae_row = result[result["Metric"] == "MAE"].iloc[0]

        # Check decimal precision (should have 1 decimal place)
        value = mae_row["MODEL_A"]
        # Value should be like "1.2", not "1.234"
        decimal_part = value.split(".")[-1]
        assert len(decimal_part) == 1

    def test_format_results_table_model_ordering(self, sample_results_file):
        """Test that model_order parameter controls column order."""
        with patch("utils.formatting.RESULTS_DIR", sample_results_file.parent):
            result = format_results_table(
                "METR-LA",
                model_order=["MODEL_B", "MODEL_A"],
                return_dataframe=True,
            )

        # MODEL_B should come before MODEL_A in columns
        columns = list(result.columns)
        model_b_idx = columns.index("MODEL_B")
        model_a_idx = columns.index("MODEL_A")
        assert model_b_idx < model_a_idx

    def test_format_results_table_filters_missing_models(self, sample_results_file):
        """Test that model_order filters out non-existent models."""
        with patch("utils.formatting.RESULTS_DIR", sample_results_file.parent):
            result = format_results_table(
                "METR-LA",
                model_order=["MODEL_A", "NONEXISTENT"],
                return_dataframe=True,
            )

        # Should only include MODEL_A
        assert "MODEL_A" in result.columns
        assert "NONEXISTENT" not in result.columns

    def test_format_results_table_missing_file_raises_error(self):
        """Test that missing results file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Patch RESULTS_DIR to point to empty tmpdir
            with patch("utils.formatting.RESULTS_DIR", tmpdir):
                with pytest.raises(FileNotFoundError, match="No results file found"):
                    format_results_table("METR-LA")

    def test_format_results_table_horizon_grouping(self, sample_results_file):
        """Test that results are grouped by horizon with separators."""
        with patch("utils.formatting.RESULTS_DIR", sample_results_file.parent):
            result = format_results_table("METR-LA", return_dataframe=True)

        # Should have rows for each horizon (15min, 30min, 1h)
        t_values = [t for t in result["T"].values if t]  # Filter empty strings
        assert "15min" in t_values or "3" in str(
            t_values
        )  # Depending on METRIC_HORIZONS format

    def test_format_results_table_empty_separator_rows(self, sample_results_file):
        """Test that empty separator rows exist between horizons."""
        with patch("utils.formatting.RESULTS_DIR", sample_results_file.parent):
            result = format_results_table("METR-LA", return_dataframe=True)

        # Should have some empty rows (separators)
        empty_rows = result[(result["T"] == "") & (result["Metric"] == "")]
        assert len(empty_rows) > 0


class TestResultsIntegration:
    """Integration tests for the full results workflow."""

    def test_full_workflow_create_backup_format(self):
        """Test complete workflow: calculate → backup → format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            results_file = tmpdir / "metr-la_results.csv"

            # Mock data
            mock_dataset = MagicMock()
            mock_dataset["test"].y = torch.randn(100, 12, 50, 2)
            mock_predictions = np.random.randn(100, 12, 50, 1)

            # Patch RESULTS_DIR in all locations: results for calculate, io for backup, formatting for format
            with patch("utils.results.RESULTS_DIR", tmpdir):
                with patch("utils.io.RESULTS_DIR", tmpdir):
                    with patch("utils.formatting.RESULTS_DIR", tmpdir):
                        with patch(
                            "utils.io.get_dataset_torch", return_value=mock_dataset
                        ):
                            with patch(
                                "utils.stgformer.get_predictions_hub",
                                return_value=mock_predictions,
                            ):
                                # Step 1: Calculate metrics
                                df = calculate_experiment_metrics(
                                    "METR-LA", "TEST_MODEL", force=True
                                )

                                assert results_file.exists()
                                assert len(df) == 1

                                # Step 2: Backup results
                                backup_path = backup_results("METR-LA")

                                assert backup_path is not None
                                assert backup_path.exists()

                                # Step 3: Format results
                                formatted = format_results_table(
                                    "METR-LA", return_dataframe=False
                                )

                                assert isinstance(formatted, str)
                                assert "TEST_MODEL" in formatted
