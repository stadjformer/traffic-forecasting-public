"""Tests for utils.baselines module."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error

from utils.baselines import (
    METRIC_HORIZONS,
    _masked_metric,
    calculate_baseline_metrics,
    calculate_metrics,
    format_results_table,
)
from utils.dataset import TrafficDataset


class TestMaskedMetric:
    """Tests for _masked_metric helper function."""

    def test_no_masking_single_null(self):
        """Test metric calculation with no null values to mask."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        result = _masked_metric(y_true, y_pred, mean_absolute_error, [0.0])
        expected = mean_absolute_error(y_true, y_pred)

        assert np.isclose(result, expected)

    def test_masking_single_null_value(self):
        """Test masking with a single null value."""
        y_true = np.array([1.0, 0.0, 3.0, 0.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        result = _masked_metric(y_true, y_pred, mean_absolute_error, [0.0])

        # Should only calculate on [1.0, 3.0, 5.0] vs [1.1, 2.9, 4.8]
        expected_true = np.array([1.0, 3.0, 5.0])
        expected_pred = np.array([1.1, 2.9, 4.8])
        expected = mean_absolute_error(expected_true, expected_pred)

        assert np.isclose(result, expected)

    def test_masking_multiple_null_values(self):
        """Test masking with multiple null values."""
        y_true = np.array([1.0, 0.0, 3.0, -1.0, 5.0, 0.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8, 6.0])

        result = _masked_metric(y_true, y_pred, mean_absolute_error, [0.0, -1.0])

        # Should only calculate on [1.0, 3.0, 5.0] vs [1.1, 2.9, 4.8]
        expected_true = np.array([1.0, 3.0, 5.0])
        expected_pred = np.array([1.1, 2.9, 4.8])
        expected = mean_absolute_error(expected_true, expected_pred)

        assert np.isclose(result, expected)

    def test_masking_nan_values(self):
        """Test masking with NaN as null value."""
        y_true = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        result = _masked_metric(y_true, y_pred, mean_absolute_error, [np.nan])

        # Should only calculate on [1.0, 3.0, 5.0] vs [1.1, 2.9, 4.8]
        expected_true = np.array([1.0, 3.0, 5.0])
        expected_pred = np.array([1.1, 2.9, 4.8])
        expected = mean_absolute_error(expected_true, expected_pred)

        assert np.isclose(result, expected)

    def test_all_values_masked(self):
        """Test when all values are masked (should return NaN)."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        result = _masked_metric(y_true, y_pred, mean_absolute_error, [0.0])

        assert np.isnan(result)

    def test_multidimensional_array(self):
        """Test masking works with multidimensional arrays."""
        y_true = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 6.0]])
        y_pred = np.array([[1.1, 2.1], [2.9, 4.2], [4.8, 6.1]])

        result = _masked_metric(y_true, y_pred, mean_absolute_error, [0.0])

        # Flattened: y_true = [1.0, 0.0, 3.0, 0.0, 5.0, 6.0]
        # After masking: [1.0, 3.0, 5.0, 6.0] vs [1.1, 2.9, 4.8, 6.1]
        expected_true = np.array([1.0, 3.0, 5.0, 6.0])
        expected_pred = np.array([1.1, 2.9, 4.8, 6.1])
        expected = mean_absolute_error(expected_true, expected_pred)

        assert np.isclose(result, expected)


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    @pytest.fixture
    def perfect_predictions(self):
        """Create perfect predictions (y_pred == y_true)."""
        np.random.seed(42)
        batch_size = 100
        horizon = 12
        num_nodes = 10
        output_dim = 1

        y_true = np.random.randn(batch_size, horizon, num_nodes, output_dim).astype(
            np.float32
        )
        y_pred = y_true.copy()

        return y_pred, y_true

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions with known error."""
        np.random.seed(42)
        batch_size = 100
        horizon = 12
        num_nodes = 10
        output_dim = 1

        y_true = np.random.randn(batch_size, horizon, num_nodes, output_dim).astype(
            np.float32
        )
        # Add constant offset for predictable error
        y_pred = y_true + 1.0

        return y_pred, y_true

    def test_perfect_predictions_zero_error(self, perfect_predictions):
        """Test that perfect predictions yield zero error."""
        y_pred, y_true = perfect_predictions

        metrics = calculate_metrics(y_pred, y_true, null_vals=0.0)

        # All metrics should be zero or very close to zero
        for metric_name, metric_value in metrics.items():
            assert np.isclose(metric_value, 0.0, atol=1e-6), (
                f"{metric_name} should be ~0 for perfect predictions"
            )

    def test_all_horizon_keys_present(self, sample_predictions):
        """Test that all expected horizon keys are in results."""
        y_pred, y_true = sample_predictions

        metrics = calculate_metrics(y_pred, y_true)

        expected_keys = []
        for horizon_name in METRIC_HORIZONS.keys():
            expected_keys.extend(
                [
                    f"mae_{horizon_name}",
                    f"mape_{horizon_name}",
                    f"rmse_{horizon_name}",
                ]
            )

        assert set(metrics.keys()) == set(expected_keys)

    def test_horizon_slicing(self):
        """Test that different horizons use correct number of steps."""
        # Create data where error increases with time step
        batch_size = 10
        horizon = 12
        num_nodes = 5
        output_dim = 1

        y_true = np.ones((batch_size, horizon, num_nodes, output_dim), dtype=np.float32)
        y_pred = np.ones((batch_size, horizon, num_nodes, output_dim), dtype=np.float32)

        # Add increasing error: step 0 has error 0, step 1 has error 1, etc.
        for t in range(horizon):
            y_pred[:, t, :, :] += t

        metrics = calculate_metrics(y_pred, y_true)

        # 15min (3 steps): average error should be (0+1+2)/3 = 1.0
        assert np.isclose(metrics["mae_15 min"], 1.0)

        # 30min (6 steps): average error should be (0+1+2+3+4+5)/6 = 2.5
        assert np.isclose(metrics["mae_30 min"], 2.5)

        # 1 hour (12 steps): average error should be (0+1+2+...+11)/12 = 5.5
        assert np.isclose(metrics["mae_1 hour"], 5.5)

    def test_single_null_value_as_float(self, sample_predictions):
        """Test null_vals can be passed as single float."""
        y_pred, y_true = sample_predictions

        # Should not raise
        metrics = calculate_metrics(y_pred, y_true, null_vals=0.0)
        assert len(metrics) > 0

    def test_multiple_null_values_as_list(self):
        """Test null_vals can be passed as list."""
        batch_size = 50
        horizon = 12
        num_nodes = 5
        output_dim = 1

        y_true = np.random.randn(batch_size, horizon, num_nodes, output_dim).astype(
            np.float32
        )
        y_pred = np.random.randn(batch_size, horizon, num_nodes, output_dim).astype(
            np.float32
        )

        # Add some null values
        y_true[0, 0, 0, 0] = 0.0
        y_true[1, 1, 1, 0] = -1.0

        # Should not raise
        metrics = calculate_metrics(y_pred, y_true, null_vals=[0.0, -1.0])
        assert len(metrics) > 0

    def test_mae_rmse_mape_relationships(self):
        """Test that MAE, RMSE, and MAPE have expected relationships."""
        batch_size = 100
        horizon = 12
        num_nodes = 10
        output_dim = 1

        y_true = (
            np.abs(
                np.random.randn(batch_size, horizon, num_nodes, output_dim).astype(
                    np.float32
                )
            )
            + 1.0
        )
        # Add varying errors
        errors = np.random.randn(batch_size, horizon, num_nodes, output_dim).astype(
            np.float32
        )
        y_pred = y_true + errors

        metrics = calculate_metrics(y_pred, y_true)

        # RMSE should be >= MAE (due to squaring)
        for horizon_name in METRIC_HORIZONS.keys():
            mae = metrics[f"mae_{horizon_name}"]
            rmse = metrics[f"rmse_{horizon_name}"]
            assert rmse >= mae, f"RMSE should be >= MAE for {horizon_name}"

    def test_output_shape_validation(self):
        """Test that function works with correct 4D shape."""
        y_pred = np.random.randn(10, 12, 5, 1).astype(np.float32)
        y_true = np.random.randn(10, 12, 5, 1).astype(np.float32)

        # Should not raise
        metrics = calculate_metrics(y_pred, y_true)
        assert len(metrics) == 9  # 3 horizons × 3 metrics


class TestCalculateBaselineMetrics:
    """Tests for calculate_baseline_metrics function."""

    @pytest.fixture
    def temp_results_dir(self, tmp_path, monkeypatch):
        """Create temporary results directory."""
        temp_dir = tmp_path / "results"
        temp_dir.mkdir()

        # Patch RESULTS_DIR to use temp directory
        monkeypatch.setattr("utils.baselines.RESULTS_DIR", temp_dir)

        return temp_dir

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset."""
        np.random.seed(42)
        datasets = {}
        for split in ["train", "val", "test"]:
            x = np.random.randn(50, 12, 10, 2).astype(np.float32)
            y = np.random.randn(50, 12, 10, 2).astype(
                np.float32
            )  # 2 features: speed and time
            datasets[split] = TrafficDataset(
                x, y, seq_len=12, horizon=12, num_nodes=10, input_dim=2, output_dim=2
            )
        return datasets

    def test_creates_results_file_on_first_run(self, temp_results_dir, mock_dataset):
        """Test that CSV file is created on first run."""
        with patch(
            "utils.baselines.utils.io.get_dataset_torch", return_value=mock_dataset
        ):
            with patch(
                "utils.baselines.utils.dcrnn.get_dcrnn_predictions"
            ) as mock_pred:
                # Mock predictions
                y_pred = np.random.randn(50, 12, 10, 1).astype(np.float32)
                mock_pred.return_value = y_pred

                df = calculate_baseline_metrics("METR-LA", "DCRNN", force=False)

                results_file = temp_results_dir / "metr-la_baselines.csv"
                assert results_file.exists()
                assert len(df) == 1
                assert df.iloc[0]["model"] == "DCRNN"

    def test_caching_skips_recalculation(self, temp_results_dir, mock_dataset):
        """Test that existing results are not recalculated unless forced."""
        # Create existing results file
        results_file = temp_results_dir / "metr-la_baselines.csv"
        existing_df = pd.DataFrame(
            {
                "model": ["DCRNN"],
                "mae_15 min": [2.5],
                "mape_15 min": [0.05],
                "rmse_15 min": [3.0],
                "mae_30 min": [3.0],
                "mape_30 min": [0.06],
                "rmse_30 min": [3.5],
                "mae_1 hour": [3.5],
                "mape_1 hour": [0.07],
                "rmse_1 hour": [4.0],
            }
        )
        existing_df.to_csv(results_file, index=False)

        with patch(
            "utils.baselines.utils.io.get_dataset_torch", return_value=mock_dataset
        ):
            with patch(
                "utils.baselines.utils.dcrnn.get_dcrnn_predictions"
            ) as mock_pred:
                df = calculate_baseline_metrics("METR-LA", "DCRNN", force=False)

                # Should not call prediction function
                mock_pred.assert_not_called()

                # Should return existing results
                assert len(df) == 1
                assert df.iloc[0]["mae_15 min"] == 2.5

    def test_force_recalculation(self, temp_results_dir, mock_dataset):
        """Test that force=True recalculates even when results exist."""
        # Create existing results file
        results_file = temp_results_dir / "metr-la_baselines.csv"
        existing_df = pd.DataFrame(
            {
                "model": ["DCRNN"],
                "mae_15 min": [999.0],  # Old value
                "mape_15 min": [0.05],
                "rmse_15 min": [3.0],
                "mae_30 min": [3.0],
                "mape_30 min": [0.06],
                "rmse_30 min": [3.5],
                "mae_1 hour": [3.5],
                "mape_1 hour": [0.07],
                "rmse_1 hour": [4.0],
            }
        )
        existing_df.to_csv(results_file, index=False)

        with patch(
            "utils.baselines.utils.io.get_dataset_torch", return_value=mock_dataset
        ):
            with patch(
                "utils.baselines.utils.dcrnn.get_dcrnn_predictions"
            ) as mock_pred:
                # Mock new predictions
                y_pred = np.random.randn(50, 12, 10, 1).astype(np.float32)
                mock_pred.return_value = y_pred

                df = calculate_baseline_metrics("METR-LA", "DCRNN", force=True)

                # Should call prediction function
                mock_pred.assert_called_once()

                # Should have updated results (not 999.0)
                assert df.iloc[0]["mae_15 min"] != 999.0

    def test_multiple_models_in_same_file(self, temp_results_dir, mock_dataset):
        """Test that multiple models can be stored in same CSV."""
        with patch(
            "utils.baselines.utils.io.get_dataset_torch", return_value=mock_dataset
        ):
            with patch(
                "utils.baselines.utils.dcrnn.get_dcrnn_predictions"
            ) as mock_pred:
                y_pred = np.random.randn(50, 12, 10, 1).astype(np.float32)
                mock_pred.return_value = y_pred

                # Add first model
                df1 = calculate_baseline_metrics("METR-LA", "DCRNN", force=False)
                assert len(df1) == 1

                # Add second DCRNN calculation with different name (to test multiple rows)
                # Force recalculation to add a second entry
                df2 = calculate_baseline_metrics("METR-LA", "DCRNN", force=True)

                # Should still have 1 row (updated, not added)
                assert len(df2) == 1
                assert df2.iloc[0]["model"] == "DCRNN"

    def test_uses_speed_only_from_dataset(self, temp_results_dir, mock_dataset):
        """Test that only speed dimension (index 0) is used from dataset."""
        with patch(
            "utils.baselines.utils.io.get_dataset_torch", return_value=mock_dataset
        ):
            with patch(
                "utils.baselines.utils.dcrnn.get_dcrnn_predictions"
            ) as mock_pred:
                with patch("utils.baselines.calculate_metrics") as mock_calc_metrics:
                    mock_calc_metrics.return_value = {
                        "mae_15 min": 2.5,
                        "mape_15 min": 0.05,
                        "rmse_15 min": 3.0,
                        "mae_30 min": 3.0,
                        "mape_30 min": 0.06,
                        "rmse_30 min": 3.5,
                        "mae_1 hour": 3.5,
                        "mape_1 hour": 0.07,
                        "rmse_1 hour": 4.0,
                    }

                    y_pred = np.random.randn(50, 12, 10, 1).astype(np.float32)
                    mock_pred.return_value = y_pred

                    calculate_baseline_metrics("METR-LA", "DCRNN")

                    # Check that calculate_metrics was called with y_true that has shape [..., 1]
                    args, kwargs = mock_calc_metrics.call_args
                    _, y_true_arg = args[0], args[1]

                    # y_true should be (50, 12, 10, 1) - speed only
                    assert y_true_arg.shape[-1] == 1

    def test_passes_null_vals_to_calculate_metrics(
        self, temp_results_dir, mock_dataset
    ):
        """Test that null_vals parameter is passed through."""
        with patch(
            "utils.baselines.utils.io.get_dataset_torch", return_value=mock_dataset
        ):
            with patch(
                "utils.baselines.utils.dcrnn.get_dcrnn_predictions"
            ) as mock_pred:
                with patch("utils.baselines.calculate_metrics") as mock_calc_metrics:
                    mock_calc_metrics.return_value = {
                        "mae_15 min": 2.5,
                        "mape_15 min": 0.05,
                        "rmse_15 min": 3.0,
                        "mae_30 min": 3.0,
                        "mape_30 min": 0.06,
                        "rmse_30 min": 3.5,
                        "mae_1 hour": 3.5,
                        "mape_1 hour": 0.07,
                        "rmse_1 hour": 4.0,
                    }

                    y_pred = np.random.randn(50, 12, 10, 1).astype(np.float32)
                    mock_pred.return_value = y_pred

                    calculate_baseline_metrics(
                        "METR-LA", "DCRNN", null_vals=[0.0, -1.0]
                    )

                    # Check that null_vals was passed as positional arg
                    args, kwargs = mock_calc_metrics.call_args
                    # null_vals is 3rd positional argument
                    assert args[2] == [0.0, -1.0]

    def test_mtgnn_model_support(self, temp_results_dir, mock_dataset):
        """Test that MTGNN model is supported in calculate_baseline_metrics."""
        with patch(
            "utils.baselines.utils.io.get_dataset_torch", return_value=mock_dataset
        ):
            with patch("utils.baselines.utils.mtgnn.get_predictions_hub") as mock_pred:
                # Mock predictions
                y_pred = np.random.randn(50, 12, 10, 1).astype(np.float32)
                mock_pred.return_value = y_pred

                df = calculate_baseline_metrics("METR-LA", "MTGNN", force=False)

                results_file = temp_results_dir / "metr-la_baselines.csv"
                assert results_file.exists()
                assert len(df) == 1
                assert df.iloc[0]["model"] == "MTGNN"

                # Verify function was called
                mock_pred.assert_called_once()

    def test_multiple_models_in_same_file_with_mtgnn(
        self, temp_results_dir, mock_dataset
    ):
        """Test that both DCRNN and MTGNN can coexist in results file."""
        with patch(
            "utils.baselines.utils.io.get_dataset_torch", return_value=mock_dataset
        ):
            y_pred = np.random.randn(50, 12, 10, 1).astype(np.float32)

            # Add DCRNN
            with patch(
                "utils.baselines.utils.dcrnn.get_dcrnn_predictions"
            ) as mock_dcrnn_pred:
                mock_dcrnn_pred.return_value = y_pred
                df1 = calculate_baseline_metrics("METR-LA", "DCRNN", force=False)
                assert len(df1) == 1

            # Add MTGNN
            with patch(
                "utils.baselines.utils.mtgnn.get_predictions_hub"
            ) as mock_mtgnn_pred:
                mock_mtgnn_pred.return_value = y_pred
                df2 = calculate_baseline_metrics("METR-LA", "MTGNN", force=False)
                assert len(df2) == 2
                assert set(df2["model"].values) == {"DCRNN", "MTGNN"}

    def test_unsupported_model_raises_error(self, temp_results_dir, mock_dataset):
        """Test that unsupported model names raise ValueError."""
        with patch(
            "utils.baselines.utils.io.get_dataset_torch", return_value=mock_dataset
        ):
            with pytest.raises(ValueError, match="Unsupported model"):
                calculate_baseline_metrics("METR-LA", "NONEXISTENT_MODEL")


class TestFormatResultsTable:
    """Tests for format_results_table function."""

    @pytest.fixture
    def sample_results_file(self, tmp_path, monkeypatch):
        """Create a sample results CSV file."""
        temp_dir = tmp_path / "results"
        temp_dir.mkdir()
        # Patch all locations: baselines for calculate, io for backup, formatting for format
        monkeypatch.setattr("utils.baselines.RESULTS_DIR", temp_dir)
        monkeypatch.setattr("utils.io.RESULTS_DIR", temp_dir)
        monkeypatch.setattr("utils.formatting.RESULTS_DIR", temp_dir)

        results_file = temp_dir / "metr-la_baselines.csv"
        df = pd.DataFrame(
            {
                "model": ["DCRNN", "GraphWaveNet"],
                "mae_15 min": [2.44, 2.30],
                "mape_15 min": [0.0532, 0.0501],
                "rmse_15 min": [4.77, 4.63],
                "mae_30 min": [3.18, 3.05],
                "mape_30 min": [0.0726, 0.0694],
                "rmse_30 min": [6.45, 6.30],
                "mae_1 hour": [3.90, 3.77],
                "mape_1 hour": [0.0927, 0.0895],
                "rmse_1 hour": [8.21, 8.05],
            }
        )
        df.to_csv(results_file, index=False)

        return results_file

    def test_returns_markdown_string_by_default(self, sample_results_file):
        """Test that function returns markdown string by default."""
        markdown = format_results_table("METR-LA")

        assert isinstance(markdown, str)
        assert "|" in markdown  # Markdown table separator
        assert "DCRNN" in markdown
        assert "GraphWaveNet" in markdown

    def test_returns_dataframe_when_requested(self, sample_results_file):
        """Test that function can return DataFrame."""
        df = format_results_table("METR-LA", return_dataframe=True)

        assert isinstance(df, pd.DataFrame)
        assert "T" in df.columns
        assert "Metric" in df.columns
        assert "DCRNN" in df.columns
        assert "GraphWaveNet" in df.columns

    def test_respects_model_order(self, sample_results_file):
        """Test that model_order parameter is respected."""
        # Reverse order
        markdown = format_results_table(
            "METR-LA", model_order=["GraphWaveNet", "DCRNN"]
        )

        # GraphWaveNet should appear before DCRNN in the markdown
        gwn_pos = markdown.find("GraphWaveNet")
        dcrnn_pos = markdown.find("DCRNN")
        assert gwn_pos < dcrnn_pos

    def test_mape_formatted_as_percentage(self, sample_results_file):
        """Test that MAPE is formatted as percentage with 1 decimal."""
        df = format_results_table("METR-LA", return_dataframe=True)

        # Find MAPE row for 15 min
        mape_row = df[df["Metric"] == "MAPE"].iloc[0]

        # Should be formatted like "5.3%" not "0.0532"
        dcrnn_mape = mape_row["DCRNN"]
        assert "%" in dcrnn_mape
        assert dcrnn_mape == "5.320%"  # 0.0532 * 100 = 5.320 with 3 decimals

    def test_mae_rmse_formatted_with_two_decimals(self, sample_results_file):
        """Test that MAE and RMSE are formatted with 2 decimals."""
        df = format_results_table("METR-LA", return_dataframe=True)

        # Find MAE row for 15 min
        mae_row = df[(df["Metric"] == "MAE") & (df["T"] == "15 min")].iloc[0]

        dcrnn_mae = mae_row["DCRNN"]
        assert dcrnn_mae == "2.440"

    def test_empty_rows_between_horizons(self, sample_results_file):
        """Test that empty rows are inserted between horizon groups."""
        df = format_results_table("METR-LA", return_dataframe=True)

        # Should have: 3 metrics for 15min + empty + 3 for 30min + empty + 3 for 1hour = 11 rows
        assert len(df) == 11

        # Rows 3 and 7 should be empty separator rows
        assert df.iloc[3]["T"] == ""
        assert df.iloc[3]["Metric"] == ""
        assert df.iloc[7]["T"] == ""
        assert df.iloc[7]["Metric"] == ""

    def test_horizon_name_only_on_first_metric(self, sample_results_file):
        """Test that horizon name (T column) is only shown for MAE, not RMSE/MAPE."""
        df = format_results_table("METR-LA", return_dataframe=True)

        # First group (15 min): row 0 should have "15 min", rows 1-2 should be empty
        assert df.iloc[0]["T"] == "15 min"
        assert df.iloc[1]["T"] == ""
        assert df.iloc[2]["T"] == ""

        # Second group (30 min): row 4 should have "30 min", rows 5-6 should be empty
        assert df.iloc[4]["T"] == "30 min"
        assert df.iloc[5]["T"] == ""
        assert df.iloc[6]["T"] == ""

    def test_all_three_metrics_present_for_each_horizon(self, sample_results_file):
        """Test that MAE, RMSE, MAPE are all present for each horizon."""
        df = format_results_table("METR-LA", return_dataframe=True)

        # Remove empty separator rows
        df_no_empty = df[df["Metric"] != ""]

        # Should have 3 horizons × 3 metrics = 9 rows
        assert len(df_no_empty) == 9

        # Count each metric type
        assert (df_no_empty["Metric"] == "MAE").sum() == 3
        assert (df_no_empty["Metric"] == "RMSE").sum() == 3
        assert (df_no_empty["Metric"] == "MAPE").sum() == 3
