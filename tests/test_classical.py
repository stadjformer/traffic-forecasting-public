"""Tests for utils.classical module (VARIMA and VAR baselines)."""

import warnings

import numpy as np
import pytest
from darts import TimeSeries

from utils.classical import (
    interpolate_missing_values,
    prepare_darts_data_global,
    prepare_darts_data_per_sensor,
)

# Suppress Darts optional dependency warnings
warnings.filterwarnings("ignore", message=".*StatsForecast.*")
warnings.filterwarnings("ignore", message=".*XGBoost.*")


class TestInterpolateMissingValues:
    """Tests for interpolate_missing_values function."""

    def test_interpolate_zeros_in_speed(self):
        """Test that 0s in speed feature are interpolated."""
        # Create data with 0s in speed
        data = np.array(
            [
                [[10.0, 0.1], [20.0, 0.2]],  # t=0
                [[0.0, 0.15], [0.0, 0.25]],  # t=1 (missing)
                [[30.0, 0.2], [40.0, 0.3]],  # t=2
            ]
        )  # Shape: [3, 2, 2]

        result = interpolate_missing_values(data)

        # Speed at t=1 should be interpolated
        # Sensor 0: (10 + 30) / 2 = 20
        # Sensor 1: (20 + 40) / 2 = 30
        assert result[1, 0, 0] == pytest.approx(20.0)
        assert result[1, 1, 0] == pytest.approx(30.0)

        # Time feature should be unchanged
        assert result[1, 0, 1] == 0.15
        assert result[1, 1, 1] == 0.25

    def test_no_zeros_unchanged(self):
        """Test that data without 0s is unchanged."""
        data = np.array(
            [
                [[10.0, 0.1], [20.0, 0.2]],
                [[15.0, 0.15], [25.0, 0.25]],
                [[30.0, 0.2], [40.0, 0.3]],
            ]
        )

        result = interpolate_missing_values(data)
        np.testing.assert_array_equal(result, data)

    def test_boundary_zeros(self):
        """Test interpolation handles 0s at boundaries."""
        # 0s at start and end
        data = np.array(
            [
                [[0.0, 0.1], [20.0, 0.2]],  # t=0 (missing)
                [[15.0, 0.15], [25.0, 0.25]],  # t=1
                [[0.0, 0.2], [40.0, 0.3]],  # t=2 (missing)
            ]
        )

        result = interpolate_missing_values(data)

        # Should use forward/backward fill at boundaries
        assert result[0, 0, 0] > 0  # Not zero anymore
        assert result[2, 0, 0] > 0  # Not zero anymore


class TestPrepareDartsData:
    """Tests for Darts data preparation functions."""

    def test_prepare_per_sensor(self):
        """Test per-sensor TimeSeries creation."""
        data = np.random.randn(100, 5, 2).astype(np.float32)  # 100 timesteps, 5 sensors

        sensor_series = prepare_darts_data_per_sensor(data)

        assert len(sensor_series) == 5
        for ts in sensor_series:
            assert isinstance(ts, TimeSeries)
            assert ts.values().shape == (100, 2)
            assert list(ts.columns) == ["speed", "time_of_day"]

    def test_prepare_global(self):
        """Test global TimeSeries creation (speed only)."""
        data = np.random.randn(100, 5, 2).astype(np.float32)  # 100 timesteps, 5 sensors

        global_ts = prepare_darts_data_global(data)

        assert isinstance(global_ts, TimeSeries)
        assert global_ts.values().shape == (100, 5)  # 5 sensors, speed only
        assert len(global_ts.columns) == 5
        # Check column names are speed-only
        assert all("speed" in col for col in global_ts.columns)


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline_interpolation(self):
        """Test interpolation pipeline on synthetic data."""
        # Create synthetic dataset
        np.random.seed(42)
        timesteps = 100
        num_sensors = 3

        # Generate data with some 0s (missing values)
        data = np.random.randn(timesteps, num_sensors, 2).astype(np.float32)
        data[:, :, 0] = np.abs(data[:, :, 0]) * 10 + 50
        data[:, :, 1] = np.tile(np.linspace(0, 1, timesteps), (num_sensors, 1)).T

        # Add some missing values (0s)
        data[10:15, 0, 0] = 0.0
        data[30:35, 1, 0] = 0.0

        # Interpolate
        data_clean = interpolate_missing_values(data, verbose=False)
        assert not np.any(data_clean[:, :, 0] == 0)  # No more 0s

        # Prepare data formats
        sensor_series = prepare_darts_data_per_sensor(data_clean)
        assert len(sensor_series) == num_sensors

        global_ts = prepare_darts_data_global(data_clean)
        assert global_ts.values().shape == (timesteps, num_sensors)  # Speed only
