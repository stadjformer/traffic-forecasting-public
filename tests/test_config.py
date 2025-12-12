"""Tests for utils.config module."""

from pathlib import Path

import pytest

from utils.config import (
    DATA_DIR,
    DCRNN_CONFIGS,
    METRIC_HORIZONS,
    MODELS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    SUPPORTED_DATASETS,
    validate_dataset_name,
)


class TestConstants:
    """Test that constants are correctly defined."""

    def test_project_root_is_path(self):
        """Test that PROJECT_ROOT is a Path object."""
        assert isinstance(PROJECT_ROOT, Path)
        assert PROJECT_ROOT.exists()

    def test_directory_constants_are_paths(self):
        """Test that directory constants are Path objects."""
        assert isinstance(DATA_DIR, Path)
        assert isinstance(MODELS_DIR, Path)
        assert isinstance(RESULTS_DIR, Path)

    def test_directory_constants_relative_to_root(self):
        """Test that directories are relative to PROJECT_ROOT."""
        assert DATA_DIR == PROJECT_ROOT / "data"
        assert MODELS_DIR == PROJECT_ROOT / "models"
        assert RESULTS_DIR == PROJECT_ROOT / "results"

    def test_supported_datasets_structure(self):
        """Test SUPPORTED_DATASETS has expected structure."""
        assert isinstance(SUPPORTED_DATASETS, dict)
        assert "METR-LA" in SUPPORTED_DATASETS
        assert "PEMS-BAY" in SUPPORTED_DATASETS
        assert SUPPORTED_DATASETS["METR-LA"] == "witgaw/METR-LA"
        assert SUPPORTED_DATASETS["PEMS-BAY"] == "witgaw/PEMS-BAY"

    def test_dcrnn_configs_structure(self):
        """Test DCRNN_CONFIGS has expected structure."""
        assert isinstance(DCRNN_CONFIGS, dict)
        assert "METR-LA" in DCRNN_CONFIGS
        assert "PEMS-BAY" in DCRNN_CONFIGS
        assert DCRNN_CONFIGS["METR-LA"] == "dcrnn_la.yaml"
        assert DCRNN_CONFIGS["PEMS-BAY"] == "dcrnn_bay.yaml"

    def test_metric_horizons_structure(self):
        """Test METRIC_HORIZONS has expected structure."""
        assert isinstance(METRIC_HORIZONS, dict)
        assert METRIC_HORIZONS["15 min"] == 3
        assert METRIC_HORIZONS["30 min"] == 6
        assert METRIC_HORIZONS["1 hour"] == 12


class TestValidateDatasetName:
    """Tests for validate_dataset_name function."""

    def test_validate_metr_la_uppercase(self):
        """Test validation with METR-LA in uppercase."""
        result = validate_dataset_name("METR-LA")
        assert result == "METR-LA"

    def test_validate_pems_bay_uppercase(self):
        """Test validation with PEMS-BAY in uppercase."""
        result = validate_dataset_name("PEMS-BAY")
        assert result == "PEMS-BAY"

    def test_validate_metr_la_lowercase(self):
        """Test validation normalizes lowercase to uppercase."""
        result = validate_dataset_name("metr-la")
        assert result == "METR-LA"

    def test_validate_pems_bay_lowercase(self):
        """Test validation normalizes lowercase to uppercase."""
        result = validate_dataset_name("pems-bay")
        assert result == "PEMS-BAY"

    def test_validate_mixed_case(self):
        """Test validation normalizes mixed case to uppercase."""
        result = validate_dataset_name("MeTr-La")
        assert result == "METR-LA"

    def test_validate_invalid_dataset_raises(self):
        """Test that invalid dataset raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            validate_dataset_name("INVALID-DATASET")

    def test_validate_empty_string_raises(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            validate_dataset_name("")

    def test_error_message_lists_valid_options(self):
        """Test that error message includes valid dataset names."""
        with pytest.raises(ValueError, match="METR-LA.*PEMS-BAY"):
            validate_dataset_name("INVALID")
