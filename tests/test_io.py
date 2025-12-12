"""Tests for utils.io module."""

from unittest.mock import MagicMock, patch

import pytest

import utils.io


class TestGetModelPaths:
    """Tests for get_model_paths helper function."""

    @patch("utils.io.HF_USERNAME_UPLOAD", "test_user")
    def test_production_paths(self):
        """Test production paths without dry-run."""
        local_dir, repo_id, is_private = utils.io.get_model_paths(
            "MTGNN", "METR-LA", dry_run=False
        )

        assert local_dir == utils.io.MODELS_DIR / "MTGNN" / "model_metr-la"
        assert repo_id == "test_user/MTGNN_METR-LA"
        assert is_private is False

    @patch("utils.io.HF_USERNAME_UPLOAD", "test_user")
    def test_dry_run_paths(self):
        """Test dry-run paths with _dry_run suffix."""
        local_dir, repo_id, is_private = utils.io.get_model_paths(
            "MTGNN", "METR-LA", dry_run=True
        )

        assert local_dir == utils.io.MODELS_DIR / "MTGNN_dry_run" / "model_metr-la"
        assert repo_id == "test_user/MTGNN_METR-LA_dry_run"
        assert is_private is True

    @patch("utils.io.HF_USERNAME_UPLOAD", "test_user")
    def test_different_model_names(self):
        """Test that different model names generate correct paths."""
        models = ["MTGNN", "GWNET", "DCRNN"]

        for model in models:
            local_dir, repo_id, is_private = utils.io.get_model_paths(
                model, "METR-LA", dry_run=False
            )

            assert local_dir == utils.io.MODELS_DIR / model / "model_metr-la"
            assert repo_id == f"test_user/{model}_METR-LA"
            assert is_private is False

    @patch("utils.io.HF_USERNAME_UPLOAD", "test_user")
    def test_different_datasets(self):
        """Test that different datasets generate correct paths."""
        datasets = ["METR-LA", "PEMS-BAY"]

        for dataset in datasets:
            local_dir, repo_id, is_private = utils.io.get_model_paths(
                "MTGNN", dataset, dry_run=False
            )

            dataset_lower = dataset.lower()
            dataset_upper = dataset.upper()

            assert local_dir == utils.io.MODELS_DIR / "MTGNN" / f"model_{dataset_lower}"
            assert repo_id == f"test_user/MTGNN_{dataset_upper}"
            assert is_private is False

    @patch("utils.io.HF_USERNAME_UPLOAD", None)
    def test_missing_username_raises_error(self):
        """Test that missing username in .env_public raises RuntimeError."""
        with pytest.raises(RuntimeError, match="HF_USERNAME_FOR_UPLOAD"):
            utils.io.get_model_paths("MTGNN", "METR-LA", dry_run=False)

    @patch("utils.io.HF_USERNAME_UPLOAD", "test_user")
    def test_case_normalization(self):
        """Test that model/dataset names are normalized correctly."""
        # Test with lowercase input
        local_dir, repo_id, _ = utils.io.get_model_paths(
            "mtgnn", "metr-la", dry_run=False
        )

        assert local_dir == utils.io.MODELS_DIR / "MTGNN" / "model_metr-la"
        assert repo_id == "test_user/MTGNN_METR-LA"

    @patch("utils.io.HF_USERNAME_UPLOAD", "test_user")
    def test_dry_run_isolation(self):
        """Test that dry-run and production paths are isolated."""
        prod_dir, prod_repo, _ = utils.io.get_model_paths(
            "MTGNN", "METR-LA", dry_run=False
        )
        dry_dir, dry_repo, _ = utils.io.get_model_paths(
            "MTGNN", "METR-LA", dry_run=True
        )

        # Ensure they're different
        assert prod_dir != dry_dir
        assert prod_repo != dry_repo

        # Ensure dry-run has suffix
        assert "_dry_run" in str(dry_dir)
        assert "_dry_run" in dry_repo

        # Ensure production doesn't have suffix
        assert "_dry_run" not in str(prod_dir)
        assert "_dry_run" not in prod_repo


class TestGetDatasetHf:
    """Tests for get_dataset_hf function."""

    @patch("utils.io._fetch_data")
    @patch("utils.io.datasets.load_dataset")
    def test_loads_dataset_from_local_parquet(self, mock_load, mock_fetch):
        """Test that dataset is loaded from local parquet files."""
        mock_dataset = {"train": [], "val": [], "test": []}
        mock_load.return_value = mock_dataset

        result = utils.io.get_dataset_hf("METR-LA")

        assert result == mock_dataset
        mock_fetch.assert_called_once()
        mock_load.assert_called_once()
        # Should load from parquet with local files
        call_args = mock_load.call_args
        assert call_args[0][0] == "parquet"
        assert "data_files" in call_args[1]

    @patch("utils.io._fetch_data")
    @patch("utils.io.datasets.load_dataset")
    def test_calls_fetch_data(self, mock_load, mock_fetch):
        """Test that _fetch_data is called to ensure data is downloaded."""
        mock_load.return_value = {"train": [], "val": [], "test": []}

        utils.io.get_dataset_hf("METR-LA", force_download=True)

        # Should call _fetch_data with force_download
        mock_fetch.assert_called_once()
        assert "METR-LA" in str(mock_fetch.call_args)


class TestGetDatasetTorch:
    """Tests for get_dataset_torch function - basic tests only."""

    def test_validates_dataset_name(self):
        """Test that invalid dataset name raises error."""
        with pytest.raises(ValueError, match="not supported"):
            utils.io.get_dataset_torch("INVALID-DATASET")


class TestGetGraphMetadata:
    """Tests for get_graph_metadata function - basic tests only."""

    def test_validates_dataset_name(self):
        """Test that invalid dataset name raises error."""
        with pytest.raises(ValueError, match="not supported"):
            utils.io.get_graph_metadata("INVALID-DATASET")


class TestValidateHfHubAccess:
    """Tests for validate_hf_hub_access function."""

    @patch("utils.io.HfApi")
    def test_validates_existing_repo(self, mock_api_class):
        """Test validation of existing repository."""
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "test"}
        mock_api.repo_info.return_value = {"id": "test/repo"}
        mock_api_class.return_value = mock_api

        # Should not raise
        utils.io.validate_hf_hub_access(
            "test/repo", create_if_missing=False, verbose=False
        )

        mock_api.whoami.assert_called_once()
        mock_api.repo_info.assert_called_once_with(
            repo_id="test/repo", repo_type="model"
        )

    @patch("utils.io.huggingface_hub.utils.RepositoryNotFoundError", Exception)
    @patch("utils.io.HfApi")
    def test_creates_repo_when_missing(self, mock_api_class):
        """Test that repository is created when missing."""
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "test"}
        # Simulate repo not found
        mock_api.repo_info.side_effect = Exception("Repository not found")
        mock_api_class.return_value = mock_api

        utils.io.validate_hf_hub_access(
            "test/new-repo", create_if_missing=True, verbose=False
        )

        mock_api.create_repo.assert_called_once()
        call_kwargs = mock_api.create_repo.call_args.kwargs
        assert call_kwargs["repo_id"] == "test/new-repo"
        assert call_kwargs["repo_type"] == "model"

    @patch("utils.io.HfApi")
    def test_invalid_repo_id_format_raises(self, mock_api_class):
        """Test that invalid repo_id format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid repo_id format"):
            utils.io.validate_hf_hub_access("invalid-repo-id", verbose=False)
