"""Tests for utils.hub module."""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from utils.hub import (
    create_model_card,
    fetch_model_from_hub,
    get_best_device,
    push_model_to_hub,
)


class TestGetBestDevice:
    """Tests for get_best_device function."""

    @patch("torch.cuda.is_available", return_value=True)
    def test_returns_cuda_when_available(self, mock_cuda):
        """Test that CUDA is returned when available."""
        device = get_best_device()
        assert device == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_returns_mps_when_cuda_unavailable(self, mock_mps, mock_cuda):
        """Test that MPS is returned when CUDA unavailable but MPS available."""
        device = get_best_device()
        assert device == "mps"

    @patch("torch.cuda.is_available", return_value=False)
    def test_returns_cpu_when_no_accelerators(self, mock_cuda):
        """Test that CPU is returned when no accelerators available."""
        with patch("torch.backends.mps", create=True) as mock_mps:
            mock_mps.is_available = MagicMock(return_value=False)
            device = get_best_device()
            assert device == "cpu"

    @patch("torch.cuda.is_available", return_value=False)
    def test_returns_cpu_when_mps_not_present(self, mock_cuda):
        """Test fallback to CPU when MPS backend doesn't exist (older PyTorch)."""
        with patch("torch.backends", spec=["cudnn"]):  # backends without mps
            device = get_best_device()
            assert device == "cpu"


class TestCreateModelCard:
    """Tests for create_model_card function."""

    def test_create_card_for_dcrnn(self):
        """Test model card generation for DCRNN."""
        card = create_model_card("DCRNN", "METR-LA")
        assert "DCRNN" in card
        assert "METR-LA" in card
        assert "Diffusion Convolutional Recurrent Neural Network" in card
        assert "tags:" in card
        assert "traffic-forecasting" in card

    def test_create_card_for_mtgnn(self):
        """Test model card generation for MTGNN."""
        card = create_model_card("MTGNN", "PEMS-BAY")
        assert "MTGNN" in card
        assert "PEMS-BAY" in card
        assert "Multivariate Time Series" in card

    def test_create_card_for_gwnet(self):
        """Test model card generation for GWNET."""
        card = create_model_card("GWNET", "METR-LA")
        assert "Graph WaveNet" in card or "Graph-WaveNet" in card
        assert "METR-LA" in card

    def test_create_card_for_stgformer(self):
        """Test model card generation for STGFORMER."""
        card = create_model_card("STGFORMER", "METR-LA")
        assert "STGformer" in card or "STGFORMER" in card
        assert "Transformer" in card or "transformer" in card

    def test_create_card_with_metrics(self):
        """Test model card includes metrics when provided."""
        metrics = {"MAE": 2.5, "RMSE": 4.3}
        card = create_model_card("DCRNN", "METR-LA", metrics)
        assert "Evaluation Metrics" in card
        assert "MAE" in card
        assert "2.5" in card
        assert "RMSE" in card
        assert "4.3" in card

    def test_create_card_without_metrics(self):
        """Test model card works without metrics."""
        card = create_model_card("DCRNN", "METR-LA")
        assert "DCRNN" in card
        # Should not have metrics section
        assert "Evaluation Metrics" not in card or "Evaluation Metrics\n\n##" in card

    def test_create_card_invalid_model_type_raises(self):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model_card("INVALID_MODEL", "METR-LA")

    def test_create_card_case_insensitive(self):
        """Test that model type is case-insensitive."""
        card = create_model_card("dcrnn", "METR-LA")
        assert "DCRNN" in card or "dcrnn" in card

    def test_create_card_for_abl_no_xavier(self):
        """Test model card generation for ABL_NO_XAVIER ablation study."""
        card = create_model_card("ABL_NO_XAVIER", "METR-LA")
        assert "ABL_NO_XAVIER" in card or "No Xavier" in card
        assert "Ablation" in card
        assert "METR-LA" in card
        assert "traffic-forecasting" in card

    def test_create_card_for_abl_no_dow(self):
        """Test model card generation for ABL_NO_DOW ablation study."""
        card = create_model_card("ABL_NO_DOW", "PEMS-BAY")
        assert "ABL_NO_DOW" in card or "No Dow" in card
        assert "Ablation" in card
        assert "PEMS-BAY" in card

    def test_create_card_for_stgformer_variant(self):
        """Test model card generation for STGFORMER variant."""
        card = create_model_card("STGFORMER_CHEB_TCN", "METR-LA")
        assert "STGFORMER" in card or "STGformer" in card
        assert "Cheb Tcn" in card or "CHEB_TCN" in card
        assert "METR-LA" in card

    def test_create_card_for_abl_no_graph(self):
        """Test model card generation for ABL_NO_GRAPH ablation study."""
        card = create_model_card("ABL_NO_GRAPH", "METR-LA")
        assert "ABL_NO_GRAPH" in card or "No Graph" in card
        assert "Ablation" in card
        assert "METR-LA" in card

    def test_create_card_for_abl_no_temporal(self):
        """Test model card generation for ABL_NO_TEMPORAL ablation study."""
        card = create_model_card("ABL_NO_TEMPORAL", "METR-LA")
        assert "ABL_NO_TEMPORAL" in card or "No Temporal" in card
        assert "Ablation" in card
        assert "METR-LA" in card


class TestFetchModelFromHub:
    """Tests for fetch_model_from_hub function."""

    @patch("utils.hub.HF_USERNAME_DOWNLOAD", "test_user")
    @patch("utils.hub.snapshot_download")
    @patch("utils.hub.MODELS_DIR", Path("/tmp/models"))
    def test_fetch_downloads_model(self, mock_snapshot, tmp_path):
        """Test that fetch_model_from_hub calls snapshot_download."""
        model_dir = tmp_path / "MTGNN" / "model_metr-la"

        with patch("utils.hub.MODELS_DIR", tmp_path):
            mock_snapshot.return_value = str(model_dir)

            fetch_model_from_hub("MTGNN", "METR-LA")

            mock_snapshot.assert_called_once()
            assert "test_user/MTGNN_METR-LA" in str(mock_snapshot.call_args)

    @patch("utils.hub.HF_USERNAME_DOWNLOAD", "test_user")
    @patch("utils.hub.snapshot_download")
    def test_fetch_uses_correct_repo_id(self, mock_snapshot, tmp_path):
        """Test that correct repo_id is constructed."""
        model_dir = tmp_path / "GWNET" / "model_pems-bay"
        model_dir.mkdir(parents=True)

        with patch("utils.hub.MODELS_DIR", tmp_path):
            mock_snapshot.return_value = str(model_dir)

            fetch_model_from_hub("GWNET", "PEMS-BAY")

            call_kwargs = mock_snapshot.call_args.kwargs
            assert call_kwargs["repo_id"] == "test_user/GWNET_PEMS-BAY"

    @patch("utils.hub.HF_USERNAME_DOWNLOAD", None)
    def test_fetch_raises_when_no_username(self):
        """Test that fetch raises error when HF_USERNAME_DOWNLOAD not set."""
        with pytest.raises(
            RuntimeError, match="HF_USERNAME_FOR_MODEL_DOWNLOAD not set"
        ):
            fetch_model_from_hub("MTGNN", "METR-LA")

    @patch("utils.hub.HF_USERNAME_DOWNLOAD", "test_user")
    @patch("utils.hub.snapshot_download")
    def test_fetch_skips_download_when_cached(self, mock_snapshot, tmp_path):
        """Test that fetch skips download when model already exists."""
        model_dir = tmp_path / "MTGNN" / "model_metr-la"
        model_dir.mkdir(parents=True)
        (
            model_dir / "model.safetensors"
        ).touch()  # Create a file to simulate cached model

        with patch("utils.hub.MODELS_DIR", tmp_path):
            result = fetch_model_from_hub("MTGNN", "METR-LA", force_download=False)

            # Should not call snapshot_download when cached
            mock_snapshot.assert_not_called()
            assert result == model_dir

    @patch("utils.hub.HF_USERNAME_DOWNLOAD", "test_user")
    @patch("utils.hub.snapshot_download")
    def test_fetch_forces_download(self, mock_snapshot, tmp_path):
        """Test that force_download=True downloads even when cached."""
        model_dir = tmp_path / "MTGNN" / "model_metr-la"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").touch()

        with patch("utils.hub.MODELS_DIR", tmp_path):
            mock_snapshot.return_value = str(model_dir)

            fetch_model_from_hub("MTGNN", "METR-LA", force_download=True)

            # Should call snapshot_download even when cached
            mock_snapshot.assert_called_once()


class TestPushModelToHub:
    """Tests for push_model_to_hub function."""

    @patch("utils.hub.HfApi")
    @patch("builtins.open", new_callable=mock_open)
    def test_push_creates_metadata(self, mock_file, mock_api_class, tmp_path):
        """Test that push creates metadata.json."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()

        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        push_model_to_hub(
            checkpoint_dir=checkpoint_dir,
            repo_id="user/model",
            model_type="MTGNN",
            dataset_name="METR-LA",
        )

        # Check that files were written
        assert mock_file.call_count >= 2  # metadata.json and README.md

    @patch("utils.hub.HfApi")
    @patch("builtins.open", new_callable=mock_open)
    def test_push_creates_repo(self, mock_file, mock_api_class, tmp_path):
        """Test that push creates repository."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()

        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        push_model_to_hub(
            checkpoint_dir=checkpoint_dir,
            repo_id="user/test-model",
            model_type="DCRNN",
            dataset_name="METR-LA",
        )

        mock_api.create_repo.assert_called_once()
        call_kwargs = mock_api.create_repo.call_args.kwargs
        assert call_kwargs["repo_id"] == "user/test-model"

    @patch("utils.hub.HfApi")
    @patch("builtins.open", new_callable=mock_open)
    def test_push_uploads_folder(self, mock_file, mock_api_class, tmp_path):
        """Test that push uploads the folder."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()

        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        push_model_to_hub(
            checkpoint_dir=checkpoint_dir,
            repo_id="user/model",
            model_type="MTGNN",
            dataset_name="METR-LA",
        )

        mock_api.upload_folder.assert_called_once()
        call_kwargs = mock_api.upload_folder.call_args.kwargs
        assert call_kwargs["repo_id"] == "user/model"

    @patch("utils.hub.HfApi")
    @patch("builtins.open", new_callable=mock_open)
    def test_push_respects_private_flag(self, mock_file, mock_api_class, tmp_path):
        """Test that private flag is passed correctly."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()

        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        push_model_to_hub(
            checkpoint_dir=checkpoint_dir,
            repo_id="user/model",
            model_type="MTGNN",
            dataset_name="METR-LA",
            private=True,
        )

        call_kwargs = mock_api.create_repo.call_args.kwargs
        assert call_kwargs["private"] is True

    @patch("utils.hub.HfApi")
    @patch("builtins.open", new_callable=mock_open)
    def test_push_returns_url(self, mock_file, mock_api_class, tmp_path):
        """Test that push returns HuggingFace URL."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()

        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        url = push_model_to_hub(
            checkpoint_dir=checkpoint_dir,
            repo_id="user/my-model",
            model_type="GWNET",
            dataset_name="PEMS-BAY",
        )

        assert url == "https://huggingface.co/user/my-model"
