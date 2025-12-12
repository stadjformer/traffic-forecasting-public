"""Tests for HuggingFace Hub validation."""

from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.utils import RepositoryNotFoundError

from utils.io import validate_hf_hub_access


class TestValidateHFHubAccess:
    """Tests for validate_hf_hub_access function."""

    def test_invalid_repo_id_format(self):
        """Test that invalid repo_id format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid repo_id format"):
            validate_hf_hub_access("invalid-repo-id")

    def test_authentication_failure(self):
        """Test that authentication failure raises RuntimeError."""
        with patch("utils.io.HfApi") as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.whoami.side_effect = Exception("Not logged in")

            with pytest.raises(RuntimeError, match="Failed to authenticate"):
                validate_hf_hub_access("username/repo-name")

    def test_repo_exists_success(self):
        """Test successful validation when repo exists."""
        with patch("utils.io.HfApi") as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.whoami.return_value = {"name": "testuser"}

            mock_repo_info = MagicMock()
            mock_repo_info.lastModified = "2024-01-01"
            mock_instance.repo_info.return_value = mock_repo_info

            mock_instance.list_repo_files.return_value = ["README.md"]

            # Should not raise
            validate_hf_hub_access("testuser/test-repo", verbose=False)

    def test_repo_not_found_creates_new(self):
        """Test that new repo is created when it doesn't exist."""
        with patch("utils.io.HfApi") as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.whoami.return_value = {"name": "testuser"}
            mock_instance.repo_info.side_effect = RepositoryNotFoundError("Not found")
            mock_instance.create_repo.return_value = None
            mock_instance.list_repo_files.return_value = []

            # Should not raise and should call create_repo
            validate_hf_hub_access(
                "testuser/test-repo", create_if_missing=True, verbose=False
            )
            mock_instance.create_repo.assert_called_once()

    def test_repo_not_found_fails_when_create_disabled(self):
        """Test that error is raised when repo doesn't exist and create_if_missing=False."""
        with patch("utils.io.HfApi") as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.whoami.return_value = {"name": "testuser"}
            mock_instance.repo_info.side_effect = RepositoryNotFoundError("Not found")

            with pytest.raises(RuntimeError, match="does not exist"):
                validate_hf_hub_access(
                    "testuser/test-repo", create_if_missing=False, verbose=False
                )

    def test_warns_on_username_mismatch(self, capsys):
        """Test that warning is printed when repo owner doesn't match authenticated user."""
        with patch("utils.io.HfApi") as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.whoami.return_value = {"name": "actualuser"}

            mock_repo_info = MagicMock()
            mock_repo_info.lastModified = "2024-01-01"
            mock_instance.repo_info.return_value = mock_repo_info
            mock_instance.list_repo_files.return_value = []

            validate_hf_hub_access("differentuser/test-repo")

            captured = capsys.readouterr()
            assert "Warning" in captured.out
            assert "doesn't match" in captured.out
