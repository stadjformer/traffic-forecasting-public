"""Simpler tests for utils.inference module - focusing on key functionality."""

import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import utils.inference


class TestGetModelPredictionsCached:
    """Tests for get_model_predictions_cached function."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory."""
        return tmp_path / "cache"

    @pytest.fixture
    def mock_predictions(self):
        """Mock predictions array."""
        return np.random.rand(10, 12, 5, 1).astype(np.float32)

    @pytest.fixture
    def mock_ground_truth(self):
        """Mock ground truth array."""
        return np.random.rand(10, 12, 5, 1).astype(np.float32)

    def test_cache_loading(self, temp_cache_dir, mock_predictions, mock_ground_truth):
        """Test that cached predictions are loaded correctly."""
        # Create a cache file
        cache_file = temp_cache_dir / "metr-la_test_model_predictions.pkl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "predictions": mock_predictions,
                    "ground_truth": mock_ground_truth,
                    "dataset_name": "METR-LA",
                    "hf_repo_prefix": "TEST_MODEL",
                },
                f,
            )

        # Load from cache
        preds, gt = utils.inference.get_model_predictions_cached(
            dataset_name="METR-LA",
            hf_repo_prefix="TEST_MODEL",
            cache_dir=temp_cache_dir,
        )

        np.testing.assert_array_equal(preds, mock_predictions)
        np.testing.assert_array_equal(gt, mock_ground_truth)

    def test_cache_file_naming(self, temp_cache_dir):
        """Test cache file naming."""
        # Without sample indices
        cache_file = temp_cache_dir / "metr-la_test_model_predictions.pkl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        mock_preds = np.zeros((5, 12, 5, 1))
        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "predictions": mock_preds,
                    "ground_truth": mock_preds,
                    "dataset_name": "METR-LA",
                    "hf_repo_prefix": "TEST_MODEL",
                },
                f,
            )

        preds, gt = utils.inference.get_model_predictions_cached(
            dataset_name="METR-LA",
            hf_repo_prefix="TEST_MODEL",
            cache_dir=temp_cache_dir,
        )

        assert cache_file.exists()
        assert preds.shape == (5, 12, 5, 1)


class TestGetLearnedAdjacencyMatrix:
    """Tests for get_learned_adjacency_matrix function."""

    def test_extract_from_graph_constructor(self):
        """Test extracting adjacency matrix from graph_constructor."""
        num_nodes = 10
        expected_adj = np.random.rand(num_nodes, num_nodes)

        # Mock model with graph_constructor
        mock_model = MagicMock()
        mock_model.adaptive_embedding = True
        mock_model.graph_constructor.get_adj.return_value = MagicMock(
            cpu=lambda: MagicMock(numpy=lambda: expected_adj)
        )

        with patch("utils.inference.load_from_hub") as mock_load:
            mock_load.return_value = (mock_model, None)

            adj = utils.inference.get_learned_adjacency_matrix(
                dataset_name="METR-LA",
                hf_repo_prefix="TEST_MODEL",
            )

            np.testing.assert_array_equal(adj, expected_adj)

    def test_graph_mode_none_returns_identity(self):
        """Test that GraphMode.NONE returns an identity matrix."""
        from stgformer.enums import GraphMode

        num_nodes = 207  # METR-LA size

        # Mock model with GraphMode.NONE
        # Need spec_set to ensure hasattr works properly
        from unittest.mock import PropertyMock

        mock_model = MagicMock()
        # Explicitly set attributes so hasattr works
        type(mock_model).graph_mode = PropertyMock(return_value=GraphMode.NONE)
        type(mock_model).num_nodes = PropertyMock(return_value=num_nodes)
        # Ensure graph_constructor doesn't exist
        del mock_model.graph_constructor

        with patch("utils.inference.load_from_hub") as mock_load:
            mock_load.return_value = (mock_model, None)

            adj = utils.inference.get_learned_adjacency_matrix(
                dataset_name="METR-LA",
                hf_repo_prefix="TEST_MODEL",
            )

            # Should be an identity matrix
            expected_identity = np.eye(num_nodes)
            np.testing.assert_array_equal(adj, expected_identity)

            # Should be square
            assert adj.shape == (num_nodes, num_nodes)

            # All diagonal elements should be 1.0
            assert np.all(np.diag(adj) == 1.0)

            # All off-diagonal elements should be 0.0
            assert np.sum(adj) == num_nodes

    def test_raises_error_for_non_learnable_graph(self):
        """Test that error is raised for models without learnable graphs."""
        # Mock model without adaptive_embedding
        mock_model = MagicMock()
        mock_model.adaptive_embedding = None

        def custom_hasattr(obj, attr):
            return False

        with (
            patch("utils.inference.load_from_hub") as mock_load,
            patch("builtins.hasattr", side_effect=custom_hasattr),
        ):
            mock_load.return_value = (mock_model, None)

            with pytest.raises(ValueError, match="does not have learnable graph"):
                utils.inference.get_learned_adjacency_matrix(
                    dataset_name="METR-LA",
                    hf_repo_prefix="TEST_MODEL",
                )


@pytest.mark.slow
class TestLearnedAdjacencyMatrixIntegration:
    """Integration tests for learned adjacency matrices from real models."""

    @pytest.mark.parametrize(
        "model_prefix,expected_k",
        [
            ("STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K8", 8),
            ("STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16", 16),
        ],
    )
    def test_k_sparse_models(self, model_prefix, expected_k):
        """Test that k-sparse models have exactly k outgoing edges per node."""
        from utils.visual import get_neighborhood_stats

        adj_matrix = utils.inference.get_learned_adjacency_matrix(
            dataset_name="METR-LA",
            hf_repo_prefix=model_prefix,
        )

        # Should be 2D
        assert adj_matrix.ndim == 2
        assert adj_matrix.shape[0] == adj_matrix.shape[1]

        # Get stats
        stats = get_neighborhood_stats(adj_matrix, weighted=False)

        # All nodes should have exactly k outgoing edges
        assert np.all(stats["out_degrees"] == expected_k), (
            f"Expected all nodes to have {expected_k} outgoing edges"
        )

        # Incoming degree should NOT be constrained to k
        # (some nodes can receive from many sources)
        assert stats["in_degrees"].max() > expected_k, (
            "Expected some nodes to have more than k incoming edges "
            "(incoming degree is not constrained)"
        )

    @pytest.mark.slow
    def test_non_sparse_model(self):
        """Test that non-sparse models are dense."""
        from utils.visual import get_neighborhood_stats

        adj_matrix = utils.inference.get_learned_adjacency_matrix(
            dataset_name="METR-LA",
            hf_repo_prefix="STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING",
        )

        n_nodes = adj_matrix.shape[0]
        stats = get_neighborhood_stats(adj_matrix, weighted=False)

        # Dense model should have all nodes connected
        assert np.all(stats["out_degrees"] == n_nodes), (
            "Expected dense adjacency (all nodes connected)"
        )

    @pytest.mark.parametrize(
        "model_prefix",
        [
            "STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K8",
            "STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16",
            "STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING",
        ],
    )
    def test_row_stochastic(self, model_prefix):
        """Test that all models produce row-stochastic adjacency matrices."""
        adj_matrix = utils.inference.get_learned_adjacency_matrix(
            dataset_name="METR-LA",
            hf_repo_prefix=model_prefix,
        )

        # Row sums should be 1.0 (softmax normalization)
        row_sums = adj_matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), (
            "Expected row-stochastic matrix (rows sum to 1)"
        )

    @pytest.mark.slow
    def test_weighted_degrees(self):
        """Test weighted degree computation on k-sparse model."""
        from utils.visual import get_neighborhood_stats

        adj_matrix = utils.inference.get_learned_adjacency_matrix(
            dataset_name="METR-LA",
            hf_repo_prefix="STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16",
        )

        stats = get_neighborhood_stats(adj_matrix, weighted=True)

        # Outgoing weighted degree should be 1.0 for all (row-stochastic)
        assert np.allclose(stats["out_degrees"], 1.0, atol=1e-5)

        # Incoming weighted degree should vary
        # (shows centrality/importance)
        assert stats["in_degrees"].std() > 0.5, (
            "Expected significant variation in incoming weighted degree "
            "(indicates hub structure)"
        )

        # Some nodes should have much higher incoming weight than average
        mean_in = stats["in_degrees"].mean()
        max_in = stats["in_degrees"].max()
        assert max_in > 2 * mean_in, (
            "Expected some hub nodes with incoming weight > 2x average"
        )
