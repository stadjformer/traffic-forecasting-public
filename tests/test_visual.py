"""Tests for visualization utilities."""

import numpy as np
import pytest

from utils.visual import get_neighborhood_stats, get_top_neighbors


class TestGetTopNeighbors:
    """Test get_top_neighbors function."""

    @pytest.fixture
    def sample_adj(self):
        """Create a simple 5x5 adjacency matrix for testing."""
        # Row-stochastic (softmax-like) matrix
        adj = np.array(
            [
                [0.0, 0.5, 0.3, 0.2, 0.0],  # Node 0 -> mainly 1,2,3
                [0.1, 0.0, 0.6, 0.2, 0.1],  # Node 1 -> mainly 2
                [0.3, 0.3, 0.0, 0.3, 0.1],  # Node 2 -> balanced
                [0.0, 0.0, 0.0, 0.0, 1.0],  # Node 3 -> only 4
                [0.25, 0.25, 0.25, 0.25, 0.0],  # Node 4 -> all equal
            ]
        )
        return adj

    def test_outgoing_neighbors(self, sample_adj):
        """Test getting outgoing neighbors."""
        neighbors, weights = get_top_neighbors(
            sample_adj, node_id=0, top_k=3, direction="outgoing"
        )
        assert len(neighbors) == 3
        assert neighbors[0] == 1  # Highest weight 0.5
        assert neighbors[1] == 2  # Second highest 0.3
        assert neighbors[2] == 3  # Third highest 0.2

    def test_incoming_neighbors(self, sample_adj):
        """Test getting incoming neighbors."""
        # Node 2 receives from: 0(0.3), 1(0.6), 3(0.0), 4(0.25)
        neighbors, weights = get_top_neighbors(
            sample_adj, node_id=2, top_k=3, direction="incoming"
        )
        assert len(neighbors) == 3
        assert neighbors[0] == 1  # Highest incoming weight 0.6
        assert neighbors[1] == 0  # Second 0.3
        assert neighbors[2] == 4  # Third 0.25

    def test_both_directions(self, sample_adj):
        """Test getting neighbors in both directions."""
        neighbors, weights = get_top_neighbors(
            sample_adj, node_id=0, top_k=5, direction="both"
        )
        # Node 0 sends to: 1,2,3  and receives from: 4
        assert len(neighbors) == 4  # Union of incoming and outgoing
        assert set(neighbors) == {1, 2, 3, 4}

    def test_invalid_direction(self, sample_adj):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="direction must be"):
            get_top_neighbors(sample_adj, node_id=0, top_k=3, direction="invalid")

    def test_node_out_of_range(self, sample_adj):
        """Test that out-of-range node ID raises error."""
        with pytest.raises(ValueError, match="out of range"):
            get_top_neighbors(sample_adj, node_id=10, top_k=3)


class TestGetNeighborhoodStats:
    """Test get_neighborhood_stats function."""

    @pytest.fixture
    def sparse_adj(self):
        """Create a k=2 sparse adjacency matrix."""
        # Each row has exactly 2 nonzero entries (k=2 sparsity)
        adj = np.array(
            [
                [0.0, 0.6, 0.4, 0.0, 0.0],  # Node 0: out to 1,2
                [0.0, 0.0, 0.7, 0.3, 0.0],  # Node 1: out to 2,3
                [0.5, 0.0, 0.0, 0.5, 0.0],  # Node 2: out to 0,3
                [0.0, 0.0, 0.0, 0.0, 1.0],  # Node 3: out to 4
                [0.8, 0.2, 0.0, 0.0, 0.0],  # Node 4: out to 0,1
            ]
        )
        return adj

    def test_unweighted_degrees(self, sparse_adj):
        """Test unweighted degree computation."""
        stats = get_neighborhood_stats(sparse_adj, weighted=False)

        # Most nodes have 2 outgoing edges (node 3 has only 1)
        assert stats["out_degrees"][0] == 2
        assert stats["out_degrees"][1] == 2
        assert stats["out_degrees"][2] == 2
        assert stats["out_degrees"][3] == 1  # Only connects to node 4
        assert stats["out_degrees"][4] == 2

        # Incoming degrees vary
        # Node 0 receives from: 2, 4 = 2
        # Node 1 receives from: 0, 4 = 2
        # Node 2 receives from: 0, 1 = 2
        # Node 3 receives from: 1, 2 = 2
        # Node 4 receives from: 3 = 1
        assert stats["in_degrees"][0] == 2
        assert stats["in_degrees"][4] == 1

    def test_weighted_degrees(self, sparse_adj):
        """Test weighted degree computation."""
        stats = get_neighborhood_stats(sparse_adj, weighted=True)

        # All outgoing weights sum to 1.0 (row-stochastic)
        assert np.allclose(stats["out_degrees"], 1.0)

        # Incoming weights vary
        # Node 0: 0.5 + 0.8 = 1.3
        assert np.isclose(stats["in_degrees"][0], 1.3)

        # Node 4: 1.0
        assert np.isclose(stats["in_degrees"][4], 1.0)

    def test_weighted_total_degree(self, sparse_adj):
        """Test total weighted degree is sum of in + out."""
        stats = get_neighborhood_stats(sparse_adj, weighted=True)

        # Total should be in + out
        expected_total = stats["in_degrees"] + stats["out_degrees"]
        assert np.allclose(stats["total_degrees"], expected_total)


class TestSparsityVerification:
    """Test that we can verify k-sparsity properties."""

    def test_k_sparse_properties(self):
        """Test properties that should hold for k-sparse matrices."""
        # Create a k=3 sparse matrix
        k = 3
        n_nodes = 10

        # Simulate what sparsify_graph does: top-k per row
        np.random.seed(42)
        dense = np.random.rand(n_nodes, n_nodes)

        # Get top-k indices per row
        top_k_indices = np.argsort(dense, axis=1)[:, -k:]

        # Create sparse matrix
        sparse = np.zeros_like(dense)
        for i in range(n_nodes):
            sparse[i, top_k_indices[i]] = dense[i, top_k_indices[i]]

        # Normalize rows to sum to 1 (like softmax)
        row_sums = sparse.sum(axis=1, keepdims=True)
        sparse = sparse / np.where(row_sums > 0, row_sums, 1)

        # Verify properties
        stats = get_neighborhood_stats(sparse, weighted=False)

        # Property 1: All nodes have exactly k outgoing edges
        assert np.all(stats["out_degrees"] == k)

        # Property 2: Row sums are 1.0 (row-stochastic)
        assert np.allclose(sparse.sum(axis=1), 1.0)

        # Property 3: Incoming degree is NOT constrained
        assert stats["in_degrees"].min() >= 0
        assert stats["in_degrees"].max() <= n_nodes
        # With random data, incoming degrees should vary
        assert stats["in_degrees"].std() > 0
