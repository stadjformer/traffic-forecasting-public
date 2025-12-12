"""Tests for model serialization and deserialization.

These tests ensure backward compatibility when refactoring by verifying:
1. Save/load round-trips produce identical outputs
2. Config persistence and reconstruction
3. Scaler state preservation
4. safetensors format integrity
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from stgformer.enums import GraphMode, PropagationMode, TemporalMode
from stgformer.model import STGFormer
from utils.stgformer import load_model, save_model
from utils.training import StandardScaler


class TestModelSerialization:
    """Tests for STGFormer model serialization and deserialization."""

    @pytest.fixture
    def sample_model_config(self):
        """Create a sample model configuration."""
        return {
            "num_nodes": 50,
            "in_steps": 12,
            "out_steps": 12,
            "input_dim": 1,  # Just speed value, no TOD
            "output_dim": 1,
            "steps_per_day": 288,
            "input_embedding_dim": 24,
            "tod_embedding_dim": 0,  # Disable TOD embeddings for simpler tests
            "dow_embedding_dim": 0,
            "spatial_embedding_dim": 0,
            "adaptive_embedding_dim": 40,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.1,
            "dropout_a": 0.3,
            "mlp_ratio": 4,
            "use_mixed_proj": True,
        }

    @pytest.fixture
    def sample_model(self, sample_model_config):
        """Create a sample STGFormer model."""
        model = STGFormer(**sample_model_config)
        model.eval()  # Set to eval mode for deterministic outputs
        return model

    @pytest.fixture
    def sample_scaler(self):
        """Create a fitted StandardScaler."""
        np.random.seed(42)
        data = torch.randn(100, 50, 1)  # (batch, nodes, features)
        scaler = StandardScaler()
        scaler.fit_transform(data)
        return scaler

    @pytest.fixture
    def sample_input(self):
        """Create sample input data for forward pass."""
        torch.manual_seed(42)
        # (batch, in_steps, num_nodes, input_dim)
        # input_dim=1 for just speed value (no TOD)
        return torch.randn(4, 12, 50, 1)

    def test_save_creates_required_files(self, sample_model, sample_scaler):
        """Test that save_model creates all required files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            save_model(sample_model, sample_scaler, save_path, "METR-LA")

            # Check all required files exist
            assert (save_path / "model.safetensors").exists()
            assert (save_path / "config.json").exists()
            assert (save_path / "scaler.json").exists()

    def test_config_json_contains_all_parameters(
        self, sample_model, sample_model_config
    ):
        """Test that config.json contains core model parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            scaler = StandardScaler()
            save_model(sample_model, scaler, save_path, "METR-LA")

            # Load and verify config
            with open(save_path / "config.json", "r") as f:
                config = json.load(f)

            # Check core parameters that are actually saved (not all __init__ params are saved)
            saved_params = [
                "num_nodes",
                "in_steps",
                "out_steps",
                "input_dim",
                "output_dim",
                "steps_per_day",
                "input_embedding_dim",
                "tod_embedding_dim",
                "dow_embedding_dim",
                "spatial_embedding_dim",
                "adaptive_embedding_dim",
                "num_heads",
                "num_layers",
                "dropout_a",
                "use_mixed_proj",
            ]

            for key in saved_params:
                assert key in config, f"Missing key: {key}"
                if key in sample_model_config:
                    expected = sample_model_config[key]
                    actual = config[key]
                    assert actual == expected, (
                        f"Mismatch for {key}: expected {expected}, got {actual}"
                    )

            # Check enum values are stored as strings
            assert config["graph_mode"] == "learned"  # Default GraphMode
            assert config["temporal_mode"] == "transformer"  # Default TemporalMode
            assert config["propagation_mode"] == "power"  # Default PropagationMode

            # Check dataset name is saved
            assert config["dataset"] == "METR-LA"

    def test_scaler_json_preserves_state(self, sample_scaler):
        """Test that scaler.json correctly preserves scaler state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            model = STGFormer(num_nodes=50, in_steps=12, out_steps=12)
            save_model(model, sample_scaler, save_path, "METR-LA")

            # Load scaler JSON
            with open(save_path / "scaler.json", "r") as f:
                scaler_data = json.load(f)

            # Verify mean and std are saved
            assert "mean" in scaler_data
            assert "std" in scaler_data

            # Verify values match
            assert torch.allclose(
                torch.tensor(scaler_data["mean"]), sample_scaler.mean, atol=1e-6
            )
            assert torch.allclose(
                torch.tensor(scaler_data["std"]), sample_scaler.std, atol=1e-6
            )

    def test_save_without_scaler(self, sample_model):
        """Test saving model without scaler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            scaler = StandardScaler()  # Not fitted
            save_model(sample_model, scaler, save_path, "METR-LA")

            # Scaler JSON should still exist but be empty
            with open(save_path / "scaler.json", "r") as f:
                scaler_data = json.load(f)

            assert scaler_data["mean"] is None
            assert scaler_data["std"] is None

    def test_load_model_round_trip(self, sample_model, sample_scaler):
        """Test save → load produces identical model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"

            # Save model
            save_model(sample_model, sample_scaler, save_path, "METR-LA")

            # Load model
            loaded_model, loaded_scaler = load_model(save_path, "METR-LA")
            loaded_model.eval()

            # Check model architecture matches
            assert loaded_model.num_nodes == sample_model.num_nodes
            assert loaded_model.in_steps == sample_model.in_steps
            assert loaded_model.out_steps == sample_model.out_steps
            assert loaded_model.model_dim == sample_model.model_dim

            # Check parameter count matches
            original_params = sum(p.numel() for p in sample_model.parameters())
            loaded_params = sum(p.numel() for p in loaded_model.parameters())
            assert loaded_params == original_params

            # Check scaler was loaded correctly
            assert torch.allclose(loaded_scaler.mean, sample_scaler.mean, atol=1e-6)
            assert torch.allclose(loaded_scaler.std, sample_scaler.std, atol=1e-6)

    def test_load_model_produces_identical_outputs(self, sample_model, sample_input):
        """Test that loaded model produces identical forward pass outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"

            # Set deterministic mode and move to CPU for consistent testing
            torch.manual_seed(42)
            sample_model = sample_model.cpu()
            sample_model.eval()
            sample_input = sample_input.cpu()

            # Get original output
            with torch.no_grad():
                original_output = sample_model(sample_input)

            # Save and load
            save_model(sample_model, StandardScaler(), save_path, "METR-LA")
            loaded_model, _ = load_model(save_path, "METR-LA", device="cpu")
            loaded_model.eval()

            # Get loaded model output
            torch.manual_seed(42)
            with torch.no_grad():
                loaded_output = loaded_model(sample_input)

            # Outputs should be identical
            assert torch.allclose(original_output, loaded_output, atol=1e-5)

    def test_load_model_without_scaler(self, sample_model):
        """Test loading model when scaler was not provided during save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"

            save_model(sample_model, StandardScaler(), save_path, "METR-LA")
            loaded_model, loaded_scaler = load_model(save_path, "METR-LA")

            # Scaler should be returned but not fitted
            assert loaded_scaler.mean is None
            assert loaded_scaler.std is None

    def test_safetensors_format_integrity(self, sample_model):
        """Test that model weights are saved in safetensors format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            save_model(sample_model, StandardScaler(), save_path, "METR-LA")

            # Check safetensors file exists and is not empty
            safetensors_path = save_path / "model.safetensors"
            assert safetensors_path.exists()
            assert safetensors_path.stat().st_size > 0

            # Verify it can be loaded (using PyTorch's load_file)
            from safetensors.torch import load_file

            state_dict = load_file(str(safetensors_path))

            # Check that state dict contains weights
            assert len(state_dict) > 0
            assert all(isinstance(v, torch.Tensor) for v in state_dict.values())

    def test_state_dict_keys_match(self, sample_model):
        """Test that saved and loaded state_dict keys match exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"

            # Get original state dict keys
            original_keys = set(sample_model.state_dict().keys())

            # Save and load
            save_model(sample_model, StandardScaler(), save_path, "METR-LA")
            loaded_model, _ = load_model(save_path, "METR-LA")

            # Get loaded state dict keys
            loaded_keys = set(loaded_model.state_dict().keys())

            # Keys should match exactly
            assert original_keys == loaded_keys

    def test_different_graph_modes_serialization(self):
        """Test serialization with LEARNED graph mode."""
        # Only test LEARNED mode to avoid geo_adj requirements
        # (SPECTRAL_INIT, GEOGRAPHIC, and HYBRID all require geo_adj)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"

            model = STGFormer(
                num_nodes=50, in_steps=12, out_steps=12, graph_mode=GraphMode.LEARNED
            )
            model.eval()

            # Save and load
            save_model(model, StandardScaler(), save_path, "METR-LA")
            loaded_model, _ = load_model(save_path, "METR-LA", device="cpu")
            loaded_model.eval()

            # Verify graph mode is preserved
            assert loaded_model.graph_mode == GraphMode.LEARNED

    def test_different_temporal_modes_serialization(self):
        """Test serialization with different temporal modes."""
        temporal_modes = [TemporalMode.TRANSFORMER, TemporalMode.TCN]

        for temporal_mode in temporal_modes:
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = Path(tmpdir) / "model"

                model = STGFormer(
                    num_nodes=50,
                    in_steps=12,
                    out_steps=12,
                    temporal_mode=temporal_mode,
                )
                model.eval()

                # Save and load
                save_model(model, StandardScaler(), save_path, "METR-LA")
                loaded_model, _ = load_model(save_path, "METR-LA")
                loaded_model.eval()

                # Verify temporal mode is preserved
                assert loaded_model.temporal_mode == temporal_mode

    def test_different_propagation_modes_serialization(self):
        """Test serialization with POWER propagation mode."""
        # Only test POWER mode to avoid geo_adj dataset node count issues
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"

            model = STGFormer(
                num_nodes=50,
                in_steps=12,
                out_steps=12,
                propagation_mode=PropagationMode.POWER,
            )
            model.eval()

            # Save and load
            save_model(model, StandardScaler(), save_path, "METR-LA")
            loaded_model, _ = load_model(save_path, "METR-LA", device="cpu")
            loaded_model.eval()

            # Verify propagation mode is preserved
            assert loaded_model.propagation_mode == PropagationMode.POWER

    def test_device_compatibility_cpu_to_gpu(self, sample_model, sample_input):
        """Test loading model from CPU checkpoint to GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"

            # Save on CPU
            sample_model.eval()
            save_model(sample_model, StandardScaler(), save_path, "METR-LA")

            # Load to GPU
            loaded_model, _ = load_model(save_path, device="cuda")
            loaded_model.eval()

            # Model should be on GPU
            assert next(loaded_model.parameters()).is_cuda

            # Forward pass should work on GPU
            gpu_input = sample_input.cuda()
            with torch.no_grad():
                gpu_output = loaded_model(gpu_input)

            assert gpu_output.is_cuda

    def test_device_compatibility_gpu_to_cpu(self, sample_model, sample_input):
        """Test loading model from GPU checkpoint to CPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"

            # Move model to GPU and save
            sample_model = sample_model.cuda()
            sample_model.eval()
            save_model(sample_model, StandardScaler(), save_path, "METR-LA")

            # Load to CPU
            loaded_model, _ = load_model(save_path, device="cpu")
            loaded_model.eval()

            # Model should be on CPU
            assert not next(loaded_model.parameters()).is_cuda

            # Forward pass should work on CPU
            with torch.no_grad():
                cpu_output = loaded_model(sample_input)

            assert not cpu_output.is_cuda

    def test_backward_compatibility_missing_optional_params(self, sample_model):
        """Test that model can be loaded even if config is missing optional parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"

            # Save model
            save_model(sample_model, StandardScaler(), save_path, "METR-LA")

            # Manually modify config to remove some optional parameters
            config_path = save_path / "config.json"
            with open(config_path, "r") as f:
                config = json.load(f)

            # Remove optional parameters (simulating old checkpoint)
            config.pop("dropout_a", None)
            config.pop("use_mixed_proj", None)

            with open(config_path, "w") as f:
                json.dump(config, f)

            # Load should still work (using defaults)
            loaded_model, _ = load_model(save_path, "METR-LA")

            # Model should load successfully
            assert loaded_model is not None

    def test_scaler_inverse_transform_after_load(self, sample_model, sample_scaler):
        """Test that scaler can inverse transform after loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"

            # Save with scaler
            save_model(sample_model, sample_scaler, save_path, "METR-LA")

            # Load scaler
            _, loaded_scaler = load_model(save_path, "METR-LA")

            # Create normalized data
            torch.manual_seed(42)
            normalized_data = torch.randn(10, 50, 1)

            # Inverse transform should work identically
            original_result = sample_scaler.inverse_transform(normalized_data)
            loaded_result = loaded_scaler.inverse_transform(normalized_data)

            assert torch.allclose(original_result, loaded_result, atol=1e-6)

    def test_multiple_save_load_cycles(self, sample_model, sample_input):
        """Test that multiple save/load cycles don't degrade model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_model = sample_model.cpu()
            sample_model.eval()
            sample_input = sample_input.cpu()
            torch.manual_seed(42)

            with torch.no_grad():
                original_output = sample_model(sample_input)

            # Perform 3 save/load cycles
            model = sample_model
            for i in range(3):
                save_path = Path(tmpdir) / f"model_{i}"
                save_model(model, StandardScaler(), save_path, "METR-LA")
                model, _ = load_model(save_path, "METR-LA", device="cpu")
                model.eval()

            # Final output should match original
            torch.manual_seed(42)
            with torch.no_grad():
                final_output = model(sample_input)

            assert torch.allclose(original_output, final_output, atol=1e-5)

    def test_dataset_name_preserved(self, sample_model):
        """Test that dataset name is preserved in config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"

            save_model(sample_model, StandardScaler(), save_path, "METR-LA")

            # Check config contains dataset name (key is "dataset", not "dataset_name")
            with open(save_path / "config.json", "r") as f:
                config = json.load(f)

            assert config["dataset"] == "METR-LA"
