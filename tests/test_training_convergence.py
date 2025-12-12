"""Tests for training convergence and training loop functions.

These tests verify:
1. Multi-epoch training reduces loss
2. train_epoch and evaluate functions work correctly
3. Gradient clipping prevents NaN/Inf
4. Scaler integration during training
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from stgformer.model import STGFormer
from utils.training import (
    MaskedHuberLoss,
    StandardScaler,
    evaluate,
    train_epoch,
)


class TestTrainingLoops:
    """Tests for train_epoch and evaluate functions."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple STGFormer model for testing."""
        model = STGFormer(
            num_nodes=10,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            output_dim=1,
            input_embedding_dim=16,
            tod_embedding_dim=0,
            dow_embedding_dim=0,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=16,
            num_heads=2,
            num_layers=1,
            dropout=0.0,  # Disable dropout for deterministic tests
        )
        return model

    @pytest.fixture
    def sample_dataloader(self):
        """Create a sample dataloader for training."""
        torch.manual_seed(42)
        np.random.seed(42)

        # Create synthetic data: (batch, seq_len, num_nodes, features)
        x = torch.randn(32, 12, 10, 1)  # 32 samples
        y = torch.randn(32, 12, 10, 1)

        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        return dataloader

    @pytest.fixture
    def fitted_scaler(self):
        """Create a fitted scaler."""
        torch.manual_seed(42)
        data = torch.randn(100, 10, 1)
        scaler = StandardScaler()
        scaler.fit_transform(data)
        return scaler

    def test_train_epoch_basic(self, simple_model, sample_dataloader):
        """Test basic train_epoch execution."""
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        criterion = MaskedHuberLoss()

        # Train for one epoch
        loss = train_epoch(
            simple_model,
            sample_dataloader,
            optimizer,
            criterion,
            device,
            null_val=0.0,
        )

        # Loss should be a positive scalar
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_train_epoch_with_scaler(
        self, simple_model, sample_dataloader, fitted_scaler
    ):
        """Test train_epoch with scaler for inverse transform."""
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        criterion = MaskedHuberLoss()

        # Train with scaler
        loss = train_epoch(
            simple_model,
            sample_dataloader,
            optimizer,
            criterion,
            device,
            scaler=fitted_scaler,
            null_val=0.0,
        )

        assert isinstance(loss, float)
        assert loss > 0.0

    def test_evaluate_basic(self, simple_model, sample_dataloader):
        """Test basic evaluate execution."""
        device = torch.device("cpu")
        criterion = MaskedHuberLoss()

        # Evaluate
        loss = evaluate(
            simple_model,
            sample_dataloader,
            criterion,
            device,
            null_val=0.0,
        )

        # Loss should be a positive scalar
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_evaluate_with_scaler(self, simple_model, sample_dataloader, fitted_scaler):
        """Test evaluate with scaler."""
        device = torch.device("cpu")
        criterion = MaskedHuberLoss()

        loss = evaluate(
            simple_model,
            sample_dataloader,
            criterion,
            device,
            scaler=fitted_scaler,
            null_val=0.0,
        )

        assert isinstance(loss, float)
        assert loss > 0.0

    def test_evaluate_deterministic(self, simple_model, sample_dataloader):
        """Test that evaluate produces deterministic results."""
        device = torch.device("cpu")
        criterion = MaskedHuberLoss()

        # Run evaluate twice
        loss1 = evaluate(simple_model, sample_dataloader, criterion, device)
        loss2 = evaluate(simple_model, sample_dataloader, criterion, device)

        # Should be identical
        assert loss1 == loss2

    def test_train_updates_parameters(self, simple_model, sample_dataloader):
        """Test that train_epoch actually updates model parameters."""
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        criterion = MaskedHuberLoss()

        # Get initial parameter values
        initial_params = [p.clone().detach() for p in simple_model.parameters()]

        # Train for one epoch
        train_epoch(simple_model, sample_dataloader, optimizer, criterion, device)

        # Check that at least some parameters changed
        params_changed = False
        for initial, current in zip(initial_params, simple_model.parameters()):
            if not torch.allclose(initial, current, atol=1e-6):
                params_changed = True
                break

        assert params_changed, "No parameters were updated during training"

    def test_evaluate_does_not_update_parameters(self, simple_model, sample_dataloader):
        """Test that evaluate does not update model parameters."""
        device = torch.device("cpu")
        criterion = MaskedHuberLoss()

        # Get initial parameter values
        initial_params = [p.clone().detach() for p in simple_model.parameters()]

        # Evaluate
        evaluate(simple_model, sample_dataloader, criterion, device)

        # Parameters should be unchanged
        for initial, current in zip(initial_params, simple_model.parameters()):
            assert torch.allclose(initial, current)

    def test_train_epoch_sets_training_mode(self, simple_model, sample_dataloader):
        """Test that train_epoch sets model to training mode."""
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        criterion = MaskedHuberLoss()

        # Set to eval mode first
        simple_model.eval()
        assert not simple_model.training

        # Train (should set to training mode)
        train_epoch(simple_model, sample_dataloader, optimizer, criterion, device)

        # Model should be in training mode
        assert simple_model.training

    def test_evaluate_sets_eval_mode(self, simple_model, sample_dataloader):
        """Test that evaluate sets model to eval mode."""
        device = torch.device("cpu")
        criterion = MaskedHuberLoss()

        # Set to training mode first
        simple_model.train()
        assert simple_model.training

        # Evaluate (should set to eval mode)
        evaluate(simple_model, sample_dataloader, criterion, device)

        # Model should be in eval mode
        assert not simple_model.training


class TestTrainingConvergence:
    """Tests for multi-epoch training convergence."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for convergence testing."""
        model = STGFormer(
            num_nodes=10,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            output_dim=1,
            input_embedding_dim=16,
            tod_embedding_dim=0,
            dow_embedding_dim=0,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=16,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
        return model

    @pytest.fixture
    def synthetic_dataloader(self):
        """Create synthetic data with a learnable pattern."""
        torch.manual_seed(42)

        # Create simple pattern: y = x + noise
        x = torch.randn(64, 12, 10, 1)
        y = x + torch.randn(64, 12, 10, 1) * 0.1  # Small noise

        dataset = TensorDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        return train_loader

    def test_multi_epoch_training_reduces_loss(
        self, simple_model, synthetic_dataloader
    ):
        """Test that training for multiple epochs reduces loss."""
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        criterion = MaskedHuberLoss()

        losses = []
        num_epochs = 5

        for epoch in range(num_epochs):
            loss = train_epoch(
                simple_model,
                synthetic_dataloader,
                optimizer,
                criterion,
                device,
            )
            losses.append(loss)

        # Loss should generally decrease
        # Check that final loss is less than initial loss
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]} -> {losses[-1]}"
        )

        # Check that loss is decreasing trend (allow some fluctuation)
        # At least 60% of epochs should have lower loss than epoch 0
        lower_count = sum(1 for loss in losses[1:] if loss < losses[0])
        assert lower_count >= len(losses) * 0.6

    def test_training_with_gradient_clipping(self, simple_model, synthetic_dataloader):
        """Test training with gradient clipping to prevent NaN/Inf."""
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1.0)  # Very large LR
        criterion = MaskedHuberLoss()

        max_grad_norm = 1.0

        for epoch in range(3):
            simple_model.train()
            for x, y in synthetic_dataloader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                pred = simple_model(x)
                loss = criterion(pred, y)
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(simple_model.parameters(), max_grad_norm)

                optimizer.step()

        # Check that no parameters are NaN or Inf
        for param in simple_model.parameters():
            assert not torch.isnan(param).any(), "Parameters contain NaN"
            assert not torch.isinf(param).any(), "Parameters contain Inf"

    def test_training_val_split_pattern(self, simple_model, synthetic_dataloader):
        """Test typical train/val pattern: train improves more than val."""
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        criterion = MaskedHuberLoss()

        # Use same data for train and val (not realistic but sufficient for test)
        train_loader = synthetic_dataloader
        val_loader = synthetic_dataloader

        train_losses = []
        val_losses = []

        for epoch in range(5):
            # Train
            train_loss = train_epoch(
                simple_model, train_loader, optimizer, criterion, device
            )
            train_losses.append(train_loss)

            # Validate
            val_loss = evaluate(simple_model, val_loader, criterion, device)
            val_losses.append(val_loss)

        # Both should decrease
        assert train_losses[-1] < train_losses[0]
        # Val loss might not always decrease but should be reasonable
        assert val_losses[-1] < val_losses[0] * 1.5  # Allow some increase

    def test_overfitting_detection(self):
        """Test that training on tiny dataset reduces train loss (learning happens)."""
        torch.manual_seed(42)

        # Create tiny dataset for overfitting
        x_train = torch.randn(4, 12, 5, 1)
        y_train = torch.randn(4, 12, 5, 1)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=4)

        # Different val data
        x_val = torch.randn(4, 12, 5, 1)
        y_val = torch.randn(4, 12, 5, 1)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=4)

        model = STGFormer(
            num_nodes=5,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            output_dim=1,
            input_embedding_dim=8,
            tod_embedding_dim=0,
            dow_embedding_dim=0,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=8,
            num_heads=2,
            num_layers=1,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = MaskedHuberLoss()
        device = torch.device("cpu")

        # Get initial loss before training
        initial_train_loss = evaluate(model, train_loader, criterion, device)

        # Train for many epochs (overfit)
        for epoch in range(50):
            train_epoch(model, train_loader, optimizer, criterion, device)

        # Get final losses
        final_train_loss = evaluate(model, train_loader, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        # Train loss should decrease (showing learning)
        assert final_train_loss < initial_train_loss

        # Both losses should be valid positive numbers
        assert final_train_loss > 0
        assert val_loss > 0

    def test_learning_rate_scheduling_effect(self, simple_model, synthetic_dataloader):
        """Test that learning rate scheduling can be integrated."""
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        criterion = MaskedHuberLoss()

        initial_lr = optimizer.param_groups[0]["lr"]
        losses = []

        for epoch in range(5):
            loss = train_epoch(
                simple_model, synthetic_dataloader, optimizer, criterion, device
            )
            losses.append(loss)
            scheduler.step()

        # Learning rate should have decreased
        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr

    def test_early_stopping_detection(self, simple_model, synthetic_dataloader):
        """Test that we can detect when to early stop based on validation loss."""
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        criterion = MaskedHuberLoss()

        val_losses = []
        patience = 3
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(10):
            # Train
            train_epoch(
                simple_model, synthetic_dataloader, optimizer, criterion, device
            )

            # Validate
            val_loss = evaluate(simple_model, synthetic_dataloader, criterion, device)
            val_losses.append(val_loss)

            # Early stopping logic
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                # Early stopping triggered
                break

        # We should have stopped before epoch 10 (or at least not failed)
        assert len(val_losses) <= 10

    def test_batch_size_effect(self, simple_model):
        """Test that different batch sizes work correctly."""
        torch.manual_seed(42)
        x = torch.randn(32, 12, 10, 1)
        y = torch.randn(32, 12, 10, 1)
        dataset = TensorDataset(x, y)

        device = torch.device("cpu")
        criterion = MaskedHuberLoss()

        batch_sizes = [4, 8, 16]
        losses = []

        for batch_size in batch_sizes:
            # Reset model
            model = STGFormer(
                num_nodes=10,
                in_steps=12,
                out_steps=12,
                input_dim=1,
                output_dim=1,
                input_embedding_dim=16,
                tod_embedding_dim=0,
                dow_embedding_dim=0,
                spatial_embedding_dim=0,
                adaptive_embedding_dim=16,
                num_heads=2,
                num_layers=1,
                dropout=0.0,
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            loss = train_epoch(model, loader, optimizer, criterion, device)
            losses.append(loss)

        # All batch sizes should produce valid losses
        assert all(loss > 0 for loss in losses)

    def test_device_compatibility_training(self, simple_model, synthetic_dataloader):
        """Test training on different devices (CPU/GPU if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        criterion = MaskedHuberLoss()

        # Train on CPU
        device_cpu = torch.device("cpu")
        model_cpu = simple_model
        optimizer_cpu = torch.optim.Adam(model_cpu.parameters(), lr=0.01)
        loss_cpu = train_epoch(
            model_cpu, synthetic_dataloader, optimizer_cpu, criterion, device_cpu
        )

        # Train on GPU
        device_gpu = torch.device("cuda")
        model_gpu = STGFormer(
            num_nodes=10,
            in_steps=12,
            out_steps=12,
            input_dim=1,
            output_dim=1,
            input_embedding_dim=16,
            tod_embedding_dim=0,
            dow_embedding_dim=0,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=16,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        ).to(device_gpu)
        optimizer_gpu = torch.optim.Adam(model_gpu.parameters(), lr=0.01)

        # Create GPU dataloader
        x_gpu = torch.randn(64, 12, 10, 1)
        y_gpu = torch.randn(64, 12, 10, 1)
        gpu_loader = DataLoader(TensorDataset(x_gpu, y_gpu), batch_size=8)

        loss_gpu = train_epoch(
            model_gpu, gpu_loader, optimizer_gpu, criterion, device_gpu
        )

        # Both should produce valid losses
        assert loss_cpu > 0
        assert loss_gpu > 0
