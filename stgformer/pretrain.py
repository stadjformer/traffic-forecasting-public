"""Masked Node Pretraining for STGFormer

Self-supervised pretraining using masked node prediction, similar to BERT's
masked language modeling but for graph-structured traffic data.

Goal: Learn spatial correlations via "if sensors A, B, C show X, then sensor D
probably shows Y" before forecasting.

Two-stage curriculum:
- Stage 1 (per-timestep): Mask random (node, timestep) positions - easier task
- Stage 2 (per-node): Mask entire nodes across all timesteps - harder task
"""

import time
from typing import Literal

import torch
from tqdm import tqdm

from stgformer.model import STGFormer


class MaskedNodePretrainer(torch.nn.Module):
    """Self-supervised pretraining via masked node prediction.

    This module wraps an STGFormer model and adds:
    - Masking logic (per-timestep or per-node)
    - Imputation prediction head
    - Missing value detection

    Args:
        model: STGFormer model to pretrain
        mask_value: Value to use for masking (default 0.0 to match missing data representation)
    """

    def __init__(self, model: STGFormer, mask_value: float = 0.0):
        super().__init__()
        self.model = model
        self.mask_value = mask_value

        # Temporal imputation head: predicts per-timestep values from spatial representation
        # Input: [batch, num_nodes, model_dim]
        # Output: [batch, num_nodes, in_steps] (speed value per node per timestep)
        self.imputation_head = torch.nn.Linear(model.model_dim, model.in_steps)

    def detect_missing(self, x: torch.Tensor) -> torch.Tensor:
        """Find existing missing values (zeros and NaN) in speed feature.

        Args:
            x: Input tensor [batch, time, nodes, features]

        Returns:
            Boolean mask [batch, time, nodes, 1] where True = missing
        """
        speed = x[..., 0:1]
        # Detect both zeros (mask_value) and NaN as missing
        return (speed == self.mask_value) | torch.isnan(speed)

    def create_mask_per_timestep(
        self,
        x: torch.Tensor,
        existing_missing: torch.Tensor,
        mask_ratio: float,
    ) -> torch.Tensor:
        """Stage 1: Mask random (node, timestep) positions.

        Easier task - model can use temporal context from same node + spatial
        context from neighbors.

        Args:
            x: Input tensor [batch, time, nodes, features]
            existing_missing: Boolean mask of already-missing values [batch, time, nodes, 1]
            mask_ratio: Fraction of valid positions to mask

        Returns:
            Boolean mask [batch, time, nodes, 1] of newly masked positions
        """
        batch_size, time_steps, num_nodes, _ = x.shape
        device = x.device

        # Valid positions are those that are NOT already missing
        valid_mask = ~existing_missing.squeeze(-1)  # [batch, time, nodes]

        # Count valid positions per sample
        num_valid = valid_mask.sum(dim=(1, 2))  # [batch]

        # Calculate number to mask per sample
        num_to_mask = (num_valid.float() * mask_ratio).long()

        # Create new mask
        new_mask = torch.zeros(
            batch_size, time_steps, num_nodes, dtype=torch.bool, device=device
        )

        for b in range(batch_size):
            # Get indices of valid positions for this sample
            valid_indices = torch.nonzero(valid_mask[b], as_tuple=False)  # [N_valid, 2]

            if len(valid_indices) == 0 or num_to_mask[b] == 0:
                continue

            # Randomly select positions to mask
            perm = torch.randperm(len(valid_indices), device=device)[: num_to_mask[b]]
            selected = valid_indices[perm]

            # Set mask
            new_mask[b, selected[:, 0], selected[:, 1]] = True

        return new_mask.unsqueeze(-1)  # [batch, time, nodes, 1]

    def create_mask_per_node(
        self,
        x: torch.Tensor,
        existing_missing: torch.Tensor,
        mask_ratio: float,
    ) -> torch.Tensor:
        """Stage 2: Mask entire nodes across ALL timesteps.

        Harder task - forces pure spatial reasoning, no temporal interpolation.
        Matches real missing data pattern (sensor failures = whole node missing).

        Args:
            x: Input tensor [batch, time, nodes, features]
            existing_missing: Boolean mask of already-missing values [batch, time, nodes, 1]
            mask_ratio: Fraction of nodes to mask

        Returns:
            Boolean mask [batch, time, nodes, 1] of newly masked positions
        """
        batch_size, time_steps, num_nodes, _ = x.shape
        device = x.device

        # A node is eligible if it has at least one observed timestep.
        node_has_valid = ~existing_missing.squeeze(-1).all(dim=1)  # [batch, nodes]

        # Count valid nodes per sample
        num_valid_nodes = node_has_valid.sum(dim=1)  # [batch]

        # Calculate number of nodes to mask per sample
        num_to_mask = (num_valid_nodes.float() * mask_ratio).long()

        # Create new mask
        new_mask = torch.zeros(
            batch_size, time_steps, num_nodes, dtype=torch.bool, device=device
        )

        for b in range(batch_size):
            # Get indices of valid nodes
            valid_node_indices = torch.nonzero(
                node_has_valid[b], as_tuple=False
            ).squeeze(-1)

            if len(valid_node_indices) == 0 or num_to_mask[b] == 0:
                continue

            # Randomly select nodes to mask
            perm = torch.randperm(len(valid_node_indices), device=device)[
                : num_to_mask[b]
            ]
            selected_nodes = valid_node_indices[perm]

            # Mask ALL timesteps for selected nodes
            new_mask[b, :, selected_nodes] = True

        return new_mask.unsqueeze(-1)  # [batch, time, nodes, 1]

    def apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to input by setting masked positions to mask_value.

        Only masks the speed feature (index 0), preserving TOD/DOW features.

        Args:
            x: Input tensor [batch, time, nodes, features]
            mask: Boolean mask [batch, time, nodes, 1]

        Returns:
            Masked input tensor with same shape
        """
        x_masked = x.clone()
        # Only mask the speed feature (index 0)
        x_masked[..., 0:1] = torch.where(mask, self.mask_value, x[..., 0:1])
        return x_masked

    def pretrain_step(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        masking_mode: Literal["per_timestep", "per_node"] = "per_timestep",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for pretraining.

        Args:
            x: Input tensor [batch, time, nodes, features]
            mask_ratio: Fraction of positions/nodes to mask
            masking_mode: "per_timestep" (stage 1) or "per_node" (stage 2)

        Returns:
            Tuple of (loss, predictions, mask):
            - loss: Scalar MSE loss on masked positions
            - predictions: [batch, nodes, 1] predicted values
            - mask: [batch, time, nodes, 1] boolean mask of masked positions
        """
        # Detect existing missing values
        existing_missing = self.detect_missing(x)

        # Create new mask based on mode
        if masking_mode == "per_timestep":
            new_mask = self.create_mask_per_timestep(x, existing_missing, mask_ratio)
        else:
            new_mask = self.create_mask_per_node(x, existing_missing, mask_ratio)

        # Apply mask to input
        x_masked = self.apply_mask(x, new_mask)

        # Get spatial representation from model
        spatial_repr = self.model.get_spatial_representation(
            x_masked
        )  # [batch, nodes, model_dim]

        # Predict values with temporal imputation head
        predictions = self.imputation_head(spatial_repr)  # [batch, nodes, in_steps]

        # Get target values (original speed values)
        target_speed = x[..., 0]  # [batch, time, nodes]

        # Reshape predictions to match target: [batch, nodes, time] -> [batch, time, nodes]
        predictions = predictions.transpose(1, 2)  # [batch, in_steps, nodes]

        # Compute loss only on newly masked positions (excluding NaN targets)
        new_mask_squeezed = new_mask.squeeze(-1)  # [batch, time, nodes]

        # Use per-timestep MSE for both masking modes to preserve temporal learning
        # Previously, per-node mode used mean aggregation which weakened temporal dynamics
        if new_mask_squeezed.any():
            pred_masked = predictions[new_mask_squeezed]
            target_masked = target_speed[new_mask_squeezed]

            # Filter out any remaining NaN values in target (safety check)
            valid_mask = ~torch.isnan(target_masked)
            if valid_mask.any():
                loss = torch.nn.functional.mse_loss(
                    pred_masked[valid_mask],
                    target_masked[valid_mask],
                )
            else:
                loss = torch.tensor(0.0, device=x.device)
        else:
            loss = torch.tensor(0.0, device=x.device)

        return loss, predictions, new_mask


def pretrain_graph_imputation(
    model: STGFormer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader | None = None,
    stage1_epochs: int = 5,
    stage1_mask_ratio: float = 0.15,
    stage2_epochs: int = 5,
    stage2_mask_ratio: float = 0.10,
    learning_rate: float = 0.001,
    device: str = "cuda",
    verbose: bool = True,
    wandb_run=None,
) -> tuple[STGFormer, torch.nn.Linear]:
    """Two-stage curriculum pretraining loop.

    Stage 1: Per-timestep masking (easier)
    Stage 2: Per-node masking (harder)

    Args:
        model: STGFormer model to pretrain
        train_loader: DataLoader yielding (x, y) tuples
        val_loader: Optional validation DataLoader (no gradient updates)
        stage1_epochs: Number of epochs for stage 1
        stage1_mask_ratio: Mask ratio for stage 1
        stage2_epochs: Number of epochs for stage 2
        stage2_mask_ratio: Mask ratio for stage 2
        learning_rate: Learning rate for optimizer
        device: Device to train on
        verbose: Whether to print progress
        wandb_run: Optional W&B run for logging

    Returns:
        Tuple of (pretrained model, imputation head)
    """
    pretrainer = MaskedNodePretrainer(model)
    pretrainer = pretrainer.to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": pretrainer.imputation_head.parameters()},
        ],
        lr=learning_rate,
    )

    def evaluate_validation(mask_ratio: float, masking_mode: str, max_batches: int = 100) -> float:
        """Evaluate validation loss without gradient updates.

        Args:
            mask_ratio: Masking ratio for validation
            masking_mode: 'per_timestep' or 'per_node'
            max_batches: Maximum batches to evaluate (default 100 for speed)
        """
        if val_loader is None:
            return 0.0

        pretrainer.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                loss, _, _ = pretrainer.pretrain_step(x, mask_ratio, masking_mode)
                val_loss += loss.item()
                num_batches += 1
                if num_batches >= max_batches:
                    break

        pretrainer.train()
        return val_loss / max(num_batches, 1)

    total_epochs = stage1_epochs + stage2_epochs
    pretrain_start_time = time.time()

    # Track global batch counter for W&B logging (continuous across epochs)
    stage1_global_batch = 0
    # stage2_global_batch is set when Stage 2 starts (continues from stage1_global_batch)

    # Stage 1: Per-timestep masking (easy)
    if stage1_epochs > 0:
        if verbose:
            print(
                f"Stage 1: Per-timestep masking ({stage1_epochs} epochs, {stage1_mask_ratio:.0%} mask ratio)"
            )
            print(f"  W&B logging: {'enabled' if wandb_run is not None else 'DISABLED (wandb_run is None)'}")

        log_interval = 100  # Log every 100 batches

        for epoch in range(stage1_epochs):
            epoch_start_time = time.time()
            pretrainer.train()
            epoch_loss = 0.0
            num_batches = 0

            batch_iter = tqdm(
                train_loader,
                desc=f"  Stage1 Epoch {epoch + 1}/{stage1_epochs}",
                leave=False,
                disable=not verbose,
            )

            for batch in batch_iter:
                x = batch[0].to(device)  # (x, y) tuple from DataLoader

                optimizer.zero_grad()
                loss, _, _ = pretrainer.pretrain_step(
                    x, stage1_mask_ratio, masking_mode="per_timestep"
                )
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Update progress bar with current loss
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

                stage1_global_batch += 1
                # Log to W&B every log_interval batches
                if wandb_run is not None and num_batches % log_interval == 0:
                    wandb_run.log(
                        {"pretrain/stage1_batch_loss": loss.item()},
                        step=stage1_global_batch,
                    )

            avg_train_loss = epoch_loss / max(num_batches, 1)
            avg_val_loss = evaluate_validation(stage1_mask_ratio, "per_timestep")
            epoch_time = time.time() - epoch_start_time

            if verbose:
                if val_loader is not None:
                    print(
                        f"  Epoch {epoch + 1}/{stage1_epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f} ({epoch_time:.1f}s)"
                    )
                else:
                    print(
                        f"  Epoch {epoch + 1}/{stage1_epochs}: Loss = {avg_train_loss:.6f} ({epoch_time:.1f}s)"
                    )

            if wandb_run is not None:
                log_dict = {
                    "pretrain/stage1_epoch": epoch + 1,
                    "pretrain/stage1_train_loss": avg_train_loss,
                    "pretrain/stage1_epoch_time": epoch_time,
                }
                if val_loader is not None:
                    log_dict["pretrain/stage1_val_loss"] = avg_val_loss
                # Use global batch step for consistent x-axis
                wandb_run.log(log_dict, step=stage1_global_batch)

    # Stage 2: Per-node masking (hard)
    if stage2_epochs > 0:
        # Continue step counter from Stage 1 for monotonic W&B steps
        stage2_global_batch = stage1_global_batch

        if verbose:
            print(
                f"Stage 2: Per-node masking ({stage2_epochs} epochs, {stage2_mask_ratio:.0%} mask ratio)"
            )

        log_interval = 100  # Log every 100 batches

        for epoch in range(stage2_epochs):
            epoch_start_time = time.time()
            pretrainer.train()
            epoch_loss = 0.0
            num_batches = 0

            batch_iter = tqdm(
                train_loader,
                desc=f"  Stage2 Epoch {epoch + 1}/{stage2_epochs}",
                leave=False,
                disable=not verbose,
            )

            for batch in batch_iter:
                x = batch[0].to(device)

                optimizer.zero_grad()
                loss, _, _ = pretrainer.pretrain_step(
                    x, stage2_mask_ratio, masking_mode="per_node"
                )
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Update progress bar with current loss
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

                stage2_global_batch += 1
                # Log to W&B every log_interval batches
                if wandb_run is not None and num_batches % log_interval == 0:
                    wandb_run.log(
                        {"pretrain/stage2_batch_loss": loss.item()},
                        step=stage2_global_batch,
                    )

            avg_train_loss = epoch_loss / max(num_batches, 1)
            avg_val_loss = evaluate_validation(stage2_mask_ratio, "per_node")
            # Also compute Stage 1 val loss to detect forgetting
            stage1_val_loss = evaluate_validation(stage1_mask_ratio, "per_timestep")
            epoch_time = time.time() - epoch_start_time

            if verbose:
                if val_loader is not None:
                    print(
                        f"  Epoch {epoch + 1}/{stage2_epochs}: Train Loss = {avg_train_loss:.6f}, "
                        f"Val Loss = {avg_val_loss:.6f}, Stage1 Val = {stage1_val_loss:.6f} ({epoch_time:.1f}s)"
                    )
                else:
                    print(
                        f"  Epoch {epoch + 1}/{stage2_epochs}: Loss = {avg_train_loss:.6f} ({epoch_time:.1f}s)"
                    )

            if wandb_run is not None:
                log_dict = {
                    "pretrain/stage2_epoch": epoch + 1,
                    "pretrain/stage2_train_loss": avg_train_loss,
                    "pretrain/stage2_epoch_time": epoch_time,
                }
                if val_loader is not None:
                    log_dict["pretrain/stage2_val_loss"] = avg_val_loss
                    log_dict["pretrain/stage1_val_during_stage2"] = stage1_val_loss
                # Use global batch step for consistent x-axis
                wandb_run.log(log_dict, step=stage2_global_batch)

    total_pretrain_time = time.time() - pretrain_start_time

    if verbose:
        print(
            f"Pretraining complete ({total_epochs} total epochs, {total_pretrain_time:.1f}s total)"
        )

    return model, pretrainer.imputation_head


def impute_missing_data(
    model: STGFormer,
    imputation_head: torch.nn.Linear,
    data: torch.Tensor,
    num_iterations: int = 1,
    batch_size: int | None = None,
    device: str | torch.device | None = None,
    mask_value: float = 0.0,
    use_normalized_data: bool = False,
) -> torch.Tensor:
    """Impute missing values using pretrained model.

    This is a preprocessing utility, not part of the training loop.

    Args:
        model: Pretrained STGFormer model
        imputation_head: Trained imputation head
        data: Input tensor [batch, time, nodes, features] or full dataset
        num_iterations: Number of imputation iterations (for iterative refinement)
        batch_size: Batch size for processing large datasets (default: process all at once)
        device: Device to use for computation
        mask_value: Value used to indicate missing data
        use_normalized_data: If True, data is normalized (mean=0, std=1) and predictions
            won't be clamped to non-negative values

    Returns:
        Data tensor with missing values imputed
    """
    model.eval()
    imputation_head.eval()

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    # Clone to avoid modifying original tensor
    data_clone = data.clone()
    original_device = data_clone.device

    # Print statistics before imputation
    missing_mask = data_clone[..., 0] == mask_value
    num_missing = missing_mask.sum().item()
    if num_missing > 0:
        print("\nImputation statistics:")
        print(f"  Missing values: {num_missing:,} / {data_clone[..., 0].numel():,}")
        non_missing_values = data_clone[..., 0][~missing_mask]
        if non_missing_values.numel() > 0:
            print("  Non-missing data stats (before imputation):")
            print(f"    Mean: {non_missing_values.mean().item():.4f}")
            print(f"    Std:  {non_missing_values.std().item():.4f}")
            print(f"    Min:  {non_missing_values.min().item():.4f}")
            print(f"    Max:  {non_missing_values.max().item():.4f}")

    # If batch_size not specified, process all at once
    if batch_size is None:
        data_on_device = data_clone.to(device)
        imputed = _impute_batch(
            model,
            imputation_head,
            data_on_device,
            num_iterations,
            mask_value,
            use_normalized_data,
        )
        result = imputed.to(original_device)
    else:
        # Otherwise, process in batches
        num_samples = data_clone.shape[0]

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_data = data_clone[start_idx:end_idx].to(device)

            # Impute this batch
            imputed_batch = _impute_batch(
                model,
                imputation_head,
                batch_data,
                num_iterations,
                mask_value,
                use_normalized_data,
            )

            # Move back to original device and store
            data_clone[start_idx:end_idx] = imputed_batch.to(original_device)

        result = data_clone

    # Print statistics after imputation
    if num_missing > 0:
        imputed_values = result[..., 0][missing_mask]
        print("  Imputed values stats (newly imputed only):")
        print(f"    Mean: {imputed_values.mean().item():.4f}")
        print(f"    Std:  {imputed_values.std().item():.4f}")
        print(f"    Min:  {imputed_values.min().item():.4f}")
        print(f"    Max:  {imputed_values.max().item():.4f}")

        # Check if any negative values were imputed (important for normalized data)
        num_negative = (imputed_values < 0).sum().item()
        if num_negative > 0:
            pct_negative = (num_negative / num_missing) * 100
            print(f"    Negative values: {num_negative:,} ({pct_negative:.1f}%)")

        # Print stats on full dataset (non-missing + imputed)
        all_values = result[..., 0]
        print("  Full dataset stats (after imputation):")
        print(f"    Mean: {all_values.mean().item():.4f}")
        print(f"    Std:  {all_values.std().item():.4f}")
        print(f"    Min:  {all_values.min().item():.4f}")
        print(f"    Max:  {all_values.max().item():.4f}")

    return result


def _impute_batch(
    model: STGFormer,
    imputation_head: torch.nn.Linear,
    data: torch.Tensor,
    num_iterations: int,
    mask_value: float,
    use_normalized_data: bool = False,
) -> torch.Tensor:
    """Impute a single batch of data.

    Args:
        model: Pretrained STGFormer model
        imputation_head: Trained imputation head
        data: Input tensor [batch, time, nodes, features]
        num_iterations: Number of imputation iterations
        mask_value: Value used to indicate missing data
        use_normalized_data: If True, data is normalized and predictions won't be clamped

    Returns:
        Data tensor with missing values imputed
    """
    with torch.no_grad():
        for _ in range(num_iterations):
            # Find missing positions
            missing_mask = data[..., 0:1] == mask_value

            if not missing_mask.any():
                break

            # Get spatial representation
            spatial_repr = model.get_spatial_representation(data)

            # Predict values with temporal imputation head
            predictions = imputation_head(spatial_repr)  # [batch, nodes, in_steps]

            # Reshape predictions to match data: [batch, nodes, time] -> [batch, time, nodes]
            predictions = predictions.transpose(1, 2).unsqueeze(
                -1
            )  # [batch, time, nodes, 1]

            # Only clamp for unnormalized data (traffic speeds can't be negative)
            # For normalized data (mean=0, std=1), negative values are valid
            if not use_normalized_data:
                predictions = predictions.clamp(min=0.0)

            # Fill missing values with per-timestep predictions
            data[..., 0:1] = torch.where(missing_mask, predictions, data[..., 0:1])

    return data
