"""Training utilities for STGFormer.

This module contains common training components:
- StandardScaler: PyTorch-based data normalization
- MaskedHuberLoss: Huber loss with null value masking
- Metrics functions: MAE, RMSE, MAPE computation with masking
- train_epoch: Single epoch training loop
- evaluate: Model evaluation loop
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# =============================================================================
# NaN Handling Utilities (shared logic for train/val/test metrics)
# =============================================================================

def prepare_predictions_and_mask(
    preds: torch.Tensor,
    labels: torch.Tensor,
    null_val: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare predictions and compute valid mask for metrics/loss computation.

    This implements the standard NaN handling policy:
    - y_pred NaN → Replace with 0 (model's fault, should be penalized)
    - y_true NaN → Mask out (missing data, not model's fault)
    - y_true == null_val → Mask out (explicit null marker)

    Args:
        preds: Model predictions
        labels: Ground truth labels
        null_val: Value to treat as null/missing in labels (default 0.0)

    Returns:
        Tuple of (cleaned_preds, valid_mask) where:
        - cleaned_preds has NaN replaced with 0
        - valid_mask is True where labels are valid (not NaN, not null_val)
    """
    # Replace NaN in predictions with 0 (model's fault if it produces NaN)
    preds_clean = torch.nan_to_num(preds, nan=0.0)

    # Mask NaN in labels (missing data, not model's fault)
    mask = ~torch.isnan(labels)

    # Also mask null_val in labels (if null_val is not NaN itself)
    if not math.isnan(null_val):
        mask = mask & (labels != null_val)

    return preds_clean, mask


# =============================================================================
# Metrics Computation Functions
# =============================================================================

def compute_masked_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    null_val: float = 0.0,
) -> Dict[str, float]:
    """Compute MAE, RMSE, and MAPE with proper masking.

    Uses prepare_predictions_and_mask() for consistent NaN handling:
    - y_pred NaN → replaced with 0 (penalized)
    - y_true NaN → masked out (not counted)

    Args:
        preds: Predicted values, shape (batch, horizon, nodes, 1) or similar
        labels: Ground truth values, same shape as preds
        null_val: Value to treat as missing/null (will be masked out).

    Returns:
        Dict with keys 'mae', 'rmse', 'mape'
    """
    # Use shared utility for consistent NaN handling
    preds_clean, mask = prepare_predictions_and_mask(preds, labels, null_val)

    if not mask.any():
        return {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0}

    # Apply mask
    preds_masked = preds_clean[mask]
    labels_masked = labels[mask]

    # MAE
    mae = torch.abs(preds_masked - labels_masked).mean().item()

    # RMSE
    rmse = torch.sqrt(((preds_masked - labels_masked) ** 2).mean()).item()

    # MAPE (avoid division by zero with small threshold)
    mape_mask = torch.abs(labels_masked) > 1e-4
    if mape_mask.any():
        mape = (torch.abs(preds_masked[mape_mask] - labels_masked[mape_mask]) /
                torch.abs(labels_masked[mape_mask])).mean().item() * 100
    else:
        mape = 0.0

    return {'mae': mae, 'rmse': rmse, 'mape': mape}


def compute_horizon_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    null_val: float = 0.0,
    horizons: List[int] = [3, 6, 12],
) -> Dict[str, float]:
    """Compute metrics for specific prediction horizons plus overall average.

    Args:
        preds: Predicted values, shape (batch, horizon, nodes, features)
        labels: Ground truth values, same shape as preds
        null_val: Value to treat as missing/null
        horizons: List of horizon indices to compute metrics for (1-indexed)
                  e.g., [3, 6, 12] for 15min, 30min, 60min with 5-min intervals

    Returns:
        Dict with keys like 'mae', 'rmse', 'mape' (overall) and
        'mae_h3', 'rmse_h3', 'mape_h3', etc. for specific horizons
    """
    metrics = {}

    # Overall metrics (all horizons)
    overall = compute_masked_metrics(preds, labels, null_val)
    metrics['mae'] = overall['mae']
    metrics['rmse'] = overall['rmse']
    metrics['mape'] = overall['mape']

    # Per-horizon metrics
    for h in horizons:
        # Horizons are 1-indexed in the paper convention, but 0-indexed in tensor
        h_idx = h - 1
        if h_idx < preds.shape[1]:
            h_metrics = compute_masked_metrics(
                preds[:, h_idx:h_idx+1, ...],
                labels[:, h_idx:h_idx+1, ...],
                null_val
            )
            metrics[f'mae_h{h}'] = h_metrics['mae']
            metrics[f'rmse_h{h}'] = h_metrics['rmse']
            metrics[f'mape_h{h}'] = h_metrics['mape']

    return metrics


def aggregate_batch_metrics(
    batch_metrics_list: List[Dict[str, float]],
    batch_sizes: List[int],
) -> Dict[str, float]:
    """Aggregate metrics across multiple batches using weighted average.

    Args:
        batch_metrics_list: List of metric dicts from each batch
        batch_sizes: List of batch sizes for weighting

    Returns:
        Aggregated metrics dict
    """
    if not batch_metrics_list:
        return {}

    total_samples = sum(batch_sizes)
    if total_samples == 0:
        return {k: 0.0 for k in batch_metrics_list[0].keys()}

    aggregated = {}
    for key in batch_metrics_list[0].keys():
        weighted_sum = sum(
            m[key] * bs for m, bs in zip(batch_metrics_list, batch_sizes)
        )
        aggregated[key] = weighted_sum / total_samples

    return aggregated


class StandardScaler:
    """Standard scaler for data normalization.

    A PyTorch-native implementation that works efficiently on GPU tensors.
    """

    def __init__(
        self, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None
    ):
        self.mean = mean
        self.std = std

    def fit_transform(
        self, data: torch.Tensor, mask_value: float | None = None
    ) -> torch.Tensor:
        """Fit scaler to data and transform it.

        Args:
            data: Input tensor to fit and transform
            mask_value: If provided, exclude values equal to this when computing mean/std.
                        NaN values are always excluded automatically.

        Returns:
            Normalized tensor
        """
        tensor = torch.as_tensor(data)

        has_nan = torch.isnan(tensor).any()

        if mask_value is not None or has_nan:
            # Use global stats when masking is needed (can't easily do per-column with masks)
            # Build valid mask: exclude NaN values (always) and mask_value (if provided)
            valid_mask = ~torch.isnan(tensor)
            if mask_value is not None:
                valid_mask = valid_mask & (tensor != mask_value)

            if valid_mask.any():
                valid_data = tensor[valid_mask]
                self.mean = valid_data.mean()
                self.std = valid_data.std(unbiased=False).clamp_min(1e-6)
            else:
                # Fallback if all values are masked/NaN
                self.mean = torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device)
                self.std = torch.tensor(1.0, dtype=tensor.dtype, device=tensor.device)
        else:
            # Original behavior: per-column stats when no masking needed
            self.mean = tensor.mean(dim=0)
            variance = tensor.var(dim=0, unbiased=False)
            self.std = torch.sqrt(variance).clamp_min(1e-6)

        return self.transform(tensor)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Transform data using fitted parameters.

        Args:
            data: Input tensor to transform

        Returns:
            Normalized tensor

        Raises:
            RuntimeError: If scaler has not been fitted
        """
        if self.mean is None or self.std is None:
            raise RuntimeError(
                "StandardScaler must be fitted before calling transform."
            )
        tensor = torch.as_tensor(data)
        mean = self.mean.to(tensor.device, tensor.dtype)
        std = self.std.to(tensor.device, tensor.dtype)
        return (tensor - mean) / std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse transform data to original scale.

        Args:
            data: Normalized tensor to inverse transform

        Returns:
            Tensor in original scale

        Raises:
            RuntimeError: If scaler has not been fitted
        """
        if self.mean is None or self.std is None:
            raise RuntimeError(
                "StandardScaler must be fitted before calling inverse_transform."
            )
        tensor = torch.as_tensor(data)
        mean = self.mean.to(tensor.device, tensor.dtype)
        std = self.std.to(tensor.device, tensor.dtype)
        return tensor * std + mean


def masked_mae_loss(
    preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0
) -> torch.Tensor:
    """Compute masked MAE loss (matches external STGFormer).

    Uses prepare_predictions_and_mask() for consistent NaN handling:
    - y_pred NaN → replaced with 0 (penalized)
    - y_true NaN → masked out (not counted)

    Args:
        preds: Predicted values
        labels: Ground truth values
        null_val: Value to treat as missing/null (will be masked out).

    Returns:
        Scalar loss value
    """
    # Use shared utility for consistent NaN handling
    preds_clean, mask = prepare_predictions_and_mask(preds, labels, null_val)

    # Normalize mask for weighted average
    mask = mask.float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds_clean - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MaskedHuberLoss(nn.Module):
    """Huber loss with masking for traffic forecasting.

    Masks out null values (e.g., missing sensor readings) before computing loss.
    Uses smooth L1 loss (Huber loss) which is less sensitive to outliers.
    """

    def __init__(self, delta: float = 1.0):
        """Initialize MaskedHuberLoss.

        Args:
            delta: Threshold at which to change between L1 and L2 loss
        """
        super().__init__()
        self.delta = delta

    def forward(
        self, preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0
    ) -> torch.Tensor:
        """Compute masked Huber loss.

        Uses prepare_predictions_and_mask() for consistent NaN handling:
        - y_pred NaN → replaced with 0 (penalized)
        - y_true NaN → masked out (not counted)

        Args:
            preds: Predicted values
            labels: Ground truth values
            null_val: Value to treat as missing/null (will be masked out).

        Returns:
            Scalar loss value
        """
        # Use shared utility for consistent NaN handling
        preds_clean, mask = prepare_predictions_and_mask(preds, labels, null_val)
        mask = mask.float()

        # Compute Huber loss element-wise
        diff = preds_clean - labels
        abs_diff = torch.abs(diff)

        # Huber loss: L2 for |diff| < delta, L1 otherwise
        quadratic = torch.clamp(abs_diff, max=self.delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear

        # Apply mask and compute mean over valid elements
        masked_loss = loss * mask
        # Replace NaN values with 0 (handles NaN * 0 = NaN issue)
        masked_loss = torch.where(
            torch.isnan(masked_loss), torch.zeros_like(masked_loss), masked_loss
        )
        num_valid = mask.sum()

        if num_valid > 0:
            return masked_loss.sum() / num_valid
        else:
            return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[StandardScaler] = None,
    null_val: float = 0.0,
) -> float:
    """Train STGFormer for one epoch.

    Args:
        model: STGFormer model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function (e.g., MaskedHuberLoss)
        device: Device to use
        scaler: Optional scaler for inverse transforming predictions
        null_val: Null value for masked loss

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward pass
        pred = model(x)

        # Compute loss (inverse transform predictions if scaler provided)
        if scaler is not None:
            pred_inv = scaler.inverse_transform(pred)
            loss = criterion(pred_inv, y, null_val)
        else:
            loss = criterion(pred, y, null_val)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[StandardScaler] = None,
    null_val: float = 0.0,
) -> float:
    """Evaluate STGFormer model.

    Args:
        model: STGFormer model
        dataloader: Evaluation data loader
        criterion: Loss function (e.g., MaskedHuberLoss)
        device: Device to use
        scaler: Optional scaler for inverse transforming predictions
        null_val: Null value for masked loss

    Returns:
        Average evaluation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            pred = model(x)

            # Compute loss (inverse transform predictions if scaler provided)
            if scaler is not None:
                pred_inv = scaler.inverse_transform(pred)
                loss = criterion(pred_inv, y, null_val)
            else:
                loss = criterion(pred, y, null_val)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)
