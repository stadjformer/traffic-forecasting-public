"""Configuration dataclasses for training STGFormer models.

This module provides strongly-typed configuration classes to replace the 40+
parameter signature in train_model(). Using dataclasses provides:
- Type hints and validation
- Default values
- Self-documenting code
- IDE autocomplete
- Easy conversion to/from dicts (for config files)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

from stgformer.model import GraphMode, PropagationMode, TemporalMode


@dataclass
class TrainingConfig:
    """Training hyperparameters and optimization settings."""

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0003
    early_stop: int = 10
    milestones: Optional[list] = field(default_factory=lambda: [20, 30])
    lr_decay_rate: float = 0.1
    clip_grad: float = 0.0
    seed: Optional[int] = 42
    verbose: bool = False
    device: Optional[str] = None
    use_torch_compile: bool = (
        True  # Enable torch.compile (disable for problematic models)
    )

    def __post_init__(self):
        """Validate training parameters."""
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.early_stop < 0:
            raise ValueError("early_stop must be non-negative")


@dataclass
class GraphConfig:
    """Graph structure configuration."""

    graph_mode: Union[str, GraphMode] = "learned"
    geo_adj: Optional[np.ndarray] = None
    lambda_hybrid: float = 0.5
    sparsity_k: Optional[int] = None

    def __post_init__(self):
        """Validate graph configuration."""
        # Convert string to enum if needed
        if isinstance(self.graph_mode, str):
            self.graph_mode = GraphMode(self.graph_mode)

        # Validate geo_adj requirements
        if self.graph_mode in (
            GraphMode.GEOGRAPHIC,
            GraphMode.HYBRID,
            GraphMode.SPECTRAL_INIT,
        ):
            if self.geo_adj is None:
                raise ValueError(f"geo_adj required for graph_mode={self.graph_mode}")

        # Validate lambda_hybrid range
        if not 0 <= self.lambda_hybrid <= 1:
            raise ValueError("lambda_hybrid must be in [0, 1]")

        # Validate sparsity_k
        if self.sparsity_k is not None and self.sparsity_k <= 0:
            raise ValueError("sparsity_k must be positive if specified")


@dataclass
class TemporalConfig:
    """Temporal processing configuration."""

    temporal_mode: Union[str, TemporalMode] = "transformer"
    # Mamba config
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    # TCN config
    tcn_num_layers: int = 3
    tcn_kernel_size: int = 3
    tcn_dilation_base: int = 2
    tcn_dropout: float = 0.1
    # Depthwise config
    depthwise_kernel_size: int = 3
    # MLP config
    mlp_hidden_dim: Optional[int] = None

    def __post_init__(self):
        """Validate temporal configuration."""
        # Convert string to enum if needed
        if isinstance(self.temporal_mode, str):
            self.temporal_mode = TemporalMode(self.temporal_mode)

        # Validate Mamba parameters
        if self.mamba_d_state <= 0:
            raise ValueError("mamba_d_state must be positive")
        if self.mamba_d_conv <= 0:
            raise ValueError("mamba_d_conv must be positive")
        if self.mamba_expand <= 0:
            raise ValueError("mamba_expand must be positive")

        # Validate TCN parameters
        if self.tcn_num_layers <= 0:
            raise ValueError("tcn_num_layers must be positive")
        if self.tcn_kernel_size <= 0:
            raise ValueError("tcn_kernel_size must be positive")
        if self.tcn_dilation_base <= 1:
            raise ValueError("tcn_dilation_base must be > 1")
        if not 0 <= self.tcn_dropout <= 1:
            raise ValueError("tcn_dropout must be in [0, 1]")

        # Validate depthwise parameters
        if self.depthwise_kernel_size <= 0:
            raise ValueError("depthwise_kernel_size must be positive")

        # Validate MLP parameters
        if self.mlp_hidden_dim is not None and self.mlp_hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be positive if specified")


@dataclass
class ModelArchConfig:
    """Model architecture overrides (optional).

    All fields are optional - only specify values you want to override from defaults.
    """

    input_embedding_dim: Optional[int] = None
    tod_embedding_dim: Optional[int] = None
    dow_embedding_dim: Optional[int] = None
    spatial_embedding_dim: Optional[int] = None
    adaptive_embedding_dim: Optional[int] = None
    num_heads: Optional[int] = None
    num_layers: Optional[int] = None
    dropout: Optional[float] = None
    dropout_a: Optional[float] = None
    mlp_ratio: Optional[int] = None
    use_mixed_proj: Optional[bool] = None
    pre_attn_kernel_size: Optional[int] = None
    prediction_head_layers: Optional[int] = None  # 1 = linear (default), 2+ = MLP

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values.

        This is useful for passing as **kwargs to model constructors.
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class PretrainConfig:
    """Pretraining configuration and optional helpers."""

    stage1_epochs: int = 5
    stage1_mask_ratio: float = 0.15
    stage2_epochs: int = 5
    stage2_mask_ratio: float = 0.10
    pretrain_batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    load_from: Optional[str] = None
    save_to: Optional[str] = None
    imputation_iterations: Optional[int] = None
    use_normalized_data: bool = False
    pretrain_data_fraction: float = 1.0  # Fraction of training data to use (1.0 = all)

    def __post_init__(self):
        """Validate pretraining parameters."""
        if self.stage1_epochs < 0:
            raise ValueError("stage1_epochs must be non-negative")
        if self.stage2_epochs < 0:
            raise ValueError("stage2_epochs must be non-negative")
        if not 0 <= self.stage1_mask_ratio <= 1:
            raise ValueError("stage1_mask_ratio must be in [0, 1]")
        if not 0 <= self.stage2_mask_ratio <= 1:
            raise ValueError("stage2_mask_ratio must be in [0, 1]")
        if self.pretrain_batch_size is not None and self.pretrain_batch_size <= 0:
            raise ValueError("pretrain_batch_size must be positive if specified")
        if self.learning_rate is not None and self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive if specified")
        if self.imputation_iterations is not None and self.imputation_iterations <= 0:
            raise ValueError("imputation_iterations must be positive if specified")
        if not 0 < self.pretrain_data_fraction <= 1:
            raise ValueError("pretrain_data_fraction must be in (0, 1]")

    def to_dict(self) -> dict:
        """Convert to dict for compatibility with existing code."""
        config = {
            "stage1_epochs": self.stage1_epochs,
            "stage1_mask_ratio": self.stage1_mask_ratio,
            "stage2_epochs": self.stage2_epochs,
            "stage2_mask_ratio": self.stage2_mask_ratio,
            "use_normalized_data": self.use_normalized_data,
            "pretrain_data_fraction": self.pretrain_data_fraction,
        }
        if self.pretrain_batch_size is not None:
            config["pretrain_batch_size"] = self.pretrain_batch_size
        if self.learning_rate is not None:
            config["learning_rate"] = self.learning_rate
        if self.load_from is not None:
            config["load_from"] = self.load_from
        if self.save_to is not None:
            config["save_to"] = self.save_to
        if self.imputation_iterations is not None:
            config["imputation_iterations"] = self.imputation_iterations
        return config


@dataclass
class STGFormerTrainingArgs:
    """Complete training configuration for STGFormer.

    This is a convenience wrapper that bundles all config objects together.
    You can either pass individual config objects to train_model() or use this
    wrapper for cleaner code.
    """

    dataset_name: str
    training: TrainingConfig = field(default_factory=TrainingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    propagation_mode: Union[str, PropagationMode] = "power"
    model_arch: Optional[ModelArchConfig] = None
    pretrain: Optional[PretrainConfig] = None
    save_dir: Optional[Path] = None
    wandb_run: Optional[object] = None
    use_imputation: bool = False
    use_zero_init: bool = True
    exclude_missing_from_norm: bool = False

    def __post_init__(self):
        """Convert string propagation_mode to enum."""
        if isinstance(self.propagation_mode, str):
            self.propagation_mode = PropagationMode(self.propagation_mode)
