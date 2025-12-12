"""Global configuration and constants for the traffic forecasting project.

This module provides a single source of truth for:
- Project directory paths
- Dataset configurations
- HuggingFace Hub settings
- Metric calculation parameters
"""

from pathlib import Path
from typing import Dict

import dotenv

# Project directory structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# HuggingFace Hub configuration
# These are loaded from .env_public for flexibility
_ENV_FILE = PROJECT_ROOT / ".env_public"
HF_USERNAME_UPLOAD = dotenv.get_key(
    dotenv_path=_ENV_FILE, key_to_get="HF_USERNAME_FOR_UPLOAD"
)
HF_USERNAME_DOWNLOAD = dotenv.get_key(
    dotenv_path=_ENV_FILE, key_to_get="HF_USERNAME_FOR_MODEL_DOWNLOAD"
)

# Dataset configuration
# Maps dataset names to HuggingFace repository IDs
SUPPORTED_DATASETS: Dict[str, str] = {
    "METR-LA": "witgaw/METR-LA",
    "PEMS-BAY": "witgaw/PEMS-BAY",
    "LARGEST-GLA": "emelle/LargeST-GLA",
}

# DCRNN-specific configuration files
DCRNN_CONFIGS: Dict[str, str] = {
    "METR-LA": "dcrnn_la.yaml",
    "PEMS-BAY": "dcrnn_bay.yaml",
}

# Metric evaluation horizons (in time steps)
# Each time step is 5 minutes, so:
# - 3 steps = 15 minutes
# - 6 steps = 30 minutes
# - 12 steps = 1 hour
METRIC_HORIZONS: Dict[str, int] = {
    "15 min": 3,
    "30 min": 6,
    "1 hour": 12,
}


def validate_dataset_name(dataset_name: str) -> str:
    """Validate and normalize dataset name.

    Args:
        dataset_name: Dataset name (case insensitive)

    Returns:
        Normalized dataset name (uppercase)

    Raises:
        ValueError: If dataset is not supported
    """
    dataset_name = dataset_name.upper()

    if dataset_name not in SUPPORTED_DATASETS.keys():
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported. "
            f"Choose one of: {list(SUPPORTED_DATASETS.keys())}"
        )

    return dataset_name


def load_experiment_config(
    config_path: str | Path, dataset_override: str | None = None
) -> Dict:
    """Load experiment configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file
        dataset_override: If provided, run only this dataset (overrides config's datasets list)

    Returns:
        Configuration dictionary with sections: datasets, model, graph, training, output

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required_sections = ["model", "training"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config missing required section: {section}")

    # Handle datasets (list or single)
    if dataset_override:
        config["datasets"] = [validate_dataset_name(dataset_override)]
    elif "datasets" in config:
        config["datasets"] = [validate_dataset_name(d) for d in config["datasets"]]
    elif "dataset" in config:
        # Backward compat: single dataset
        config["datasets"] = [validate_dataset_name(config["dataset"])]
    else:
        raise ValueError("Config must specify 'datasets' list or 'dataset'")

    # Set defaults for optional sections
    if "graph" not in config:
        config["graph"] = {}
    config["graph"].setdefault("mode", "learned")
    config["graph"].setdefault("lambda_hybrid", 0.5)
    config["graph"].setdefault("sparsity_k", None)

    if "temporal" not in config:
        config["temporal"] = {}
    config["temporal"].setdefault("mode", "transformer")

    if "output" not in config:
        config["output"] = {}
    config["output"].setdefault("save_dir", None)

    # Set training defaults
    config["training"].setdefault("use_torch_compile", True)

    # Optional tagging for experiment subsets
    config.setdefault("experiment_subset", "")

    return config
