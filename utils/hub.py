"""HuggingFace Hub utilities for model management.

This module provides shared utilities for:
- Device detection (CUDA/MPS/CPU)
- Model card generation
- Model download/upload helpers
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from huggingface_hub import HfApi, snapshot_download

from utils.config import HF_USERNAME_DOWNLOAD, MODELS_DIR


def get_best_device() -> str:
    """Get the best available device for training/inference.

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def create_model_card(
    model_type: str,
    dataset_name: str,
    metrics: Optional[Dict[str, float]] = None,
    description: Optional[str] = None,
) -> str:
    """Create a standardized README.md model card for HuggingFace Hub.

    Args:
        model_type: Model type ("DCRNN", "MTGNN", "GWNET", "STGFORMER", or custom prefix)
        dataset_name: Dataset name (e.g., "METR-LA")
        metrics: Optional dict of evaluation metrics
        description: Optional custom description (used for STGFORMER variants)

    Returns:
        Formatted model card as markdown string
    """
    model_descriptions = {
        "DCRNN": {
            "title": "Diffusion Convolutional Recurrent Neural Network",
            "description": """This model uses a graph neural network architecture that combines:
- Diffusion convolution to capture spatial dependencies on road networks
- Recurrent neural networks (GRU) for temporal modeling
- Sequence-to-sequence learning for multi-step ahead forecasting""",
            "citation": """```bibtex
@inproceedings{li2018dcrnn,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2018}
}
```""",
            "usage": f"""```python
from dcrnn_pytorch.model.pytorch.dcrnn_supervisor import DCRNNSupervisor
from utils.dcrnn import load_from_hub

# Download checkpoint
checkpoint_path = load_from_hub("your-username/dcrnn-{dataset_name.lower()}")

# Load model (requires config and adj_mx from checkpoint)
# See documentation for full loading example
```""",
        },
        "MTGNN": {
            "title": "Multivariate Time Series Forecasting with Graph Neural Networks",
            "description": """This model uses a graph neural network architecture that combines:
- Graph learning to automatically discover spatial dependencies
- Temporal convolution for modeling temporal patterns
- Mix-hop propagation for capturing multi-scale spatial patterns""",
            "citation": """```bibtex
@inproceedings{wu2020connecting,
  title={Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks},
  author={Wu, Zonghan and Pan, Shirui and Long, Guodong and Jiang, Jing and Chang, Xiaojun and Zhang, Chengqi},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
  pages={753--763},
  year={2020}
}
```""",
            "usage": f"""```python
from utils.mtgnn import load_from_hub

# Load model from Hub
model = load_from_hub("{dataset_name}")

# Get predictions
import numpy as np
x = np.random.randn(10, 2, 207, 12)  # (batch, features, nodes, seq_len)
predictions = model.predict(x)
```""",
        },
        "GWNET": {
            "title": "Graph WaveNet",
            "description": """This model uses a graph neural network architecture that combines:
- Adaptive adjacency matrix learning
- Spatial graph convolution for capturing spatial dependencies
- Temporal convolution with dilated causal convolutions
- Multi-scale temporal receptive field""",
            "citation": """```bibtex
@inproceedings{wu2019graph,
  title={Graph WaveNet for Deep Spatial-Temporal Graph Modeling},
  author={Wu, Zonghan and Pan, Shirui and Long, Guodong and Jiang, Jing and Zhang, Chengqi},
  booktitle={Proceedings of the 28th International Joint Conference on Artificial Intelligence},
  pages={1907--1913},
  year={2019}
}
```""",
            "usage": f"""```python
from utils.gwnet import load_from_hub

# Load model from Hub
model = load_from_hub("{dataset_name}")

# Get predictions
import numpy as np
x = np.random.randn(10, 12, 207, 2)  # (batch, seq_len, nodes, features)
predictions = model.predict(x)
```""",
        },
        "STGFORMER": {
            "title": "Spatial-Temporal Graph Transformer",
            "description": """This model uses a graph neural network architecture that combines:
- Spatial graph attention for capturing spatial dependencies
- Temporal self-attention for modeling temporal patterns
- Transformer architecture for long-range dependencies""",
            "citation": """```bibtex
@inproceedings{lan2022stgformer,
  title={STGformer: Spatial-Temporal Graph Transformer for Traffic Forecasting},
  author={Lan, Shengnan and Ma, Yong and Huang, Weijia and Wang, Wanwei and Yang, Hui and Li, Peng},
  booktitle={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022}
}
```""",
            "usage": f"""```python
from utils.stgformer_external import load_from_hub

# Load model from Hub
model = load_from_hub("{dataset_name}")

# Get predictions (see documentation for details)
```""",
        },
        "STGFORMER_INTERNAL": {
            "title": "Spatial-Temporal Graph Transformer (Internal Implementation)",
            "description": """Internal implementation of STGFormer, a hybrid graph transformer that combines:
- Fast linear attention O(N+T) for global spatio-temporal patterns
- Graph convolution for local spatial structure
- Adaptive graph learning from node embeddings
- Unified spatio-temporal processing with decomposed attention""",
            "citation": """```bibtex
@inproceedings{lan2022stgformer,
  title={STGformer: Spatial-Temporal Graph Transformer for Traffic Forecasting},
  author={Lan, Shengnan and Ma, Yong and Huang, Weijia and Wang, Wanwei and Yang, Hui and Li, Peng},
  booktitle={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022}
}
```""",
            "usage": f"""```python
from utils.stgformer import load_from_hub

# Load model from Hub
model, scaler = load_from_hub("{dataset_name}", hf_repo_prefix="{model_type}")

# Get predictions
from utils.stgformer import get_predictions
predictions = get_predictions(model, scaler, test_dataset)
```""",
        },
    }

    model_type_upper = model_type.upper()

    # Handle STGFORMER variants (e.g., STGFORMER_MAMBA, STGFORMER_CHEBYSHEV)
    # and ablation experiments (e.g., ABL_NO_XAVIER, ABL_NO_DOW)
    # by inheriting from STGFORMER_INTERNAL with custom title/description
    if model_type_upper not in model_descriptions:
        if model_type_upper.startswith("STGFORMER") or model_type_upper.startswith(
            "ABL_"
        ):
            # Use STGFORMER_INTERNAL as base, customize title and description
            info = model_descriptions["STGFORMER_INTERNAL"].copy()
            if model_type_upper.startswith("ABL_"):
                # Ablation study variant
                variant_name = (
                    model_type_upper.replace("ABL_", "").replace("_", " ").title()
                )
                info["title"] = (
                    f"Spatial-Temporal Graph Transformer (Ablation: {variant_name})"
                )
            else:
                # Regular STGFORMER variant
                variant_name = (
                    model_type_upper.replace("STGFORMER_", "").replace("_", " ").title()
                )
                info["title"] = f"Spatial-Temporal Graph Transformer ({variant_name})"
            if description:
                info["description"] = description
            # Update usage example with correct prefix
            info["usage"] = f"""```python
from utils.stgformer import load_from_hub

# Load model from Hub
model, scaler = load_from_hub("{dataset_name}", hf_repo_prefix="{model_type_upper}")

# Get predictions
from utils.stgformer import get_predictions
predictions = get_predictions(model, scaler, test_dataset)
```"""
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. Supported: {list(model_descriptions.keys())} "
                "or STGFORMER_*/ABL_* variants"
            )
    else:
        info = model_descriptions[model_type_upper]

    # Override description if provided
    if description and model_type_upper in model_descriptions:
        info = info.copy()
        info["description"] = description

    # Build metrics section
    metrics_section = ""
    if metrics:
        metrics_section = "## Evaluation Metrics\n\n"
        for metric_name, value in metrics.items():
            metrics_section += f"- **{metric_name}**: {value:.4f}\n"

    # Generate model card
    model_card = f"""---
tags:
- traffic-forecasting
- time-series
- graph-neural-network
- {model_type.lower()}
datasets:
- {dataset_name.lower()}
---

# {info["title"]} - {dataset_name}

{info["title"]} ({model_type_upper}) trained on {dataset_name} dataset for traffic speed forecasting.

## Model Description

{info["description"]}

{metrics_section}

## Dataset

**{dataset_name}**: Traffic speed data from highway sensors.

## Usage

{info["usage"]}

## Training

Model was trained using the {model_type_upper} implementation with default hyperparameters.

## Citation

If you use this model, please cite the original {model_type_upper} paper:

{info["citation"]}

## License

This model checkpoint is released under the same license as the training code.
"""

    return model_card


def fetch_model_from_hub(
    model_type: str,
    dataset_name: str,
    force_download: bool = False,
    verbose: bool = False,
) -> Path:
    """Download model from HuggingFace Hub.

    Args:
        model_type: Model type ("DCRNN", "MTGNN", "GWNET", "STGFORMER")
        dataset_name: Dataset name (e.g., "METR-LA")
        force_download: Force re-download even if cached
        verbose: Print download progress

    Returns:
        Path to downloaded model directory
    """
    from utils.config import validate_dataset_name

    dataset_name = validate_dataset_name(dataset_name)
    model_type_upper = model_type.upper()

    # Construct repo ID and local path
    if not HF_USERNAME_DOWNLOAD:
        raise RuntimeError(
            "HF_USERNAME_FOR_MODEL_DOWNLOAD not set in .env_public. "
            "Cannot download models from HuggingFace Hub."
        )

    repo_id = f"{HF_USERNAME_DOWNLOAD}/{model_type_upper}_{dataset_name}"
    model_dir = MODELS_DIR / model_type_upper / f"model_{dataset_name.lower()}"

    # Download if needed
    if force_download or not model_dir.exists() or not any(model_dir.iterdir()):
        if verbose:
            print(f"Downloading {model_type_upper} model from {repo_id}...")

        model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=model_dir,
            force_download=force_download,
        )

        if verbose:
            print(f"Downloaded to: {model_dir}")

    return model_dir


def push_model_to_hub(
    checkpoint_dir: Path,
    repo_id: str,
    model_type: str,
    dataset_name: str,
    metrics: Optional[Dict[str, float]] = None,
    commit_message: Optional[str] = None,
    private: bool = False,
    description: Optional[str] = None,
) -> str:
    """Push a model checkpoint to HuggingFace Hub.

    Args:
        checkpoint_dir: Path to local checkpoint directory
        repo_id: HuggingFace repo ID (e.g., "username/MTGNN_METR-LA")
        model_type: Model type for metadata and card generation
        dataset_name: Dataset name for metadata and card generation
        metrics: Optional dict of evaluation metrics
        commit_message: Optional custom commit message
        private: Whether to create a private repository
        description: Optional custom description for model card

    Returns:
        URL to the uploaded model on HuggingFace Hub
    """
    import json

    checkpoint_dir = Path(checkpoint_dir)

    # Create metadata file
    metadata = {
        "dataset": dataset_name,
        "upload_date": datetime.now().isoformat(),
        "metrics": metrics or {},
        "framework": "PyTorch",
        "model_type": model_type.upper(),
    }

    metadata_path = checkpoint_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Create model card
    model_card = create_model_card(model_type, dataset_name, metrics, description)
    readme_path = checkpoint_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(model_card)

    # Upload to Hub
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id, repo_type="model", private=private, exist_ok=True
        )
    except Exception as e:
        print(f"Note: {e}")

    # Upload folder
    if commit_message is None:
        commit_message = f"Upload {model_type.upper()} model trained on {dataset_name}"

    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )

    return f"https://huggingface.co/{repo_id}"
