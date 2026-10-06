"""Portable configuration and exact inference-weight loading."""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


class Config(dict):
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


def config(value):
    return (
        Config({k: config(v) for k, v in value.items()})
        if isinstance(value, dict)
        else value
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(root=None, device="cpu", mpnn_source=None, mpnn_weights=None):
    from dcflow.model.network import DCFlow

    root = Path(root or Path(__file__).resolve().parents[1])
    cfg = config(json.loads((root / "config/model.json").read_text()))
    source = mpnn_source or os.environ.get("PROTEINMPNN_SOURCE")
    weights = mpnn_weights or os.environ.get("PROTEINMPNN_WEIGHTS")
    if not source:
        raise ValueError(
            "Set PROTEINMPNN_SOURCE to the downloaded ProteinMPNN directory; see EXTERNAL_MODELS.md"
        )
    weights = weights or str(Path(source) / "vanilla_model_weights/v_48_020.pt")
    cfg.mpnn_prior.source_dir, cfg.mpnn_prior.checkpoint_path = source, weights
    model = DCFlow(cfg)
    path = root / "checkpoints/dcflow.pt"
    manifest = json.loads((root / "checkpoints/manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("DCFlow checkpoint SHA256 mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    # External, frozen weights are loaded by the adapter, core weights strictly here.
    external = {
        k: v for k, v in model.state_dict().items() if k.startswith("mpnn_prior.")
    }
    model.load_state_dict({**state, **external}, strict=True)
    return model.to(device).eval().requires_grad_(False)
