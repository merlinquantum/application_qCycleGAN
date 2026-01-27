"""Utilities for loading experiment configs from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from .defaults import ClassicalConfig, DatasetConfig, QuantumConfig, TrainingConfig

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def _load_raw_config(config_path: Path) -> dict:
    """Load raw dictionary data from a YAML or JSON file."""
    text = config_path.read_text()
    if config_path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError(
                "PyYAML is required to parse .yaml config files. Install with `pip install pyyaml`."
            )
        return yaml.safe_load(text)
    return json.loads(text)


def load_configs(config_path: str | Path) -> Tuple[QuantumConfig, DatasetConfig, TrainingConfig, ClassicalConfig]:
    """
    Load experiment configuration triples (quantum, dataset, training).

    Args:
        config_path: Path to a YAML/JSON config file containing keys
            ``quantum``, ``dataset`` and ``training``.

    Returns:
        Tuple of (QuantumConfig, DatasetConfig, TrainingConfig, ClassicalConfig).
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = _load_raw_config(path)
    quantum_section = dict(raw.get("quantum", {}))
    if "dynamic" in quantum_section and "quantum" not in quantum_section:
        quantum_section["quantum"] = quantum_section.pop("dynamic")
    classical_section = dict(raw.get("classical", {}))
    quantum = QuantumConfig(**quantum_section)
    dataset = DatasetConfig(**raw.get("dataset", {}))
    training = TrainingConfig(**raw.get("training", {}))
    classical = ClassicalConfig(**classical_section)

    # Validate eagerly to fail fast when configs are invalid.
    quantum.validate()
    dataset.validate()
    training.validate()
    classical.validate()
    return quantum, dataset, training, classical
