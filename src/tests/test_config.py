"""Tests for configuration dataclasses and loader."""

from pathlib import Path

import pytest

from config import ClassicalConfig, DatasetConfig, QuantumConfig, TrainingConfig, load_configs


def test_quantum_config_validation_failure():
    """QuantumConfig should guard against impossible layouts."""
    cfg = QuantumConfig(num_modes=2, num_photons=5)
    with pytest.raises(ValueError):
        cfg.validate()


def test_dataset_config_validation_failure():
    """DatasetConfig should ensure core fields exist."""
    cfg = DatasetConfig(dataset_folder="", train_batch_size=-1)
    with pytest.raises(ValueError):
        cfg.validate()


def test_training_config_validation_failure():
    """TrainingConfig should validate numeric hyperparameters."""
    cfg = TrainingConfig(learning_rate=-0.5)
    with pytest.raises(ValueError):
        cfg.validate()


def test_load_configs_from_file():
    """Full config loader should hydrate dataclasses from YAML."""
    config_path = Path(__file__).parent.parent / "config" / "experiments" / "default.yaml"
    quantum, dataset, training, classical = load_configs(config_path)

    assert quantum.num_modes == 20
    assert quantum.processes == 4
    assert dataset.dataset_folder.endswith("dataset_full_scale/")
    assert dataset.training_images == pytest.approx(1.0)
    assert training.learning_rate == pytest.approx(5e-6)
    assert training.tracker_project_name == "gparmar_unpaired_d2n"
    assert training.unet_trained is False
    assert isinstance(classical, ClassicalConfig)
    assert classical.cl_comp is False


def test_parse_args_unpaired_training_propagates_save_step_checkpoints():
    """CLI parser should propagate training.save_step_checkpoints from YAML config."""
    from my_utils.training_utils import parse_args_unpaired_training

    config_path = Path(__file__).parent.parent / "config" / "experiments" / "classical_run.yaml"
    args = parse_args_unpaired_training(["--experiment_config", str(config_path)])

    assert args.save_step_checkpoints is True
    assert args.training_config.save_step_checkpoints is True
