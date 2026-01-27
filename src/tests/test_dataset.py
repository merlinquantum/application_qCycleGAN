"""Tests for the refactored unpaired dataset pipeline."""

from types import SimpleNamespace
from pathlib import Path

import torch
import pytest

from config import DatasetConfig
from data import UnpairedDataset, build_transform


class DummyTokenizer:
    """Minimal tokenizer stub mirroring the HF API used in datasets."""

    model_max_length = 16

    def __call__(self, text, max_length, padding, truncation, return_tensors):
        del text, padding, truncation, return_tensors  # Unused in stub
        return SimpleNamespace(input_ids=torch.zeros(1, max_length, dtype=torch.long))


@pytest.fixture
def dataset_root():
    return Path(__file__).parent.parent / "data" / "small_bdd"


@pytest.fixture
def dataset_config(dataset_root):
    return DatasetConfig(
        dataset_name="dummy",
        dataset_folder=str(dataset_root),
        train_img_prep="resize_64",
        val_img_prep="resize_64",
        train_batch_size=1,
        dataloader_num_workers=0,
        quantum_dims=(4, 16, 16),
        prompt_src="Driving in the night",
        prompt_tgt="Driving in the day",
        training_images=1.0,
    )


def test_unpaired_dataset_length_and_shapes(dataset_config):
    """Dataset __len__ and __getitem__ should honor transforms and prompts."""
    tokenizer = DummyTokenizer()
    dataset = UnpairedDataset(config=dataset_config, split="train", tokenizer=tokenizer)

    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["pixel_values_src"].shape[0] == 3
    assert sample["pixel_values_tgt"].shape[0] == 3
    assert sample["caption_src"] == dataset.fixed_caption_src
    assert sample["caption_tgt"] == dataset.fixed_caption_tgt


def test_unpaired_dataset_fraction(dataset_config):
    """Passing a fraction should downscale the dataset size."""
    tokenizer = DummyTokenizer()
    dataset = UnpairedDataset(
        config=dataset_config,
        split="train",
        tokenizer=tokenizer,
        fraction=0.25,
    )
    assert len(dataset) <= int((len(dataset.image_paths_src) + len(dataset.image_paths_tgt)) * 0.25) + 1


def test_build_transform_unknown_preset():
    """Unknown presets should raise a ValueError."""
    with pytest.raises(ValueError):
        build_transform("does_not_exist")
