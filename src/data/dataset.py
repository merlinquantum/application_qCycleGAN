"""Refactored unpaired dataset implementation."""

from __future__ import annotations

import os
import shutil
import random
import atexit
from pathlib import Path
from typing import List, Sequence

import torch
from PIL import Image
import torchvision.transforms.functional as F
from datasets import load_dataset, load_dataset_builder
from tqdm import tqdm

from config import DatasetConfig
from .transforms import build_transform


IMG_EXTENSIONS: Sequence[str] = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif")


class UnpairedDataset(torch.utils.data.Dataset):
    """Unpaired dataset with shared preprocessing and prompt handling."""

    def __init__(
        self,
        config: DatasetConfig,
        split: str = "train",
        tokenizer=None,
        fraction: float | None = None,
    ) -> None:
        super().__init__()
        if tokenizer is None:
            raise ValueError("tokenizer must be provided for UnpairedDataset.")
        split = split.lower()
        if split not in {"train", "test"}:
            raise ValueError("split must be either 'train' or 'test'")

        self.config = config
        self.split = split
        self.tokenizer = tokenizer
        self.root = Path(config.dataset_folder)
        self.source_folder = self._resolve_folder("A")
        self.target_folder = self._resolve_folder("B")

        self.fixed_caption_src, self.input_ids_src = self._load_prompt(
            self.root / "fixed_prompt_a.txt",
        )
        self.fixed_caption_tgt, self.input_ids_tgt = self._load_prompt(
            self.root / "fixed_prompt_b.txt",
        )

        self.image_paths_src = self._discover_images(self.source_folder)
        self.image_paths_tgt = self._discover_images(self.target_folder)
        if not self.image_paths_src:
            raise FileNotFoundError(f"No source images found in {self.source_folder}")
        if not self.image_paths_tgt:
            raise FileNotFoundError(f"No target images found in {self.target_folder}")

        transform_name = (
            config.train_img_prep if split == "train" else config.val_img_prep
        )
        self.transform = build_transform(transform_name)

        default_fraction = config.training_images if split == "train" else 1.0
        self.fraction = float(fraction or default_fraction)
        if not 0 < self.fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")

        total = len(self.image_paths_src) + len(self.image_paths_tgt)
        self.num_items = max(1, int(total * self.fraction))

    def _resolve_folder(self, domain_suffix: str) -> Path:
        folder_name = f"{self.split}_{domain_suffix}"
        folder = self.root / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"Expected folder {folder} to exist.")
        return folder

    def _discover_images(self, folder: Path) -> List[str]:
        matches: List[str] = []
        for pattern in IMG_EXTENSIONS:
            matches.extend(sorted(str(path) for path in folder.glob(pattern)))
        return matches

    def _load_prompt(self, path: Path):
        text = path.read_text().strip()
        tokens = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return text, tokens

    def __len__(self) -> int:
        return self.num_items

    def _select_source_path(self, index: int) -> str:
        if index < len(self.image_paths_src):
            return self.image_paths_src[index]
        return random.choice(self.image_paths_src)

    def _select_target_path(self) -> str:
        return random.choice(self.image_paths_tgt)

    def __getitem__(self, index: int):
        img_path_src = self._select_source_path(index)
        img_path_tgt = self._select_target_path()
        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")

        img_t_src = F.to_tensor(self.transform(img_pil_src))
        img_t_tgt = F.to_tensor(self.transform(img_pil_tgt))
        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])

        return {
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt,
            "caption_src": self.fixed_caption_src,
            "caption_tgt": self.fixed_caption_tgt,
            "input_ids_src": self.input_ids_src,
            "input_ids_tgt": self.input_ids_tgt,
            "path_src": img_path_src,
            "path_tgt": img_path_tgt,
        }


def load_bdd_dataset(
    dataset_name,
    output_dir,
    prompt_src="Driving in the night",
    prompt_tgt="Driving in the day",
    cleanup: bool = False,
    train_fraction: float | None = None,
):
    """Download HuggingFace dataset and materialize it to disk."""
    print("[Dataset] Loading dataset from source")
    os.makedirs(output_dir, exist_ok=True)
    builder = load_dataset_builder(dataset_name)
    split_names = list(builder.info.splits.keys()) or ["train"]
    for split_name in split_names:
        fraction = train_fraction if split_name.lower().startswith("train") else None
        split_spec = split_name
        if fraction is not None and fraction < 1.0:
            percent = max(int(fraction * 100), 1)
            split_spec = f"{split_name}[:{percent}%]"
            print(f"[Dataset] Downloading subset {split_spec} of {dataset_name}")
        split_data = load_dataset(dataset_name, split=split_spec)
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for i, example in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
            image: Image.Image = example["image"]
            save_path = os.path.join(split_dir, f"{i:05d}.jpg")
            image.save(save_path)
    text_a = prompt_src or "Driving in the night"
    file_path = os.path.join(output_dir, "fixed_prompt_a.txt")
    with open(file_path, 'w') as file:
        file.write(text_a)
    text_b = prompt_tgt or "Driving in the day"
    file_path = os.path.join(output_dir, "fixed_prompt_b.txt")
    with open(file_path, 'w') as file:
        file.write(text_b)
    if cleanup:
        def _cleanup():
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
                print(f"[Dataset] Cleaned up dataset at {output_dir}")
        atexit.register(_cleanup)
    return "Dataset Loaded"


def load_small_dataset(
    dataset_name,
    output_dir,
    cleanup: bool = False,
    train_fraction: float | None = None,
):
    """Convenience wrapper matching the notebook helper."""
    return load_bdd_dataset(
        dataset_name,
        output_dir,
        cleanup=cleanup,
        train_fraction=train_fraction,
    )
