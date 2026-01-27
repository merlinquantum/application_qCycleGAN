"""Quantum-aware dataset that augments the base unpaired dataset."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fu
import torchvision.transforms.functional as F
from PIL import Image

from config import DatasetConfig
from .dataset import UnpairedDataset
from .transforms import build_transform


def read_embeddings_from_h5(path: str) -> Tuple[List[str], List[str]]:
    h5_a = os.path.join(path, "train_A.h5")
    h5_b = os.path.join(path, "train_B.h5")
    with h5py.File(h5_a, "r") as handle_a:
        q_embs_a = list(handle_a.keys())
        print(f"[Dataset/QEmb] Number of quantum embeddings for A = {len(q_embs_a)}")
    with h5py.File(h5_b, "r") as handle_b:
        q_embs_b = list(handle_b.keys())
        print(f"[Dataset/QEmb] Number of quantum embeddings for B = {len(q_embs_b)}")
    return q_embs_a, q_embs_b


def write_embeddings_to_csv(embs_a: List[str], embs_b: List[str], output_dir: str) -> None:
    embs_a_copy, embs_b_copy = embs_a.copy(), embs_b.copy()
    max_len = max(len(embs_a_copy), len(embs_b_copy))
    if len(embs_a_copy) < max_len:
        embs_a_copy += [np.nan] * (max_len - len(embs_a_copy))
    if len(embs_b_copy) < max_len:
        embs_b_copy += [np.nan] * (max_len - len(embs_b_copy))
    df = pd.DataFrame({"q_embs_A": embs_a_copy, "q_embs_B": embs_b_copy})
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(output_dir) / "q_embs_used.csv", index=False)
    print("[Dataset/QEmb] Embeddings saved to CSV")


def read_from_emb16(tensor_path: str) -> torch.Tensor:
    with open(tensor_path, "rb") as handle:
        q_emb = handle.read()
    q_emb = np.frombuffer(q_emb, dtype=np.float32).reshape(3, 16, 16)
    return torch.tensor(q_emb)


def read_from_emb32(tensor_path: str) -> torch.Tensor:
    tensor = torch.load(tensor_path)
    return tensor


def image_fail(list_f: List[List[str]], fold: str, path: str) -> bool:
    img_name = os.path.basename(path)
    return [fold, img_name] in list_f


class UnpairedDataset_Quantum(UnpairedDataset):
    """
    Dataset that optionally serves pre-computed quantum embeddings alongside images.
    """

    def __init__(
        self,
        config: DatasetConfig,
        split: str,
        tokenizer,
        image_prep: str | None = None,
        q_emb_path: str | None = None,
        output_dir: str | None = None,
        annotations_path: str | None = None,
        annotations_on_image: bool = False,
        q_fail: bool = True,
        partial: bool = False,
    ) -> None:
        super().__init__(config=config, split=split, tokenizer=tokenizer)
        if image_prep:
            self.transform = build_transform(image_prep)

        self.q_emb_path = q_emb_path
        self.annotations_on_images = annotations_on_image
        self.annotations_path = annotations_path
        self.q_fail = q_fail
        self.partial = partial

        if not self.annotations_on_images:
            if q_emb_path is None or output_dir is None:
                raise ValueError("q_emb_path and output_dir are required when annotations_on_image=False.")
            self.q_embs_A, self.q_embs_B = read_embeddings_from_h5(self.q_emb_path)
            write_embeddings_to_csv(self.q_embs_A, self.q_embs_B, output_dir)
        else:
            if annotations_path is None:
                raise ValueError("annotations_path must be provided when annotations_on_image=True.")

        if self.q_fail:
            df = pd.read_csv("/mnt/bmw-challenge-volume/home/jupyter-pemeriau/q_embs/fails_empty.csv")
            self.list_empty = df.values.tolist()
        else:
            self.list_empty = []

    def __len__(self) -> int:
        if self.annotations_on_images:
            total = len(self.image_paths_src) + len(self.image_paths_tgt)
        else:
            total = len(self.q_embs_A) + len(self.q_embs_B)
        if self.partial:
            total = int(0.25 * total)
        return total

    def _select_quantum_names(self, index: int) -> Tuple[str, str]:
        if index < len(self.q_embs_A):
            src_name = self.q_embs_A[index]
        else:
            src_name = random.choice(self.q_embs_A)
        tgt_name = random.choice(self.q_embs_B)
        return src_name, tgt_name

    def _load_quantum_tensors_from_h5(self, src_name: str, tgt_name: str):
        qt_t_src = qt_t_tgt = None
        with h5py.File(os.path.join(self.q_emb_path, "train_A.h5"), "r") as f_src:
            if src_name in f_src:
                qt_t_src = torch.tensor(f_src[src_name])
            else:
                print(f"[Dataset/QEmb] Warning: {src_name} not found in q_embs_A")
        with h5py.File(os.path.join(self.q_emb_path, "train_B.h5"), "r") as f_tgt:
            if tgt_name in f_tgt:
                qt_t_tgt = torch.tensor(f_tgt[tgt_name])
            else:
                print(f"[Dataset/QEmb] Warning: {tgt_name} not found in q_embs_B")
        return qt_t_src, qt_t_tgt

    def _select_image_with_failures(self, candidates: List[str], fold: str) -> str:
        img_path = random.choice(candidates)
        while image_fail(self.list_empty, fold, os.path.basename(img_path)):
            img_path = random.choice(candidates)
        return img_path

    def __getitem__(self, index: int):
        if not self.annotations_on_images:
            src_name, tgt_name = self._select_quantum_names(index)
            img_path_src = os.path.join(self.source_folder, src_name)
            img_path_tgt = os.path.join(self.target_folder, tgt_name)
            qt_t_src, qt_t_tgt = self._load_quantum_tensors_from_h5(src_name, tgt_name)
        else:
            if index < len(self.image_paths_src):
                img_path_src = self.image_paths_src[index]
            else:
                img_path_src = random.choice(self.image_paths_src)
            img_path_tgt = random.choice(self.image_paths_tgt)

            if self.q_fail:
                img_path_src = self._select_image_with_failures(self.image_paths_src, "trainA")
                img_path_tgt = self._select_image_with_failures(self.image_paths_tgt, "trainB")

            q_train_a = os.path.join(self.annotations_path, "A", "trainA")
            q_train_b = os.path.join(self.annotations_path, "B", "trainB")

            if not self.q_fail:
                qt_t_src_path = os.path.join(q_train_a, f"{os.path.basename(img_path_src[:-4])}.emb16")
                qt_t_tgt_path = os.path.join(q_train_b, f"{os.path.basename(img_path_tgt[:-4])}.emb16")
                qt_t_src = read_from_emb16(qt_t_src_path)
                qt_t_tgt = read_from_emb16(qt_t_tgt_path)
            else:
                qt_t_src_path = os.path.join(q_train_a, f"{os.path.basename(img_path_src[:-4])}.emb32")
                qt_t_tgt_path = os.path.join(q_train_b, f"{os.path.basename(img_path_tgt[:-4])}.emb32")
                qt_t_src = read_from_emb32(qt_t_src_path)
                qt_t_tgt = read_from_emb32(qt_t_tgt_path)

            qt_t_src = Fu.interpolate(qt_t_src.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False)
            qt_t_tgt = Fu.interpolate(qt_t_tgt.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False)
            qt_t_src = qt_t_src.squeeze(0)
            qt_t_tgt = qt_t_tgt.squeeze(0)

        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")
        img_t_src = F.to_tensor(self.transform(img_pil_src))
        img_t_tgt = F.to_tensor(self.transform(img_pil_tgt))
        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])

        if self.annotations_on_images:
            img_t_src = torch.cat((img_t_src, qt_t_src), dim=0)
            img_t_tgt = torch.cat((img_t_tgt, qt_t_tgt), dim=0)

        return {
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt,
            "quantic_values_src": qt_t_src,
            "quantic_values_tgt": qt_t_tgt,
            "caption_src": self.fixed_caption_src,
            "caption_tgt": self.fixed_caption_tgt,
            "input_ids_src": self.input_ids_src,
            "input_ids_tgt": self.input_ids_tgt,
        }
