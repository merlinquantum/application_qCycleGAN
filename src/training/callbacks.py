"""Callback interfaces for training orchestration."""

from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import wandb
from cleanfid.fid import get_folder_features, frechet_distance
from peft.utils import get_peft_model_state_dict
from loguru import logger

from my_utils.device_utils import empty_cache
from my_utils.dino_struct import DinoStructureLoss


def _to_jsonable(value):  # type: ignore[override]
    """Best-effort conversion of common ML types to JSON-serializable objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(val) for val in value]
    return str(value)


class Callback:
    """Base callback class with optional hooks."""

    def on_step_end(self, trainer, state, logs):
        """Called at the end of each optimizer step."""

    def on_epoch_end(self, trainer, state):
        """Called at the end of each epoch."""


class LambdaCallback(Callback):
    """Convenience callback that proxies hooks to simple callables."""

    def __init__(self, on_step_end=None, on_epoch_end=None):
        self._on_step_end = on_step_end
        self._on_epoch_end = on_epoch_end

    def on_step_end(self, trainer, state, logs):
        if self._on_step_end is not None:
            self._on_step_end(trainer, state, logs)

    def on_epoch_end(self, trainer, state):
        if self._on_epoch_end is not None:
            self._on_epoch_end(trainer, state)


class PeriodicCallback(Callback):
    """Helper for callbacks that execute every N steps."""

    def __init__(self, frequency: int):
        self.frequency = frequency or 0

    def _should_run(self, trainer, state) -> bool:
        if not trainer.accelerator.sync_gradients:
            return False
        if not trainer.accelerator.is_main_process:
            return False
        if self.frequency <= 0:
            return False
        return state.global_step % self.frequency == 0


class VisualizationCallback(PeriodicCallback):
    """Log qualitative samples to the active tracker."""

    def __init__(self, frequency: int, tracker_name: str = "wandb"):
        super().__init__(frequency)
        self.tracker_name = tracker_name

    def on_step_end(self, trainer, state, logs):
        if not self._should_run(trainer, state):
            return

        batch = trainer.last_batch
        outputs = trainer.last_outputs
        if not batch or not outputs:
            return

        viz_img_a = batch["pixel_values_src"].to(dtype=trainer.weight_dtype)[:, :3]
        viz_img_b = batch["pixel_values_tgt"].to(dtype=trainer.weight_dtype)[:, :3]
        cyc_rec_a = outputs.get("cyc_rec_a")
        cyc_rec_b = outputs.get("cyc_rec_b")
        fake_a = outputs.get("fake_a")
        fake_b = outputs.get("fake_b")
        if any(t is None for t in (cyc_rec_a, cyc_rec_b, fake_a, fake_b)):
            return

        bsz = viz_img_a.shape[0]
        log_dict = {
            "train/real_a": [wandb.Image(viz_img_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
            "train/real_b": [wandb.Image(viz_img_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
            "train/rec_a": [wandb.Image(cyc_rec_a[idx][:3].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
            "train/rec_b": [wandb.Image(cyc_rec_b[idx][:3].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
            "train/fake_b": [wandb.Image(fake_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
            "train/fake_a": [wandb.Image(fake_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
        }
        for tracker in trainer.accelerator.trackers:
            if tracker.name == self.tracker_name:
                tracker.log(log_dict, step=state.global_step)
                gc.collect()
                empty_cache()


class ModelSaver:
    """Utility to persist model components in a unified format."""

    def __init__(
        self,
        output_dir: str,
        lora_modules_encoder: Sequence[str],
        lora_modules_decoder: Sequence[str],
        lora_modules_others: Sequence[str],
        lora_rank_unet: int,
        vae_lora_target_modules: Sequence[str],
        lora_rank_vae: int,
        quantum_enabled: bool,
        boson_sampler,
        checkpoint_subdir: str = "checkpoints",
        enable_step_checkpoints: bool = False,
    ):
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, checkpoint_subdir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.lora_modules_encoder = list(lora_modules_encoder)
        self.lora_modules_decoder = list(lora_modules_decoder)
        self.lora_modules_others = list(lora_modules_others)
        self.lora_rank_unet = lora_rank_unet
        self.vae_lora_target_modules = list(vae_lora_target_modules)
        self.lora_rank_vae = lora_rank_vae
        self.quantum_enabled = quantum_enabled
        self.boson_sampler = boson_sampler
        self.enable_step_checkpoints = enable_step_checkpoints

    def save(self, trainer, step: int, filename: Optional[str] = None) -> str:
        """Persist the current model state and return the checkpoint path."""
        if filename is None and not self.enable_step_checkpoints:
            logger.debug(
                "[ModelSaver] Skipping checkpoint at step {} (step checkpoints disabled).",
                step,
            )
            return ""
        eval_unet = trainer.accelerator.unwrap_model(trainer.unet)
        eval_vae_enc = trainer.accelerator.unwrap_model(trainer.vae_enc)
        eval_vae_dec = trainer.accelerator.unwrap_model(trainer.vae_dec)
        outf = os.path.join(self.checkpoint_dir, filename or f"model_{step}.pkl")
        sd = {
            "l_target_modules_encoder": self.lora_modules_encoder,
            "l_target_modules_decoder": self.lora_modules_decoder,
            "l_modules_others": self.lora_modules_others,
            "rank_unet": self.lora_rank_unet,
            "sd_encoder": get_peft_model_state_dict(eval_unet, adapter_name="default_encoder"),
            "sd_decoder": get_peft_model_state_dict(eval_unet, adapter_name="default_decoder"),
            "sd_other": get_peft_model_state_dict(eval_unet, adapter_name="default_others"),
            "rank_vae": self.lora_rank_vae,
            "vae_lora_target_modules": self.vae_lora_target_modules,
            "sd_vae_enc": eval_vae_enc.state_dict(),
            "sd_vae_dec": eval_vae_dec.state_dict(),
        }
        if self.quantum_enabled and self.boson_sampler is not None:
            sd["quantum_params"] = self.boson_sampler.model.state_dict()
        torch.save(sd, outf, _use_new_zipfile_serialization=False)
        return outf


class CheckpointCallback(PeriodicCallback):
    """Persist LoRA and VAE weights periodically."""

    def __init__(
        self,
        frequency: int,
        output_dir: str,
        max_train_steps: int,
        lora_modules_encoder: Sequence[str],
        lora_modules_decoder: Sequence[str],
        lora_modules_others: Sequence[str],
        lora_rank_unet: int,
        vae_lora_target_modules: Sequence[str],
        lora_rank_vae: int,
        quantum_enabled: bool,
        boson_sampler,
    ):
        super().__init__(frequency)
        self.output_dir = output_dir
        self.max_train_steps = max_train_steps
        self.lora_modules_encoder = list(lora_modules_encoder)
        self.lora_modules_decoder = list(lora_modules_decoder)
        self.lora_modules_others = list(lora_modules_others)
        self.lora_rank_unet = lora_rank_unet
        self.vae_lora_target_modules = list(vae_lora_target_modules)
        self.lora_rank_vae = lora_rank_vae
        self.quantum_enabled = quantum_enabled
        self.boson_sampler = boson_sampler
        self.model_saver = ModelSaver(
            output_dir=output_dir,
            lora_modules_encoder=self.lora_modules_encoder,
            lora_modules_decoder=self.lora_modules_decoder,
            lora_modules_others=self.lora_modules_others,
            lora_rank_unet=self.lora_rank_unet,
            vae_lora_target_modules=self.vae_lora_target_modules,
            lora_rank_vae=self.lora_rank_vae,
            quantum_enabled=self.quantum_enabled,
            boson_sampler=self.boson_sampler,
            enable_step_checkpoints=True,
        )

    def _should_run(self, trainer, state):
        if state.global_step == self.max_train_steps:
            return trainer.accelerator.is_main_process
        return super()._should_run(trainer, state)

    def on_step_end(self, trainer, state, logs):
        if not self._should_run(trainer, state):
            return

        self.model_saver.save(trainer, state.global_step)
        gc.collect()
        empty_cache()


class ValidationCallback(PeriodicCallback):
    """Run expensive validation metrics (FID + DINO)."""

    def __init__(
        self,
        frequency: int,
        output_dir: str,
        validation_transform: Callable[[Image.Image], Image.Image],
        src_image_paths: Sequence[str],
        tgt_image_paths: Sequence[str],
        feat_model,
        fid_device: str,
        a2b_ref_stats: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
        b2a_ref_stats: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
        validation_num_images: int,
        keep_generated_samples: bool = False,
        max_saved_samples: int = 100,
        model_saver: Optional[ModelSaver] = None,
        metrics_filename: str = "training_validation_metrics.json",
        best_metrics_filename: str = "best_metrics.json",
    ):
        super().__init__(frequency)
        self.output_dir = output_dir
        self.validation_transform = validation_transform
        self.src_image_paths = list(src_image_paths)
        self.tgt_image_paths = list(tgt_image_paths)
        self.feat_model = feat_model
        self.fid_device = fid_device
        self.a2b_ref_mu, self.a2b_ref_sigma = a2b_ref_stats
        self.b2a_ref_mu, self.b2a_ref_sigma = b2a_ref_stats
        self.validation_num_images = validation_num_images
        self._to_tensor = transforms.ToTensor()
        self._normalize = transforms.Normalize([0.5], [0.5])
        self.keep_generated_samples = keep_generated_samples
        self.max_saved_samples = max_saved_samples
        self.model_saver = model_saver
        self.metrics_path = os.path.join(self.output_dir, metrics_filename)
        self.best_metrics_path = os.path.join(self.output_dir, best_metrics_filename)
        self.best_fid_mean = float("inf")
        self.best_fid_step: Optional[int] = None
        self.best_dino_mean = float("inf")
        self.best_dino_step: Optional[int] = None
        self._load_best_metrics()

    def _has_reference_stats(self) -> bool:
        return (
            self.feat_model is not None
            and self.a2b_ref_mu is not None
            and self.a2b_ref_sigma is not None
            and self.b2a_ref_mu is not None
            and self.b2a_ref_sigma is not None
        )

    def on_step_end(self, trainer, state, logs):
        if not self._should_run(trainer, state):
            return
        self._run_validation(trainer, state)

    def run_once(self, trainer, state):
        """Execute validation immediately, bypassing frequency checks."""
        self._run_validation(trainer, state)

    def _run_validation(self, trainer, state):
        if not self._has_reference_stats():
            return
        device = trainer.accelerator.device
        eval_unet = trainer.accelerator.unwrap_model(trainer.unet)
        eval_vae_enc = trainer.accelerator.unwrap_model(trainer.vae_enc)
        eval_vae_dec = trainer.accelerator.unwrap_model(trainer.vae_dec)
        _timesteps = torch.tensor(
            [trainer.noise_scheduler.config.num_train_timesteps - 1],
            device=device,
        ).long()
        net_dino = DinoStructureLoss(device=device)
        fid_base = os.path.join(self.output_dir, f"fid-{state.global_step}")
        fid_output_dir = os.path.join(fid_base, "samples_a2b")
        os.makedirs(fid_output_dir, exist_ok=True)
        l_dino_scores_a2b: List[float] = []

        for idx, input_img_path in enumerate(tqdm(self.src_image_paths)):
            if idx > self.validation_num_images and self.validation_num_images > 0:
                break
            outf = os.path.join(fid_output_dir, f"{idx}.png")
            with torch.no_grad():
                input_img = self.validation_transform(Image.open(input_img_path).convert("RGB"))
                img_a = self._to_tensor(input_img)
                img_a = self._normalize(img_a).unsqueeze(0).to(device)
                use_annotation = getattr(trainer, "annotation_active", trainer.args.quantum)
                if use_annotation:
                    eval_fake_b, _ = trainer.cyclegan_d.forward_with_networks_dynamic(
                        img_a,
                        "a2b",
                        eval_vae_enc,
                        eval_unet,
                        eval_vae_dec,
                        trainer.noise_scheduler,
                        _timesteps,
                        trainer.fixed_a2b_emb_base[0:1],
                        bs=trainer.boson_sampler,
                        device=device,
                    )
                else:
                    eval_fake_b = trainer.cyclegan_d.forward_with_networks(
                        img_a,
                        "a2b",
                        eval_vae_enc,
                        eval_unet,
                        eval_vae_dec,
                        trainer.noise_scheduler,
                        _timesteps,
                        trainer.fixed_a2b_emb_base[0:1],
                    )
                eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b[0] * 0.5 + 0.5)
                eval_fake_b_pil.save(outf)
                a = net_dino.preprocess(input_img).unsqueeze(0)
                b = net_dino.preprocess(eval_fake_b_pil).unsqueeze(0)
                dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
                l_dino_scores_a2b.append(dino_ssim)

        dino_score_a2b = np.mean(l_dino_scores_a2b) if l_dino_scores_a2b else 0.0
        gen_features = get_folder_features(
            fid_output_dir,
            model=self.feat_model,
            num_workers=0,
            num=None,
            shuffle=False,
            seed=0,
            batch_size=8,
            device=torch.device(self.fid_device),
            mode="clean",
            custom_fn_resize=None,
            description="",
            verbose=True,
            custom_image_tranform=None,
        )
        ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
        score_fid_a2b = frechet_distance(self.a2b_ref_mu, self.a2b_ref_sigma, ed_mu, ed_sigma)
        self._cleanup_generated_samples(fid_output_dir)

        fid_output_dir = os.path.join(fid_base, "samples_b2a")
        os.makedirs(fid_output_dir, exist_ok=True)
        l_dino_scores_b2a: List[float] = []

        for idx, input_img_path in enumerate(tqdm(self.tgt_image_paths)):
            if idx > self.validation_num_images and self.validation_num_images > 0:
                break
            outf = os.path.join(fid_output_dir, f"{idx}.png")
            with torch.no_grad():
                input_img = self.validation_transform(Image.open(input_img_path).convert("RGB"))
                img_b = self._to_tensor(input_img)
                img_b = self._normalize(img_b).unsqueeze(0).to(device)
                use_annotation = getattr(trainer, "annotation_active", trainer.args.quantum)
                if use_annotation:
                    eval_fake_a, _ = trainer.cyclegan_d.forward_with_networks_dynamic(
                        img_b,
                        "b2a",
                        eval_vae_enc,
                        eval_unet,
                        eval_vae_dec,
                        trainer.noise_scheduler,
                        _timesteps,
                        trainer.fixed_b2a_emb_base[0:1],
                        bs=trainer.boson_sampler,
                        device=device,
                    )
                else:
                    eval_fake_a = trainer.cyclegan_d.forward_with_networks(
                        img_b,
                        "b2a",
                        eval_vae_enc,
                        eval_unet,
                        eval_vae_dec,
                        trainer.noise_scheduler,
                        _timesteps,
                        trainer.fixed_b2a_emb_base[0:1],
                    )
                eval_fake_a_pil = transforms.ToPILImage()(eval_fake_a[0] * 0.5 + 0.5)
                eval_fake_a_pil.save(outf)
                a = net_dino.preprocess(input_img).unsqueeze(0)
                b = net_dino.preprocess(eval_fake_a_pil).unsqueeze(0)
                dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
                l_dino_scores_b2a.append(dino_ssim)

        dino_score_b2a = np.mean(l_dino_scores_b2a) if l_dino_scores_b2a else 0.0
        gen_features = get_folder_features(
            fid_output_dir,
            model=self.feat_model,
            num_workers=0,
            num=None,
            shuffle=False,
            seed=0,
            batch_size=8,
            device=torch.device(self.fid_device),
            mode="clean",
            custom_fn_resize=None,
            description="",
            verbose=True,
            custom_image_tranform=None,
        )
        ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
        score_fid_b2a = frechet_distance(self.b2a_ref_mu, self.b2a_ref_sigma, ed_mu, ed_sigma)
        self._cleanup_generated_samples(fid_output_dir)

        fid_mean = (score_fid_a2b + score_fid_b2a) / 2.0
        dino_mean = (dino_score_a2b + dino_score_b2a) / 2.0
        fid_improved, dino_improved = self._update_best_metrics(fid_mean, dino_mean, trainer, state)
        best_fid_step = self.best_fid_step if self.best_fid_step is not None else -1
        best_dino_step = self.best_dino_step if self.best_dino_step is not None else -1
        logs = {
            "val/fid_a2b": score_fid_a2b,
            "val/fid_b2a": score_fid_b2a,
            "val/fid_mean": fid_mean,
            "val/dino_struct_a2b": dino_score_a2b,
            "val/dino_struct_b2a": dino_score_b2a,
            "val/dino_struct_mean": dino_mean,
            "val/best_fid_mean": self.best_fid_mean,
            "val/best_dino_struct_mean": self.best_dino_mean,
            "val/best_fid_step": best_fid_step,
            "val/best_dino_step": best_dino_step,
            "val/saved_best_model": bool(fid_improved),
        }
        trainer.accelerator.log(_to_jsonable(logs), step=state.global_step)
        metrics_entry = {
            "step": state.global_step,
            "fid_a2b": float(score_fid_a2b),
            "fid_b2a": float(score_fid_b2a),
            "fid_mean": float(fid_mean),
            "dino_struct_a2b": float(dino_score_a2b),
            "dino_struct_b2a": float(dino_score_b2a),
            "dino_struct_mean": float(dino_mean),
            "best_fid_mean": float(self.best_fid_mean if np.isfinite(self.best_fid_mean) else fid_mean),
            "best_dino_struct_mean": float(self.best_dino_mean if np.isfinite(self.best_dino_mean) else dino_mean),
            "saved_best_model": fid_improved,
            "best_fid_step": self.best_fid_step,
            "best_dino_step": self.best_dino_step,
        }
        self._append_metrics_entry(metrics_entry)
        logger.info(
            "[Validation] Step {} metrics -> FID(a2b={:.3f}, b2a={:.3f}, mean={:.3f}) | "
            "DINO(a2b={:.4f}, b2a={:.4f}, mean={:.4f}) | best_FID={:.3f}@{} best_DINO={:.4f}@{}{}",
            state.global_step,
            score_fid_a2b,
            score_fid_b2a,
            fid_mean,
            dino_score_a2b,
            dino_score_b2a,
            dino_mean,
            self.best_fid_mean,
            self.best_fid_step if self.best_fid_step is not None else "-",
            self.best_dino_mean,
            self.best_dino_step if self.best_dino_step is not None else "-",
            " [new best]" if fid_improved or dino_improved else "",
        )
        del net_dino

    def _update_best_metrics(self, fid_mean: float, dino_mean: float, trainer, state) -> Tuple[bool, bool]:
        fid_improved = fid_mean < self.best_fid_mean
        dino_improved = dino_mean < self.best_dino_mean
        if fid_improved:
            self.best_fid_mean = fid_mean
            self.best_fid_step = state.global_step
            logger.info(
                "[Validation] New best mean FID {:.3f} at step {}.",
                self.best_fid_mean,
                state.global_step,
            )
        if dino_improved:
            self.best_dino_mean = dino_mean
            self.best_dino_step = state.global_step
            logger.info(
                "[Validation] New best mean DINO {:.4f} at step {}.",
                self.best_dino_mean,
                state.global_step,
            )
        if fid_improved or dino_improved:
            self._write_best_metrics()
        if fid_improved and self.model_saver is not None:
            checkpoint_path = self.model_saver.save(trainer, state.global_step, filename="best_model.pkl")
            if checkpoint_path:
                logger.info("[Validation] Saved new best model to {}", checkpoint_path)
            gc.collect()
            empty_cache()
        return fid_improved, dino_improved

    def _append_metrics_entry(self, entry: Dict[str, object]) -> None:
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        history: List[Dict[str, object]] = []
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                    if isinstance(existing, list):
                        history = existing
            except (json.JSONDecodeError, OSError):
                history = []
        history.append(_to_jsonable(entry))
        with open(self.metrics_path, "w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(history), handle, indent=2)

    def _load_best_metrics(self) -> None:
        if not os.path.exists(self.best_metrics_path):
            return
        try:
            with open(self.best_metrics_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return
        best_fid = data.get("best_fid_mean")
        if isinstance(best_fid, (int, float)):
            self.best_fid_mean = float(best_fid)
        best_fid_step = data.get("best_fid_step")
        if isinstance(best_fid_step, int):
            self.best_fid_step = best_fid_step
        best_dino = data.get("best_dino_struct_mean")
        if isinstance(best_dino, (int, float)):
            self.best_dino_mean = float(best_dino)
        best_dino_step = data.get("best_dino_step")
        if isinstance(best_dino_step, int):
            self.best_dino_step = best_dino_step

    def _write_best_metrics(self) -> None:
        os.makedirs(os.path.dirname(self.best_metrics_path), exist_ok=True)
        payload = {
            "best_fid_mean": self.best_fid_mean,
            "best_fid_step": self.best_fid_step,
            "best_dino_struct_mean": self.best_dino_mean,
            "best_dino_step": self.best_dino_step,
        }
        with open(self.best_metrics_path, "w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(payload), handle, indent=2)

    def _cleanup_generated_samples(self, directory: str) -> None:
        if not os.path.isdir(directory):
            return
        if not self.keep_generated_samples:
            shutil.rmtree(directory, ignore_errors=True)
            return
        allowed_ext = {".png", ".jpg", ".jpeg", ".bmp"}
        paths = sorted(
            p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in allowed_ext
        )
        for path in paths[self.max_saved_samples:]:
            try:
                path.unlink()
            except OSError:
                pass
