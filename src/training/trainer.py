"""CycleGAN training loop abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import TrainingConfig
from .losses import CycleGANLosses


@dataclass
class TrainerState:
    """Mutable state tracked across training."""

    epoch: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None


class CycleGANTrainer:
    """Encapsulates the CycleGAN training loop."""

    def __init__(
        self,
        args,
        training_config: TrainingConfig,
        accelerator,
        loss_helper: CycleGANLosses,
        train_dataloader: DataLoader,
        net_disc_a,
        net_disc_b,
        unet,
        vae_enc,
        vae_dec,
        cyclegan_d,
        noise_scheduler,
        fixed_a2b_emb_base,
        fixed_b2a_emb_base,
        optimizer_gen,
        optimizer_disc,
        lr_scheduler_gen,
        lr_scheduler_disc,
        params_gen,
        boson_sampler=None,
        callbacks: Optional[List] = None,
        weight_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.args = args
        self.config = training_config
        self.accelerator = accelerator
        self.loss_helper = loss_helper
        self.train_dataloader = train_dataloader
        self.net_disc_a = net_disc_a
        self.net_disc_b = net_disc_b
        self.unet = unet
        self.vae_enc = vae_enc
        self.vae_dec = vae_dec
        # Note: vae_a2b and vae_b2a are accessed through vae_enc/vae_dec wrappers
        self.cyclegan_d = cyclegan_d
        self.noise_scheduler = noise_scheduler
        self.fixed_a2b_emb_base = fixed_a2b_emb_base
        self.fixed_b2a_emb_base = fixed_b2a_emb_base
        self.optimizer_gen = optimizer_gen
        self.optimizer_disc = optimizer_disc
        self.lr_scheduler_gen = lr_scheduler_gen
        self.lr_scheduler_disc = lr_scheduler_disc
        self.params_gen = params_gen
        self.boson_sampler = boson_sampler
        self.callbacks = callbacks or []
        self.weight_dtype = weight_dtype
        self.annotation_active = getattr(args, "annotation_active", args.quantum)

        self.state = TrainerState()
        self.progress_bar = None
        self.last_batch = None
        self.last_outputs: Dict[str, torch.Tensor] = {}

    def train(self):
        """Run the training loop."""
        total_steps = self.config.max_train_steps or len(self.train_dataloader)
        self.progress_bar = tqdm(
            range(total_steps),
            initial=self.state.global_step,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )

        self._log_trainable_param_summary()

        for epoch in range(self.state.epoch, self.config.max_train_epochs):
            self.state.epoch = epoch
            for step, batch in enumerate(self.train_dataloader):
                logs, skipped = self._train_step(batch)
                if skipped:
                    continue

                # Track "global_step" as actual optimizer steps (i.e. when gradients are synced).
                if self.accelerator.sync_gradients:
                    self.state.global_step += 1
                    self.accelerator.log(logs, step=self.state.global_step)
                    if self.progress_bar is not None:
                        self.progress_bar.update(1)
                    self._run_step_callbacks(logs)

                if self.state.global_step >= total_steps:
                    break

            self._run_epoch_callbacks()
            if self.state.global_step >= total_steps:
                break

        if self.progress_bar is not None:
            self.progress_bar.close()

    # pylint: disable=too-many-statements,too-many-locals
    def _train_step(self, batch) -> Tuple[Dict[str, float], bool]:
        """Perform one training iteration."""
        accelerator = self.accelerator
        args = self.args
        logs: Dict[str, float] = {}
        skipped = False

        img_a = batch["pixel_values_src"].to(dtype=self.weight_dtype)
        img_b = batch["pixel_values_tgt"].to(dtype=self.weight_dtype)
        t_start = time.time()

        l_accumulate = [self.unet, self.net_disc_a, self.net_disc_b, self.vae_enc, self.vae_dec]
        with accelerator.accumulate(*l_accumulate):
            bsz = img_a.shape[0]
            fixed_a2b_emb = self.fixed_a2b_emb_base.repeat(bsz, 1, 1).to(dtype=self.weight_dtype)
            fixed_b2a_emb = self.fixed_b2a_emb_base.repeat(bsz, 1, 1).to(dtype=self.weight_dtype)
            timesteps = torch.tensor(
                [self.noise_scheduler.config.num_train_timesteps - 1] * bsz,
                device=img_a.device,
            ).long()

            # Cycle consistency
            if self.annotation_active:
                cyc_fake_b, _ = self.cyclegan_d.forward_with_networks_dynamic(
                    img_a,
                    "a2b",
                    self.vae_enc,
                    self.unet,
                    self.vae_dec,
                    self.noise_scheduler,
                    timesteps,
                    fixed_a2b_emb,
                    bs=self.boson_sampler,
                    device=accelerator.device,
                    accelerator=accelerator,
                )
                if torch.isnan(cyc_fake_b).any():
                    accelerator.print("[Trainer] Skipping step: Real A -> Fake B produced NaNs")
                    skipped = True
                    return logs, skipped
                cyc_rec_a, _ = self.cyclegan_d.forward_with_networks_dynamic(
                    cyc_fake_b,
                    "b2a",
                    self.vae_enc,
                    self.unet,
                    self.vae_dec,
                    self.noise_scheduler,
                    timesteps,
                    text_emb=fixed_b2a_emb,
                    bs=self.boson_sampler,
                    device=accelerator.device,
                    accelerator=accelerator,
                )
                if torch.isnan(cyc_rec_a).any():
                    accelerator.print("[Trainer] Skipping step: Fake B -> Rec A produced NaNs")
                    skipped = True
                    return logs, skipped
            else:
                cyc_fake_b = self.cyclegan_d.forward_with_networks(
                    img_a,
                    "a2b",
                    self.vae_enc,
                    self.unet,
                    self.vae_dec,
                    self.noise_scheduler,
                    timesteps,
                    fixed_a2b_emb,
                )
                cyc_rec_a = self.cyclegan_d.forward_with_networks(
                    cyc_fake_b,
                    "b2a",
                    self.vae_enc,
                    self.unet,
                    self.vae_dec,
                    self.noise_scheduler,
                    timesteps,
                    fixed_b2a_emb,
                )

            loss_cycle_a = self.loss_helper.cycle_loss(cyc_rec_a, img_a)

            if self.annotation_active:
                cyc_fake_a, _ = self.cyclegan_d.forward_with_networks_dynamic(
                    img_b,
                    "b2a",
                    self.vae_enc,
                    self.unet,
                    self.vae_dec,
                    self.noise_scheduler,
                    timesteps,
                    fixed_b2a_emb,
                    bs=self.boson_sampler,
                    device=accelerator.device,
                    accelerator=accelerator,
                )
                if torch.isnan(cyc_fake_a).any():
                    accelerator.print("[Trainer] Skipping step: Real B -> Fake A produced NaNs")
                    skipped = True
                    return logs, skipped
                cyc_rec_b, _ = self.cyclegan_d.forward_with_networks_dynamic(
                    cyc_fake_a,
                    "a2b",
                    self.vae_enc,
                    self.unet,
                    self.vae_dec,
                    self.noise_scheduler,
                    timesteps,
                    text_emb=fixed_a2b_emb,
                    bs=self.boson_sampler,
                    device=accelerator.device,
                    accelerator=accelerator,
                )
                if torch.isnan(cyc_rec_b).any():
                    accelerator.print("[Trainer] Skipping step: Fake A -> Rec B produced NaNs")
                    skipped = True
                    return logs, skipped
            else:
                cyc_fake_a = self.cyclegan_d.forward_with_networks(
                    img_b,
                    "b2a",
                    self.vae_enc,
                    self.unet,
                    self.vae_dec,
                    self.noise_scheduler,
                    timesteps,
                    fixed_b2a_emb,
                )
                cyc_rec_b = self.cyclegan_d.forward_with_networks(
                    cyc_fake_a,
                    "a2b",
                    self.vae_enc,
                    self.unet,
                    self.vae_dec,
                    self.noise_scheduler,
                    timesteps,
                    fixed_a2b_emb,
                )

            loss_cycle_b = self.loss_helper.cycle_loss(cyc_rec_b, img_b)
            accelerator.backward(loss_cycle_a + loss_cycle_b, retain_graph=True)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(self.params_gen, args.max_grad_norm)
            self.optimizer_gen.step()
            self.lr_scheduler_gen.step()
            self.optimizer_gen.zero_grad()

            if self.annotation_active:
                fake_a, _ = self.cyclegan_d.forward_with_networks_dynamic(
                    img_b,
                    "b2a",
                    self.vae_enc,
                    self.unet,
                    self.vae_dec,
                    self.noise_scheduler,
                    timesteps,
                    fixed_b2a_emb,
                    bs=self.boson_sampler,
                    device=accelerator.device,
                )
                fake_b, _ = self.cyclegan_d.forward_with_networks_dynamic(
                    img_a,
                    "a2b",
                    self.vae_enc,
                    self.unet,
                    self.vae_dec,
                    self.noise_scheduler,
                    timesteps,
                    fixed_a2b_emb,
                    bs=self.boson_sampler,
                    device=accelerator.device,
                )
            else:
                fake_a = self.cyclegan_d.forward_with_networks(
                    img_b,
                    "b2a",
                    self.vae_enc,
                    self.unet,
                    self.vae_dec,
                    self.noise_scheduler,
                    timesteps,
                    fixed_b2a_emb,
                )
                fake_b = self.cyclegan_d.forward_with_networks(
                    img_a,
                    "a2b",
                    self.vae_enc,
                    self.unet,
                    self.vae_dec,
                    self.noise_scheduler,
                    timesteps,
                    fixed_a2b_emb,
                )

            loss_gan_a = self.loss_helper.gan_loss(self.net_disc_a(fake_b, for_G=True))
            loss_gan_b = self.loss_helper.gan_loss(self.net_disc_b(fake_a, for_G=True))
            accelerator.backward(loss_gan_a + loss_gan_b, retain_graph=True)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(self.params_gen, args.max_grad_norm)
            self.optimizer_gen.step()
            self.lr_scheduler_gen.step()
            self.optimizer_gen.zero_grad()
            self.optimizer_disc.zero_grad()

            if self.annotation_active:
                idt_a, _ = self.cyclegan_d.forward_with_networks_dynamic(
                    img_b,
                    "a2b",
                    self.vae_enc,
                    self.unet,
                    self.vae_dec,
                    self.noise_scheduler,
                    timesteps,
                    fixed_a2b_emb,
                    bs=self.boson_sampler,
                    device=accelerator.device,
                )
            else:
                idt_a = self.cyclegan_d.forward_with_networks(
                    img_b,
                    "a2b",
                    self.vae_enc,
                    self.unet,
                    self.vae_dec,
                    self.noise_scheduler,
                    timesteps,
                    fixed_a2b_emb,
                )
            loss_idt_a = self.loss_helper.identity_loss(idt_a, img_b)

            if self.annotation_active:
                idt_b, _ = self.cyclegan_d.forward_with_networks_dynamic(
                    img_a,
                    "b2a",
                    self.vae_enc,
                    self.unet,
                    self.vae_dec,
                    self.noise_scheduler,
                    timesteps,
                    fixed_b2a_emb,
                    bs=self.boson_sampler,
                    device=accelerator.device,
                )
            else:
                idt_b = self.cyclegan_d.forward_with_networks(
                    img_a,
                    "b2a",
                    self.vae_enc,
                    self.unet,
                    self.vae_dec,
                    self.noise_scheduler,
                    timesteps,
                    fixed_b2a_emb,
                )
            loss_idt_b = self.loss_helper.identity_loss(idt_b, img_a)
            loss_g_idt = loss_idt_a + loss_idt_b
            accelerator.backward(loss_g_idt, retain_graph=False)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(self.params_gen, args.max_grad_norm)
            self.optimizer_gen.step()
            self.lr_scheduler_gen.step()
            self.optimizer_gen.zero_grad()

            loss_D_A_fake = self.loss_helper.gan_loss(self.net_disc_a(fake_b.detach(), for_real=False))
            loss_D_B_fake = self.loss_helper.gan_loss(self.net_disc_b(fake_a.detach(), for_real=False))
            loss_D_fake = (loss_D_A_fake + loss_D_B_fake) * 0.5
            accelerator.backward(loss_D_fake, retain_graph=False)
            if accelerator.sync_gradients:
                params_to_clip = list(self.net_disc_a.parameters()) + list(self.net_disc_b.parameters())
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            self.optimizer_disc.step()
            self.lr_scheduler_disc.step()
            self.optimizer_disc.zero_grad()

            loss_D_A_real = self.loss_helper.gan_loss(self.net_disc_a(img_b, for_real=True))
            loss_D_B_real = self.loss_helper.gan_loss(self.net_disc_b(img_a, for_real=True))
            loss_D_real = (loss_D_A_real + loss_D_B_real) * 0.5
            accelerator.backward(loss_D_real, retain_graph=False)
            if accelerator.sync_gradients:
                params_to_clip = list(self.net_disc_a.parameters()) + list(self.net_disc_b.parameters())
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            self.optimizer_disc.step()
            self.lr_scheduler_disc.step()
            self.optimizer_disc.zero_grad()

        logs["cycle_a"] = loss_cycle_a.detach().item()
        logs["cycle_b"] = loss_cycle_b.detach().item()
        logs["gan_a"] = loss_gan_a.detach().item()
        logs["gan_b"] = loss_gan_b.detach().item()
        logs["disc_a"] = (loss_D_A_fake + loss_D_A_real).detach().item()
        logs["disc_b"] = (loss_D_B_fake + loss_D_B_real).detach().item()
        logs["idt_a"] = loss_idt_a.detach().item()
        logs["idt_b"] = loss_idt_b.detach().item()
        logs["step_time"] = time.time() - t_start

        self.last_batch = batch
        self.last_outputs = {
            "cyc_rec_a": cyc_rec_a.detach(),
            "cyc_rec_b": cyc_rec_b.detach(),
            "fake_a": fake_a.detach(),
            "fake_b": fake_b.detach(),
        }
        
        return logs, skipped

    def _log_trainable_param_summary(self) -> None:
        """Print per-module trainable parameter counts before training."""
        def _counts(module: Optional[torch.nn.Module]) -> Tuple[int, int]:
            if module is None:
                return 0, 0
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return trainable, total

        modules = {
            "UNet": self.unet,
            "VAE_A2B": self.vae_enc.vae,  # Access through wrapper to avoid duplicate references
            "VAE_B2A": self.vae_enc.vae_b2a,  # Access through wrapper to avoid duplicate references
            "Disc_A": self.net_disc_a,
            "Disc_B": self.net_disc_b,
        }
        if self.boson_sampler is not None:
            modules["BosonSampler"] = self.boson_sampler

        self.accelerator.print("\n[Trainer] Trainable parameter summary before first step:")
        for name, module in modules.items():
            trainable, total = _counts(module)
            self.accelerator.print(f" - {name}: {trainable:,}/{total:,} parameters trainable")
        self.accelerator.print("")

    def _run_step_callbacks(self, logs):
        for callback in self.callbacks:
            callback.on_step_end(self, self.state, logs)

    def _run_epoch_callbacks(self):
        for callback in self.callbacks:
            callback.on_epoch_end(self, self.state)
