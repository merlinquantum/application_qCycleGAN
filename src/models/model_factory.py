"""Model assembly helpers for CycleGAN training."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from huggingface_hub import hf_hub_download
from loguru import logger

from config import ClassicalConfig, QuantumConfig, TrainingConfig
from models.cyclegan_turbo import (
    CycleGAN_Turbo,
    VAE_decode,
    VAE_encode,
    initialize_unet,
    initialize_vae,
)


@dataclass
class ModelComponents:
    """Bundle aggregating neural components used during training."""

    unet: torch.nn.Module
    vae_enc: torch.nn.Module
    vae_dec: torch.nn.Module
    vae_a2b: torch.nn.Module
    vae_b2a: torch.nn.Module
    lora_modules_encoder: List[str]
    lora_modules_decoder: List[str]
    lora_modules_others: List[str]
    vae_lora_target_modules: List[str]
    cyclegan: Optional[CycleGAN_Turbo] = None
    adaptation_module: Optional[torch.nn.Module] = None
    quantum_encoder: Optional[torch.nn.Module] = None


class ModelFactory:
    """Factory that builds the UNet/VAE stack either from scratch or checkpoints."""

    @staticmethod
    def build(
        accelerator,
        training_cfg: TrainingConfig,
        quantum_cfg: QuantumConfig,
        classical_cfg: ClassicalConfig | None = None,
    ) -> ModelComponents:
        """Create UNet/VAE/Lora components from configs."""
        classical_cfg = classical_cfg or ClassicalConfig()
        if quantum_cfg.start_path or classical_cfg.cl_comp:
            return ModelFactory._build_from_checkpoint(accelerator, training_cfg, quantum_cfg)
        return ModelFactory._build_from_scratch(accelerator, training_cfg, quantum_cfg)

    @staticmethod
    def _build_from_checkpoint(accelerator, training_cfg: TrainingConfig, quantum_cfg: QuantumConfig) -> ModelComponents:
        """Load pretrained weights and attach optional quantum adaptation module."""
        _, lora_enc, lora_dec, lora_others = initialize_unet(
            training_cfg.lora_rank_unet,
            return_lora_module_names=True,
        )
        _, vae_lora_targets = initialize_vae(
            training_cfg.lora_rank_vae,
            return_lora_module_names=True,
            dynamic=quantum_cfg.quantum,
        )
        model_path = ModelFactory._resolve_pretrained_path(training_cfg, quantum_cfg)
        cyclegan = CycleGAN_Turbo(accelerator=accelerator, pretrained_path=model_path)
        vae_enc = cyclegan.vae_enc
        vae_dec = cyclegan.vae_dec
        vae_a2b = cyclegan.vae
        vae_b2a = cyclegan.vae_b2a
        unet = cyclegan.unet
        # training these layers for more adaptability
        unet.conv_in.requires_grad_(True)
        logger.info("[ModelFactory] UNet.conv_in requires_grad set to True")
        vae_a2b.post_quant_conv.requires_grad_(True)
        logger.info("[ModelFactory] VAE A2B post_quant_conv requires_grad set to True")
        vae_b2a.post_quant_conv.requires_grad_(True)
        logger.info("[ModelFactory] VAE B2A post_quant_conv requires_grad set to True")
        adaptation_module = None
        if quantum_cfg.quantum:
            adaptation_module = ModelFactory._load_adaptation_module(model_path, accelerator.device)

        return ModelComponents(
            unet=unet,
            vae_enc=vae_enc,
            vae_dec=vae_dec,
            vae_a2b=vae_a2b,
            vae_b2a=vae_b2a,
            lora_modules_encoder=lora_enc,
            lora_modules_decoder=lora_dec,
            lora_modules_others=lora_others,
            vae_lora_target_modules=vae_lora_targets,
            cyclegan=cyclegan,
            adaptation_module=adaptation_module,
        )

    @staticmethod
    def _build_from_scratch(accelerator, training_cfg: TrainingConfig, quantum_cfg: QuantumConfig) -> ModelComponents:
        """Initialize UNet/VAE with LoRA layers when no checkpoint is provided."""
        logger.info(
            '[ModelFactory] Building from scratch using base "stabilityai/sd-turbo" weights (no CycleGAN-Turbo checkpoint).'
        )
        unet, lora_enc, lora_dec, lora_others = initialize_unet(
            training_cfg.lora_rank_unet,
            return_lora_module_names=True,
        )
        vae_a2b, vae_lora_targets = initialize_vae(
            training_cfg.lora_rank_vae,
            return_lora_module_names=True,
            dynamic=quantum_cfg.quantum,
        )
        device = accelerator.device
        weight_dtype = torch.float32
        vae_a2b.to(device, dtype=weight_dtype)
        unet.to(device, dtype=weight_dtype)
        unet.conv_in.requires_grad_(True)
        logger.info("[ModelFactory] UNet.conv_in requires_grad set to True")
        vae_b2a = copy.deepcopy(vae_a2b)
        vae_enc = VAE_encode(vae_a2b, vae_b2a=vae_b2a)
        vae_dec = VAE_decode(vae_a2b, vae_b2a=vae_b2a)

        return ModelComponents(
            unet=unet,
            vae_enc=vae_enc,
            vae_dec=vae_dec,
            vae_a2b=vae_a2b,
            vae_b2a=vae_b2a,
            lora_modules_encoder=lora_enc,
            lora_modules_decoder=lora_dec,
            lora_modules_others=lora_others,
            vae_lora_target_modules=vae_lora_targets,
            cyclegan=None,
            adaptation_module=None,
        )
    @staticmethod
    def _resolve_pretrained_path(training_cfg: TrainingConfig, quantum_cfg: QuantumConfig) -> str:
        """Turn HF IDs or local paths into a concrete checkpoint path."""
        if quantum_cfg.start_path:
            return ModelFactory._resolve_path(quantum_cfg.start_path)
        checkpoint = training_cfg.hf_model_path
        if checkpoint:
            if checkpoint.startswith("hf://"):
                repo_id, filename = checkpoint[5:].rsplit("/", 1)
                return hf_hub_download(repo_id=repo_id, filename=filename)
            return hf_hub_download(
                repo_id="quandelagl/img-to-img-turbo-pretrained",
                filename=checkpoint,
            )
        raise ValueError("No pretrained path or hf_model_path provided for checkpoint loading.")

    @staticmethod
    def _resolve_path(path: str) -> str:
        """Download Hugging Face checkpoints when needed and return local path."""
        if path.startswith("hf://"):
            repo_id, filename = path[5:].rsplit("/", 1)
            return hf_hub_download(repo_id=repo_id, filename=filename)
        return path

    @staticmethod
    def _load_adaptation_module(model_path: str, device: torch.device) -> Optional[torch.nn.Module]:
        """Load the lightweight conv adaptation module from the classical checkpoint if present."""
        try:
            checkpoint = torch.load(model_path, map_location=device)
        except Exception:  # pragma: no cover - defensive
            return None
        if "conv_ad" not in checkpoint:
            return None
        module = torch.nn.Conv2d(8, 4, kernel_size=1, stride=1)
        module.load_state_dict(checkpoint["conv_ad"])
        return module.to(device)


class RandomAblationSampler(torch.nn.Module):
    """Simple random projection used to ablate the Boson sampler."""

    def __init__(
        self,
        dims: Sequence[int],
        output_dim: int = 1024,
        hidden_dim: int = 2048,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        trimmed_dims = tuple(dims[1:]) if len(dims) >= 4 else tuple(dims)
        if not trimmed_dims:
            raise ValueError("RandomAblationSampler requires non-empty dims.")
        if hidden_dim < 0:
            raise ValueError("RandomAblationSampler requires hidden_dim > 0.")
        self.dims = trimmed_dims
        self.input_dim = math.prod(self.dims)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if self.hidden_dim > 0:
            logger.info(
                "[ModelFactor - Random] Random MLP defined with hidden dimension of {}",
                hidden_dim,
            )
            self.model = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, self.output_dim),
            )
        else:
            logger.info(
                "[ModelFactory - Random] Linear layer defined",
                hidden_dim,
            )
            self.model = torch.nn.Linear(self.input_dim, self.output_dim)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        flat = x.view(batch_size, -1)
        if flat.shape[1] != self.input_dim:
            raise ValueError(
                f"RandomAblationSampler expected flattened dim {self.input_dim}, "
                f"got {flat.shape[1]}."
            )
        #if self.hidden_dim > 0:
        return torch.softmax(self.model(flat), dim = -1)
        #else:
            #return self.model(flat)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Delegate checkpoint loading to the underlying MLP."""
        return self.model.load_state_dict(state_dict, strict=strict)

    def state_dict(self, destination=None, prefix="", keep_vars=False):  # pragma: no cover - delegation
        return self.model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)


class TrainableParamsManager:
    """Utility helpers that manage gradients and parameter selection."""

    @staticmethod
    def freeze_vae_encoder(components: ModelComponents) -> None:
        """Freeze the VAE encoding path while keeping the decoding path trainable.

        Intended behavior for `quantum`, `cl_comp`, and `random_ablation` runs:
        - Encoder (and quant_conv) are fully frozen.
        - Decoder remains *LoRA-trained* (vae_skip adapter) while base decoder weights stay frozen,
          except:
            - `decoder.conv_in` and `decoder.conv_out` are fully trained (base weights enabled),
            - `decoder.skip_conv_{1..4}` are trained (these are added skip adapters),
            - `post_quant_conv` is trained (maps latent -> decoder space).
        """
        def _enable_module(module: torch.nn.Module) -> None:
            if module is None:
                return
            module.requires_grad_(True)

        def _enable_base_layer_if_present(module: torch.nn.Module) -> None:
            if module is None:
                return
            # PEFT LoRA-wrapped convs expose the underlying conv as `.base_layer`.
            base = getattr(module, "base_layer", None)
            (base or module).requires_grad_(True)

        for label, vae in (("A->B", components.vae_a2b), ("B->A", components.vae_b2a)):
            if vae is None:
                continue
            # Start from a conservative baseline: freeze everything, then selectively unfreeze.
            vae.requires_grad_(False)

            # Encoder (and quant_conv) stays frozen.
            if hasattr(vae, "encoder") and vae.encoder is not None:
                vae.encoder.requires_grad_(False)
            if hasattr(vae, "quant_conv") and vae.quant_conv is not None:
                vae.quant_conv.requires_grad_(False)

            # Decoder LoRA params (vae_skip adapter) remain trainable; base decoder weights remain frozen.
            for name, param in vae.named_parameters():
                if name.startswith("decoder.") and "lora" in name and "vae_skip" in name:
                    param.requires_grad_(True)

            # Ensure the skip conv adapters are trained (they are part of the decoder path).
            decoder = getattr(vae, "decoder", None)
            for idx in range(1, 5):
                skip = getattr(decoder, f"skip_conv_{idx}", None) if decoder is not None else None
                if skip is not None:
                    _enable_module(skip)

            # Fully train decoder IO convs (base weights, even when LoRA-wrapped).
            if decoder is not None:
                _enable_base_layer_if_present(getattr(decoder, "conv_in", None))
                _enable_base_layer_if_present(getattr(decoder, "conv_out", None))

            # Train latent->decoder mapping conv.
            if hasattr(vae, "post_quant_conv") and vae.post_quant_conv is not None:
                _enable_base_layer_if_present(vae.post_quant_conv)

            logger.info(
                "[ModelFactory] VAE {} frozen encoder; trainable decoder LoRA + conv_in/out + skip_convs + post_quant_conv",
                label,
            )

    @staticmethod
    def trainable_params(
        components: ModelComponents,
        quantum_cfg: QuantumConfig,
        boson_sampler=None,
        train_unet: bool = False,
    ) -> List[torch.nn.Parameter]:
        """Return the list of trainable tensors, optionally including quantum add-ons."""
        extra_params: List[torch.nn.Parameter] = []
        # Always propagate provided boson-sampler params; requires_grad flags decide if they train.
        boson_for_training = boson_sampler if boson_sampler is not None else None
        if boson_for_training is not None:
            extra_params.extend(p for p in boson_for_training.parameters() if p.requires_grad)
        if components.adaptation_module is not None:
            extra_params.extend(p for p in components.adaptation_module.parameters() if p.requires_grad)

        if components.cyclegan is not None:
            return components.cyclegan.get_traininable_params(
                components.unet,
                components.vae_a2b,
                components.vae_b2a,
                boson_for_training,
                dynamic=False,
                train_full_unet=train_unet,
            ) + extra_params
        base = CycleGAN_Turbo.get_traininable_params(
            components.unet,
            components.vae_a2b,
            components.vae_b2a,
            boson_for_training,
            dynamic=quantum_cfg.quantum,
            train_full_unet=train_unet,
        )
        return base + extra_params

    @staticmethod
    def describe(components: ModelComponents) -> None:
        def _count(module: Optional[torch.nn.Module]) -> int:
            if module is None:
                return 0
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        logger.info("[ModelFactory] === Parameter Freezing / LoRA Overview ===")
        frozen_notes = []
        if components.vae_a2b is not None and not any(p.requires_grad for p in components.vae_a2b.encoder.parameters()):
            frozen_notes.append("VAE A->B encoder")
        if components.vae_b2a is not None and not any(p.requires_grad for p in components.vae_b2a.encoder.parameters()):
            frozen_notes.append("VAE B->A encoder")
        logger.info(
            "[ModelFactory] Frozen modules: {}",
            ", ".join(frozen_notes) if frozen_notes else "none",
        )

        trainable_notes = [
            "UNet.conv_in",
            "UNet LoRA adapters (encoder/decoder/others)",
            "VAE decoder skip_convs (1-4)",
            "VAE post_quant_conv (both directions)",
        ]
        logger.info("[ModelFactory] Trainable base layers: {}", ", ".join(trainable_notes))
        logger.info(
            "[ModelFactory] LoRA modules (encoder/decoder/other): {}/{}/{}",
            len(components.lora_modules_encoder),
            len(components.lora_modules_decoder),
            len(components.lora_modules_others),
        )
        logger.info("[ModelFactory] UNet trainable params: {:,}", _count(components.unet))
        logger.info("[ModelFactory] VAE A2B trainable params: {:,}", _count(components.vae_a2b))
        logger.info("[ModelFactory] VAE B2A trainable params: {:,}", _count(components.vae_b2a))
        if components.adaptation_module is not None:
            logger.info(
                "[ModelFactory] Adaptation module trainable params: {:,}",
                _count(components.adaptation_module),
            )
        if components.cyclegan is not None:
            logger.info("[ModelFactory] Loaded pretrained CycleGAN weights")
        logger.info("[ModelFactory] =========================================")
