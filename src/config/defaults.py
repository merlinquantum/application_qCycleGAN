"""Dataclass-based configuration objects with lightweight validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class QuantumConfig:
    """Configuration for the quantum encoder."""

    num_modes: int = 20
    num_photons: int = 3
    epsilon: float = 1e-5
    computation_space: str = "UNBUNCHED"
    trainable_parameters: List[str] = field(default_factory=lambda: ["phi"])
    quantum: bool = False
    quantum_trained: bool = False
    start_path: Optional[str] = None
    processes: int = 4
    sort_encoding: bool = True
    use_haar_unitary: bool = False

    def validate(self) -> None:
        """Ensure configuration values are consistent."""
        if self.num_modes < self.num_photons:
            raise ValueError(
                f"num_modes ({self.num_modes}) must be >= num_photons ({self.num_photons})"
            )
        if self.num_modes <= 0 or self.num_photons <= 0:
            raise ValueError("num_modes and num_photons must be positive integers")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.processes <= 0:
            raise ValueError("processes must be a positive integer")


@dataclass
class ClassicalConfig:
    """Configuration for classical comparison/ablation settings."""

    cl_comp: bool = False
    random_ablation: bool = False
    random_trained: bool = False
    random_ablation_hidden_dim: int = 2048

    def validate(self) -> None:
        """Basic logical checks for classical toggles."""
        if self.random_trained and not self.random_ablation:
            raise ValueError("random_trained=True requires random_ablation=True")
        if self.random_ablation_hidden_dim < 0:
            raise ValueError("random_ablation_hidden_dim must be a positive integer")


@dataclass
class DatasetConfig:
    """Configuration for dataset and preprocessing pipeline."""

    dataset_name: str = "quandelagl/bdd-100k"
    dataset_folder: str = ""
    train_img_prep: str = "no_resize"
    val_img_prep: str = "no_resize"
    train_batch_size: int = 4
    dataloader_num_workers: int = 0
    quantum_dims: Tuple[int, ...] = (4, 16, 16)
    prompt_src: str = "Driving in the night"
    prompt_tgt: str = "Driving in the day"
    training_images: float = 1.0
    cleanup: bool = False

    def validate(self) -> None:
        """Basic sanity checks for dataset paths and sizes."""
        if not self.dataset_folder:
            raise ValueError("dataset_folder must be provided")
        if not self.dataset_name:
            raise ValueError("dataset_name must be provided")
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")
        if self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers cannot be negative")
        if len(self.quantum_dims) != 3:
            raise ValueError("quantum_dims must be a tuple of length 3 (C, H, W)")
        if not 0 < self.training_images <= 1.0:
            raise ValueError("training_images must be in (0, 1]")


@dataclass
class TrainingConfig:
    """Training loop hyperparameters and runtime options."""

    learning_rate: float = 5e-6
    max_train_epochs: int | None = 100
    max_train_steps: int | None = None
    validation_steps: int = 500
    lambda_gan: float = 0.5
    lambda_idt: float = 1.0
    lambda_cycle: float = 1.0
    lambda_cycle_lpips: float = 10.0
    lambda_idt_lpips: float = 1.0
    gan_disc_type: str = "vagan_clip"
    gan_loss_type: str = "multilevel_sigmoid"
    hf_model_path: str = "model_251.pkl"
    revision: Optional[str] = None
    variant: Optional[str] = None
    lora_rank_unet: int = 128
    lora_rank_vae: int = 4
    unet_trained: bool = False
    viz_freq: int = 20
    output_dir: str = "a"
    report_to: str = "wandb"
    tracker_project_name: str = ""
    validation_num_images: int = -1
    checkpointing_steps: int = 1000
    save_step_checkpoints: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 10.0
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    seed: int = 42
    gradient_accumulation_steps: int = 1
    allow_tf32: bool = False
    gradient_checkpointing: bool = False
    enable_xformers_memory_efficient_attention: bool = False

    def validate(self) -> None:
        """Ensure hyperparameters are in valid ranges."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.validation_steps <= 0:
            raise ValueError("validation_steps must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_train_epochs is not None and self.max_train_epochs <= 0:
            raise ValueError("max_train_epochs must be positive when provided")
        if self.max_train_steps is not None and self.max_train_steps <= 0:
            raise ValueError("max_train_steps must be positive when provided")
