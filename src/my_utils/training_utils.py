import os
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms.functional as F
from config import ClassicalConfig, DatasetConfig, QuantumConfig, TrainingConfig, load_configs
from data import (
    UnpairedDataset,
    UnpairedDataset_Quantum,
    build_transform,
    image_fail,
    read_from_emb16,
    read_from_emb32,
)


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def parse_args_paired_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=5, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_clipsim", default=5.0, type=float)

    # dataset options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo", help="The name of the wandb project to log to.")

    # details about the model architecture
    parser.add_argument("--pretrained_model_path", default=None)
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=128, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # training details
    parser.add_argument("--output_dir", default = None)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def parse_args_unpaired_training(input_args=None):
    """
    Parses command-line arguments used for configuring an unpaired session (CycleGAN-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """

    def _normalize_quantum_dims(value):
        if isinstance(value, str):
            stripped = value.strip().strip("()[]")
            parts = [part.strip() for part in stripped.split(",") if part.strip()]
            return tuple(int(part) for part in parts)
        return tuple(value)

    def _defaults_from_configs(bundle):
        quantum_cfg, dataset_cfg, training_cfg, classical_cfg = bundle
        defaults = {
            "seed": training_cfg.seed,
            "gan_disc_type": training_cfg.gan_disc_type,
            "gan_loss_type": training_cfg.gan_loss_type,
            "lambda_gan": training_cfg.lambda_gan,
            "lambda_idt": training_cfg.lambda_idt,
            "lambda_cycle": training_cfg.lambda_cycle,
            "lambda_cycle_lpips": training_cfg.lambda_cycle_lpips,
            "lambda_idt_lpips": training_cfg.lambda_idt_lpips,
            "dataset": dataset_cfg.dataset_name,
            "dataset_folder": dataset_cfg.dataset_folder,
            "train_img_prep": dataset_cfg.train_img_prep,
            "val_img_prep": dataset_cfg.val_img_prep,
            "train_batch_size": dataset_cfg.train_batch_size,
            "dataloader_num_workers": dataset_cfg.dataloader_num_workers,
            "quantum_dims": tuple(dataset_cfg.quantum_dims),
            "prompt_src": dataset_cfg.prompt_src,
            "prompt_tgt": dataset_cfg.prompt_tgt,
            "training_images": dataset_cfg.training_images,
            "cleanup_dataset": dataset_cfg.cleanup,
            "max_train_epochs": training_cfg.max_train_epochs,
            "max_train_steps": training_cfg.max_train_steps,
            "hf_model_path": training_cfg.hf_model_path,
            "revision": training_cfg.revision,
            "variant": training_cfg.variant,
            "lora_rank_unet": training_cfg.lora_rank_unet,
            "lora_rank_vae": training_cfg.lora_rank_vae,
            "unet_trained": training_cfg.unet_trained,
            "viz_freq": training_cfg.viz_freq,
            "output_dir": training_cfg.output_dir,
            "report_to": training_cfg.report_to,
            "tracker_project_name": training_cfg.tracker_project_name,
            "validation_steps": training_cfg.validation_steps,
            "validation_num_images": training_cfg.validation_num_images,
            "checkpointing_steps": training_cfg.checkpointing_steps,
            "save_step_checkpoints": training_cfg.save_step_checkpoints,
            "learning_rate": training_cfg.learning_rate,
            "adam_beta1": training_cfg.adam_beta1,
            "adam_beta2": training_cfg.adam_beta2,
            "adam_weight_decay": training_cfg.adam_weight_decay,
            "adam_epsilon": training_cfg.adam_epsilon,
            "max_grad_norm": training_cfg.max_grad_norm,
            "lr_scheduler": training_cfg.lr_scheduler,
            "lr_warmup_steps": training_cfg.lr_warmup_steps,
            "lr_num_cycles": training_cfg.lr_num_cycles,
            "lr_power": training_cfg.lr_power,
            "gradient_accumulation_steps": training_cfg.gradient_accumulation_steps,
            "allow_tf32": training_cfg.allow_tf32,
            "gradient_checkpointing": training_cfg.gradient_checkpointing,
            "enable_xformers_memory_efficient_attention": training_cfg.enable_xformers_memory_efficient_attention,
            "quantum": quantum_cfg.quantum,
            "cl_comp": classical_cfg.cl_comp,
            "quantum_start_path": quantum_cfg.start_path,
            "quantum_processes": quantum_cfg.processes,
            "random_ablation": classical_cfg.random_ablation,
            "random_trained": classical_cfg.random_trained,
            "random_ablation_hidden_dim": classical_cfg.random_ablation_hidden_dim,
            "use_haar_unitary": quantum_cfg.use_haar_unitary,
        }
        return defaults

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--experiment_config",
        type=str,
        default="config/experiments/default.yaml",
        help="Path to a YAML/JSON config file with quantum/dataset/training sections.",
    )
    config_parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Legacy flat JSON config file. Overrides --experiment_config when provided.",
    )
    config_args, _ = config_parser.parse_known_args()

    config_bundle = None
    legacy_overrides = {}
    config_path = config_args.config_file or config_args.experiment_config
    if config_path:
        path = Path(config_path)
        if path.suffix in {".yaml", ".yml"}:
            config_bundle = load_configs(path)
        else:
            with open(path, "r", encoding="utf-8") as cfg_fp:
                raw = json.load(cfg_fp)
            if all(key in raw for key in ("quantum", "dataset", "training")):
                quantum_section = dict(raw.get("quantum", {}))
                if "dynamic" in quantum_section and "quantum" not in quantum_section:
                    quantum_section["quantum"] = quantum_section.pop("dynamic")
                config_bundle = (
                    QuantumConfig(**quantum_section),
                    DatasetConfig(**raw.get("dataset", {})),
                    TrainingConfig(**raw.get("training", {})),
                    ClassicalConfig(**raw.get("classical", {})),
                )
                for cfg in config_bundle:
                    cfg.validate()
            else:
                legacy_overrides = raw

    parser = argparse.ArgumentParser(
        description="Simple example of a ControlNet training script.",
        parents=[config_parser],
    )

    # fixed random seed
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_idt", default=1, type=float)
    parser.add_argument("--lambda_cycle", default=1, type=float)
    parser.add_argument("--lambda_cycle_lpips", default=10.0, type=float)
    parser.add_argument("--lambda_idt_lpips", default=1.0, type=float)

    # args for dataset and dataloader options
    parser.add_argument("--dataset", type=str, default="quandelagl/bdd-100k",
                        help="Hugging Face dataset repo id to download when preparing the dataset folder.")
    parser.add_argument("--dataset_folder", type=str, default=None)
    parser.add_argument("--train_img_prep", default=None)
    parser.add_argument("--val_img_prep", default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--max_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)

    # args for the model
    parser.add_argument("--hf_model_path", default="model_251.pkl")
    parser.add_argument("--revision", default=None, type=str)
    parser.add_argument("--variant", default=None, type=str)
    parser.add_argument("--lora_rank_unet", default=128, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument(
        "--unet_trained",
        type=_str_to_bool,
        default=False,
        help="Train the entire UNet (True) or only the LoRA adapters/conv_in layers (False).",
    )

    # args for validation and logging
    parser.add_argument("--viz_freq", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default = 'a' )#, required=True)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--tracker_project_name", type=str, default=None)
    parser.add_argument("--validation_steps", type=int, default=500,)
    parser.add_argument("--validation_num_images", type=int, default=-1, help="Number of images to use for validation. -1 to use all images.")
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument(
        "--save_step_checkpoints",
        type=_str_to_bool,
        default=False,
        help="Save step-based checkpoints (model_<step>.pkl) every --checkpointing_steps.",
    )

    # args for the optimization options
    parser.add_argument("--learning_rate", type=float, default=5e-6,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.",)
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # memory saving options
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--gradient_checkpointing", action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")


    # dynamic conditional quantum embeddings with the VAE frozen
    parser.add_argument(
        "--quantum",
        type=_str_to_bool,
        default=False,
        help="Enable quantum embeddings and use the Boson sampler during training.",
    )
    parser.add_argument(
        "--quantum_trained",
        type=_str_to_bool,
        default=False,
        help="Enable gradient updates for the Boson sampler parameters.",
    )
    parser.add_argument(
        "--random_ablation",
        type=_str_to_bool,
        default=False,
        help="Replace the Boson sampler with a random linear projection for ablation studies.",
    )
    parser.add_argument(
        "--random_trained",
        type=_str_to_bool,
        default=False,
        help="Train the random ablation layer when --random_ablation is enabled.",
    )
    parser.add_argument(
        "--random_ablation_hidden_dim",
        type=int,
        default=2048,
        help="Hidden dimension size for the random ablation MLP.",
    )
    parser.add_argument(
        "--quantum_dynamic",
        dest="quantum",
        type=_str_to_bool,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--cl_comp", type=bool, default=False,
                        help="Define if we use an initialized classical model for experiment comparison")
    parser.add_argument("--quantum_start_path", type=str,
                        help="Path to pretrained VAE encoder")
    parser.add_argument("--quantum_dims", type=tuple, default=(4, 16, 16), help="Dimensions of the quantum encoder")
    parser.add_argument("--quantum_processes", type=int, default=4,
                        help="Number of threads to use for the Boson Sampler")
    parser.add_argument(
        "--use_haar_unitary",
        action="store_true",
        default=False,
        help="Use a Haar-random unitary for the boson sampler circuit.",
    )
    parser.add_argument("--training_images", type = float, default = 1., help="Part of the training images to be used")
    parser.add_argument("--prompt_src", type=str, default="Driving in the night",
                        help="Fixed text prompt describing images in domain A/train_A.")
    parser.add_argument("--prompt_tgt", type=str, default="Driving in the day",
                        help="Fixed text prompt describing images in domain B/train_B.")
    parser.add_argument("--cleanup_dataset", action="store_true",
                        help="Delete downloaded dataset after training completes.")

    if config_bundle:
        parser.set_defaults(**_defaults_from_configs(config_bundle))
    if legacy_overrides:
        if "quantum_dynamic" in legacy_overrides and "quantum" not in legacy_overrides:
            legacy_overrides["quantum"] = legacy_overrides.pop("quantum_dynamic")
        if "quantum_dims" in legacy_overrides and isinstance(legacy_overrides["quantum_dims"], list):
            legacy_overrides["quantum_dims"] = tuple(legacy_overrides["quantum_dims"])
        parser.set_defaults(**legacy_overrides)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    args.experiment_config = args.config_file or args.experiment_config
    args.quantum_dynamic = args.quantum

    # normalize types
    if isinstance(args.quantum_dims, list):
        args.quantum_dims = tuple(args.quantum_dims)
    elif not isinstance(args.quantum_dims, tuple):
        args.quantum_dims = _normalize_quantum_dims(args.quantum_dims)

    required_fields = ["dataset_folder", "train_img_prep", "val_img_prep", "tracker_project_name"]
    missing_fields = [field for field in required_fields if getattr(args, field) in (None, "")]
    if missing_fields:
        parser.error(f"Missing required arguments: {', '.join(missing_fields)}. Provide them via CLI or the config file.")

    if config_bundle:
        base_quantum, base_dataset, base_training, base_classical = config_bundle
    else:
        base_quantum, base_dataset, base_training, base_classical = (
            QuantumConfig(),
            DatasetConfig(),
            TrainingConfig(),
            ClassicalConfig(),
        )

    quantum_config = QuantumConfig(
        num_modes=base_quantum.num_modes,
        num_photons=base_quantum.num_photons,
        epsilon=base_quantum.epsilon,
        computation_space=base_quantum.computation_space,
        trainable_parameters=list(base_quantum.trainable_parameters),
        quantum=args.quantum,
        quantum_trained=args.quantum_trained,
        start_path=args.quantum_start_path,
        processes=args.quantum_processes,
        sort_encoding=base_quantum.sort_encoding,
        use_haar_unitary=args.use_haar_unitary,
    )
    classical_config = ClassicalConfig(
        cl_comp=args.cl_comp,
        random_ablation=args.random_ablation,
        random_trained=args.random_trained,
        random_ablation_hidden_dim=args.random_ablation_hidden_dim,
    )
    dataset_config = DatasetConfig(
        dataset_name=args.dataset,
        dataset_folder=args.dataset_folder,
        train_img_prep=args.train_img_prep,
        val_img_prep=args.val_img_prep,
        train_batch_size=args.train_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        quantum_dims=tuple(args.quantum_dims),
        prompt_src=args.prompt_src,
        prompt_tgt=args.prompt_tgt,
        training_images=args.training_images,
        cleanup=args.cleanup_dataset,
    )
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        max_train_epochs=args.max_train_epochs,
        max_train_steps=args.max_train_steps,
        validation_steps=args.validation_steps,
        lambda_gan=args.lambda_gan,
        lambda_idt=args.lambda_idt,
        lambda_cycle=args.lambda_cycle,
        lambda_cycle_lpips=args.lambda_cycle_lpips,
        lambda_idt_lpips=args.lambda_idt_lpips,
        gan_disc_type=args.gan_disc_type,
        gan_loss_type=args.gan_loss_type,
        hf_model_path=args.hf_model_path,
        revision=args.revision,
        variant=args.variant,
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        viz_freq=args.viz_freq,
        output_dir=args.output_dir,
        report_to=args.report_to,
        tracker_project_name=args.tracker_project_name,
        validation_num_images=args.validation_num_images,
        checkpointing_steps=args.checkpointing_steps,
        save_step_checkpoints=args.save_step_checkpoints,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_weight_decay=args.adam_weight_decay,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_num_cycles=args.lr_num_cycles,
        lr_power=args.lr_power,
        seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        allow_tf32=args.allow_tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        enable_xformers_memory_efficient_attention=args.enable_xformers_memory_efficient_attention,
        unet_trained=args.unet_trained,
    )
    quantum_config.validate()
    dataset_config.validate()
    training_config.validate()
    classical_config.validate()
    args.quantum_config = quantum_config
    args.dataset_config = dataset_config
    args.training_config = training_config
    args.classical_config = classical_config

    return args


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        Itialize the paired dataset object for loading and transforming paired data samples
        from specified dataset folders.

        This constructor sets up the paths to input and output folders based on the specified 'split',
        loads the captions (or prompts) for the input images, and prepares the transformations and
        tokenizer to be applied on the data.

        Parameters:
        - dataset_folder (str): The root folder containing the dataset, expected to include
                                sub-folders for different splits (e.g., 'train_A', 'train_B').
        - split (str): The dataset split to use ('train' or 'test'), used to select the appropriate
                       sub-folders and caption files within the dataset folder.
        - image_prep (str): The image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        if split == "train":
            self.input_folder = os.path.join(dataset_folder, "train_A")
            self.output_folder = os.path.join(dataset_folder, "train_B")
            captions = os.path.join(dataset_folder, "train_prompts.json")
        elif split == "test":
            self.input_folder = os.path.join(dataset_folder, "test_A")
            self.output_folder = os.path.join(dataset_folder, "test_B")
            captions = os.path.join(dataset_folder, "test_prompts.json")
        with open(captions, "r") as f:
            self.captions = json.load(f)
        self.img_names = list(self.captions.keys())
        self.T = build_transform(image_prep)
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.captions)

    def __getitem__(self, idx):
        """
        Retrieves a dataset item given its index. Each item consists of an input image, 
        its corresponding output image, the captions associated with the input image, 
        and the tokenized form of this caption.

        This method performs the necessary preprocessing on both the input and output images, 
        including scaling and normalization, as well as tokenizing the caption using a provided tokenizer.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        dict: A dictionary containing the following key-value pairs:
            - "output_pixel_values": a tensor of the preprocessed output image with pixel values 
            scaled to [-1, 1].
            - "conditioning_pixel_values": a tensor of the preprocessed input image with pixel values 
            scaled to [0, 1].
            - "caption": the text caption.
            - "input_ids": a tensor of the tokenized caption.

        Note:
        The actual preprocessing steps (scaling and normalization) for images are defined externally 
        and passed to this class through the `image_prep` parameter during initialization. The 
        tokenization process relies on the `tokenizer` also provided at initialization, which 
        should be compatible with the models intended to be used with this dataset.
        """
        img_name = self.img_names[idx]
        input_img = Image.open(os.path.join(self.input_folder, img_name))
        output_img = Image.open(os.path.join(self.output_folder, img_name))
        caption = self.captions[img_name]

        # input images scaled to 0,1
        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t)
        # output images scaled to -1,1
        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        input_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "input_ids": input_ids,
        }


def get_next_id(filename="id_store.txt"):
    try:
        with open(filename, "r") as f:
            last_id = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        last_id = 0

    next_id = last_id + 1

    with open(filename, "w") as f:
        f.write(str(next_id))

    return next_id

# tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", revision=None, use_fast=False,)
# dataset_train = UnpairedDataset_Quantum(dataset_folder="../data/dataset_full_scale/", image_prep="resize_128", split="train", tokenizer=tokenizer, 
#                                 q_emb_path = "/home/jupyter-pemeriau/q_embs/emb_128_dims_16_16_ckpt1001", 
#                                 output_dir = "/home/jupyter-pemeriau/q_embs/emb_128_dims_16_16_ckpt1001" )

# for k in range(len(dataset_train)):
#     a = dataset_train[k]
