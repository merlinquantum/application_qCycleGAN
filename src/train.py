import os
import copy
from datetime import datetime
from dataclasses import asdict, is_dataclass
from pathlib import Path

import lpips
import torch
from glob import glob
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.optimization import get_scheduler
from cleanfid.fid import get_folder_features, build_feature_extractor, frechet_distance
import vision_aided_loss
from model import make_1step_sched
from models.cyclegan_turbo import CycleGAN_Turbo
from data import UnpairedDataset, build_transform, load_bdd_dataset
from config import ClassicalConfig
from my_utils.training_utils import parse_args_unpaired_training, get_next_id
from training import (
    CycleGANLosses,
    CycleGANTrainer,
    VisualizationCallback,
    ValidationCallback,
    ModelSaver,
    CheckpointCallback,
)
from models.quantum_encoder import BosonSampler
from models.model_factory import ModelFactory, TrainableParamsManager, RandomAblationSampler
from loguru import logger


def _to_jsonable_config(value):  # type: ignore[override]
    """Convert config payloads to tracker-friendly (JSON-serializable) structures."""
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
    if is_dataclass(value):
        return _to_jsonable_config(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable_config(v) for v in value]
    return str(value)


def main(args):
    base_quantum_cfg = args.quantum_config
    dataset_cfg = args.dataset_config
    training_cfg = args.training_config
    classical_cfg = getattr(args, "classical_config", ClassicalConfig())
    if classical_cfg.random_ablation and base_quantum_cfg.quantum:
        raise ValueError("random_ablation cannot be enabled when quantum mode is True.")
    quantum_cfg = copy.deepcopy(base_quantum_cfg)
    annotation_active = base_quantum_cfg.quantum or classical_cfg.random_ablation
    quantum_cfg.quantum = annotation_active
    set_seed(training_cfg.seed)
    sampler = None
    args.annotation_active = annotation_active
    args.quantum_runtime_config = quantum_cfg
    if annotation_active:
        logger.info("[Init] --- qCycleGAN training ---")
        logger.info(
            "[Quantum Setup] Defining the quantum encoder with dims={} and num_processes={}",
            dataset_cfg.quantum_dims,
            quantum_cfg.processes,
        )
        # Set the random seeds
        torch.manual_seed(training_cfg.seed)
        if classical_cfg.random_ablation:
            sampler = RandomAblationSampler(
                (quantum_cfg.processes,) + dataset_cfg.quantum_dims,
                hidden_dim=classical_cfg.random_ablation_hidden_dim,
            )
            logger.info("[Quantum Setup] Random ablation layer defined (output_dim=1024)")
        elif base_quantum_cfg.quantum:
            sampler = BosonSampler.from_config(
                (quantum_cfg.processes,) + dataset_cfg.quantum_dims,
                config=quantum_cfg,
            )
            logger.info("[Quantum Setup] Boson Sampler defined: {}", type(sampler).__name__)

    accelerator = Accelerator(
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        log_with=training_cfg.report_to,
    )
    device = accelerator.device
    device_str = device.type
    logger.info("Accelerator initialized: device={}, processes={}", device, accelerator.num_processes)

    # set up id and name of experiment
    exp_id = get_next_id()
    base_output_dir = training_cfg.output_dir
    if base_output_dir == 'a':
        base_output_dir = "/scratch/all_outputs"
    os.makedirs(base_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(base_output_dir, timestamp)
    training_cfg.output_dir = output_dir
    args.output_dir = output_dir
    args.exp_id = exp_id
    if accelerator.is_main_process:
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", revision=training_cfg.revision,
                                              use_fast=False, )
    noise_scheduler_1step = make_1step_sched(device=device)
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").to(device)

    weight_dtype = torch.float32
    if sampler is not None:
        grad_flag = (
            classical_cfg.random_trained if classical_cfg.random_ablation else quantum_cfg.quantum_trained
        )
        sampler = sampler.to(device)
        sampler.requires_grad_(grad_flag)
        logger.debug(
            "[Sampler] {} moved to device {}; requires_grad={}",
            type(sampler).__name__,
            device,
            grad_flag,
        )
    text_encoder.to(device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)

    if training_cfg.gan_disc_type == "vagan_clip":
        disc_device = device_str if device_str in ("cuda", "cpu", "mps") else "cpu"
        net_disc_a = vision_aided_loss.Discriminator(cv_type='clip', loss_type=training_cfg.gan_loss_type, device=disc_device)
        net_disc_a.cv_ensemble.requires_grad_(False)  # Freeze feature extractor
        net_disc_b = vision_aided_loss.Discriminator(cv_type='clip', loss_type=training_cfg.gan_loss_type, device=disc_device)
        net_disc_b.cv_ensemble.requires_grad_(False)  # Freeze feature extractor


    logger.info("[ModelFactory] Building components via ModelFactory")
    components = ModelFactory.build(accelerator, training_cfg, quantum_cfg, classical_cfg)
    # Freeze VAE encoder for quantum or classical-comparison runs to mimic quantum setup.
    if quantum_cfg.quantum or classical_cfg.cl_comp or classical_cfg.random_ablation:
        TrainableParamsManager.freeze_vae_encoder(components)
    params_gen = TrainableParamsManager.trainable_params(
        components,
        quantum_cfg,
        sampler,
        train_unet=training_cfg.unet_trained,
    )
    TrainableParamsManager.describe(components)
    unet = components.unet
    vae_enc = components.vae_enc
    vae_dec = components.vae_dec
    # Note: vae_a2b and vae_b2a are already inside vae_enc/vae_dec wrappers
    # Accessing them separately would create duplicate references in memory (CUDA OOM risk)
    l_modules_unet_encoder = components.lora_modules_encoder
    l_modules_unet_decoder = components.lora_modules_decoder
    l_modules_unet_others = components.lora_modules_others
    vae_lora_target_modules = components.vae_lora_target_modules
    cyclegan_d = components.cyclegan or CycleGAN_Turbo
    vae_a2b_ref = vae_enc.vae
    vae_b2a_ref = vae_enc.vae_b2a
    logger.info(
        "[ModelFactory] TOTAL parameters = {}",
        sum(p.numel() for p in unet.parameters())
        + sum(p.numel() for p in vae_a2b_ref.parameters())
        + sum(p.numel() for p in vae_b2a_ref.parameters()),
    )
    logger.info(
        "[ModelFactory] TOTAL trainable parameters = {}",
        sum(p.numel() for p in unet.parameters() if p.requires_grad) + sum(p.numel() for p in vae_a2b_ref.parameters() if p.requires_grad) + sum(p.numel() for p in vae_b2a_ref.parameters() if p.requires_grad),
    )
    if training_cfg.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()

    if training_cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if training_cfg.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # TEST
    # print(f"------------------------ After loading the weights, VAE conv in = {vae_enc.vae.encoder.conv_in.lora_A.vae_skip.weight}")
    optimizer_gen = torch.optim.AdamW(
        params_gen,
        lr=training_cfg.learning_rate,
        betas=(training_cfg.adam_beta1, training_cfg.adam_beta2),
        weight_decay=training_cfg.adam_weight_decay,
        eps=training_cfg.adam_epsilon,
    )

    params_disc = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
    optimizer_disc = torch.optim.AdamW(
        params_disc,
        lr=training_cfg.learning_rate,
        betas=(training_cfg.adam_beta1, training_cfg.adam_beta2),
        weight_decay=training_cfg.adam_weight_decay,
        eps=training_cfg.adam_epsilon,
    )

    _ = load_bdd_dataset(
        dataset_cfg.dataset_name,
        dataset_cfg.dataset_folder,
        dataset_cfg.prompt_src,
        dataset_cfg.prompt_tgt,
        cleanup=dataset_cfg.cleanup,
        train_fraction=dataset_cfg.training_images,
    )
    dataset_train = UnpairedDataset(
        config=dataset_cfg,
        split="train",
        tokenizer=tokenizer,
    )
    #here, we could lower the number of images used
    logger.info("[Data Loading] Dataset loaded")

    logger.info("[Data Loading] Length of dataset_train = {}", len(dataset_train))
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=dataset_cfg.train_batch_size,
        shuffle=True,
        num_workers=dataset_cfg.dataloader_num_workers,
    )
    max_train_steps = training_cfg.max_train_steps or len(train_dataloader)
    logger.info("[Data Loading] DataLoader prepared")
    T_val = build_transform(dataset_cfg.val_img_prep)
    fixed_caption_src = dataset_train.fixed_caption_src
    fixed_caption_tgt = dataset_train.fixed_caption_tgt
    l_images_src_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_src_test.extend(glob(os.path.join(dataset_cfg.dataset_folder, "test_A", ext)))
    l_images_src_test_2 = [el for el in l_images_src_test if os.path.basename(el)]# not in list_failures]
    l_images_src_test = l_images_src_test_2

    l_images_tgt_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_tgt_test.extend(glob(os.path.join(dataset_cfg.dataset_folder, "test_B", ext)))

    l_images_src_test, l_images_tgt_test = sorted(l_images_src_test), sorted(l_images_tgt_test)
    #l_images_src_test, l_images_tgt_test = l_images_src_test[:100], l_images_tgt_test[:100]
    logger.info(
        "[Validation Prep] Working with {} src and {} tgt test images",
        len(l_images_src_test),
        len(l_images_tgt_test),
    )
    logger.debug("UNet conv_in: {}", unet.conv_in)
    # make the reference FID statistics
    logger.info("[Validation Prep] Building reference FID statistics")
    fid_device = device_str if device_str in ("cuda", "cpu") else "cpu"
    feat_model = None
    a2b_ref_mu = None
    a2b_ref_sigma = None
    b2a_ref_mu = None
    b2a_ref_sigma = None
    if accelerator.is_main_process:
        feat_model = build_feature_extractor("clean", fid_device, use_dataparallel=False)
        """
        FID reference statistics for A -> B translation
        """
        output_dir_ref = os.path.join(output_dir, "fid_reference_a2b")
        os.makedirs(output_dir_ref, exist_ok=True)
        # transform all images according to the validation transform and save them
        for _path in tqdm(l_images_tgt_test):
            _img = T_val(Image.open(_path).convert("RGB"))
            outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(".jpg", ".png")
            if not os.path.exists(outf):
                _img.save(outf)
        # compute the features for the reference images
        ref_features = get_folder_features(output_dir_ref, model=feat_model, num_workers=0, num=None,
                                           shuffle=False, seed=0, batch_size=8, device=torch.device(fid_device),
                                           mode="clean", custom_fn_resize=None, description="", verbose=True,
                                           custom_image_tranform=None)
        a2b_ref_mu, a2b_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)
        """
        FID reference statistics for B -> A translation
        """
        # transform all images according to the validation transform and save them
        output_dir_ref = os.path.join(output_dir, "fid_reference_b2a")
        os.makedirs(output_dir_ref, exist_ok=True)
        for _path in tqdm(l_images_src_test):
            _img = T_val(Image.open(_path).convert("RGB"))
            outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(".jpg", ".png")
            if not os.path.exists(outf):
                _img.save(outf)
        # compute the features for the reference images
        ref_features = get_folder_features(output_dir_ref, model=feat_model, num_workers=0, num=None,
                                           shuffle=False, seed=0, batch_size=8, device=torch.device(fid_device),
                                           mode="clean", custom_fn_resize=None, description="", verbose=True,
                                           custom_image_tranform=None)
        b2a_ref_mu, b2a_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)

    logger.info("[Optimization] Defining schedulers for generator and discriminator")
    lr_scheduler_gen = get_scheduler(
        training_cfg.lr_scheduler,
        optimizer=optimizer_gen,
        num_warmup_steps=training_cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=training_cfg.lr_num_cycles,
        power=training_cfg.lr_power,
    )
    lr_scheduler_disc = get_scheduler(
        training_cfg.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=training_cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=training_cfg.lr_num_cycles,
        power=training_cfg.lr_power,
    )

    net_lpips = lpips.LPIPS(net='vgg').to(device)
    net_lpips.requires_grad_(False)

    logger.info("[Embeddings] Defining fixed text embeddings")
    fixed_a2b_tokens = \
    tokenizer(fixed_caption_tgt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
              return_tensors="pt").input_ids[0].to(device)
    fixed_a2b_emb_base = text_encoder(fixed_a2b_tokens.unsqueeze(0))[0].detach()
    fixed_b2a_tokens = \
    tokenizer(fixed_caption_src, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
              return_tensors="pt").input_ids[0].to(device)
    fixed_b2a_emb_base = text_encoder(fixed_b2a_tokens.unsqueeze(0))[0].detach()
    del text_encoder, tokenizer  # free up some memory

    unet, vae_enc, vae_dec, net_disc_a, net_disc_b = accelerator.prepare(unet, vae_enc, vae_dec, net_disc_a, net_disc_b)
    net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc = accelerator.prepare(
        net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc
    )
    loss_helper = CycleGANLosses(training_cfg, lpips_module=net_lpips)
    if accelerator.is_main_process:
        config = _to_jsonable_config(dict(vars(args)))
        config["run_name"] = f"qCycleGAN-{exp_id}"
        accelerator.init_trackers(training_cfg.tracker_project_name, config=config)
        tracker_names = [getattr(tracker, "name", type(tracker).__name__) for tracker in accelerator.trackers]
        logger.info("[Tracking] Initialized trackers: {}", tracker_names if tracker_names else "none")
    # checking if the VAE_encoder is frozen on the pretrained weights
    logger.info(
        "[ModelFactory] Trainable params -> vae_enc={}, a2b={}, b2a={}",
        sum(p.numel() for p in vae_enc.parameters() if p.requires_grad),
        sum(p.numel() for p in vae_a2b_ref.encoder.parameters() if p.requires_grad),
        sum(p.numel() for p in vae_b2a_ref.encoder.parameters() if p.requires_grad),
    )
    # turn off eff. attn for the disc
    for name, module in net_disc_a.named_modules():
        if "attn" in name:
            module.fused_attn = False
    for name, module in net_disc_b.named_modules():
        if "attn" in name:
            module.fused_attn = False

    model_saver = ModelSaver(
        output_dir=output_dir,
        lora_modules_encoder=l_modules_unet_encoder,
        lora_modules_decoder=l_modules_unet_decoder,
        lora_modules_others=l_modules_unet_others,
        lora_rank_unet=training_cfg.lora_rank_unet,
        vae_lora_target_modules=vae_lora_target_modules,
        lora_rank_vae=training_cfg.lora_rank_vae,
        quantum_enabled=quantum_cfg.quantum,
        boson_sampler=sampler,
        enable_step_checkpoints=training_cfg.save_step_checkpoints,
    )

    callbacks = [
        VisualizationCallback(frequency=training_cfg.viz_freq),
        ValidationCallback(
            frequency=training_cfg.validation_steps,
            output_dir=output_dir,
            validation_transform=T_val,
            src_image_paths=l_images_src_test,
            tgt_image_paths=l_images_tgt_test,
            feat_model=feat_model,
            fid_device=fid_device,
            a2b_ref_stats=(a2b_ref_mu, a2b_ref_sigma),
            b2a_ref_stats=(b2a_ref_mu, b2a_ref_sigma),
            validation_num_images=training_cfg.validation_num_images,
            model_saver=model_saver,
        ),
    ]
    if training_cfg.save_step_checkpoints and training_cfg.checkpointing_steps > 0:
        callbacks.append(
            CheckpointCallback(
                frequency=training_cfg.checkpointing_steps,
                output_dir=output_dir,
                max_train_steps=max_train_steps,
                lora_modules_encoder=l_modules_unet_encoder,
                lora_modules_decoder=l_modules_unet_decoder,
                lora_modules_others=l_modules_unet_others,
                lora_rank_unet=training_cfg.lora_rank_unet,
                vae_lora_target_modules=vae_lora_target_modules,
                lora_rank_vae=training_cfg.lora_rank_vae,
                quantum_enabled=quantum_cfg.quantum,
                boson_sampler=sampler,
            )
        )

    trainer = CycleGANTrainer(
        args=args,
        training_config=training_cfg,
        accelerator=accelerator,
        loss_helper=loss_helper,
        train_dataloader=train_dataloader,
        net_disc_a=net_disc_a,
        net_disc_b=net_disc_b,
        unet=unet,
        vae_enc=vae_enc,
        vae_dec=vae_dec,
        cyclegan_d=cyclegan_d,
        noise_scheduler=noise_scheduler_1step,
        fixed_a2b_emb_base=fixed_a2b_emb_base,
        fixed_b2a_emb_base=fixed_b2a_emb_base,
        optimizer_gen=optimizer_gen,
        optimizer_disc=optimizer_disc,
        lr_scheduler_gen=lr_scheduler_gen,
        lr_scheduler_disc=lr_scheduler_disc,
        params_gen=params_gen,
        boson_sampler=sampler,
        callbacks=callbacks,
        weight_dtype=weight_dtype,
    )
    trainer.train()



if __name__ == "__main__":
    args = parse_args_unpaired_training()
    if getattr(args, "experiment_config", None):
        logger.info("[Config] Using experiment config: {}", args.experiment_config)
    main(args)
