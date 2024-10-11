import argparse
import copy
import logging
import math
import os
import os.path as osp
import pathlib
import random
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import diffusers
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from einops import rearrange
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from diffusers.utils.import_utils import is_torch_npu_available

from datasets import TalkingFaceVideo
from modules import UNet2DConditionModel, UNet3DConditionModel, VKpsGuider, AudioProjection, ReferenceAttentionControl
from utils import seed_everything
from pipelines.utils import zero_module

warnings.filterwarnings("ignore")

logger = get_logger(__name__, log_level="INFO")


class Net(nn.Module):
    def __init__(
            self,
            reference_net: UNet2DConditionModel,
            denoising_unet: UNet3DConditionModel,
            v_kps_guider: VKpsGuider,
            audio_projection: AudioProjection,
            reference_control_writer: ReferenceAttentionControl,
            reference_control_reader: ReferenceAttentionControl,
            device,
            weight_dtype,
            kps_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.reference_net = reference_net
        self.denoising_unet = denoising_unet
        self.v_kps_guider = v_kps_guider
        self.audio_projection = audio_projection
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.device = device
        self.weight_dtype = weight_dtype
        self.kps_drop_rate = kps_drop_rate

    def forward(
            self,
            noisy_latents,
            timesteps,
            reference_image_latents,
            audio_frame_embeddings,
            kps_images,
            do_unconditional_forward: bool = False,
    ):
        kps_features = self.v_kps_guider(kps_images)
        if do_unconditional_forward:
            kps_features = torch.zeros_like(kps_features)
        elif self.kps_drop_rate != 0.0:
            drop_mask = torch.rand(kps_features.shape[0]) < self.kps_drop_rate
            kps_features[drop_mask, ...] = 0.

        batch_size, num_frames, num_embeds, dim = audio_frame_embeddings.shape
        audio_frame_embeddings = audio_frame_embeddings.reshape(-1, num_embeds, dim)
        audio_frame_embeddings = self.audio_projection(audio_frame_embeddings)
        _, num_embeds, dim = audio_frame_embeddings.shape
        audio_frame_embeddings = audio_frame_embeddings.reshape(batch_size, num_frames, num_embeds, dim)
        if do_unconditional_forward:
            audio_frame_embeddings = torch.zeros_like(audio_frame_embeddings)

        ref_timesteps = torch.zeros_like(timesteps)
        ref_encoder_hidden_states = torch.zeros(
            (batch_size, 1, 768),
            dtype=kps_features.dtype,
            device=kps_features.device,
        )
        self.reference_net(
            reference_image_latents,
            ref_timesteps,
            encoder_hidden_states=ref_encoder_hidden_states,
            return_dict=False,
        )

        self.reference_control_reader.update(
            self.reference_control_writer,
            do_classifier_free_guidance=False,
            do_unconditional_forward=do_unconditional_forward,
            dtype=self.weight_dtype,
        )

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            kps_features=kps_features,
            encoder_hidden_states=audio_frame_embeddings.reshape(-1, num_embeds, dim),
        ).sample

        return model_pred


def get_denoising_unet_state_dict(old_state_dict, state_dict_type):
    new_denoising_unet_state_dict = copy.deepcopy(old_state_dict)
    if state_dict_type == 'old_attn':
        for name in old_state_dict.keys():
            if 'norm1' in name:
                new_denoising_unet_state_dict[name.replace('norm1', 'norm1_5')] = old_state_dict[name]
            if 'attn1' in name:
                new_denoising_unet_state_dict[name.replace('attn1', 'attn1_5')] = old_state_dict[name]
            if 'attn2.to_q' in name:
                new_denoising_unet_state_dict[name] = old_state_dict[name.replace(
                    'attn2.to_q',
                    'attn2.processor.to_q_aud'
                )]
            if 'attn2.to_k' in name:
                new_denoising_unet_state_dict[name] = old_state_dict[name.replace(
                    'attn2.to_k',
                    'attn2.processor.to_k_aud'
                )]
            if 'attn2.to_v' in name:
                new_denoising_unet_state_dict[name] = old_state_dict[name.replace(
                    'attn2.to_v',
                    'attn2.processor.to_v_aud'
                )]
            if 'attn2.to_out' in name:
                new_denoising_unet_state_dict[name] = old_state_dict[name.replace(
                    'attn2.to_out',
                    'attn2.processor.to_out_aud'
                )]
    elif state_dict_type == 'moore_pretrained':
        for name in old_state_dict.keys():
            if 'norm1' in name:
                new_denoising_unet_state_dict[name.replace('norm1', 'norm1_5')] = old_state_dict[name]
            if 'attn1' in name:
                new_denoising_unet_state_dict[name.replace('attn1', 'attn1_5')] = old_state_dict[name]
    elif state_dict_type == 'new_attn':
        pass
    else:
        raise ValueError(f'The state_dict_type {state_dict_type} is not supported. '
                         f'Only support "moore_pretrained", "old_attn", and "new_attn".')
    return new_denoising_unet_state_dict


def get_module_params(module):
    params = list(module.parameters())
    num_params = sum(p.numel() for p in params) / 1e6

    params_trainable = list(filter(lambda p: p.requires_grad, module.parameters()))
    num_trainable_params = sum(p.numel() for p in params_trainable) / 1e6

    return num_params, num_trainable_params


def count_params(net: Net):
    num_params, num_trainable_params = get_module_params(net.reference_net)
    logger.info(f"#parameters of ReferenceNet is {num_params:.3f} M ({num_trainable_params:.3f} M is trainable).")

    num_params, num_trainable_params = get_module_params(net.denoising_unet)
    logger.info(f"#parameters of Denoising U-Net is {num_params:.3f} M ({num_trainable_params:.3f} M is trainable).")

    num_params, num_trainable_params = get_module_params(net.v_kps_guider)
    logger.info(f"#parameters of V-Kps Guider is {num_params:.3f} M ({num_trainable_params:.3f} M is trainable).")

    num_params, num_trainable_params = get_module_params(net.audio_projection)
    logger.info(f"#parameters of Audio Projection is {num_params:.3f} M ({num_trainable_params:.3f} M is trainable).")


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/training/stage2.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    exp_name = '.'.join(Path(args.config).name.split('.')[:-1])

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    local_rank = accelerator.device

    save_dir = f"{cfg.output_dir}/{exp_name}"
    if accelerator.is_main_process:
        pathlib.Path(save_dir).mkdir(exist_ok=True, parents=True)
        os.makedirs(f"{save_dir}/log", exist_ok=True)
        # copy config yaml
        # print(cfg)
        OmegaConf.save(cfg, f"{save_dir}/log/config.yaml")

    inference_config_path = "./inference_v2.yaml"
    inference_config = OmegaConf.load(inference_config_path)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    elif cfg.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        raise ValueError(f"Do not support weight dtype: {cfg.weight_dtype} during training")

    # initialize the noise scheduler
    scheduler_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        scheduler_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    noise_scheduler = DDIMScheduler(**scheduler_kwargs)

    # initialize the pretrained fixed modules
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(device=local_rank, dtype=weight_dtype)
    audio_encoder = Wav2Vec2Model.from_pretrained(cfg.audio_encoder_path).to(dtype=weight_dtype, device=local_rank)
    # audio_processor = Wav2Vec2Processor.from_pretrained(cfg.audio_encoder_path)

    # initialize our modules
    reference_net = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device=local_rank, dtype=weight_dtype)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        cfg.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs),
    ).to(device=local_rank, dtype=weight_dtype)

    v_kps_guider = VKpsGuider(
        conditioning_embedding_channels=320,
        block_out_channels=(16, 32, 96, 256),
    ).to(device=local_rank, dtype=weight_dtype)

    if cfg.data.audio_embeddings_type == 'global':
        inp_dim = 768
    else:
        raise ValueError(f'Do not support {cfg.data.audio_embeddings_type}. '
                         f'Now only support "global".')
    mid_dim = denoising_unet.config.cross_attention_dim
    out_dim = denoising_unet.config.cross_attention_dim
    inp_seq_len = 2 * (2 * cfg.data.num_padding_audio_frames + 1)
    out_seq_len = 2 * cfg.data.num_padding_audio_frames + 1
    audio_projection = AudioProjection(
        dim=mid_dim,
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=out_seq_len,
        embedding_dim=inp_dim,
        output_dim=out_dim,
        ff_mult=4,
        max_seq_len=inp_seq_len,
    ).to(device=local_rank, dtype=weight_dtype)

    # load parameters to our modules
    if cfg.reference_net_path != '':
        info = reference_net.load_state_dict(torch.load(cfg.reference_net_path, map_location="cpu"), strict=False)
        logger.info(
            f"Loaded ReferenceNet from {cfg.reference_net_path}. Info: {info}"
        )

    if cfg.v_kps_guider_path != '':
        info = v_kps_guider.load_state_dict(torch.load(cfg.v_kps_guider_path, map_location="cpu"))
        logger.info(
            f"Loaded VKpsGuider from {cfg.v_kps_guider_path}. Info: {info}"
        )

    if cfg.audio_projection_path != '':
        info = audio_projection.load_state_dict(torch.load(cfg.audio_projection_path, map_location="cpu"))
        logger.info(
            f"Loaded AudioProjection from {cfg.audio_projection_path}. Info: {info}"
        )

    if cfg.denoising_unet_path != '':
        state_dict = torch.load(cfg.denoising_unet_path, map_location="cpu")
        new_state_dict = get_denoising_unet_state_dict(state_dict, cfg.denoising_unet_state_dict_type)
        info = denoising_unet.load_state_dict(new_state_dict, strict=False)
        logger.info(
            f"Loaded Denoising U-Net from {cfg.denoising_unet_path} in type of {cfg.denoising_unet_state_dict_type}. "
            f"Info: {info}"
        )

    if cfg.motion_module_path != '':
        state_dict = torch.load(cfg.motion_module_path, map_location="cpu")
        m, u = denoising_unet.load_state_dict(state_dict, strict=False)
        logger.info(
            f"Loaded Motion Module from {cfg.motion_module_path}. "
            f"Info: ### missing keys: {len(m)}; ### unexpected keys: {len(u)};"
        )

    if cfg.train_stage == 'stage_1':
        for name, params in denoising_unet.named_parameters():
            if 'temporal_transformer.proj_out' in name:
                zero_module(params)
                # logger.info(name)
            if 'attn2.to_out' in name:
                zero_module(params)
                # logger.info(name)
    elif cfg.train_stage == 'stage_2':
        for name, params in denoising_unet.named_parameters():
            if 'temporal_transformer.proj_out' in name:
                zero_module(params)
                # logger.info(name)
            if 'attn2.to_out' in name:
                zero_module(params)
                # logger.info(name)
    elif cfg.train_stage == 'stage_2_resume':
        pass
    elif cfg.train_stage == 'stage_3':
        for name, params in denoising_unet.named_parameters():
            if 'temporal_transformer.proj_out' in name:
                zero_module(params)
                # logger.info(name)
            if 'attn2.to_out' in name:
                zero_module(params)
                # logger.info(name)
    else:
        raise NotImplementedError(f"{cfg.train_stage} not implement")

    if accelerator.is_main_process:
        print(f'#############')
        for k, v in denoising_unet.named_parameters():
            if 'temporal_transformer.proj_out' in k:
                print(k, torch.sum(v))
            if 'attn2.to_out' in k:
                print(k, torch.sum(v))

    # set gradient state of all modules
    vae.requires_grad_(False)
    audio_encoder.requires_grad_(False)

    for name, param in reference_net.named_parameters():
        if "up_blocks.3" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(cfg.module_training.reference_net)
    denoising_unet.requires_grad_(cfg.module_training.denoising_unet)
    v_kps_guider.requires_grad_(cfg.module_training.v_kps_guider)
    audio_projection.requires_grad_(cfg.module_training.audio_projection)
    for name, module in denoising_unet.named_modules():
        if "motion_modules" in name:
            for params in module.parameters():
                params.requires_grad = cfg.module_training.motion_module
        if "attentions" in name and ("attn2" in name or "norm2" in name):
            # logger.info(name)
            for params in module.parameters():
                params.requires_grad = cfg.module_training.audio_projection

    # initialize the reference control writer and reader
    reference_control_writer = ReferenceAttentionControl(
        reference_net,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
        reference_drop_rate=cfg.data.reference_drop_rate,
    )

    net = Net(
        reference_net=reference_net,
        denoising_unet=denoising_unet,
        v_kps_guider=v_kps_guider,
        audio_projection=audio_projection,
        reference_control_writer=reference_control_writer,
        reference_control_reader=reference_control_reader,
        device=local_rank,
        weight_dtype=weight_dtype,
        kps_drop_rate=cfg.data.kps_drop_rate,
    )

    if cfg.solver.reference_net_gradient_checkpointing:
        reference_net.enable_gradient_checkpointing()
    if cfg.solver.denoising_unet_gradient_checkpointing:
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # initialize the optimizer and lr scheduler
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps * cfg.solver.gradient_accumulation_steps,
    )

    if 'kps_type' not in cfg.data:
        cfg.data.kps_type = 'v'

    dataset = TalkingFaceVideo(
        image_size=(cfg.data.train_height, cfg.data.train_width),
        meta_paths=cfg.data.meta_paths,
        flip_rate=cfg.data.flip_rate,
        sample_rate=cfg.data.sample_rate,
        num_frames=cfg.data.num_frames,
        reference_margin=cfg.data.reference_margin,
        num_padding_audio_frames=cfg.data.num_padding_audio_frames,
        kps_type=cfg.data.kps_type,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=4)

    (net, optimizer, dataloader, lr_scheduler) = accelerator.prepare(net, optimizer, dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(dataloader) / cfg.solver.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(cfg.solver.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0
    if accelerator.is_main_process:
        count_params(net)

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, cfg.solver.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"{exp_name}, Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        t_data_start = time.time()
        for step, batch in enumerate(dataloader):
            t_data = time.time() - t_data_start
            with accelerator.accumulate(net):
                target_images = batch["target_images"].to(weight_dtype)
                with torch.no_grad():
                    length = target_images.shape[2]
                    target_images = rearrange(target_images, "b c f h w -> (b f) c h w")
                    latents = vae.encode(target_images).latent_dist.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=length)
                    latents = latents * 0.18215

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0:
                    noise += cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1, 1),
                        device=latents.device,
                    )
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                with torch.no_grad():
                    reference_image = batch["reference_image"].to(dtype=vae.dtype, device=vae.device)
                    reference_image_latents = vae.encode(reference_image).latent_dist.sample()  # (bs, d, 64, 64)
                    reference_image_latents = reference_image_latents * 0.18215

                    kps_images = batch["kps_images"].to(local_rank, dtype=weight_dtype)  # (bs, f, c, H, W)
                    audio_frame_embeddings = batch['audio_frame_embeddings'].to(local_rank, dtype=weight_dtype)

                    lip_masks = batch["lip_masks"].to(dtype=vae.dtype, device=vae.device)

                # add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

                model_pred = net(
                    noisy_latents,
                    timesteps,
                    reference_image_latents,
                    audio_frame_embeddings,
                    kps_images,
                    do_unconditional_forward=random.random() < cfg.uncond_ratio,
                )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                if 'lip_loss_weight' in cfg.data:
                    loss *= ((cfg.data.lip_loss_weight - 1) * lip_masks + 1.)

                if cfg.snr_gamma != 0:
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                            torch.stack([snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.solver.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "td": f"{t_data:.2f}s",
            }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break

            # save model after each epoch
            if global_step % cfg.checkpointing_steps == 1:
                save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                # accelerator.save_state(save_path)

                if accelerator.is_main_process:
                    if cfg.save_all or cfg.module_training.denoising_unet:
                        module = accelerator.unwrap_model(net.denoising_unet)
                        save_module_checkpoint(module, save_dir, "denoising_unet", global_step)
                    elif cfg.module_training.motion_module or cfg.module_training.audio_projection:
                        module = accelerator.unwrap_model(net.denoising_unet)
                        save_module_checkpoint(module, save_dir, "denoising_unet", global_step)

                    if cfg.save_all or cfg.module_training.reference_net:
                        module = accelerator.unwrap_model(net.reference_net)
                        save_module_checkpoint(module, save_dir, "reference_net", global_step)

                    if cfg.save_all or cfg.module_training.v_kps_guider:
                        module = accelerator.unwrap_model(net.v_kps_guider)
                        save_module_checkpoint(module, save_dir, "v_kps_guider", global_step)

                    if cfg.save_all or cfg.module_training.audio_projection:
                        module = accelerator.unwrap_model(net.audio_projection)
                        save_module_checkpoint(module, save_dir, "audio_projection", global_step)

                    if cfg.save_all or cfg.module_training.motion_module:
                        module = accelerator.unwrap_model(net.denoising_unet)
                        save_motion_module_checkpoint(module, save_dir, "motion_module", global_step)

    # save model after each epoch
    # if accelerator.is_main_process:
    save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)

    if accelerator.is_main_process:
        if cfg.save_all or cfg.module_training.denoising_unet:
            module = accelerator.unwrap_model(net.denoising_unet)
            save_module_checkpoint(module, save_dir, "denoising_unet", global_step)
        elif cfg.module_training.motion_module or cfg.module_training.audio_projection:
            module = accelerator.unwrap_model(net.denoising_unet)
            save_module_checkpoint(module, save_dir, "denoising_unet", global_step)

        if cfg.save_all or cfg.module_training.reference_net:
            module = accelerator.unwrap_model(net.reference_net)
            save_module_checkpoint(module, save_dir, "reference_net", global_step)

        if cfg.save_all or cfg.module_training.v_kps_guider:
            module = accelerator.unwrap_model(net.v_kps_guider)
            save_module_checkpoint(module, save_dir, "v_kps_guider", global_step)

        if cfg.save_all or cfg.module_training.audio_projection:
            module = accelerator.unwrap_model(net.audio_projection)
            save_module_checkpoint(module, save_dir, "audio_projection", global_step)

        if cfg.save_all or cfg.module_training.motion_module:
            module = accelerator.unwrap_model(net.denoising_unet)
            save_motion_module_checkpoint(module, save_dir, "motion_module", global_step)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def save_module_checkpoint(module, save_dir, prefix, ckpt_num):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    state_dict = module.state_dict()
    new_state_dict = {}
    for n, p in state_dict.items():
        new_state_dict[n] = p.clone()
    torch.save(new_state_dict, save_path)


def save_motion_module_checkpoint(model, save_dir, prefix, ckpt_num):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    mm_state_dict = OrderedDict()
    state_dict = model.state_dict()
    for key in state_dict:
        if "motion_module" in key:
            mm_state_dict[key] = state_dict[key].clone()

    torch.save(mm_state_dict, save_path)


if __name__ == "__main__":
    main()
