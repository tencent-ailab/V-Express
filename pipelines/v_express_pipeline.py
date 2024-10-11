# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/pipelines/pipeline_animation.py
import inspect
import math
from typing import Callable, List, Optional, Union

import torch
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm
from transformers import CLIPImageProcessor

from modules import ReferenceAttentionControl
from .context import get_context_scheduler


def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class VExpressPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
            self,
            vae,
            reference_net,
            denoising_unet,
            v_kps_guider,
            audio_processor,
            audio_encoder,
            audio_projection,
            scheduler: Union[
                DDIMScheduler,
                PNDMScheduler,
                LMSDiscreteScheduler,
                EulerDiscreteScheduler,
                EulerAncestralDiscreteScheduler,
                DPMSolverMultistepScheduler,
            ],
            image_proj_model=None,
            tokenizer=None,
            text_encoder=None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            reference_net=reference_net,
            denoising_unet=denoising_unet,
            v_kps_guider=v_kps_guider,
            audio_processor=audio_processor,
            audio_encoder=audio_encoder,
            audio_projection=audio_projection,
            scheduler=scheduler,
            image_proj_model=image_proj_model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.reference_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.condition_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    @torch.no_grad()
    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in tqdm(range(latents.shape[0]), desc='Decoding latents into frames'):
            image = self.vae.decode(latents[frame_idx: frame_idx + 1].to(self.vae.device)).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().float()
            video.append(image)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)

        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
            self,
            batch_size,
            num_channels_latents,
            width,
            height,
            video_length,
            dtype,
            device,
            generator,
            latents=None
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )

        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _encode_prompt(
            self,
            prompt,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
            )

        if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_videos_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                    hasattr(self.text_encoder.config, "use_attention_mask")
                    and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_videos_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]

        return timesteps, num_inference_steps - t_start

    def prepare_reference_latent(self, reference_image, height, width):
        reference_image_tensor = self.reference_image_processor.preprocess(reference_image, height=height, width=width)
        reference_image_tensor = reference_image_tensor.to(dtype=self.dtype, device=self.device)
        reference_image_latents = self.vae.encode(reference_image_tensor).latent_dist.mean
        reference_image_latents = reference_image_latents * 0.18215
        return reference_image_latents

    def prepare_kps_feature(self, kps_images, height, width, do_classifier_free_guidance):
        kps_image_tensors = []
        for idx, kps_image in enumerate(kps_images):
            kps_image_tensor = self.condition_image_processor.preprocess(kps_image, height=height, width=width)
            kps_image_tensor = kps_image_tensor.unsqueeze(2)  # [bs, c, 1, h, w]
            kps_image_tensors.append(kps_image_tensor)
        kps_images_tensor = torch.cat(kps_image_tensors, dim=2)  # [bs, c, t, h, w]

        bs = 16
        num_forward = math.ceil(kps_images_tensor.shape[2] / bs)
        kps_feature = []
        for i in range(num_forward):
            tensor = kps_images_tensor[:, :, i * bs:(i + 1) * bs, ...].to(device=self.device, dtype=self.dtype)
            feature = self.v_kps_guider(tensor).cpu()
            kps_feature.append(feature)
            torch.cuda.empty_cache()
        kps_feature = torch.cat(kps_feature, dim=2)

        if do_classifier_free_guidance:
            uc_kps_feature = torch.zeros_like(kps_feature)
            kps_feature = torch.cat([uc_kps_feature, kps_feature], dim=0)

        return kps_feature

    def prepare_audio_embeddings(self, audio_waveform, video_length, num_pad_audio_frames, do_classifier_free_guidance):
        audio_waveform = self.audio_processor(audio_waveform, return_tensors="pt", sampling_rate=16000)['input_values']
        audio_waveform = audio_waveform.to(self.device, self.dtype)
        audio_embeddings = self.audio_encoder(audio_waveform).last_hidden_state  # [1, num_embeds, d]
        in_dtype = audio_embeddings.dtype

        audio_embeddings = audio_embeddings.to(dtype=torch.float32)
        audio_embeddings = torch.nn.functional.interpolate(
            audio_embeddings.permute(0, 2, 1),
            size=2 * video_length,
            mode='linear',
        )[0, :, :].permute(1, 0)  # [2*vid_len, dim]
        audio_embeddings = audio_embeddings.to(dtype=in_dtype)

        audio_embeddings = torch.cat([
            torch.zeros_like(audio_embeddings)[:2 * num_pad_audio_frames, :],
            audio_embeddings,
            torch.zeros_like(audio_embeddings)[:2 * num_pad_audio_frames, :],
        ], dim=0)  # [2*num_pad+2*vid_len+2*num_pad, dim]

        frame_audio_embeddings = []
        for frame_idx in range(video_length):
            start_sample = frame_idx
            end_sample = frame_idx + 2 * num_pad_audio_frames

            frame_audio_embedding = audio_embeddings[2 * start_sample:2 * (end_sample + 1), :]  # [2*num_pad+1, dim]
            frame_audio_embeddings.append(frame_audio_embedding)
        audio_embeddings = torch.stack(frame_audio_embeddings, dim=0)  # [vid_len, 2*num_pad+1, dim]

        audio_embeddings = self.audio_projection(audio_embeddings).unsqueeze(0)
        if do_classifier_free_guidance:
            uc_audio_embeddings = torch.zeros_like(audio_embeddings)
            audio_embeddings = torch.cat([uc_audio_embeddings, audio_embeddings], dim=0)
        return audio_embeddings

    def mean_overlap(
        self,
        reference_image,
        kps_images,
        audio_waveform,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        strength=1.,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        context_schedule="uniform",
        context_frames=24,
        context_overlap=4,
        reference_attention_weight=1.,
        audio_attention_weight=1.,
        num_pad_audio_frames=2,
        do_multi_devices_inference=False,
        save_gpu_memory=False,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0
        batch_size = 1

        # Prepare timesteps
        timesteps = None
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        reference_control_writer = ReferenceAttentionControl(
            self.reference_net,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
            reference_attention_weight=reference_attention_weight,
            audio_attention_weight=audio_attention_weight,
        )

        num_channels_latents = self.denoising_unet.in_channels
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        reference_image_latents = self.prepare_reference_latent(reference_image, height, width)
        kps_feature = self.prepare_kps_feature(kps_images, height, width, do_classifier_free_guidance)
        if save_gpu_memory:
            del self.v_kps_guider
        torch.cuda.empty_cache()
        audio_embeddings = self.prepare_audio_embeddings(
            audio_waveform,
            video_length,
            num_pad_audio_frames,
            do_classifier_free_guidance,
        )
        if save_gpu_memory:
            del self.audio_processor, self.audio_encoder, self.audio_projection
        torch.cuda.empty_cache()

        context_scheduler = get_context_scheduler(context_schedule)
        context_queue = list(
            context_scheduler(
                step=0,
                num_frames=video_length,
                context_size=context_frames,
                context_stride=1,
                context_overlap=context_overlap,
                closed_loop=False,
            )
        )

        num_frame_context = torch.zeros(video_length, device=device, dtype=torch.long)
        for context in context_queue:
            num_frame_context[context] += 1

        encoder_hidden_states = torch.zeros((1, 1, 768), dtype=self.dtype, device=self.device)
        self.reference_net(
            reference_image_latents,
            timestep=0,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )
        reference_control_reader.update(reference_control_writer, do_classifier_free_guidance, dtype=self.dtype)
        if save_gpu_memory:
            del self.reference_net
        torch.cuda.empty_cache()

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            self.dtype,
            torch.device('cpu'),
            generator,
        )  # [bs, c, l, h, w]

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                context_counter = torch.zeros(video_length, device=device, dtype=torch.long)
                noise_preds = [None] * video_length
                for context_idx, context in enumerate(context_queue):
                    latent_kps_feature = kps_feature[:, :, context].to(device, self.dtype)

                    latent_audio_embeddings = audio_embeddings[:, context, ...]

                    _, _, num_tokens, dim = latent_audio_embeddings.shape
                    latent_audio_embeddings = latent_audio_embeddings.reshape(-1, num_tokens, dim)

                    input_latents = latents[:, :, context, ...].to(device)
                    input_latents = input_latents.repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    input_latents = self.scheduler.scale_model_input(input_latents, t)
                    noise_pred = self.denoising_unet(
                        input_latents,
                        t,
                        encoder_hidden_states=latent_audio_embeddings.reshape(-1, num_tokens, dim),
                        kps_features=latent_kps_feature,
                        return_dict=False,
                    )[0]
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    context_counter[context] += 1
                    noise_pred /= num_frame_context[context][None, None, :, None, None]
                    step_frame_ids = []
                    step_noise_preds = []
                    for latent_idx, frame_idx in enumerate(context):
                        if noise_preds[frame_idx] is None:
                            noise_preds[frame_idx] = noise_pred[:, :, latent_idx, ...]
                        else:
                            noise_preds[frame_idx] += noise_pred[:, :, latent_idx, ...]
                        if context_counter[frame_idx] == num_frame_context[frame_idx]:
                            step_frame_ids.append(frame_idx)
                            step_noise_preds.append(noise_preds[frame_idx])
                            noise_preds[frame_idx] = None
                    step_noise_preds = torch.stack(step_noise_preds, dim=2)
                    output_latents = self.scheduler.step(
                        step_noise_preds,
                        t,
                        latents[:, :, step_frame_ids, ...].to(device),
                        **extra_step_kwargs,
                    ).prev_sample
                    latents[:, :, step_frame_ids, ...] = output_latents.cpu()

                    progress_bar.set_description(
                        f'Denoising Step Index: {i + 1} / {len(timesteps)}, '
                        f'Context Index: {context_idx + 1} / {len(context_queue)}'
                    )

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        reference_control_reader.clear()
        reference_control_writer.clear()

        video_tensor = self.decode_latents(latents)
        return video_tensor

    @torch.no_grad()
    def __call__(
        self,
        reference_image,
        kps_images,
        audio_waveform,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        strength=1.,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        context_schedule="uniform",
        context_frames=24,
        context_overlap=4,
        reference_attention_weight=1.,
        audio_attention_weight=1.,
        num_pad_audio_frames=2,
        do_multi_devices_inference=False,
        save_gpu_memory=False,
        **kwargs,
    ):
        return self.mean_overlap(
            reference_image=reference_image,
            kps_images=kps_images,
            audio_waveform=audio_waveform,
            width=width,
            height=height,
            video_length=video_length,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            context_schedule=context_schedule,
            context_frames=context_frames,
            context_overlap=context_overlap,
            reference_attention_weight=reference_attention_weight,
            audio_attention_weight=audio_attention_weight,
            num_pad_audio_frames=num_pad_audio_frames,
            do_multi_devices_inference=do_multi_devices_inference,
            save_gpu_memory=save_gpu_memory,
            **kwargs,
        )
