import torch

from PIL import Image
from einops import rearrange
from typing import Callable, Dict, List, Optional, Union, Tuple
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import WanLoraLoaderMixin
from ppmem.model.scheduler import FlowUniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

# Import Separate VAE for no temporal compression support
from ppmem.model.vae import AutoencoderKLWanSeparate

from ppmem.model.dit import WanTransformerPlana3DModel
from ppmem.model.planes import extract_plane_params_from_pred_prim
from ppmem._logging import logger


def data_seq_to_patch(
    patch_size: list,
    data_seq: torch.Tensor,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
    batch_size: int = 1,
):
    p_t, p_h, p_w = patch_size
    post_patch_num_frames = latent_num_frames // p_t
    post_patch_height = latent_height // p_h
    post_patch_width = latent_width // p_w

    data_patch = data_seq.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    data_patch = data_patch.permute(0, 7, 1, 4, 2, 5, 3, 6)
    data_patch = data_patch.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return data_patch


def t5_prompt_embeds(
    tokenizer,
    text_encoder: UMT5EncoderModel,
    prompt: Union[str, List[str]],
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    """UMT5 embeddings of one prompt, zero-padded to max_sequence_length.

    A function rather than only a pipeline method, so the prompt can be encoded before
    the DiT is built and the text encoder freed (ppmem/engine/builder.py).
    """
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = 1

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)

    return prompt_embeds


class PPMemImageToVideoPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Generates one PPMem chunk: an image-to-video Wan2.2 denoising loop conditioned on the
    projected RGB/mask of the plane memory, the depth to follow and the prefix frames
    (history, very first, context), followed by one geometry pass that predicts the
    chunk's planes.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer, text_encoder:
            UMT5 tokenizer and encoder. PPMem passes None for both: the engine encodes the
            prompt once and the pipeline takes `prompt_embeds`.
        transformer ([`WanTransformerPlana3DModel`]):
            The PPMem DiT: denoises the latents and predicts planes.
        scheduler ([`FlowUniPCMultistepScheduler`]):
            Flow-matching UniPC sampler.
        vae ([`AutoencoderKLWanSeparate`]):
            Wan2.2 VAE, used per frame (one latent per frame).
        pipe_dtype (`torch.dtype`):
            dtype the transformer runs in.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWanSeparate,
        scheduler: FlowUniPCMultistepScheduler,
        transformer: WanTransformerPlana3DModel = None,
        pipe_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.pipe_dtype = pipe_dtype

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        return t5_prompt_embeds(
            self.tokenizer,
            self.text_encoder,
            prompt,
            max_sequence_length=max_sequence_length,
            device=device or self._execution_device,
            dtype=dtype,
        )

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = 1

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if image is not None and image_embeds is not None:
            raise ValueError(
                f"Cannot forward both `image`: {image} and `image_embeds`: {image_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if image is None and image_embeds is None:
            raise ValueError(
                "Provide either `image` or `image_embeds`. Cannot leave both `image` and `image_embeds` undefined."
            )
        if image is not None and not isinstance(image, torch.Tensor) and not isinstance(image, Image.Image):
            raise ValueError(f"`image` has to be of type `torch.Tensor` or `PIL.Image.Image` but is {type(image)}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and not isinstance(prompt, str):
            raise ValueError(f"`prompt` has to be of type `str` but is {type(prompt)}")
        elif negative_prompt is not None and not isinstance(negative_prompt, str):
            raise ValueError(
                f"`negative_prompt` has to be of type `str` but is {type(negative_prompt)}"
            )

    def prepare_latents(
        self,
        image: PipelineImageInput = None,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]
        video_condition = image
        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        # Encode each frame on its own (one latent per frame)
        if isinstance(self.vae, AutoencoderKLWanSeparate) and hasattr(self.vae, 'encode_no_temporal_compression'):
            latent_condition = self.vae.encode_no_temporal_compression(video_condition).latent_dist.mode()
        else:
            latent_condition = self.vae.encode(video_condition).latent_dist.mode()
        latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        first_frame_mask = torch.ones(
            1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device
        )
        first_frame_mask[:, :, 0] = 0
        return latents, latent_condition, first_frame_mask

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[Union[Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        latent_proj_rgb_1: Optional[torch.Tensor] = None,
        latent_proj_rgb_2: Optional[torch.Tensor] = None,
        latent_proj_mask_1: Optional[torch.Tensor] = None,
        latent_proj_mask_2: Optional[torch.Tensor] = None,
        latents_context: Optional[torch.Tensor] = None,
        latents_very_first: Optional[torch.Tensor] = None,
        latents_history: Optional[torch.Tensor] = None,
        latents_depth: Optional[torch.Tensor] = None,
        latents_context_depth: Optional[torch.Tensor] = None,
        latents_hv_depth: Optional[torch.Tensor] = None,
        raw_next_ar_depth: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics_c2w: Optional[torch.Tensor] = None,
    ):
        r"""
        Generate one video and optionally extract plane geometry.

        Args:
            image (`PIL.Image.Image` or `torch.Tensor`):
                The single input image used for VAE conditioning.
            prompt, negative_prompt (`str`, *optional*):
                Only usable when the pipeline was built with a tokenizer and text encoder;
                PPMem builds it without them and passes `prompt_embeds` /
                `negative_prompt_embeds` instead.
            height (`int`, defaults to `480`):
                The height of the generated video.
            width (`int`, defaults to `832`):
                The width of the generated video.
            num_frames (`int`, defaults to `81`):
                Frame count in the Wan convention, (latent frames - 1) * 4 + 1. The VAE here
                maps frames 1:1, so VideoGenerator passes (segment_size - 1) * 4 + 1.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages videos that are closely linked to
                the text `prompt`, usually at the expense of lower visual quality.
            generator (`torch.Generator`, *optional*):
                The random generator used to sample the initial latents.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds, negative_prompt_embeds (`torch.Tensor`):
                UMT5 embeddings of the prompt and of the (empty) negative prompt.
            image_embeds (`torch.Tensor`, *optional*):
                Unused for generation; leave as None and supply `image`.
            output_type (`str`, *optional*, defaults to `"np"`):
                Output format passed to `video_processor.postprocess_video` for decoded videos.
                Use `"latent"` to return latents without VAE decoding.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.
            latent_proj_rgb_1, latent_proj_rgb_2, latent_proj_mask_1, latent_proj_mask_2 (`torch.Tensor`):
                Latents of the plane memory's two projected source views and their coverage
                masks, [c, f, h, w] each; concatenated to the input channels.
            latents_history, latents_very_first, latents_context (`torch.Tensor`, *optional*):
                Prefix frames, [c, n, h, w] each. The sequence is
                [history | very_first | context | main]; prefix frames stay clean (not
                denoised) and get fixed RoPE time indices -3, -2 and -1.
            latents_depth (`torch.Tensor`, *optional*):
                Depth latents of the main frames, [c, f, h, w], concatenated to the input
                channels. The denoising steps run with these channels zeroed and LoRA off;
                the geometry pass after the last step uses them, with LoRA on. Required when
                prefix frames are present.
            latents_context_depth (`torch.Tensor`, *optional*):
                Pre-encoded depth maps in latent space for context frames.
                Required when context prefix frames are present. Shape should be [c, n_ctx, h, w],
                with the same frame count and order as `latents_context`.
            latents_hv_depth (`torch.Tensor`, *optional*):
                Pre-encoded depth maps in latent space for history and very-first prefix frames.
                Required when either prefix is present. Shape should be [c, n_history + n_vf, h, w],
                with history frames followed by very-first frames.
            raw_next_ar_depth (`torch.Tensor`, *optional*):
                Metric depth of the main frames, [f, H, W], for depth token fusion in the
                geometry pass.
            intrinsics (`torch.Tensor`, *optional*):
                Camera intrinsics matrix, shaped [3, 3] or repeated as [N, 3, 3].
                Corresponds to the target video resolution (height x width).
                Use the same matrix throughout the segment; primitive rendering
                shares the first matrix across all frames.
                Enables plane extraction. With `return_dict=True`, results are attached as `plane_params`.
            extrinsics_c2w (`torch.Tensor`, *optional*):
                Per-frame camera-to-world extrinsics. Shape [N, 4, 4].
                N should match the number of latent frames (excluding context).
                Passed through in the output for downstream reprojection use.

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                With `return_dict=True`, returns `WanPipelineOutput` with `frames` containing decoded videos
                or latents according to `output_type`. Its `plane_params` attribute contains geometry results
                when plane predictions are available, otherwise `None`.
                With `return_dict=False`, returns the single-element tuple `(video,)` without geometry results.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct

        # 2. One video per call. The batch dimension is fixed at 1.
        # Reject batched inputs explicitly to avoid silently using only the first video.
        if isinstance(prompt, (list, tuple)):
            raise TypeError(
                f"prompt must be a single string; got {type(prompt).__name__} of "
                f"length {len(prompt)}. This pipeline generates one video per call."
            )
        if isinstance(negative_prompt, (list, tuple)):
            raise TypeError(
                f"negative_prompt must be a single string; got "
                f"{type(negative_prompt).__name__} of length {len(negative_prompt)}."
            )
        if prompt_embeds is not None and prompt_embeds.shape[0] != 1:
            raise ValueError(
                f"prompt_embeds must have batch size 1; got {prompt_embeds.shape[0]}."
            )
        if isinstance(generator, list):
            raise TypeError(
                f"generator must be a single torch.Generator; got a list of "
                f"length {len(generator)}."
            )

        batch_size = 1

        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        num_history_frames = latents_history.shape[1] if latents_history is not None else 0
        num_very_first_frames = latents_very_first.shape[1] if latents_very_first is not None else 0
        num_context_frames = latents_context.shape[1] if latents_context is not None else 0
        total_prefix_frames = num_history_frames + num_very_first_frames + num_context_frames
        if total_prefix_frames > 0:
            if latents_depth is None:
                raise ValueError("latents_depth is required when prefix frames are present")
            depth_prefix_parts = []
            for name, depth, expected_frames in (
                ("latents_hv_depth", latents_hv_depth, num_history_frames + num_very_first_frames),
                ("latents_context_depth", latents_context_depth, num_context_frames),
            ):
                if expected_frames == 0:
                    continue
                if depth is None:
                    raise ValueError(f"{name} is required for {expected_frames} prefix frames")
                if depth.shape[1] != expected_frames:
                    raise ValueError(f"{name} has {depth.shape[1]} frames, expected {expected_frames}")
                depth_prefix_parts.append(depth.to(latents_depth.dtype))
            latents_depth = torch.cat(depth_prefix_parts + [latents_depth], dim=1)

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._interrupt = False

        device = self._execution_device

        # 3. Prompt embeddings (encoded once by the engine and passed in)
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        transformer_dtype = self.pipe_dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=self.vae.dtype)
        latents_outputs = self.prepare_latents(
            image,
            batch_size,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )
        # first_frame_mask is 0 on the frames that stay clean (the condition image and the
        # prefix frames) and 1 on the frames being denoised; it also zeroes their timestep.
        latents, condition, first_frame_mask = latents_outputs
        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # Handle prefix frames: [history | very_first | context | main]
        # Build each prefix part that is provided, in order.
        prefix_parts = []
        if latents_history is not None:
            prefix_parts.append(latents_history.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).to(latents.dtype))
        if latents_very_first is not None:
            prefix_parts.append(latents_very_first.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).to(latents.dtype))
        if latents_context is not None:
            prefix_parts.append(latents_context.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).to(latents.dtype))

        if total_prefix_frames > 0:
            prefix_expanded = torch.cat(prefix_parts, dim=2)  # [b, c, total_prefix, h, w]

            # Concatenate prefix at the beginning: [history | very_first | context | main]
            latents = torch.cat([prefix_expanded, latents], dim=2)

            # Update condition and masks to include prefix frames
            original_num_main_frames = latents.shape[2] - total_prefix_frames
            condition_expanded = condition.expand(-1, -1, original_num_main_frames, -1, -1)
            condition = torch.cat([prefix_expanded, condition_expanded], dim=2)
            # Prefix frames: mask=0 (clean, timestep=0)
            prefix_mask = torch.zeros(
                first_frame_mask.shape[0], first_frame_mask.shape[1], total_prefix_frames,
                first_frame_mask.shape[3], first_frame_mask.shape[4],
                dtype=first_frame_mask.dtype, device=device
            )
            first_frame_mask = torch.cat([prefix_mask, first_frame_mask], dim=2)
            mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # Handle proj_rgb and proj_mask prefix frames padding (zeros for all prefix frames)
        if total_prefix_frames > 0 and latent_proj_rgb_1 is not None:
            pfx_shape_rgb = list(latent_proj_rgb_1.shape)
            pfx_shape_rgb[1] = total_prefix_frames
            pfx_shape_mask = list(latent_proj_mask_1.shape[:1]) + [total_prefix_frames] + list(latent_proj_mask_1.shape[2:])
            latent_proj_rgb_1 = torch.cat([torch.zeros(pfx_shape_rgb, dtype=latent_proj_rgb_1.dtype, device=device), latent_proj_rgb_1], dim=1)
            latent_proj_rgb_2 = torch.cat([torch.zeros(pfx_shape_rgb, dtype=latent_proj_rgb_2.dtype, device=device), latent_proj_rgb_2], dim=1)
            latent_proj_mask_1 = torch.cat([torch.zeros(pfx_shape_mask, dtype=latent_proj_mask_1.dtype, device=device), latent_proj_mask_1], dim=1)
            latent_proj_mask_2 = torch.cat([torch.zeros(pfx_shape_mask, dtype=latent_proj_mask_2.dtype, device=device), latent_proj_mask_2], dim=1)

        # 6. Denoising loop
        _pred_prim_high_list = None
        _rendered_dict_list = None   # rendered depth/normal maps from rasterizer
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # Pre-expand conditioning tensors (avoid repeated expand+cat in loop)
        _bsz = latents.shape[0]
        _cond_parts = []
        if latent_proj_rgb_1 is not None:
            _cond_parts.extend([
                latent_proj_rgb_1.unsqueeze(0).expand(_bsz, -1, -1, -1, -1).to(transformer_dtype),
                latent_proj_rgb_2.unsqueeze(0).expand(_bsz, -1, -1, -1, -1).to(transformer_dtype),
                latent_proj_mask_1.unsqueeze(0).expand(_bsz, -1, -1, -1, -1).to(transformer_dtype),
                latent_proj_mask_2.unsqueeze(0).expand(_bsz, -1, -1, -1, -1).to(transformer_dtype),
            ])
        if latents_depth is not None:
            _cond_parts.append(latents_depth.unsqueeze(0).expand(_bsz, -1, -1, -1, -1).to(transformer_dtype))
        _precat_cond = torch.cat(_cond_parts, dim=1) if _cond_parts else None
        # With LoRA modules present, denoise with LoRA disabled and zero depth channels.
        # If intrinsics are provided, follow with one geometry forward at the final
        # timestep using LoRA and the real depth conditioning in _precat_cond.
        _precat_cond_no_depth = None
        if _precat_cond is not None and latents_depth is not None:
            _depth_ch = latents_depth.shape[0]  # C (e.g. 48 for separate VAE)
            _precat_cond_no_depth = _precat_cond.clone()
            _precat_cond_no_depth[:, -_depth_ch:] = 0

        try:
            from ppmem.model.lora import set_lora_enabled, LoRALinear
            _has_lora = any(isinstance(m, LoRALinear) for m in self.transformer.modules())
        except Exception:
            set_lora_enabled = None
            _has_lora = False
        _lora_extra_depth_pass = False
        _use_no_depth_precat = False
        if _has_lora:
            set_lora_enabled(self.transformer, False)
            _lora_extra_depth_pass = True
            if _precat_cond_no_depth is not None:
                _use_no_depth_precat = True

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                current_model = self.transformer
                current_guidance_scale = guidance_scale

                # Per-token timesteps: 0 on clean frames, t elsewhere
                batch_size, latent_channel, latent_num_frames, latent_height, latent_width = latents.shape
                first_frame_mask = first_frame_mask.expand_as(latents)
                latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
                latent_model_input = latent_model_input.to(transformer_dtype)
                temp_ts = first_frame_mask[:, 0:1, :, ::2, ::2].flatten(1) * t  # [b, f*h/2*w/2]
                timestep = torch.cat([i for i in temp_ts], dim=0)

                actual_latent_num_frames = latent_num_frames  # Includes context frames

                _cur_precat = _precat_cond_no_depth if _use_no_depth_precat else _precat_cond
                if _cur_precat is not None:
                    latent_model_input = torch.cat([latent_model_input, _cur_precat], dim=1)

                input_dict = dict()
                input_dict["n_seq"] = batch_size
                input_dict["timestep"] = timestep
                input_dict["hidden_states"] = latent_model_input.chunk(batch_size, dim=0)
                input_dict["encoder_hidden_states"] = rearrange(prompt_embeds, 'b l c -> 1 (b l) c')
                input_dict["latent_num_frames_lst"] = [actual_latent_num_frames] * batch_size
                input_dict["latent_height_lst"] = [latent_height] * batch_size
                input_dict["latent_width_lst"] = [latent_width] * batch_size
                input_dict["latent_len_lst"] = [actual_latent_num_frames*latent_height*latent_width] * batch_size
                input_dict["text_emb_len_lst"] = [prompt_embeds.shape[1]] * batch_size
                # Add prefix frame counts for RoPE position encoding
                input_dict["num_context_frames_lst"] = [num_context_frames] * batch_size
                input_dict["num_very_first_frames_lst"] = [num_very_first_frames] * batch_size
                input_dict["num_history_frames_lst"] = [num_history_frames] * batch_size
                if raw_next_ar_depth is not None:
                    # Depth token fusion takes a list with one [N, H, W] depth per video
                    if not isinstance(raw_next_ar_depth, list):
                        input_dict["raw_next_ar_depth"] = [raw_next_ar_depth] * batch_size
                    else:
                        if len(raw_next_ar_depth) < batch_size:
                            input_dict["raw_next_ar_depth"] = raw_next_ar_depth * batch_size
                        else:
                            input_dict["raw_next_ar_depth"] = raw_next_ar_depth

                # Without an extra geometry pass, extract planes on the last denoising step.
                is_last_step = (i == len(timesteps) - 1)
                if is_last_step and intrinsics is not None and not _lora_extra_depth_pass:
                    # Prepare intrinsics for the model: List of [N, 3, 3], one per batch element
                    num_latent_frames_no_ctx = actual_latent_num_frames - total_prefix_frames
                    if intrinsics.dim() == 2:
                        # [3, 3] → replicate to [N, 3, 3]
                        intrinsics_for_model = intrinsics.unsqueeze(0).expand(
                            num_latent_frames_no_ctx, -1, -1
                        ).contiguous()
                    else:
                        intrinsics_for_model = intrinsics
                    input_dict['intrinsics'] = [intrinsics_for_model] * batch_size
                    input_dict['original_height'] = [height] * batch_size
                    input_dict['original_width'] = [width] * batch_size

                    # The plane branch converts the velocity prediction into a latent
                    # prediction and adds it to hidden_states_feat before the plane
                    # decoder; that needs noisy_latents and timesteps_value.
                    input_dict['noisy_latents'] = input_dict['hidden_states']
                    input_dict['timesteps_value'] = [
                        t.reshape(1) for _ in range(batch_size)
                    ]
                    # Pass inference scheduler to model for velocity->latent conversion
                    input_dict['inference_scheduler'] = self.scheduler

                with current_model.cache_context("cond"):
                    model_output_full = current_model(
                        input_dict,
                    )
                    noise_pred = model_output_full[0]

                    # Extract plane params from cond pass on last step
                    if is_last_step and intrinsics is not None:
                        # forward returns (noise_pred, pred_prim_high, rendered_dict)
                        _pred_prim_high_list = model_output_full[1]   # List[dict], one per video
                        _rendered_dict_list = model_output_full[2]    # List[dict] or None
                    else:
                        _pred_prim_high_list = None

                if self.do_classifier_free_guidance:
                    input_dict_cfg = dict()
                    input_dict_cfg["n_seq"] = batch_size
                    input_dict_cfg["timestep"] = timestep
                    input_dict_cfg["hidden_states"] = latent_model_input.chunk(batch_size, dim=0)  # proj_rgb / proj_mask are already concatenated in
                    input_dict_cfg["encoder_hidden_states"] = rearrange(negative_prompt_embeds, 'b l c -> 1 (b l) c')
                    input_dict_cfg["latent_num_frames_lst"] = [actual_latent_num_frames] * batch_size
                    input_dict_cfg["latent_height_lst"] = [latent_height] * batch_size
                    input_dict_cfg["latent_width_lst"] = [latent_width] * batch_size
                    input_dict_cfg["latent_len_lst"] = [actual_latent_num_frames*latent_height*latent_width] * batch_size
                    input_dict_cfg["text_emb_len_lst"] = [negative_prompt_embeds.shape[1]] * batch_size
                    # Add prefix frame counts for RoPE position encoding
                    input_dict_cfg["num_context_frames_lst"] = [num_context_frames] * batch_size
                    input_dict_cfg["num_very_first_frames_lst"] = [num_very_first_frames] * batch_size
                    input_dict_cfg["num_history_frames_lst"] = [num_history_frames] * batch_size

                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            input_dict_cfg,
                        )[0]
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                noise_pred = data_seq_to_patch(
                    self.transformer.model_args.patch_size,
                    noise_pred,
                    actual_latent_num_frames,  # Use actual frames including context
                    latent_height,
                    latent_width,
                    batch_size=batch_size,
                )
                # If we have context frames, we need to slice them out from the prediction
                # because we only want to update the generated frames, not the prefix
                if total_prefix_frames > 0:
                    noise_pred = noise_pred[:, :, total_prefix_frames:, :, :]

                if total_prefix_frames > 0:
                    latents_prefix_part = latents[:, :, :total_prefix_frames, :, :]
                    latents_generated_part = latents[:, :, total_prefix_frames:, :, :]
                    latents_generated_part = self.scheduler.step(noise_pred, t, latents_generated_part, return_dict=False)[0]
                    latents = torch.cat([latents_prefix_part, latents_generated_part], dim=2)
                else:
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if _lora_extra_depth_pass and intrinsics is not None:
            set_lora_enabled(self.transformer, True)

            t_depth = timesteps[-1]  # final timestep (least noisy)
            depth_model = self.transformer

            batch_size_d = latents.shape[0]
            _, _, lnf_d, lh_d, lw_d = latents.shape
            fm_d = first_frame_mask.expand_as(latents)
            lmi_d = ((1 - fm_d) * condition + fm_d * latents).to(transformer_dtype)
            ts_d = fm_d[:, 0:1, :, ::2, ::2].flatten(1) * t_depth
            ts_d = torch.cat([x for x in ts_d], dim=0)

            if _precat_cond is not None:
                lmi_d = torch.cat([lmi_d, _precat_cond], dim=1)

            id_d = {
                "n_seq": batch_size_d,
                "timestep": ts_d,
                "hidden_states": lmi_d.chunk(batch_size_d, dim=0),
                "encoder_hidden_states": rearrange(prompt_embeds, 'b l c -> 1 (b l) c'),
                "latent_num_frames_lst": [lnf_d] * batch_size_d,
                "latent_height_lst": [lh_d] * batch_size_d,
                "latent_width_lst": [lw_d] * batch_size_d,
                "latent_len_lst": [lnf_d * lh_d * lw_d] * batch_size_d,
                "text_emb_len_lst": [prompt_embeds.shape[1]] * batch_size_d,
                "num_context_frames_lst": [num_context_frames] * batch_size_d,
                "num_very_first_frames_lst": [num_very_first_frames] * batch_size_d,
                "num_history_frames_lst": [num_history_frames] * batch_size_d,
            }
            if raw_next_ar_depth is not None:
                rd = [raw_next_ar_depth] * batch_size_d if not isinstance(raw_next_ar_depth, list) else raw_next_ar_depth
                id_d["raw_next_ar_depth"] = rd[:batch_size_d] if len(rd) >= batch_size_d else rd * batch_size_d

            nf_no_ctx = lnf_d - total_prefix_frames
            if intrinsics.dim() == 2:
                intr_d = intrinsics.unsqueeze(0).expand(nf_no_ctx, -1, -1).contiguous()
            else:
                intr_d = intrinsics
            id_d['intrinsics'] = [intr_d] * batch_size_d
            id_d['original_height'] = [height] * batch_size_d
            id_d['original_width'] = [width] * batch_size_d
            id_d['noisy_latents'] = id_d['hidden_states']
            id_d['timesteps_value'] = [t_depth.reshape(1) for _ in range(batch_size_d)]
            id_d['inference_scheduler'] = self.scheduler

            with depth_model.cache_context("cond"):
                depth_out = depth_model(id_d)
                _pred_prim_high_list = depth_out[1]
                _rendered_dict_list = depth_out[2]

            set_lora_enabled(self.transformer, False)

        latents = (1 - first_frame_mask) * condition + first_frame_mask * latents
        # If we have context frames, they should remain unchanged (already in condition)
        # The first_frame_mask already ensures context frames (mask=0) keep their original values

        # Remove all prefix frames from latents before decoding (we only want to return generated frames)
        if total_prefix_frames > 0:
            latents = latents[:, :, total_prefix_frames:, :, :]

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean

            # Decode each latent frame on its own
            if isinstance(self.vae, AutoencoderKLWanSeparate) and hasattr(self.vae, 'decode_no_temporal_compression'):
                video = self.vae.decode_no_temporal_compression(latents, return_dict=False)[0]
            else:
                video = self.vae.decode(latents, return_dict=False)[0]

            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Build plane params output if available
        plane_params_output = None
        if _pred_prim_high_list is not None:
            plane_params_output = []
            for prim_dict in _pred_prim_high_list:
                plane_params_output.append(extract_plane_params_from_pred_prim(prim_dict))
            pp_out = {
                'plane_params_per_video': plane_params_output,
                'intrinsics': intrinsics,
            }
            if extrinsics_c2w is not None:
                pp_out['extrinsics_c2w'] = extrinsics_c2w
            if _rendered_dict_list is not None:
                pp_out['rendered_dict_list'] = _rendered_dict_list
            plane_params_output = pp_out

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        output = WanPipelineOutput(frames=video)
        output.plane_params = plane_params_output
        return output
