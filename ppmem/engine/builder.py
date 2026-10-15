"""Inference-only model builder."""

import os

import torch
from torch._utils import _get_available_device_type, _get_device_module

from ppmem.config import MODEL
from ppmem.model.loaders import load_text_encoder, load_tokenizer, load_vae
from ppmem.model.dit import WanTransformerPlana3DModel
from ppmem.model.scheduler import FlowUniPCMultistepScheduler
from ppmem._logging import logger

from ppmem.engine import depth_latent
from ppmem.engine.determinism import set_determinism


def resolve_device(device=None, device_type: str = None) -> torch.device:
    """Use the explicit device, or device 0 of the requested or detected type.

    CUDA device indices refer to the GPUs exposed by CUDA_VISIBLE_DEVICES.
    """
    if isinstance(device, torch.device):
        return device
    if device is None:
        device = f"{device_type or _get_available_device_type() or 'cuda'}:0"
    return torch.device(device)


class InferenceEngine:
    """Encode the prompt, then build the model, VAE, scheduler, and pipeline components
    for inference."""

    def __init__(self, job_config, prompt, checkpoint_dir, device=None):
        self.job_config = job_config

        device_type = _get_available_device_type() or "cuda"
        device_module = _get_device_module(device_type)
        self.device = resolve_device(device, device_type)
        device_module.set_device(self.device)

        # ---- single-GPU guard ----
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            raise NotImplementedError(
                f"Inference is single-GPU only, got WORLD_SIZE={world_size}. "
                "The multi-GPU parallelism code (FSDP / TP / PP / EP / CP) is not included."
            )

        # Deterministic-algorithm flags have to be set before the model is built.
        set_determinism(job_config.run.deterministic)

        # bf16, except the _keep_in_fp32_modules (see cast_for_inference).
        self.model_dtype = torch.bfloat16

        # ---- prompt ----
        # Every chunk is generated from the same prompt, so encode it and the empty
        # negative prompt once, before the DiT is built, and free the text encoder. Its
        # weights (10.6 GiB in bf16) would otherwise stay on the GPU for the whole run.
        from ppmem.model.pipeline import t5_prompt_embeds
        tokenizer = load_tokenizer(checkpoint_dir, subfolder="tokenizer")
        text_encoder = load_text_encoder(
            checkpoint_dir,
            subfolder="text_encoder",
            torch_dtype=self.model_dtype,
            torch_device=self.device,
        )
        with torch.no_grad():
            self.prompt_embeds = t5_prompt_embeds(tokenizer, text_encoder, prompt, device=self.device)
            self.negative_prompt_embeds = t5_prompt_embeds(tokenizer, text_encoder, "", device=self.device)
        del tokenizer, text_encoder
        device_module.empty_cache()

        self.model_args = MODEL

        logger.info("[builder] Building WanTransformerPlana3DModel")
        with torch.device("meta"):
            model = WanTransformerPlana3DModel(self.model_args)

        # Set the inference dtypes while the model is still on the meta device, so the
        # weights are materialized directly in them instead of in float32 first.
        # init_weights draws values that the checkpoint overwrites; cast_for_inference
        # then casts the buffers init_weights recreates in float32 and adds the float32
        # hooks.
        model.cast_tensors(self.model_dtype)
        model.to_empty(device=device_type)
        with torch.no_grad():
            model.init_weights()
        self.model_parts = [model.cast_for_inference(self.model_dtype)]

        # ---- eval_scheduler ----
        self.eval_scheduler = FlowUniPCMultistepScheduler(shift=job_config.run.flow_shift)

        # ---- pipeline_cls ----
        from ppmem.model.pipeline import PPMemImageToVideoPipeline
        self.pipeline_cls = PPMemImageToVideoPipeline

        # ---- VAE ----
        # Shared with the pipeline; neither side mutates it. Its weights come from here
        # and nowhere else: the published checkpoint does not carry a second copy.
        self.vae = load_vae(
            checkpoint_dir,
            subfolder="vae",
            torch_dtype=self.model_dtype,
            torch_device=self.device,
        )
        self.vae.requires_grad_(False)

        # The forward in ppmem/model/dit.py reads self.vae.config.latents_mean / latents_std.
        for model_part in self.model_parts:
            if hasattr(model_part, "vae"):
                model_part.vae = self.vae

# Assigned, not imported inside the class body: that would also bind the module object.
InferenceEngine.normalize_depth = depth_latent.normalize_depth
InferenceEngine.encode_depth_to_latent = depth_latent.encode_depth_to_latent
