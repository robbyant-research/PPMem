import torch

from ppmem.model.vae import AutoencoderKLWanSeparate


@torch.no_grad()
def encode_video_no_temporal_compression(
    vae: AutoencoderKLWanSeparate,
    videos: torch.Tensor,
    device,
    dtype: torch.dtype,
):
    """
    Encode video without temporal compression - each frame becomes one latent.
    T frames -> T latents (instead of 1 + (T-1)//4 latents)

    """
    videos = videos.to(device=device, dtype=dtype)
    if isinstance(vae, AutoencoderKLWanSeparate):
        # diffusers' encode returns un-normalized latents; normalize here so the
        # return value matches the WAN reference.
        cfg = vae.config
        latents_mean = (
            torch.tensor(cfg.latents_mean).view(1, cfg.z_dim, 1, 1, 1).to(videos.device, dtype)
        )
        latents_std = (
            (1.0 / torch.tensor(cfg.latents_std)).view(1, cfg.z_dim, 1, 1, 1).to(videos.device, dtype)
        )
        latents = vae.encode_no_temporal_compression(videos).latent_dist.mode().to(dtype)
        latents = (latents - latents_mean) * latents_std
    else:
        raise ValueError("Unknown vae type")
    return latents
