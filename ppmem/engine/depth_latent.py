"""Depth normalization and depth-to-latent encoding, attached to InferenceEngine."""

import math
import torch


@torch.no_grad()
def normalize_depth(
    self,
    depth: torch.Tensor,
    method: str = "linear",
    depth_range: tuple = (0.2, 10.0),
    output_range: tuple = (-1.0, 1.0),
) -> torch.Tensor:
    """
    Normalize depth values to a target range.

    Args:
        depth: Input depth tensor, shape [N, H, W] or [N, 1, H, W]
        method: "linear" maps depth_range onto output_range directly;
            "log" maps it in log space, giving near depths more resolution.
        depth_range: (min_depth, max_depth) in original depth values
        output_range: (min_val, max_val) for normalized output

    Returns:
        Normalized depth tensor with same shape as input
    """
    if method == "linear":
        # Replace invalid values (NaN, Inf, or <= 0) with depth_range[0] before clamping
        # This prevents NaN propagation through VAE encoding
        invalid_mask = (depth <= 0) | torch.isnan(depth) | torch.isinf(depth)
        depth = torch.where(invalid_mask,
                           torch.tensor(depth_range[0], dtype=depth.dtype, device=depth.device),
                           depth)
        depth = torch.clamp(depth, depth_range[0], depth_range[1])
        # Linear normalization: [depth_range[0], depth_range[1]] -> [output_range[0], output_range[1]]
        normalized = (depth - depth_range[0]) / (depth_range[1] - depth_range[0])  # [0, 1]
        normalized = normalized * (output_range[1] - output_range[0]) + output_range[0]  # [output_range[0], output_range[1]]
        return normalized
    elif method == "log":
        # Log-scale normalization: more uniform distribution in [-1, 1]
        # Near depths get more resolution than far depths
        invalid_mask = (depth <= 0) | torch.isnan(depth) | torch.isinf(depth)
        depth = torch.where(invalid_mask,
                           torch.tensor(depth_range[0], dtype=depth.dtype, device=depth.device),
                           depth)
        depth = torch.clamp(depth, depth_range[0], depth_range[1])
        log_depth = torch.log(depth)
        log_min = math.log(depth_range[0])
        log_max = math.log(depth_range[1])
        normalized = (log_depth - log_min) / (log_max - log_min)  # [0, 1]
        normalized = normalized * (output_range[1] - output_range[0]) + output_range[0]
        return normalized
    elif method == "inverse":
        raise NotImplementedError("Inverse depth normalization not implemented yet")
    else:
        raise ValueError(f"Unknown normalization method: {method}")


@torch.no_grad()
def encode_depth_to_latent(
    self,
    depth: torch.Tensor,
    norm_method: str = "linear",
    depth_range: tuple = (0.2, 10.0),
) -> torch.Tensor:
    """
    Encode depth map to VAE latent space.

    Args:
        depth: Input depth tensor, shape [N, H, W] or [N, C, H, W]
        norm_method: Normalization method (see normalize_depth)
        depth_range: Depth range for normalization
    Returns:
        Encoded depth latent, shape [1, z_dim, F_latent, h, w]
    """
    if not hasattr(self, 'vae') or self.vae is None:
        raise RuntimeError(
            "VAE is not loaded. Cannot encode depth to latent."
        )

    # Accept [1, N, H, W] too
    if depth.ndim == 4 and depth.shape[0] == 1:
        depth = depth.squeeze(0)
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]

    depth = self.normalize_depth(depth, method=norm_method, depth_range=depth_range, output_range=(-1.0, 1.0))

    # VAE expects 3-channel input: [N, 1, H, W] -> [N, 3, H, W]
    depth_3ch = depth.repeat(1, 3, 1, 1)

    # VAE expects 5D input [B, C, T, H, W]: [N, 3, H, W] -> [1, 3, N, H, W]
    depth_5d = depth_3ch.unsqueeze(0).permute(0, 2, 1, 3, 4)
    depth_5d = depth_5d.to(dtype=self.model_dtype, device=self.device)

    # One latent per frame. Posterior mean, as for the RGB conditions; the posterior std
    # is tiny compared with the latent scale.
    latent = self.vae.encode_no_temporal_compression(depth_5d).latent_dist.mode()

    # Latent mean/std normalization
    z_dim = self.vae.config.z_dim
    latents_mean = (
        torch.tensor(self.vae.config.latents_mean)
        .view(1, z_dim, 1, 1, 1)
        .to(latent.device, latent.dtype)
    )
    latents_std = (
        (1.0 / torch.tensor(self.vae.config.latents_std))
        .view(1, z_dim, 1, 1, 1)
        .to(latent.device, latent.dtype)
    )
    latent = (latent - latents_mean) * latents_std

    return latent  # [1, z_dim, N, h, w]
