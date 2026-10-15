"""Latent Encoder for Incremental Video Generation"""

import torch
import numpy as np
from einops import rearrange

from ._encode import encode_video_no_temporal_compression


class LatentEncoder:
    """Encode RGB images to latent space using VAE.

    """

    def __init__(self, vae, device='cuda', dtype=torch.bfloat16):
        """
        Args:
            vae: an already-loaded AutoencoderKLWanSeparate (required).
            device: Device to run on
            dtype: Data type for encoding

        """
        self.device = device
        self.dtype = dtype
        self.vae = vae

    def _prepare_video_tensor(self, images, normalize=True):
        """
        Convert images to a (1, 3, F, H, W) tensor on device.

        Args:
            images: (N, H, W, 3) numpy array [0-255] or (N, 3, H, W) tensor [-1, 1]
            normalize: Whether to normalize from [0-255] to [-1, 1]
        """
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = images[None, ...]  # (H, W, 3) -> (1, H, W, 3)

            # Convert to tensor
            images = torch.from_numpy(images).float()  # (N, H, W, 3)

            if normalize:
                # Normalize [0, 255] -> [-1, 1]
                images = images / 255.0 * 2.0 - 1.0

            # (N, H, W, 3) -> (N, 3, H, W)
            images = images.permute(0, 3, 1, 2)

        # (N, 3, H, W) -> (3, N, H, W) -> (1, 3, N, H, W)
        video_tensor = images.permute(1, 0, 2, 3).unsqueeze(0)

        return video_tensor.to(device=self.device, dtype=self.dtype)

    def _encode_video_tensor(self, video_tensor):
        """Encode a (1, 3, F, H, W) tensor per-frame (F -> F)."""
        return encode_video_no_temporal_compression(
            vae=self.vae,
            videos=video_tensor,
            device=self.device,
            dtype=self.dtype,
        )

    def encode_images_batch(self, images_dict, normalize=True, keys_no_normalize=None):
        """
        Encode multiple image sequences

        Args:
            images_dict: Dict of {name: images_array}
                        images_array: (N, H, W, 3) numpy array [0-255] or [-1, 1] for mask
            normalize: Whether to normalize [0,255] -> [-1,1] (default True for RGB)
            keys_no_normalize: Optional set/list of keys that are already [-1,1] (e.g. mask), skip /255*2-1

        Returns:
            latents_dict: Dict of {name: latents}
        """
        latents_dict = {}
        no_norm = set(keys_no_normalize or [])

        for name, images in images_dict.items():
            use_normalize = normalize and (name not in no_norm)
            video_tensor = self._prepare_video_tensor(images, normalize=use_normalize)
            # Encode all frames
            latents_dict[name] = self._encode_video_tensor(video_tensor)

        return latents_dict

    def rearrange_latents(self, latents_dict):
        """
        Rearrange latents from (B, C, F, H, W) to (B, F*H*W, C)
        
        Args:
            latents_dict: Dict of {name: (B, C, F, H, W) tensor}
        
        Returns:
            rearranged_dict: Dict of {name: (B, F*H*W, C) tensor}
        """
        rearranged = {}
        for name, latents in latents_dict.items():
            rearranged[name] = rearrange(latents, 'b c f h w -> b (f h w) c')
        return rearranged
