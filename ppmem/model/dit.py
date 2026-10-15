import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from einops import rearrange
from torch.utils._pytree import tree_map
from typing import Any, List, Dict, Optional, Union, Tuple

from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders import FromOriginalModelMixin
from diffusers.models.attention import AttentionMixin
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

from ppmem._logging import logger
from .attention import build_attn_mask


# `model_args` is the config's model section itself, not a second copy of it.
from ppmem.config import Model as ModelArgs
from .blocks import WanTransformerBlock
from .embedding import WanTimeTextImageEmbedding, WanRotaryPosEmbed
from .heads import PlaneHead
from .lora import LoRALinear, apply_lora_to_block, parse_block_range, parse_lora_targets
from .planes import normalize_quaternion, quat_to_rot


FREEZE_TRAINABLE_PREFIXES = (
    "plane_head",
    "feature_decoder",
    "decoder_out_proj",
    "decoder_mid_proj",
    "proj_out_feat",
    "depth_token_fusion",
)

try:
    from diff_rect_rasterization import RectRasterizationSettings, RectRasterizer
    RASTERIZER_AVAILABLE = True
except ImportError:
    RASTERIZER_AVAILABLE = False
    logger.warning("diff_rect_rasterization not available; render_primitives will not work.")


def compute_patch_depth_stats(depth, patch_h, patch_w, valid_threshold=0.01):
    """
    Compute per-patch depth statistics for sparse depth maps.
    Only averages over valid (non-zero) pixels, preserving metric scale.

    Args:
        depth: [N, H, W] raw depth map, 0 = invalid/hole
        patch_h, patch_w: pixel region each token covers in the original image
        valid_threshold: minimum depth value to be considered valid

    Returns:
        mean_depth: [N, h_out, w_out] mean of valid depth pixels per patch (0 if no valid pixels)
        coverage:   [N, h_out, w_out] fraction of valid pixels per patch [0, 1]
    """
    valid = (depth > valid_threshold).float()
    depth_valid = depth * valid

    N, H, W = depth.shape
    h_out = H // patch_h
    w_out = W // patch_w

    # Crop to exact patch grid (drop remainder pixels)
    depth_patches = depth_valid[:, :h_out * patch_h, :w_out * patch_w]
    valid_patches = valid[:, :h_out * patch_h, :w_out * patch_w]

    # Reshape to [N, h_out, patch_h, w_out, patch_w] and sum over patch dims
    depth_patches = depth_patches.reshape(N, h_out, patch_h, w_out, patch_w)
    valid_patches = valid_patches.reshape(N, h_out, patch_h, w_out, patch_w)

    valid_sum = valid_patches.sum(dim=(2, 4))   # [N, h_out, w_out]
    depth_sum = depth_patches.sum(dim=(2, 4))   # [N, h_out, w_out]

    coverage = valid_sum / (patch_h * patch_w)
    mean_depth = depth_sum / (valid_sum + 1e-8)
    mean_depth = mean_depth * (valid_sum > 0).float()  # zero where no valid pixels

    return mean_depth, coverage


class FlashAttnVarlenEncoderLayer(nn.Module):
    """
    Drop-in replacement for nn.TransformerEncoderLayer(norm_first=True)
    that uses flash_attn_varlen_func for variable-length batched attention.

    Parameter names match nn.TransformerEncoderLayer exactly for checkpoint compatibility:
      self_attn.in_proj_weight, self_attn.in_proj_bias,
      self_attn.out_proj.weight, self_attn.out_proj.bias,
      linear1.weight/bias, linear2.weight/bias,
      norm1.weight/bias, norm2.weight/bias
    """
    def __init__(self, d_model, nhead, dim_feedforward, activation='gelu'):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Self-attention (manual QKV projection, matching nn.MultiheadAttention param names)
        self.self_attn = nn.Module()
        self.self_attn.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.self_attn.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        self.self_attn.out_proj = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # LayerNorms (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()

    def forward(self, x, cu_seqlens, max_seqlen):
        """
        Args:
            x: [total_tokens, D] — all sequences concatenated
            cu_seqlens: [n_seq + 1], int32
            max_seqlen: int
        Returns: [total_tokens, D]
        """
        from flash_attn_interface import flash_attn_varlen_func

        x_norm = self.norm1(x)

        qkv = F.linear(x_norm, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)  # each [total_tokens, D]

        # Reshape for flash_attn: [total_tokens, nhead, head_dim]
        q = q.view(-1, self.nhead, self.head_dim)
        k = k.view(-1, self.nhead, self.head_dim)
        v = v.view(-1, self.nhead, self.head_dim)

        attn_out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
        )
        attn_out = attn_out.view(-1, self.d_model)  # [total_tokens, D]

        x = x + self.self_attn.out_proj(attn_out)

        x_norm = self.norm2(x)
        x = x + self.linear2(self.activation(self.linear1(x_norm)))

        return x


class FlashAttnVarlenEncoder(nn.Module):
    """
    Drop-in replacement for nn.TransformerEncoder using flash_attn_varlen.
    """
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, activation='gelu'):
        super().__init__()
        self.layers = nn.ModuleList([
            FlashAttnVarlenEncoderLayer(d_model, nhead, dim_feedforward, activation)
            for _ in range(num_layers)
        ])

    def forward(self, x, cu_seqlens, max_seqlen):
        """
        Args:
            x: [total_tokens, D]
            cu_seqlens: [n_seq + 1], int32
            max_seqlen: int
        Returns: [total_tokens, D]
        """
        for layer in self.layers:
            x = layer(x, cu_seqlens, max_seqlen)
        return x


class DepthTokenFusionModule(nn.Module):
    """
    Token-based depth fusion.
    Ref: arxiv 2601.17895v1

    Depth tokenization: Conv2d patch embedding (in_chans=1)
    Modality encoding: additive constants (+1 feat, +2 depth) on shared spatial PE
    Masking: retain depth tokens whose coverage is not below 0.01.
    Backbone: FlashAttnVarlenEncoder (768 dim, 12 heads, 12 layers)
    """
    VIT_DIM = 768
    VIT_HEADS = 12
    VIT_LAYERS = 12
    VIT_FFN_DIM = 3072  # 768 * 4

    def __init__(self, feat_dim, patch_size=16, pe_grid_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.pe_grid_size = pe_grid_size  # Square PE grid, bicubic interpolated to actual resolution
        D = self.VIT_DIM

        # Feature projection: feat_dim <-> ViT dim
        self.proj_in = nn.Linear(feat_dim, D)
        self.proj_out = nn.Linear(D, feat_dim)

        # Depth patch embedding: raw depth [1, 1, H, W] -> [1, D, H_lat, W_lat]
        self.depth_patch_embed = nn.Conv2d(1, D, kernel_size=patch_size, stride=patch_size, bias=True)

        # Shared 2D spatial PE (DINOv2 style: square grid, bicubic interpolated to target resolution)
        self.spatial_pe = nn.Parameter(torch.zeros(1, pe_grid_size, pe_grid_size, D))

        # No learnable modality embedding: feat tokens get +1, depth tokens get +2,
        # added to the spatial PE.

        # ViT-Base encoder (flash_attn_varlen for variable-length batched attention)
        self.encoder = FlashAttnVarlenEncoder(
            d_model=D, nhead=self.VIT_HEADS, dim_feedforward=self.VIT_FFN_DIM,
            num_layers=self.VIT_LAYERS, activation='gelu',
        )
        self.final_norm = nn.LayerNorm(D)


    def reset_parameters(self):
        # === 1. DINOv2-style init (trunc_normal for Linear, kaiming for Conv2d) ===
        # proj_in: trunc_normal (DINOv2 style for all Linear layers)
        nn.init.trunc_normal_(self.proj_in.weight, std=0.02)
        nn.init.zeros_(self.proj_in.bias)

        # ViT encoder: trunc_normal for Linear/MHA, default for LayerNorm
        for layer in self.encoder.layers:
            nn.init.trunc_normal_(layer.self_attn.in_proj_weight, std=0.02)
            nn.init.zeros_(layer.self_attn.in_proj_bias)
            nn.init.trunc_normal_(layer.self_attn.out_proj.weight, std=0.02)
            nn.init.zeros_(layer.self_attn.out_proj.bias)
            nn.init.trunc_normal_(layer.linear1.weight, std=0.02)
            nn.init.zeros_(layer.linear1.bias)
            nn.init.trunc_normal_(layer.linear2.weight, std=0.02)
            nn.init.zeros_(layer.linear2.bias)
            # LayerNorms (PyTorch default: weight=1, bias=0)
            nn.init.ones_(layer.norm1.weight)
            nn.init.zeros_(layer.norm1.bias)
            nn.init.ones_(layer.norm2.weight)
            nn.init.zeros_(layer.norm2.bias)

        nn.init.ones_(self.final_norm.weight)
        nn.init.zeros_(self.final_norm.bias)

        # spatial_pe: trunc_normal (DINOv2 pos_embed style)
        nn.init.trunc_normal_(self.spatial_pe, std=0.02)

        # depth_patch_embed: kaiming uniform (DINOv2 PatchEmbed Conv2d style)
        nn.init.kaiming_uniform_(self.depth_patch_embed.weight, a=math.sqrt(5))
        nn.init.zeros_(self.depth_patch_embed.bias)

        nn.init.trunc_normal_(self.proj_out.weight, std=0.02)
        nn.init.zeros_(self.proj_out.bias)

    def interpolate_pos_encoding(self, h_tokens, w_tokens):
        """Bicubic interpolate 2D spatial PE to target (h_tokens, w_tokens).
        Following DINOv2's interpolate_pos_encoding_without_cls."""
        D = self.VIT_DIM
        M = self.pe_grid_size
        if h_tokens == M and w_tokens == M:
            return self.spatial_pe.reshape(1, M * M, D)
        pe = self.spatial_pe.permute(0, 3, 1, 2).float()  # [1, D, M, M]
        pe = F.interpolate(pe, size=(h_tokens, w_tokens), mode='bicubic', align_corners=False)
        pe = pe.permute(0, 2, 3, 1).reshape(1, h_tokens * w_tokens, D)  # [1, S, D]
        return pe.to(self.spatial_pe.dtype)

    def forward(self, feat_tokens, raw_depth_frames, coverage, h_tokens, w_tokens):
        """
        Depth fusion with flash_attn_varlen. Supports per-frame and batched input.

        Per-frame:
            feat_tokens: [S, C], raw_depth_frames: [H, W], coverage: [S]
            Returns: [S, C]
        Batched:
            feat_tokens: [N, S, C], raw_depth_frames: [N, H, W], coverage: [N, S]
            Returns: [N, S, C]
        """
        batched = feat_tokens.dim() == 3
        if not batched:
            feat_tokens = feat_tokens.unsqueeze(0)
            raw_depth_frames = raw_depth_frames.unsqueeze(0)
            coverage = coverage.unsqueeze(0)

        N, S, _ = feat_tokens.shape
        device = feat_tokens.device

        model_dtype = self.proj_in.weight.dtype
        feat_tokens = feat_tokens.to(model_dtype)

        pos_enc = self.interpolate_pos_encoding(h_tokens, w_tokens)[0]  # [S, D]

        # Feature tokens: [N, S, C] -> [N, S, D] + PE + modality
        feat_vit = self.proj_in(feat_tokens) + pos_enc.unsqueeze(0) + 1

        # Depth tokens: [N, 1, H, W] -> Conv2d -> [N, S, D] + PE + modality
        depth_2d = raw_depth_frames.unsqueeze(1).to(feat_tokens.dtype)
        depth_vit_all = self.depth_patch_embed(depth_2d).flatten(2).transpose(1, 2)
        depth_vit_all = depth_vit_all + pos_enc.unsqueeze(0) + 2

        # Keep every token the pool actually covers; drop only the empty ones.
        # This comparison retains NaN coverage; coverage >= 0.01 would discard it.
        keep = ~(coverage < 0.01)

        # Build varlen concat: [feat_0, depth_kept_0, feat_1, depth_kept_1, ...]
        # Each sequence has length S + K_i (variable K_i)
        seq_parts = []
        seq_lens = []
        for i in range(N):
            depth_kept_i = depth_vit_all[i][keep[i]]  # [K_i, D]
            seq_parts.append(feat_vit[i])       # [S, D]
            seq_parts.append(depth_kept_i)      # [K_i, D]
            seq_lens.append(S + depth_kept_i.shape[0])

        combined = torch.cat(seq_parts, dim=0)  # [total_tokens, D]

        cu_seqlens = torch.zeros(N + 1, dtype=torch.int32, device=device)
        cu_seqlens[1:] = torch.tensor(seq_lens, dtype=torch.int32, device=device).cumsum(0)
        max_seqlen = max(seq_lens)

        # Encoder (flash_attn_varlen)
        combined = self.encoder(combined, cu_seqlens, max_seqlen)
        combined = self.final_norm(combined)

        # Split output back, extract feat tokens (first S per sequence)
        enhanced_list = []
        # cu_seqlens is the cumsum of seq_lens; taking the offsets from the Python list
        # avoids a device sync per sequence.
        start = 0
        for i in range(N):
            enhanced_list.append(combined[start:start + S])  # [S, D]
            start += seq_lens[i]
        enhanced_vit = torch.stack(enhanced_list)  # [N, S, D]

        enhanced = self.proj_out(enhanced_vit)  # [N, S, C]

        if not batched:
            enhanced = enhanced.squeeze(0)
        return enhanced

def compute_gradient_magnitude(x, mode='sobel'):
    """
    Compute gradient magnitude for boundary detection.
    
    Args:
        x: [B, H, W, C] tensor
        mode: 'sobel' or 'simple'
    
    Returns:
        total_magnitude: [B, H, W] tensor
    """
    # x: [B, H, W, C]
    x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

    if mode == 'sobel':
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

        C = x.shape[1]
        sobel_x = sobel_x.repeat(C, 1, 1, 1)
        sobel_y = sobel_y.repeat(C, 1, 1, 1)

        x_padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='replicate')

        gradient_x = torch.nn.functional.conv2d(x_padded, sobel_x, groups=C)  # [B, C, H, W]
        gradient_y = torch.nn.functional.conv2d(x_padded, sobel_y, groups=C)  # [B, C, H, W]

        magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)  # [B, C, H, W]
        total_magnitude = magnitude.sum(dim=1)  # [B, H, W]

    elif mode == 'simple':
        gradient_x = x[:, :, :, 1:] - x[:, :, :, :-1]  # [B, C, H, W-1]
        gradient_y = x[:, :, 1:, :] - x[:, :, :-1, :]  # [B, C, H-1, W]

        gradient_x = torch.nn.functional.pad(gradient_x, (0, 1, 0, 0))  # [B, C, H, W]
        gradient_y = torch.nn.functional.pad(gradient_y, (0, 0, 0, 1))  # [B, C, H, W]

        magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)
        total_magnitude = magnitude.sum(dim=1)  # [B, H, W]
    else:
        raise NotImplementedError

    return total_magnitude


def generate_gradient_mask(normal_map, ratio=0.1, split_threshold=None):
    """
    Generate binary mask for high-gradient regions (boundaries).
    
    Args:
        normal_map: [B, H, W, 3] or [T, H, W, 3] normal map tensor
        ratio: threshold ratio for gradient magnitude (default 0.1)
        split_threshold: if provided, overrides ratio
    
    Returns:
        mask: [B, H, W] or [T, H, W] boolean tensor (True = boundary/high gradient)
    """
    threshold = split_threshold if split_threshold is not None else ratio
    grad_magnitude = compute_gradient_magnitude(normal_map)
    mask = grad_magnitude > threshold
    return mask


class WanCausalConv3d(nn.Conv3d):
    """
    A custom 3D causal convolution layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x):
        x = F.pad(x, self._padding)
        return super().forward(x)


class WanRMS_norm(nn.Module):
    """
    A custom RMS normalization layer.
    """

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class WanResidualBlock(nn.Module):
    """
    A custom residual block module (from WAN VAE).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.nonlinearity = nn.SiLU() if non_linearity == "silu" else nn.ReLU()

        # layers
        self.norm1 = WanRMS_norm(in_dim, images=False)
        self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanRMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = WanCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        h = self.conv_shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)

        x = self.dropout(x)
        x = self.conv2(x)

        return x + h


class WanAttentionBlock(nn.Module):
    """
    Single-head spatial self-attention within each frame (from the WAN VAE).
    Operates per-frame on [B*T, C, H, W] with Conv2d.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = WanRMS_norm(dim, images=True)  # for 2D [B*T, C, H, W]
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        batch_size, channels, time, height, width = x.size()

        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        x = F.scaled_dot_product_attention(q, k, v)

        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)
        x = self.proj(x)

        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)

        return x + identity


class DupUp3D(nn.Module):
    """
    Duplicate-upsample 3D: repeat channels and reshape to expand T and/or H,W.
    From WAN VAE WanResidualUpBlock avg_shortcut (no learnable params).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int = 1,
        factor_s: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk: bool = False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class WanLikeFeatureDecoder(nn.Module):
    """
    Mini WAN-like decoder for feature upsampling.
    Uses only the first up_block.

    Follows the structure of the WAN VAE decoder:
      conv_in(post_quant_conv(z)) + extra_conv_in(feats)
      -> mid_block (ResBlock -> Attention -> ResBlock)
      -> up_block (3 ResBlocks + 2x spatial upsample Conv2d -> out_dim, plus an averaging shortcut)
    """

    def __init__(self, z_dim, feat_dim, hidden_dim=512, dropout=0.0):
        super().__init__()
        self.out_dim = hidden_dim // 2

        self.post_quant_conv = WanCausalConv3d(z_dim, z_dim, 1)
        self.conv_in = WanCausalConv3d(z_dim, hidden_dim, 3, padding=1)
        self.extra_conv_in = WanCausalConv3d(feat_dim, hidden_dim, 3, padding=1)
        self.norm = WanRMS_norm(hidden_dim, images=False)

        # mid_block: ResBlock -> Attention -> ResBlock (matches WAN mid_block)
        self.mid_resnets = nn.ModuleList([
            WanResidualBlock(hidden_dim, hidden_dim, dropout),
            WanResidualBlock(hidden_dim, hidden_dim, dropout),
        ])
        self.mid_attention = WanAttentionBlock(hidden_dim)

        # up_block: 3 ResBlocks (matches WAN up_blocks[0].resnets)
        self.up_resnets = nn.ModuleList([
            WanResidualBlock(hidden_dim, hidden_dim, dropout),
            WanResidualBlock(hidden_dim, hidden_dim, dropout),
            WanResidualBlock(hidden_dim, hidden_dim, dropout),
        ])
        # avg_shortcut for Wan 2.2 (WanResidualUpBlock): spatial 2x only, no temporal
        self.up_avg_shortcut = DupUp3D(
            in_channels=hidden_dim,
            out_channels=self.out_dim,
            factor_t=1,
            factor_s=2,
        )
        # 2x spatial upsample + Conv2d (matches WAN upsamplers[0].resample)
        self.nonlinearity = nn.SiLU()
        self.up_conv = nn.Conv2d(hidden_dim, self.out_dim, 3, padding=1)

    def reset_parameters(self):
        """Initialize all parameters to valid defaults.
        
        gamma must be re-initialized here: the model is built on the meta device and
        to_empty() wipes the gamma=1 set in __init__.
        """
        for module in self.modules():
            if isinstance(module, WanRMS_norm):
                if module.gamma is not None:
                    nn.init.ones_(module.gamma)
                if isinstance(module.bias, nn.Parameter):
                    nn.init.zeros_(module.bias)
        nn.init.kaiming_normal_(self.up_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.up_conv.bias is not None:
            nn.init.zeros_(self.up_conv.bias)
        nn.init.zeros_(self.extra_conv_in.weight)
        nn.init.zeros_(self.extra_conv_in.bias)

    def forward(self, feats, z):
        """
        Args:
            feats: [B, feat_dim, T, H, W] - unpatchified features
            z: [B, z_dim, T, H, W] - unpatchified latent predictions
        Returns:
            mid_out: [B, hidden_dim, T, H, W] - mid_block output (low-res)
            up_out:  [B, out_dim, T, 2H, 2W] - upsampled output (high-res)
        """
        x = self.conv_in(self.post_quant_conv(z)) + self.extra_conv_in(feats)

        # mid_block: ResBlock -> Attention -> ResBlock
        x = self.mid_resnets[0](x)
        x = self.mid_attention(x)
        x = self.mid_resnets[1](x)

        mid_out = x

        # up_block: save input for avg_shortcut, then resnets -> norm -> upsample
        up_input = x
        for block in self.up_resnets:
            x = block(x)

        # norm and SiLU act on each pixel on its own, so running them before the nearest 2x
        # upsample gives the same values on a quarter of the pixels.
        x = self.norm(x)
        x = self.nonlinearity(x)

        # 2x spatial upsample (per-frame Conv2d, like WanResample "upsample2d")
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.up_conv(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', b=b, t=t)

        # Wan 2.2 avg_shortcut: add upsampled shortcut from block input
        x = x + self.up_avg_shortcut(up_input, first_chunk=False)

        return mid_out, x


def _fp32_inputs(module, args, kwargs):
    """Forward pre-hook for a _keep_in_fp32_modules module: run it in float32."""
    ref = next((a for a in (*args, *kwargs.values()) if torch.is_tensor(a) and a.is_floating_point()), None)
    module._fp32_caller_dtype = None if ref is None else ref.dtype
    up = lambda a: a.float() if torch.is_tensor(a) and a.is_floating_point() else a
    return tuple(up(a) for a in args), {k: up(v) for k, v in kwargs.items()}


def _caller_dtype_outputs(module, args, kwargs, output):
    """Forward hook: hand the outputs back in the dtype the module's input came in."""
    dtype = module._fp32_caller_dtype
    if dtype is None or dtype == torch.float32:
        return output
    return tree_map(lambda o: o.to(dtype) if torch.is_tensor(o) and o.is_floating_point() else o, output)


class WanTransformerPlana3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin):
    _skip_layerwise_casting_patterns = ["patch_embedding_mlp", "patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1", "norm2", "norm3",
        "plane_head",               # Geometric parameter prediction (depth, radii, rotation)
        "feature_decoder",          # WAN-like feature decoder
        "decoder_out_proj",         # Decoder output projection (high-res)
        "decoder_mid_proj",         # Decoder mid projection (low-res)
        "proj_out",                 # Final output projection
        "proj_out_feat",            # Feature output projection
        "patch_embedding_mlp",      # Input patch embedding
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    def __init__(
        self,
        model_args: ModelArgs,
    ) -> None:
        super().__init__()
        self.model_args = model_args
        inner_dim = model_args.num_attention_heads * model_args.attention_head_dim
        out_channels = model_args.out_channels or model_args.in_channels
        out_channels_feat = model_args.out_channels_feat or model_args.in_channels

        self.scheduler = None

        # Set by the builder; the geometry pass reads its latents_mean / latents_std.
        self.vae = None

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(model_args.attention_head_dim, model_args.patch_size)
        self.patch_embedding_mlp = nn.Linear(model_args.in_channels * model_args.patch_size[0] * model_args.patch_size[1] * model_args.patch_size[2], inner_dim)

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=model_args.freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=model_args.text_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    model_args.ffn_dim,
                    model_args.num_attention_heads,
                    model_args.cross_attn_norm,
                    model_args.eps,
                    model_args.added_kv_proj_dim,
                )
                for _ in range(model_args.num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, model_args.eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(model_args.patch_size))
        self.proj_out_feat = nn.Linear(inner_dim, out_channels_feat * math.prod(model_args.patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        # 5. WAN-like feature decoder
        decoder_hidden_dim = model_args.decoder_hidden_dim
        # Original decoder: mid_block + up_block[0] only (no temporal upsampling)
        self.feature_decoder = WanLikeFeatureDecoder(
            z_dim=out_channels,
            feat_dim=out_channels_feat,
            hidden_dim=decoder_hidden_dim,
        )

        # 5b. Depth token fusion
        self.depth_token_fusion = DepthTokenFusionModule(
            feat_dim=out_channels_feat,  # 240
        )

        # 6. Plane head for predicting plane parameters.
        # Features arrive unpatchified as [B, L*4, out_channels_feat], so dec_dim is
        # out_channels_feat.
        plane_head_hidden_dim = min(1024, model_args.ffn_dim)
        plane_head_dec_dim = out_channels_feat

        self.plane_head = PlaneHead(
            dec_dim=plane_head_dec_dim,
            hidden_dim=plane_head_hidden_dim,
            depth_mode=model_args.depth_mode,
        )

        # Projections from decoder dims → plane_head dec_dim
        decoder_hidden_dim = model_args.decoder_hidden_dim
        decoder_out_dim = decoder_hidden_dim // 2  # matches WAN upsampler output (512)
        self.decoder_out_proj = nn.Linear(decoder_out_dim, plane_head_dec_dim)
        # Low-res path: use mid_block output (1024 ch)
        self.decoder_mid_proj = nn.Linear(decoder_hidden_dim, plane_head_dec_dim)


        # Plane normal standard (z-axis in camera space); cast to float32 where it is used
        self.register_buffer('plane_normal_standard', torch.tensor([0., 0., 1.], dtype=torch.float32).reshape(1, 3, 1))

        # Cache for pixel coordinates
        self.pixel_coords_dict = {}

        # Background color for rendering; the rasterizer gets it as float32
        self.register_buffer('bg', torch.tensor([0., 0., 0.], dtype=torch.float32))

        # Injects the depth LoRA. The requires_grad bookkeeping around it is inert at inference.
        self._apply_freeze_for_depth(model_args)

    def _apply_freeze_for_depth(self, model_args) -> None:
        """Freeze the DiT backbone, keep the depth modules trainable, partial-freeze
        patch_embedding_mlp, and inject LoRA on attn1.

        The requires_grad bookkeeping is inert at inference.
        """

        for name, p in self.named_parameters():
            top = name.split(".", 1)[0]
            if top in FREEZE_TRAINABLE_PREFIXES:
                continue
            p.requires_grad_(False)

        # patch_embedding_mlp takes the RGB conditioning channels first and the depth
        # latent after them, so the whole weight stays trainable while the bias does not.
        self.patch_embedding_mlp.weight.requires_grad_(True)
        if self.patch_embedding_mlp.bias is not None:
            self.patch_embedding_mlp.bias.requires_grad_(False)
        # 3) Optional LoRA on blocks. Defaults to attn1 only; can be extended
        #    to include attn2 / ffn via lora_targets config.
        lora_spec = model_args.lora_on_dit_blocks or ''
        lora_rank = int(model_args.lora_rank)
        lora_alpha = int(model_args.lora_alpha)
        lora_targets_spec = model_args.lora_targets or 'attn1'
        lora_targets = parse_lora_targets(lora_targets_spec)
        lora_idx = parse_block_range(lora_spec, num_blocks=len(self.blocks))
        for i in lora_idx:
            apply_lora_to_block(self.blocks[i], rank=lora_rank, alpha=lora_alpha, targets=lora_targets)

        logger.debug(f"LoRA: blocks={lora_idx} targets={lora_targets} "
                    f"(rank={lora_rank}, alpha={lora_alpha})")

    def cast_for_inference(self, dtype):
        """Cast to dtype, except _keep_in_fp32_modules, which stay and run in float32.

        diffusers' from_pretrained keeps those modules in float32; model.to(dtype)
        would cast them too. Each kept module receives float32 inputs and returns its
        outputs in the dtype its input came in, so the code around it is unchanged:
        patch_embedding_mlp still feeds bfloat16 to the blocks, and the heads, whose
        inputs are already float32, return float32.
        """
        keep = set(self._keep_in_fp32_modules or ())
        self.cast_tensors(dtype)
        for name, module in self.named_modules():
            parts = name.split(".")
            if keep.intersection(parts[-1:]) and not keep.intersection(parts[:-1]):  # outermost only
                module.register_forward_pre_hook(_fp32_inputs, with_kwargs=True)
                module.register_forward_hook(_caller_dtype_outputs, with_kwargs=True)
        return self

    def cast_tensors(self, dtype):
        """Cast every floating parameter and buffer outside _keep_in_fp32_modules to dtype.

        The dtype half of cast_for_inference. It also works on the meta device, which
        lets the builder materialize the weights in dtype instead of float32.
        """
        keep = set(self._keep_in_fp32_modules or ())
        with torch.no_grad():
            for name, t in [*self.named_parameters(), *self.named_buffers()]:
                if t.is_floating_point() and not keep.intersection(name.split(".")):
                    t.data = t.data.to(dtype)
        return self

    def init_weights(
        self,
    ):
        """Initialize model weights following the original Wan implementation."""
        # Buffers must be re-materialized: to_empty() leaves them uninitialized.
        # This must be done BEFORE any other initialization
        self.plane_normal_standard = torch.tensor([0., 0., 1.], dtype=torch.float32, device=self.plane_normal_standard.device).reshape(1, 3, 1)
        self.bg = torch.tensor([0., 0., 0.], dtype=torch.float32, device=self.bg.device)

        nn.init.normal_(self.patch_embedding_mlp.weight, std=0.02)
        if self.patch_embedding_mlp.bias is not None:
            nn.init.constant_(self.patch_embedding_mlp.bias, 0)

        # Initialize condition embedder (time/text/image embeddings)
        self.condition_embedder.init_weights()

        for block in self.blocks:
            block.init_weights()

        # LoRA is initialized here, after materialization: the model is built on the
        # meta device, where trunc_normal_ inside LoRALinear.__init__ is a no-op.
        for blk in self.blocks:
            for m in blk.modules():
                if isinstance(m, LoRALinear):
                    with torch.no_grad():
                        nn.init.trunc_normal_(m.lora_A, std=0.02)
                        nn.init.zeros_(m.lora_B)

        self.norm_out.reset_parameters()
        nn.init.normal_(self.proj_out.weight, std=0.02)
        if self.proj_out.bias is not None:
            nn.init.constant_(self.proj_out.bias, 0)

        nn.init.normal_(self.proj_out_feat.weight, std=0.02)
        if self.proj_out_feat.bias is not None:
            nn.init.constant_(self.proj_out_feat.bias, 0)

        nn.init.normal_(self.scale_shift_table, std=0.02)

        self.rope.init_weights()

        if hasattr(self, 'feature_decoder'):
            self.feature_decoder.reset_parameters()
            nn.init.zeros_(self.decoder_out_proj.bias)
            nn.init.normal_(self.decoder_out_proj.weight, std=0.02)
            nn.init.zeros_(self.decoder_mid_proj.bias)
            nn.init.normal_(self.decoder_mid_proj.weight, std=0.02)

        if hasattr(self, 'depth_token_fusion') and self.depth_token_fusion is not None:
            self.depth_token_fusion.reset_parameters()

        if hasattr(self, 'plane_head') and self.plane_head is not None:
            self.plane_head.reset_parameters()
        else:
            logger.warning("[init_weights] plane_head not found or is None!")

    def _build_rotary_emb(self, latent_num_frames_lst, latent_height_lst, latent_width_lst,
                          num_history_frames_lst, num_very_first_frames_lst,
                          num_context_frames_lst, device):
        rotary_emb = []
        for i, (f, h, w) in enumerate(zip(latent_num_frames_lst, latent_height_lst, latent_width_lst)):
            f_patch = int(f) // self.model_args.patch_size[0]
            h_patch = int(h) // self.model_args.patch_size[1]
            w_patch = int(w) // self.model_args.patch_size[2]

            n_history  = int(num_history_frames_lst[i])  if num_history_frames_lst  is not None else 0
            n_vf       = int(num_very_first_frames_lst[i]) if num_very_first_frames_lst is not None else 0
            n_ctx      = int(num_context_frames_lst[i])  if num_context_frames_lst  is not None else 0
            n_main_patch = f_patch - n_history - n_vf - n_ctx

            parts = []
            if n_history > 0:
                idx = -3
                parts.append(self.rope(n_history, h_patch, w_patch, device=device, f_const_idx=idx))
            if n_vf > 0:
                idx = -2
                parts.append(self.rope(n_vf, h_patch, w_patch, device=device, f_const_idx=idx))
            if n_ctx > 0:
                parts.append(self.rope(n_ctx, h_patch, w_patch, device=device, f_const_idx=-1))
            parts.append(self.rope(n_main_patch, h_patch, w_patch, device=device, f_start_idx=0))
            rope_emb = torch.cat(parts, dim=0)
            rotary_emb.append(rope_emb)

        rotary_emb = torch.cat(rotary_emb)[None, ...]
        return rotary_emb

    def forward_pre_process(
        self,
        timestep: torch.Tensor,
        hidden_states: List[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        latent_num_frames_lst: List[int] = None,
        latent_height_lst: List[int] = None,
        latent_width_lst: List[int] = None,
        text_emb_len_lst: List[int] = None,
        num_context_frames_lst: List[int] = None,
        num_very_first_frames_lst: List[int] = None,
        num_history_frames_lst: List[int] = None,
    ):

        # RoPE
        # Frame ordering in sequence: [history | very_first | regular_ctx | main_frames]
        # RoPE strategy:
        #   history=-3, very_first=-2, regular_ctx=-1
        # The table depends only on these counts and sizes, which stay the same for every
        # forward of a chunk; it is built on the CPU in float64, so build it once per chunk.
        rope_key = (
            tuple(int(x) for x in latent_num_frames_lst), tuple(int(x) for x in latent_height_lst),
            tuple(int(x) for x in latent_width_lst),
            tuple(int(x) for x in num_history_frames_lst) if num_history_frames_lst is not None else None,
            tuple(int(x) for x in num_very_first_frames_lst) if num_very_first_frames_lst is not None else None,
            tuple(int(x) for x in num_context_frames_lst) if num_context_frames_lst is not None else None,
            timestep.device,
        )
        cached = getattr(self, '_rope_cache', None)
        if cached is not None and cached[0] == rope_key:
            rotary_emb = cached[1]
        else:
            rotary_emb = self._build_rotary_emb(
                latent_num_frames_lst, latent_height_lst, latent_width_lst,
                num_history_frames_lst, num_very_first_frames_lst, num_context_frames_lst,
                timestep.device)
            self._rope_cache = (rope_key, rotary_emb)

        # Patchify and embed
        hidden_states = [
            rearrange(
                i,
                '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)',
                c1=self.model_args.patch_size[0],
                c2=self.model_args.patch_size[1],
                c3=self.model_args.patch_size[2],
            ) for i in hidden_states
        ]
        hidden_states = torch.cat(hidden_states, dim=1) # [1, (L1+...+Ln), C]
        hidden_states = self.patch_embedding_mlp(hidden_states)

        # Query / key sequence lengths for the attention mask
        q_seq_len_list = [
            int((f//self.model_args.patch_size[0]) * (h//self.model_args.patch_size[1]) * (w//self.model_args.patch_size[2]))
            for f, h, w in zip(latent_num_frames_lst, latent_height_lst, latent_width_lst)
        ]
        kv_seq_len_list = [int(i) for i in text_emb_len_lst]
        n_seq = len(q_seq_len_list)

        # Condition embedding. The timestep comes per token ([L1+...+Ln]), as the Wan2.2
        # TI2V-5B model expects; one timestep per sequence is repeated to every token.
        (
            temb,
            timestep_proj,
            encoder_hidden_states,
        ) = self.condition_embedder(timestep, encoder_hidden_states)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        if temb.shape[0] != sum(q_seq_len_list):
            temb_cache = []
            for item, q_len in zip(temb, q_seq_len_list):
                temb_cache.append(item[None, ...].repeat(q_len, 1))
            temb = torch.cat(temb_cache)
        if timestep_proj.shape[0] != sum(q_seq_len_list):
            timestep_proj_cache = []
            for item, q_len in zip(timestep_proj, q_seq_len_list):
                timestep_proj_cache.append(item[None, ...].repeat(q_len, 1, 1))
            timestep_proj = torch.cat(timestep_proj_cache)

        # Attention mask
        build_attn_mask(
            n_seq=n_seq,
            q_seq_len_list=q_seq_len_list,
            kv_seq_len_list=kv_seq_len_list,
            device=hidden_states.device,
        )

        padding_size = 0

        return hidden_states, encoder_hidden_states, temb, timestep_proj, rotary_emb, padding_size

    def get_primitives(
        self,
        pred: dict,
        latent_height: int,
        latent_width: int,
        intrinsics: torch.Tensor,
        is_upsampled: bool = False,
        original_height: Optional[int] = None,
        original_width: Optional[int] = None,
        depth_mode: str = "linear",
        min_depth: float = 0.2,
        max_depth: float = 10.0,
    ):
        """
        Convert plane predictions to primitives in camera space.
        Everything stays in camera coordinates (no c2w transform).
        
        Args:
            pred: dict with 'pred_center_depth', 'pred_radii', 'pred_rot'
            latent_height: height dimension (before patching, after VAE encoding)
            latent_width: width dimension (before patching, after VAE encoding)
            intrinsics: [N, 3, 3] camera intrinsics, one matrix per predicted frame.
                Values correspond to the original video resolution and are scaled
                to the patch grid here. The frame count must match the predictions.
            is_upsampled: whether this is for upsampled features
            original_height: original video height (if None, assumes intrinsics matches latent_height)
            original_width: original video width (if None, assumes intrinsics matches latent_width)
            depth_mode: "linear" or "log" - depth representation mode
            min_depth: minimum depth in metric space (meters)
            max_depth: maximum depth in metric space (meters)
        
        Returns:
            dict with processed primitives (depth, radii, rot, normals, centers in camera space)
        """
        pred_plane_depth = pred['pred_center_depth']  # B, N, 1

        # If log mode, convert log depth to metric depth for 3D coordinate computation
        if depth_mode == "log":
            # pred_plane_depth is in log space: log(depth)
            # Convert to metric space: depth = exp(log_depth)
            pred_plane_depth_metric = torch.exp(pred_plane_depth)
            # Clamp to valid range
            pred_plane_depth_metric = torch.clamp(pred_plane_depth_metric, min=min_depth, max=max_depth)
        else:
            # Linear mode: clamp directly
            pred_plane_depth_metric = torch.clamp(pred_plane_depth, min=min_depth, max=max_depth)

        bs, plane_num, _ = pred_plane_depth.shape

        # float32: linalg.inv below does not support bfloat16
        intrinsics = intrinsics.float()

        # Handle intrinsics shape: should be [N, 3, 3] but might have extra batch dim
        if intrinsics.dim() == 4:
            # Shape is [1, N, 3, 3], squeeze batch dimension
            intrinsics = intrinsics.squeeze(0)  # [N, 3, 3]
        assert intrinsics.dim() == 3, f"Expected intrinsics to be [N, 3, 3], got {intrinsics.shape}"

        pred_plane_rot_q_normed = normalize_quaternion(pred['pred_rot'])  # B, N, 4
        pred_plane_rot_matrix = quat_to_rot(pred_plane_rot_q_normed.view(-1, 4))  # B*N, 3, 3

        # Compute plane normal in camera space (no c2w transform!)
        # Convert plane_normal_standard to match pred_plane_rot_matrix dtype
        plane_normal_std = self.plane_normal_standard.to(pred_plane_rot_matrix.dtype)
        pred_plane_normal = torch.bmm(
            pred_plane_rot_matrix,
            plane_normal_std.expand(bs * plane_num, 3, 1)
        ).squeeze(-1)  # B*N, 3
        pred_plane_normal = pred_plane_normal.reshape(bs, plane_num, 3)  # B, N, 3

        pred_plane_radii = torch.clamp(pred['pred_radii'], min=1e-3)  # B, N, 2
        pred_plane_radii_padding = torch.cat([pred_plane_radii, pred_plane_radii], dim=-1)  # B, N, 4

        p_t, p_h, p_w = self.model_args.patch_size
        h_patch = latent_height // p_h
        w_patch = latent_width // p_w

        # Spatial resolution is multiplied by patch_size
        h_patch *= p_h  # 2x for patch_size (1, 2, 2)
        w_patch *= p_w  # 2x for patch_size (1, 2, 2)

        if is_upsampled:
            h_patch *= 2
            w_patch *= 2

        # Handle per-frame intrinsics
        # intrinsics shape: [N, 3, 3] where N is number of ORIGINAL frames (without context frames)
        # IMPORTANT: Use intrinsics.shape[0] as ground truth, NOT latent_num_frames
        N_frames = intrinsics.shape[0]
        f_patch = N_frames

        # Double-check dimensions match predictions
        expected_seq_len = f_patch * h_patch * w_patch
        actual_seq_len = pred_plane_depth.shape[1]
        if expected_seq_len != actual_seq_len:
            raise ValueError(
                f"Dimension mismatch: pred has {actual_seq_len} patches but expected "
                f"{expected_seq_len} (f={f_patch}, h={h_patch}, w={w_patch}). "
                f"This likely means predictions include context frames but intrinsics doesn't."
            )

        # Scale intrinsics from original resolution to patch resolution
        # If original_height/width not provided, assume intrinsics already matches latent resolution
        if original_height is None or original_width is None:
            # Intrinsics matches latent resolution, scale to patch resolution
            if is_upsampled:
                scale_h = float(h_patch) / (latent_height * 2)  # latent_height to upsampled
                scale_w = float(w_patch) / (latent_width * 2)
            else:
                scale_h = float(h_patch) / latent_height
                scale_w = float(w_patch) / latent_width
        else:
            # Intrinsics matches original video resolution, scale to patch resolution
            scale_h = float(h_patch) / original_height
            scale_w = float(w_patch) / original_width

        intrinsics_patches = intrinsics.clone()  # B, 3, 3
        intrinsics_patches[:, 0, :] *= scale_w  # fx and cx
        intrinsics_patches[:, 1, :] *= scale_h  # fy and cy

        # Generate pixel coordinates for ALL patches (T*H*W)
        # We generate coordinates for each frame independently
        pixel_coords_key = f'f{int(f_patch)}_h{int(h_patch)}_w{int(w_patch)}'
        if pixel_coords_key not in self.pixel_coords_dict:
            y, x = torch.meshgrid(
                torch.arange(int(h_patch), device=pred_plane_depth.device),
                torch.arange(int(w_patch), device=pred_plane_depth.device),
                indexing='ij'
            )
            # Stack as [3, h_patch, w_patch]
            # Use pixel center coordinates (0.5, 1.5, 2.5...) instead of corner (0, 1, 2...)
            spatial_coords = torch.stack([x + 0.5, y + 0.5, torch.ones_like(x)], dim=0).float()

            # Repeat for each frame: [3, f_patch, h_patch, w_patch]
            spatial_coords = spatial_coords.unsqueeze(1).repeat(1, f_patch, 1, 1)

            # Reshape to [3, f_patch*h_patch*w_patch]
            pixel_coords = spatial_coords.reshape(3, -1)
            self.pixel_coords_dict[pixel_coords_key] = pixel_coords.cpu()

        # Cached on the device where they are used; after the first call .to() is a no-op.
        pixel_coords = self.pixel_coords_dict[pixel_coords_key].to(pred_plane_depth.device)
        self.pixel_coords_dict[pixel_coords_key] = pixel_coords
        pixel_coords = pixel_coords.unsqueeze(0).repeat(bs, 1, 1).float()  # [B, 3, T*H*W]

        # Backproject to camera rays (handle per-frame intrinsics)
        # pixel_coords: [B, 3, T*H*W] -> reshape to [B, 3, f_patch, H*W]
        # intrinsics_patches: [f_patch, 3, 3]
        HW = h_patch * w_patch
        pixel_coords_frames = pixel_coords.reshape(bs, 3, f_patch, HW)  # [B, 3, f_patch, H*W]

        # Rearrange to [B, f_patch, 3, H*W] for per-frame processing
        pixel_coords_frames = pixel_coords_frames.permute(0, 2, 1, 3)  # [B, f_patch, 3, H*W]

        # Apply per-frame intrinsics: reshape for batch matrix multiply
        # [B*f_patch, 3, H*W] and [f_patch, 3, 3] -> [B*f_patch, 3, H*W]
        pixel_coords_flat = pixel_coords_frames.reshape(bs * f_patch, 3, HW)  # [B*f_patch, 3, H*W]
        # torch.linalg.inv does not support BFloat16; convert to float32
        intrinsics_inv = torch.linalg.inv(intrinsics_patches.float())  # [f_patch, 3, 3]

        # Check for NaN/Inf in the inverted intrinsics (falls back to identity)
        if torch.isnan(intrinsics_inv).any() or torch.isinf(intrinsics_inv).any():
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[CRITICAL] NaN/Inf detected in intrinsics inverse! Using identity matrix as fallback.")
            # Replace with identity matrix to prevent crash
            intrinsics_inv = torch.eye(3, device=intrinsics_inv.device, dtype=intrinsics_inv.dtype).unsqueeze(0).repeat(f_patch, 1, 1)

        intrinsics_inv_expanded = intrinsics_inv.unsqueeze(0).repeat(bs, 1, 1, 1).reshape(bs * f_patch, 3, 3)  # [B*f_patch, 3, 3]

        camera_coords_flat = torch.bmm(intrinsics_inv_expanded, pixel_coords_flat)  # [B*f_patch, 3, H*W]

        # Reshape back to [B, 3, T*H*W]
        camera_coords = camera_coords_flat.reshape(bs, f_patch, 3, HW).permute(0, 2, 1, 3).reshape(bs, 3, f_patch * HW)  # [B, 3, T*H*W]

        # Compute 3D centers in camera space (no world transform!)
        # Use metric depth for 3D coordinate calculation
        camera_coords_depth = camera_coords * pred_plane_depth_metric.permute(0, 2, 1)  # [B, 3, T*H*W]
        pred_plane_center_camera = camera_coords_depth.permute(0, 2, 1)  # [B, T*H*W, 3]

        # Generate boundary mask for selective primitive rendering
        pred_plane_normal_patches = pred_plane_normal.reshape(bs, f_patch, h_patch, w_patch, 3)

        # Compute boundary mask per frame: [B, T, H, W]
        boundary_mask_list = []
        for t in range(f_patch):
            # Get normals for this frame: [B, H, W, 3]
            normal_frame = pred_plane_normal_patches[:, t, :, :, :].detach()
            mask_frame = generate_gradient_mask(normal_frame, ratio=0.1)
            boundary_mask_list.append(mask_frame)

        boundary_mask = torch.stack(boundary_mask_list, dim=1)

        res = dict()
        res['pred_plane_rot_q_normed'] = pred_plane_rot_q_normed
        res['pred_plane_radii_padding'] = pred_plane_radii_padding
        res['pred_plane_center_camera'] = pred_plane_center_camera  # Camera space, not world!
        res['pred_plane_normal_camera'] = pred_plane_normal  # Camera space normal
        res['pred_depth_patches'] = pred_plane_depth  # B, T*H*W, 1 - original prediction (log or linear)
        res['pred_depth_patches_metric'] = pred_plane_depth_metric  # B, T*H*W, 1 - metric space (for rendering)
        res['intrinsics_patches'] = intrinsics_patches
        res['boundary_mask'] = boundary_mask  # [B, T, H, W] - True = boundary/high gradient

        return res

    def render_primitives(
        self,
        pred_prim_low: dict,
        pred_prim_high: dict,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
        intrinsics: torch.Tensor,
        image_height: int,
        image_width: int,
        original_height: int,
        original_width: int,
        depth_mode: str = "linear",
        max_depth: float = 10.0,
    ):
        """
        Render plane primitives using rectangular Gaussian splatting.
        Renders each frame independently so geometry never mixes across time.
        
        Args:
            pred_prim_low: Low-res primitives from get_primitives()
            pred_prim_high: High-res primitives from get_primitives()
            latent_num_frames: number of frames in latent
            latent_height: height in latent space (after VAE encoding)
            latent_width: width in latent space (after VAE encoding)
            intrinsics: [T, 3, 3] matrices from the single-video caller, at the
                original video resolution. The first matrix is used for every
                rendered frame, so intrinsics must be constant across the segment.
            image_height: target rendered image height
            image_width: target rendered image width
            original_height: original video height that intrinsics corresponds to
            original_width: original video width that intrinsics corresponds to
        
        Returns:
            dict with rendered depth maps (list of T maps per batch element) and primitives info
        """
        if not RASTERIZER_AVAILABLE:
            raise RuntimeError("diff_rect_rasterization is required for render_primitives")

        bs = pred_prim_low['pred_plane_center_camera'].shape[0]

        # Calculate patch dimensions
        p_t, p_h, p_w = self.model_args.patch_size
        f_patch = latent_num_frames // p_t
        h_patch = latent_height // p_h
        w_patch = latent_width // p_w

        # Spatial resolution is multiplied by patch_size
        h_patch *= p_h  # 2x for patch_size (1, 2, 2)
        w_patch *= p_w  # 2x for patch_size (1, 2, 2)

        h_patch_high = h_patch * 2
        w_patch_high = w_patch * 2

        allmaps_list = []  # [batch_idx][frame_idx] -> depth_map

        final_plane_normal_list = []
        final_plane_center_camera_list = []
        final_plane_radii_list = []
        final_plane_rot_q_normed_list = []

        # bg must be float32 for the rasterizer; the model is cast to bf16 at build time.
        bg_float32 = self.bg.float()

        for b in range(bs):
            batch_allmaps = []

            # Extract primitives for this batch element [T*H*W, ...]
            # float32: the rasterizer does not take bf16
            plane_center_low_all = pred_prim_low['pred_plane_center_camera'][b].float()  # [T*H*W, 3]
            plane_radii_low_all = pred_prim_low['pred_plane_radii_padding'][b].float()
            plane_rot_q_low_all = pred_prim_low['pred_plane_rot_q_normed'][b].float()

            plane_center_high_all = pred_prim_high['pred_plane_center_camera'][b].float()
            plane_radii_high_all = pred_prim_high['pred_plane_radii_padding'][b].float()
            plane_rot_q_high_all = pred_prim_high['pred_plane_rot_q_normed'][b].float()

            # Reshape from [T*H*W, ...] to [T, H*W, ...]
            plane_center_low = plane_center_low_all.reshape(f_patch, h_patch * w_patch, 3)
            plane_radii_low = plane_radii_low_all.reshape(f_patch, h_patch * w_patch, 4)
            plane_rot_q_low = plane_rot_q_low_all.reshape(f_patch, h_patch * w_patch, 4)

            plane_center_high = plane_center_high_all.reshape(f_patch, h_patch_high * w_patch_high, 3)
            plane_radii_high = plane_radii_high_all.reshape(f_patch, h_patch_high * w_patch_high, 4)
            plane_rot_q_high = plane_rot_q_high_all.reshape(f_patch, h_patch_high * w_patch_high, 4)

            # Extract boundary masks for this batch element [T, H, W]
            boundary_mask_low = pred_prim_low['boundary_mask'][b]  # [T, H, W]

            # Upsample low-res boundary mask to high-res (nearest neighbor)
            # [T, H, W] -> [T, H*2, W*2]
            boundary_mask_high = torch.nn.functional.interpolate(
                boundary_mask_low.float().unsqueeze(1),  # [T, 1, H, W]
                size=(h_patch_high, w_patch_high),
                mode='nearest'
            ).squeeze(1).bool()  # [T, H*2, W*2]

            # The caller supplies one video (b=0); its first K is shared across frames.
            intrinsic_b = intrinsics[b]  # 3, 3

            # FOV calculation
            # FOV (tanfov) is a physical camera property that should NOT change with render resolution
            #
            # Step 1: Calculate tanfov using ORIGINAL resolution (this is the true camera FOV)
            fx_orig = intrinsic_b[0, 0]
            fy_orig = intrinsic_b[1, 1]
            tanfovx = original_width / (2.0 * fx_orig)
            tanfovy = original_height / (2.0 * fy_orig)

            # Step 2: Scale intrinsics to match RENDER resolution for projection
            scale_x = image_width / original_width
            scale_y = image_height / original_height
            cx = intrinsic_b[0, 2] * scale_x
            cy = intrinsic_b[1, 2] * scale_y

            # Camera is at origin in camera space (identity w2c)
            # All rasterizer inputs are float32
            raster_cam_w2c = torch.eye(4, device=plane_center_low_all.device, dtype=torch.float32)
            raster_cam_center = torch.zeros(3, device=plane_center_low_all.device, dtype=torch.float32)
            raster_img_center = torch.tensor([cx, cy], device=plane_center_low_all.device, dtype=torch.float32)

            znear, zfar = 0.01, 10.0
            proj_matrix = self._get_projection_matrix(znear, zfar, tanfovx, tanfovy).to(plane_center_low_all.device)
            # The 3DGS rasterizer is row-major (p_clip = p @ fullproj), so both the view and
            # the projection matrix are passed transposed. w2c is the identity here; P still
            # has to be transposed.
            raster_cam_w2c = raster_cam_w2c.T.contiguous()  # identity.T == identity
            raster_cam_fullproj = (raster_cam_w2c @ proj_matrix.T).contiguous().float()

            splat_weight = 300.0

            for t in range(f_patch):
                # Get primitives for this specific frame [H*W, ...]
                plane_center_low_t = plane_center_low[t]  # [H*W, 3]
                plane_radii_low_t = plane_radii_low[t]
                plane_rot_q_low_t = plane_rot_q_low[t]

                plane_center_high_t = plane_center_high[t]
                plane_radii_high_t = plane_radii_high[t]
                plane_rot_q_high_t = plane_rot_q_high[t]

                # Get boundary mask for this frame [H, W] for low, [H*2, W*2] for high
                boundary_mask_low_t = boundary_mask_low[t]  # [H, W]
                boundary_mask_high_t = boundary_mask_high[t]  # [H*2, W*2]

                # Merge primitives using the boundary mask
                # Low-res: keep NON-boundary areas (False in mask)
                mask_low_flat = boundary_mask_low_t.reshape(-1)  # [H*W]
                selected_low_center = plane_center_low_t[~mask_low_flat]  # Keep False areas
                selected_low_q = plane_rot_q_low_t[~mask_low_flat]
                selected_low_radii = plane_radii_low_t[~mask_low_flat]

                # High-res: keep boundary areas (True in mask)
                mask_high_flat = boundary_mask_high_t.reshape(-1)  # [H*2 * W*2]
                selected_high_center = plane_center_high_t[mask_high_flat]  # Keep True areas
                selected_high_q = plane_rot_q_high_t[mask_high_flat]
                selected_high_radii = plane_radii_high_t[mask_high_flat]

                # Concatenate: low-res (smooth areas) + high-res (boundary areas)
                plane_center_merged = torch.cat([selected_low_center, selected_high_center], dim=0)
                plane_rot_q_merged = torch.cat([selected_low_q, selected_high_q], dim=0)
                plane_radii_merged = torch.cat([selected_low_radii, selected_high_radii], dim=0)

                primitive_num = plane_center_merged.shape[0]

                plane_rot_matrix_merged = quat_to_rot(plane_rot_q_merged)  # N, 3, 3
                # Convert plane_normal_standard to match plane_rot_matrix_merged dtype
                plane_normal_std = self.plane_normal_standard.to(plane_rot_matrix_merged.dtype)
                plane_normal_merged = torch.bmm(
                    plane_rot_matrix_merged,
                    plane_normal_std.expand(primitive_num, 3, 1)
                ).squeeze(-1)

                final_plane_normal_list.append(plane_normal_merged)
                final_plane_center_camera_list.append(plane_center_merged)
                final_plane_rot_q_normed_list.append(plane_rot_q_merged)
                final_plane_radii_list.append(plane_radii_merged)

                import logging
                _rast_logger = logging.getLogger(__name__)

                def _sanitize_grad(grad):
                    """Replace non-finite gradients with zero."""
                    if grad is not None and (torch.isnan(grad).any() or torch.isinf(grad).any()):
                        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                    return grad

                # --- Sanitize merged rasterizer inputs ---
                # Adjust clamp range based on depth_mode
                max_coord = max_depth if depth_mode == "log" else 50.0
                plane_center_merged = torch.nan_to_num(plane_center_merged, nan=0.0, posinf=max_coord, neginf=-max_coord)
                plane_center_merged = torch.clamp(plane_center_merged, -max_coord, max_coord)
                plane_radii_merged = torch.nan_to_num(plane_radii_merged, nan=0.01, posinf=5.0, neginf=0.01)
                plane_radii_merged = torch.clamp(plane_radii_merged, 1e-2, 5.0)
                plane_rot_q_merged = torch.nan_to_num(plane_rot_q_merged, nan=0.0, posinf=1.0, neginf=-1.0)

                screenspace_points = torch.zeros_like(
                    plane_center_merged,
                    dtype=plane_center_merged.dtype,
                    requires_grad=True
                )
                try:
                    screenspace_points.retain_grad()
                except:
                    pass

                # image_height / image_width must be Python ints, not tensors:
                # The C++ rasterizer's tile-based culling breaks with tensor inputs.
                _image_height_int = int(image_height.item()) if isinstance(image_height, torch.Tensor) else int(image_height)
                _image_width_int = int(image_width.item()) if isinstance(image_width, torch.Tensor) else int(image_width)

                raster_settings = RectRasterizationSettings(
                    image_height=_image_height_int,
                    image_width=_image_width_int,
                    tanfovx=tanfovx.item() if isinstance(tanfovx, torch.Tensor) else tanfovx,
                    tanfovy=tanfovy.item() if isinstance(tanfovy, torch.Tensor) else tanfovy,
                    bg=bg_float32,
                    scale_modifier=1.0,
                    viewmatrix=raster_cam_w2c,
                    projmatrix=raster_cam_fullproj,
                    sh_degree=0,
                    campos=raster_cam_center,
                    prefiltered=False,
                    debug=False,
                    lambdaw=splat_weight * 5.0,
                    image_center=raster_img_center,
                    scales2=plane_radii_merged[:, :2].detach()
                )

                # Sanitize rasterizer gradients.
                if plane_center_merged.requires_grad:
                    plane_center_merged.register_hook(_sanitize_grad)
                if plane_radii_merged.requires_grad:
                    plane_radii_merged.register_hook(_sanitize_grad)
                if plane_rot_q_merged.requires_grad:
                    plane_rot_q_merged.register_hook(_sanitize_grad)

                rasterizer = RectRasterizer(raster_settings=raster_settings)
                rgb, _, allmap = rasterizer(
                    means3D=plane_center_merged,
                    means2D=screenspace_points,
                    shs=None,
                    colors_precomp=torch.rand_like(plane_center_merged),
                    opacities=torch.ones_like(plane_center_merged)[:, :1],
                    scales=plane_radii_merged,
                    rotations=plane_rot_q_merged,
                    cov3D_precomp=None
                )

                # Sanitize rasterizer output gradients.
                if allmap.requires_grad:
                    allmap.register_hook(_sanitize_grad)

                # --- Check rasterizer output for NaN/Inf ---
                if torch.isnan(allmap).any() or torch.isinf(allmap).any():
                    _rast_logger.warning(
                        f"[RASTER OUTPUT NaN] merged allmap b={b}, t={t}: "
                        f"nan_count={torch.isnan(allmap).sum().item()}, inf_count={torch.isinf(allmap).sum().item()}, "
                        f"center range=[{plane_center_merged.min().item():.4f}, {plane_center_merged.max().item():.4f}], "
                        f"radii range=[{plane_radii_merged.min().item():.4f}, {plane_radii_merged.max().item():.4f}]"
                    )
                    allmap = torch.nan_to_num(allmap, nan=0.0, posinf=0.0, neginf=0.0)

                batch_allmaps.append(allmap)


            allmaps_list.append(batch_allmaps)

        res_dict = {}
        res_dict["allmaps_list"] = allmaps_list  # [batch][frame]

        res_dict["plane_normal_list"] = final_plane_normal_list
        res_dict["plane_center_camera_list"] = final_plane_center_camera_list
        res_dict["plane_radii_list"] = final_plane_radii_list
        res_dict["plane_rot_q_normed_list"] = final_plane_rot_q_normed_list

        return res_dict

    def _get_projection_matrix(self, znear, zfar, tanfovx, tanfovy):
        """
        Projection matrix using CV (Computer Vision) coordinate system convention.
        
        IMPORTANT: Uses +Z forward (CV convention), NOT -Z forward (OpenGL convention).
        
        CV Coordinates:
            +X: right
            +Y: down
            +Z: forward (into the scene)
        
        Points with positive Z values are in front of the camera.
        """
        top = tanfovy * znear
        bottom = -top
        right = tanfovx * znear
        left = -right

        # float32 for the rasterizer
        P = torch.zeros(4, 4, dtype=torch.float32)

        # CV coordinate system: z_sign = 1.0 (points have +Z in front of camera)
        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign  # +1.0 for CV (not -1.0 like OpenGL)
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)

        return P

    def forward(
        self,
        input_dict: Optional[Dict[str, Any]],
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict noise and optionally extract and render plane geometry.

        The pipeline supplies one video, with per-video values wrapped in sequences.

        Required input_dict entries:
            timestep: Diffusion timesteps.
            hidden_states: A sequence containing a [1, C, T, H, W] latent tensor.
            encoder_hidden_states: Packed text embeddings.
            latent_num_frames_lst: Frame counts, including any prefix frames.
            latent_height_lst, latent_width_lst: Latent spatial dimensions.
            text_emb_len_lst: Text sequence lengths.

        Geometry inputs:
            intrinsics: A list containing [T_main, 3, 3] matrices at the original
                video resolution. Primitive rendering uses the first matrix for
                every frame, so the segment must use constant intrinsics.
            original_height, original_width: Per-video dimensions for those matrices.
            noisy_latents, timesteps_value, inference_scheduler: Supplied by the
                pipeline for velocity-to-latent conversion in the geometry decoder.
            raw_next_ar_depth: Optional depth maps for depth-token fusion.

        Optional num_context_frames_lst, num_very_first_frames_lst and
        num_history_frames_lst specify prefix lengths for positional encoding
        and removal before geometry decoding.

        Returns:
            (output, pred_prim_high, rendered_dict): noise predictions, per-video
            plane dictionaries, and rendered maps. Without intrinsics, the latter
            two values are None. Rendered maps also require the rasterizer.
        """
        timestep = input_dict["timestep"]
        hidden_states = input_dict["hidden_states"]
        encoder_hidden_states = input_dict.get("encoder_hidden_states", None)
        # 1. Input patchify / embedding
        (
            hidden_states,
            encoder_hidden_states,
            temb,
            timestep_proj,
            rotary_emb,
            padding_size,
        ) = self.forward_pre_process(
            timestep=timestep,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            latent_num_frames_lst=input_dict["latent_num_frames_lst"],
            latent_height_lst=input_dict["latent_height_lst"],
            latent_width_lst=input_dict["latent_width_lst"],
            text_emb_len_lst=input_dict["text_emb_len_lst"],
            num_context_frames_lst=input_dict.get("num_context_frames_lst", None),
            num_very_first_frames_lst=input_dict.get("num_very_first_frames_lst", None),
            num_history_frames_lst=input_dict.get("num_history_frames_lst", None),
        )

        # 2. Transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 3. Output norm, projection (but not upsample yet if we need to mix first)
        temb_scale_shift_table = self.scale_shift_table + temb[:, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, 'l n c -> n l c').chunk(2, dim=0)
        hidden_states_normalized = self.norm_out(hidden_states) * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states_normalized)

        # Expand each patch's four subpatches consecutively to preserve spatial order.
        output = rearrange(hidden_states, '1 l (n c) -> 1 (l n) c', n=4)

        # Everything below this point feeds plane_head and the rasterizer. Without
        # intrinsics the return is (output, None, None), so stop here.
        intrinsics_input = input_dict.get('intrinsics', None)
        if intrinsics_input is not None:
            assert isinstance(intrinsics_input, list), "intrinsics must be a List of [3, 3] tensors"
            if len(intrinsics_input) != len(input_dict['latent_num_frames_lst']):
                _cnt = getattr(self, '_intrinsics_len_warn_count', 0)
                if _cnt < 20:
                    self._intrinsics_len_warn_count = _cnt + 1
                    logger.warning(
                        f"[SKIP_PLANE_ALL #{_cnt+1}] intrinsics list len={len(intrinsics_input)} "
                        f"!= n_videos={len(input_dict['latent_num_frames_lst'])}. "
                        f"Skipping plane extraction for the whole batch."
                    )
                intrinsics_input = None
        if intrinsics_input is None:
            return (output, None, None)

        hidden_states_feat = self.proj_out_feat(hidden_states_normalized)
        # The geometry pass below sets the run's memory peak; free what the blocks needed
        # and it does not (the timestep modulation alone is ~2 GB at full history).
        del temb, timestep_proj, temb_scale_shift_table, shift, scale
        del hidden_states_normalized, encoder_hidden_states, rotary_emb

        latents_pred_patch = None

        # Optionally add velocity prediction (converted to latent) with plane features
        # Add latents_pred to hidden_states_feat BEFORE upsample
        noisy_latents = input_dict.get('noisy_latents', None)
        timesteps_value = input_dict.get('timesteps_value', None)

        # Prioritize inference_scheduler from input_dict (used during inference)
        scheduler = input_dict.get('inference_scheduler', self.scheduler)

        if noisy_latents is not None and timesteps_value is not None and scheduler is not None:
            # Convert velocity prediction to latent prediction
            # Reference: latents_pred = noisy_latents - sigma * v_pred

            t_tensor = torch.stack([t.reshape(()) for t in timesteps_value])  # [n_seq]

            if hasattr(scheduler, 'index_for_timestep'):
                sigma_tensor = torch.stack([
                    scheduler.sigmas[scheduler.index_for_timestep(_t)]
                    for _t in t_tensor.unbind(0)
                ], dim=0).to(device=output.device, dtype=output.dtype)
            else:
                sigmas = []
                for idx, t in enumerate(timesteps_value):
                    t_value = t.item() if isinstance(t, torch.Tensor) else t
                    if abs(t_value) < 1e-6:
                        sigma = torch.tensor(0.0, device=output.device, dtype=output.dtype)
                    else:
                        scheduler_timesteps = scheduler.timesteps.to(output.device)
                        diffs = torch.abs(scheduler_timesteps - t_value)
                        closest_idx = torch.argmin(diffs)
                        sigma = scheduler.sigmas[closest_idx]
                    sigmas.append(sigma)
                sigma_tensor = torch.stack(sigmas).to(device=output.device, dtype=output.dtype)

            # Patchify noisy_latents to match proj_out's (n c) output convention
            # proj_out output is in (n, c) = (subpatch, channel) order
            # So we use (c1 c2 c3 c) here, NOT (c c1 c2 c3) which is the input convention
            p_t, p_h, p_w = self.model_args.patch_size
            noisy_latents_patchified_list = [
                rearrange(
                    nl,
                    '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c1 c2 c3 c)',
                    c1=p_t, c2=p_h, c3=p_w,
                ) for nl in noisy_latents
            ]
            noisy_latents_concat = torch.cat(noisy_latents_patchified_list, dim=1)

            # Broadcast each video's sigma to all its patch tokens, including prefixes.
            sigma_expanded = []
            for i, nl_patch in enumerate(noisy_latents_patchified_list):
                patchified_len = nl_patch.shape[1]
                per_token_sigma = sigma_tensor[i].expand(patchified_len).clone()

                sigma_expanded.append(per_token_sigma)
            sigma_expanded = torch.cat(sigma_expanded, dim=0).unsqueeze(0).unsqueeze(-1)  # [1, L_patchified, 1]

            # latents_pred = noisy_latents - sigma * v_pred
            # Both hidden_states and noisy_latents_concat are in (n c) = (c1 c2 c3 c) convention
            # hidden_states: [1, L_total, out_channels * prod(patch_size)]
            # noisy_latents_concat: [1, L_total, in_channels * prod(patch_size)]
            # Truncation must happen on the inner c dim (per subpatch), NOT the flattened dim
            total_patch_size = math.prod(self.model_args.patch_size)  # e.g. 4 for (1,2,2)
            C_out = hidden_states.shape[-1] // total_patch_size   # out_channels
            C_in = noisy_latents_concat.shape[-1] // total_patch_size  # in_channels
            assert C_in >= C_out, \
                f"noisy_latents per-patch channels ({C_in}) < hidden_states per-patch channels ({C_out})"
            if C_in == C_out:
                noisy_latents_for_pred = noisy_latents_concat
            else:
                # Reshape to [1, L, n_subpatches, C_in], truncate c, flatten back
                noisy_latents_for_pred = (
                    noisy_latents_concat
                    .view(1, -1, total_patch_size, C_in)[:, :, :, :C_out]
                    .reshape(1, -1, total_patch_size * C_out)
                )

            latents_pred_patch = noisy_latents_for_pred - sigma_expanded * hidden_states

            # Save originals for WAN-like decoder (before dim matching and mixing)
            latents_pred_patch_orig = latents_pred_patch  # [1, L, C_out * patch_prod]
            hidden_states_feat_pre_mix = hidden_states_feat  # [1, L, C_feat]

            # Add latents_pred to hidden_states_feat
            C_feat = hidden_states_feat.shape[-1]
            C_pred = latents_pred_patch.shape[-1]

            if C_pred != C_feat:
                # Project latents_pred to match hidden_states_feat dimensions
                # Repeat along the channel dim when C_feat is a multiple of C_pred
                if C_feat % C_pred == 0:
                    repeat_factor = C_feat // C_pred
                    latents_pred_patch = latents_pred_patch.repeat(1, 1, repeat_factor)
                else:
                    # Use zero padding if not divisible
                    padding = C_feat - C_pred
                    latents_pred_patch = torch.nn.functional.pad(
                        latents_pred_patch,
                        (0, padding),
                        mode='constant',
                        value=0
                    )

            hidden_states_feat = hidden_states_feat + latents_pred_patch.detach()

        p_t, p_h, p_w = self.model_args.patch_size
        total_frames = sum(input_dict["latent_num_frames_lst"])
        h = input_dict["latent_height_lst"][0]
        w = input_dict["latent_width_lst"][0]
        f_patch_total = total_frames // p_t
        h_patch = h // p_h
        w_patch = w // p_w

        if latents_pred_patch is not None:
            # --- WAN-like decoder path ---
            # Unpatchify hidden_states_feat (pre-mix) to 5D: [1, f*h*w, pt*ph*pw*C] -> [1, C, T, H, W]
            feats_5d = rearrange(
                hidden_states_feat_pre_mix,
                '1 (f h w) (pt ph pw c) -> 1 c (f pt) (h ph) (w pw)',
                f=f_patch_total, h=h_patch, w=w_patch, pt=p_t, ph=p_h, pw=p_w
            )
            # Unpatchify latents_pred_patch (original, before dim matching) to 5D
            z_5d = rearrange(
                latents_pred_patch_orig.detach(),
                '1 (f h w) (pt ph pw c) -> 1 c (f pt) (h ph) (w pw)',
                f=f_patch_total, h=h_patch, w=w_patch, pt=p_t, ph=p_h, pw=p_w
            )

            # Latent unscale: z * latents_std + latents_mean
            _ms_key = (z_5d.device, z_5d.dtype)
            if getattr(self, '_latent_scale_key', None) != _ms_key:
                self._latents_mean = torch.tensor(self.vae.config.latents_mean).view(
                    1, -1, 1, 1, 1).to(z_5d.device, z_5d.dtype)
                self._latents_std = torch.tensor(self.vae.config.latents_std).view(
                    1, -1, 1, 1, 1).to(z_5d.device, z_5d.dtype)
                self._latent_scale_key = _ms_key
            latents_mean, latents_std = self._latents_mean, self._latents_std
            z_5d_unscaled = z_5d * latents_std + latents_mean

            # === Depth token fusion ===
            # Fuse depth before decoding, at the input latent's spatial resolution.
            raw_depths = input_dict.get('raw_next_ar_depth', None)
            if raw_depths is not None:
                n_videos_dtf = len(input_dict['latent_num_frames_lst'])
                orig_h_lst = input_dict.get('original_height', [288] * n_videos_dtf)
                lat_h_lst = input_dict.get('latent_height_lst', [h] * n_videos_dtf)

                # Depth token fusion: collect all main frames into a single batched call
                enhanced_frames = list(feats_5d[0].unbind(dim=1))  # T × [C, H, W]

                all_t_indices = []
                all_feat_tokens = []
                all_raw_depths = []
                all_coverages = []
                H_lat_common = None
                W_lat_common = None

                offset_t = 0
                for vi in range(n_videos_dtf):
                    total_f = int(input_dict['latent_num_frames_lst'][vi])
                    prefix_f = sum(
                        int(input_dict.get(k, [0] * n_videos_dtf)[vi])
                        for k in ['num_context_frames_lst', 'num_very_first_frames_lst', 'num_history_frames_lst']
                    )
                    main_f = total_f - prefix_f
                    H_lat = int(lat_h_lst[vi])
                    W_lat = int(input_dict['latent_width_lst'][vi])
                    orig_h_i = int(orig_h_lst[vi]) if isinstance(orig_h_lst[vi], (int, float)) else int(orig_h_lst[vi].item())
                    ppt = orig_h_i // H_lat

                    if H_lat_common is None:
                        H_lat_common, W_lat_common = H_lat, W_lat

                    raw_d = raw_depths[vi].to(feats_5d.device)

                    _, cov = compute_patch_depth_stats(raw_d, ppt, ppt)

                    assert raw_d.shape[0] == main_f, \
                        f"raw_depth frames {raw_d.shape[0]} != main_f {main_f}"

                    for fi in range(main_f):
                        t_idx = offset_t + prefix_f + fi
                        all_t_indices.append(t_idx)
                        all_feat_tokens.append(rearrange(enhanced_frames[t_idx], 'c h w -> (h w) c'))
                        all_raw_depths.append(raw_d[fi])
                        all_coverages.append(cov[fi].reshape(-1))

                    offset_t += total_f

                # Single batched depth token fusion call
                feat_batch = torch.stack(all_feat_tokens)     # [N_total, S, C]
                depth_batch = torch.stack(all_raw_depths)     # [N_total, H, W]
                cov_batch = torch.stack(all_coverages)        # [N_total, S]

                enhanced_batch = self.depth_token_fusion(
                    feat_batch, depth_batch, cov_batch,
                    h_tokens=H_lat_common, w_tokens=W_lat_common,
                )  # [N_total, S, C]

                for idx, t_idx in enumerate(all_t_indices):
                    enhanced_frames[t_idx] = rearrange(
                        enhanced_batch[idx], '(h w) c -> c h w', h=H_lat_common, w=W_lat_common
                    )

                feats_5d = torch.stack(enhanced_frames, dim=1).unsqueeze(0)

            # --- WAN-like decoder path ---
            # Strip each video's prefix (history + very_first + context) and send only main to the
            # decoder. Cross-video isolation comes from the per-video batch stacking.
            n_videos = len(input_dict['latent_num_frames_lst'])

            main_parts_feats = []
            main_parts_z = []
            per_video_main_frames = []
            offset_t = 0
            for vi in range(n_videos):
                nf_i = int(input_dict['latent_num_frames_lst'][vi])
                num_ctx_i = int(input_dict.get('num_context_frames_lst', [0] * n_videos)[vi])
                num_vf_i = int(input_dict.get('num_very_first_frames_lst', [0] * n_videos)[vi])
                num_hist_i = int(input_dict.get('num_history_frames_lst', [0] * n_videos)[vi])
                prefix_i = num_ctx_i + num_vf_i + num_hist_i
                main_f_i = nf_i - prefix_i
                per_video_main_frames.append(main_f_i)
                # Layout: [history ... | very_first | context ... | main ...]; the prefix is stripped.
                # Slice: [offset_t + prefix_i : offset_t + nf_i]
                slice_start = offset_t + prefix_i
                slice_end = offset_t + nf_i
                main_parts_feats.append(feats_5d[:, :, slice_start:slice_end, :, :])
                main_parts_z.append(z_5d_unscaled[:, :, slice_start:slice_end, :, :])
                offset_t += nf_i

            # Pad each video to max main frames at temporal END, stack to batch dim
            max_eff_f = max(per_video_main_frames)
            _, C_feat, _, H_dec, W_dec = feats_5d.shape
            _, C_z, _, _, _ = z_5d_unscaled.shape

            def _pad_temporal_end(t, max_t):
                pad_t = max_t - t.shape[2]
                return F.pad(t, (0, 0, 0, 0, 0, pad_t)) if pad_t > 0 else t

            feats_batch = torch.cat([_pad_temporal_end(f, max_eff_f) for f in main_parts_feats], dim=0)  # [N, C, max_f, H, W]
            z_batch = torch.cat([_pad_temporal_end(z, max_eff_f) for z in main_parts_z], dim=0)

            mid_out_batch, up_out_batch = self.feature_decoder(feats_batch, z_batch)

            # Crop each video back to its own length (zero-padded to max above)
            mid_parts = []
            up_parts = []
            for vi in range(n_videos):
                main_i = per_video_main_frames[vi]
                mid_parts.append(mid_out_batch[vi:vi+1, :, :main_i, :, :])
                up_parts.append(up_out_batch[vi:vi+1, :, :main_i, :, :])
            mid_out = torch.cat(mid_parts, dim=2)  # [1, C, T_main_total, H, W]
            up_out = torch.cat(up_parts, dim=2)

            # High-res: reshape to sequence and project (main frames only)
            up_out_seq = rearrange(up_out, '1 c t h w -> 1 (t h w) c')
            output_feat_upsampled = self.decoder_out_proj(up_out_seq)

            # Low-res: use mid_block output (main frames only)
            output_feat = rearrange(mid_out, '1 c t h w -> 1 (t h w) c')
            output_feat = self.decoder_mid_proj(output_feat)

        else:
            raise ValueError(
                "The plane branch needs noisy_latents, timesteps_value and inference_scheduler "
                "in input_dict; the pipeline passes all three with the intrinsics."
            )
        # PlaneHead expects 'dec_feat' [B, N, C] where C = plane_head_dec_dim:
        #   output_feat:           [1, T*H*W,     plane_head_dec_dim]  mid_block output, pixel level
        #   output_feat_upsampled: [1, T*2H*2W,   plane_head_dec_dim]  up_block output, 2x pixel level

        pred_planes_low = self.plane_head(output_feat)
        pred_planes_high = self.plane_head(output_feat_upsampled)

        # Process plane predictions into primitives (camera space)
        # Predictions are unpacked per video.
        if intrinsics_input is not None:
            n_videos = len(input_dict['latent_num_frames_lst'])

            # Get original video resolution if provided (also must be Lists)
            original_height_input = input_dict.get('original_height', None)
            original_width_input = input_dict.get('original_width', None)

            pred_prim_low_list = []
            pred_prim_high_list = []
            cur_idx_low = 0
            cur_idx_high = 0

            for i in range(n_videos):
                latent_num_frames = int(input_dict['latent_num_frames_lst'][i])
                latent_height = int(input_dict['latent_height_lst'][i])
                latent_width = int(input_dict['latent_width_lst'][i])

                # Get intrinsics for this video (must be from List)
                # intrinsics_i is [N, 3, 3] where N is number of ORIGINAL frames (without context)
                intrinsics_i = intrinsics_input[i]  # [N, 3, 3]

                # Compute original frame count from intrinsics (ground truth)
                # latent_num_frames includes all prefix frames (history + very_first + context);
                # intrinsics only cover the main frames
                num_context = int(input_dict.get('num_context_frames_lst', [0] * n_videos)[i])
                num_vf = int(input_dict.get('num_very_first_frames_lst', [0] * n_videos)[i])
                num_history = int(input_dict.get('num_history_frames_lst', [0] * n_videos)[i])

                p_t, p_h, p_w = self.model_args.patch_size

                original_num_frames = latent_num_frames - num_context - num_vf - num_history

                # Tokens this video occupies, so the cursor can advance even when it is skipped
                unpatchify_multiplier = p_h * p_w
                seq_len_low = original_num_frames * (latent_height // p_h) * (latent_width // p_w) * unpatchify_multiplier
                seq_len_high = original_num_frames * (latent_height // p_h * 2) * (latent_width // p_w * 2) * unpatchify_multiplier

                # Verify intrinsics matches original frame count
                expected_frames = intrinsics_i.shape[1] if intrinsics_i.dim() == 4 else intrinsics_i.shape[0]
                original_num_frames = int(original_num_frames)
                expected_frames = int(expected_frames)
                if expected_frames != original_num_frames:
                    _cnt = getattr(self, '_frame_mismatch_warn_count', 0)
                    if _cnt < 20:
                        self._frame_mismatch_warn_count = _cnt + 1
                        logger.warning(
                            f"[SKIP_PLANE #{_cnt+1}] Video {i}: "
                            f"intrinsics has {expected_frames} frames but "
                            f"original_num_frames is {original_num_frames} "
                            f"(latent: {latent_num_frames}, ctx={num_context}, vf={num_vf}, hist={num_history}). "
                            f"Skipping plane extraction for this video."
                        )
                    cur_idx_low += seq_len_low
                    cur_idx_high += seq_len_high
                    continue

                # WAN decoder path: prefix already stripped before decoder,
                # output_feat contains only main frames. No prefix skip needed.
                pass
                # Compute sequence lengths using ORIGINAL frames (no context)
                unpatchify_multiplier = p_h * p_w
                seq_len_low = (original_num_frames // p_t) * (latent_height // p_h) * (latent_width // p_w) * unpatchify_multiplier
                seq_len_high = (original_num_frames // p_t) * (latent_height // p_h * 2) * (latent_width // p_w * 2) * unpatchify_multiplier

                # Extract predictions for this video (only original frames, exclude context)
                pred_planes_low_i = {
                    key: val[:, cur_idx_low:cur_idx_low+seq_len_low, :] if val.dim() == 3 else val[cur_idx_low:cur_idx_low+seq_len_low]
                    for key, val in pred_planes_low.items()
                }
                pred_planes_high_i = {
                    key: val[:, cur_idx_high:cur_idx_high+seq_len_high, :] if val.dim() == 3 else val[cur_idx_high:cur_idx_high+seq_len_high]
                    for key, val in pred_planes_high.items()
                }

                # Get original resolution for this video
                # original_height/width are lists (like latent_height_lst), directly index
                original_height_i = original_height_input[i] if original_height_input is not None else None
                original_width_i = original_width_input[i] if original_width_input is not None else None

                # Process primitives for this video (use ORIGINAL frame count)
                pred_prim_low_i = self.get_primitives(
                    pred_planes_low_i,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    intrinsics=intrinsics_i,
                    is_upsampled=False,
                    original_height=original_height_i,
                    original_width=original_width_i,
                    depth_mode=self.model_args.depth_mode,
                    min_depth=self.model_args.min_depth,
                    max_depth=self.model_args.max_depth,
                )
                pred_prim_high_i = self.get_primitives(
                    pred_planes_high_i,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    intrinsics=intrinsics_i,
                    is_upsampled=True,
                    original_height=original_height_i,
                    original_width=original_width_i,
                    depth_mode=self.model_args.depth_mode,
                    min_depth=self.model_args.min_depth,
                    max_depth=self.model_args.max_depth,
                )

                pred_prim_low_list.append(pred_prim_low_i)
                pred_prim_high_list.append(pred_prim_high_i)

                cur_idx_low += seq_len_low
                cur_idx_high += seq_len_high

            # Always a List, one entry per video
            pred_prim_low = pred_prim_low_list
            pred_prim_high = pred_prim_high_list
        else:
            pred_prim_low = None
            pred_prim_high = None

        # Optionally render primitives (requires intrinsics and render flag)
        if intrinsics_input is not None and RASTERIZER_AVAILABLE:
            n_videos = len(input_dict['latent_num_frames_lst'])

            # Render each video separately
            rendered_dict_list = []

            for i in range(n_videos):
                latent_num_frames = int(input_dict['latent_num_frames_lst'][i])
                latent_height = int(input_dict['latent_height_lst'][i])
                latent_width = int(input_dict['latent_width_lst'][i])

                # Main frame count (excluding the prefix frames: history + very_first + context)
                num_context = int(input_dict.get('num_context_frames_lst', [0] * n_videos)[i])
                num_vf = int(input_dict.get('num_very_first_frames_lst', [0] * n_videos)[i])
                num_history = int(input_dict.get('num_history_frames_lst', [0] * n_videos)[i])
                original_num_frames = latent_num_frames - num_context - num_vf - num_history

                # Get original resolution for this video (required for correct FOV calculation)
                original_height_input = input_dict.get('original_height')
                original_width_input = input_dict.get('original_width')
                original_height_i = original_height_input[i] if original_height_input else latent_height
                original_width_i = original_width_input[i] if original_width_input else latent_width

                image_height = original_height_i
                image_width = original_width_i

                intrinsics_render = intrinsics_input[i]
                if intrinsics_render.dim() == 2:
                    intrinsics_render = intrinsics_render.unsqueeze(0)

                rendered_dict_i = self.render_primitives(
                    pred_prim_low=pred_prim_low[i],
                    pred_prim_high=pred_prim_high[i],
                    latent_num_frames=original_num_frames,  # Use original frames (no context)
                    latent_height=latent_height,
                    latent_width=latent_width,
                    intrinsics=intrinsics_render,
                    image_height=image_height,
                    image_width=image_width,
                    original_height=original_height_i,
                    original_width=original_width_i,
                    depth_mode=self.model_args.depth_mode,
                    max_depth=self.model_args.max_depth,
                )
                rendered_dict_list.append(rendered_dict_i)

            rendered_dict = rendered_dict_list
        else:
            rendered_dict = None

        # Returns:
        #      [0] output          -- noise prediction
        #      [1] pred_prim_high  -- List[dict], one high-res primitive set per video
        #      [2] rendered_dict   -- rasterized depth / normal / alpha of the low-res planes
        return (output, pred_prim_high, rendered_dict)
