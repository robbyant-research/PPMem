import torch
import torch.nn as nn

from typing import Optional
from einops import rearrange

from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.normalization import FP32LayerNorm

from .attention import WanAttention, WanAttnProcessor


@maybe_allow_in_graph
class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(attn_type="self-attention"),
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(attn_type="cross-attention"),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def init_weights(self):
        """Initialize weights for all submodules in the transformer block."""
        self.attn1.init_weights()
        self.attn2.init_weights()

        for module in self.ffn.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        nn.init.normal_(self.scale_shift_table, std=0.02)

        self.norm1.reset_parameters()
        if not isinstance(self.norm2, nn.Identity):
            self.norm2.reset_parameters()
        self.norm3.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:

        temb_scale_shift_table = self.scale_shift_table.to(dtype=temb.dtype, device=temb.device) + temb
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = rearrange(temb_scale_shift_table, 'l n c -> n l c').chunk(6, dim=0)

        # 1. Self-attention
        norm_hidden_states = self.norm1(hidden_states) * (1 + scale_msa) + shift_msa
        attn_output = self.attn1(norm_hidden_states, None, rotary_emb)
        hidden_states = hidden_states + attn_output * gate_msa

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states) * (1 + c_scale_msa) + c_shift_msa
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = hidden_states + ff_output * c_gate_msa

        return hidden_states
