"""Attention for the WAN DiT: the varlen op (FlexFlashAttnVarlen) and the
WanAttention / WanAttnProcessor layers on top of it.
"""

import torch
import torch.nn as nn
from typing import Optional
from typing_extensions import List
from diffusers.models.attention import AttentionModuleMixin

try:
    from flash_attn_interface import flash_attn_varlen_func
except ImportError:
    # FlashAttention-3 builds kernels for sm_80 and sm_90a only, so it is not available on e.g. Blackwell GeForce /
    # RTX PRO GPUs (sm_120) or on Windows. Use PyTorch SDPA on each segment instead: the same attention, slower.
    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                               causal=False):
        """[total_q, heads, dim] attention in which tokens of segment i (cu_seqlens[i]:cu_seqlens[i + 1])
        attend only to keys of segment i, as flash_attn_varlen_func. Non-causal only (PPMem's use)."""
        if causal:
            raise NotImplementedError("the SDPA fallback supports causal=False only")
        cu_q, cu_k = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
        out = q.new_empty(q.shape[0], q.shape[1], v.shape[2])
        for i in range(len(cu_q) - 1):
            q_i = q[cu_q[i]:cu_q[i + 1]].transpose(0, 1)[None]
            k_i = k[cu_k[i]:cu_k[i + 1]].transpose(0, 1)[None]
            v_i = v[cu_k[i]:cu_k[i + 1]].transpose(0, 1)[None]
            out[cu_q[i]:cu_q[i + 1]] = torch.nn.functional.scaled_dot_product_attention(q_i, k_i, v_i)[0].transpose(0, 1)
        return out

try:
    from magi_attention.common.ranges import AttnRanges
    from magi_attention.functional.flex_flash_attn import flex_flash_attn_func
except ModuleNotFoundError as exc:
    if exc.name != "magi_attention":
        raise
    AttnRanges = None
    flex_flash_attn_func = None


def _use_magi(device):
    return flex_flash_attn_func is not None and torch.cuda.get_device_capability(device)[0] >= 9


class BaseAttnVarlen(torch.nn.Module):
    def __init__(
        self,
        attn_type: str = "self-attention",
    ) -> None:
        super().__init__()
        if attn_type not in ["self-attention", "cross-attention"]:
            raise ValueError(f"Unrecognized attn_type {attn_type}.")
        self.attn_type = attn_type

    def convert_type(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        q_varlen = query[0]
        k_varlen = key[0]
        v_varlen = value[0]

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)
        return q_varlen, k_varlen, v_varlen

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype=torch.bfloat16,
    ):
        q_varlen, k_varlen, v_varlen = self.convert_type(query, key, value, dtype)
        if self.attn_type == "self-attention":
            x_out = self.self_attn(q_varlen, k_varlen, v_varlen)
        elif self.attn_type == "cross-attention":
            x_out = self.cross_attn(q_varlen, k_varlen, v_varlen)
        else:
            raise ValueError(f"Unrecognized attn_type {self.attn_type}.")
        return x_out

    def self_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        raise NotImplementedError("Not implement self.self_attn")

    def cross_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        raise NotImplementedError("Not implement self.cross_attn")

    @staticmethod
    @torch.no_grad()
    def init_attn_params(
        self,
    ) -> None:
        raise NotImplementedError("Not implement self.init_attn_params")


class FlexFlashAttnVarlen(BaseAttnVarlen):
    is_causal_selfattn: bool = False
    is_causal_crossattn: bool = False
    # flash attn params
    max_seqlen_q: int = None
    cu_seqlens_q: torch.Tensor = None
    max_seqlen_kv: int = None
    cu_seqlens_kv: torch.Tensor = None
    # magi attn params
    ## self-attention
    selfattn_ranges: AttnRanges = None
    selfattn_mask_type: List[int] = None
    ## cross-attention
    crossattn_q_ranges: AttnRanges = None
    crossattn_kv_ranges: AttnRanges = None
    crossattn_mask_type: List[int] = None

    def self_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        if _use_magi(query.device):
            x_out, _ = flex_flash_attn_func(
                query, key, value,
                q_ranges=FlexFlashAttnVarlen.selfattn_ranges.to_tensor(query.device),
                k_ranges=FlexFlashAttnVarlen.selfattn_ranges.to_tensor(query.device),
                attn_type_map=FlexFlashAttnVarlen.selfattn_mask_type,
            )
        else: # otherwise we use flash attn varlen
            x_out = flash_attn_varlen_func(
                query, key, value,
                cu_seqlens_q=FlexFlashAttnVarlen.cu_seqlens_q,
                cu_seqlens_k=FlexFlashAttnVarlen.cu_seqlens_q,
                max_seqlen_q=FlexFlashAttnVarlen.max_seqlen_q,
                max_seqlen_k=FlexFlashAttnVarlen.max_seqlen_q,
                causal=FlexFlashAttnVarlen.is_causal_selfattn,
            )
        x_out = x_out[None, ...] # add batch_size dim
        return x_out

    def cross_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        if _use_magi(query.device):
            x_out, _ = flex_flash_attn_func(
                query, key, value,
                q_ranges=FlexFlashAttnVarlen.crossattn_q_ranges.to_tensor(query.device),
                k_ranges=FlexFlashAttnVarlen.crossattn_kv_ranges.to_tensor(query.device),
                attn_type_map=FlexFlashAttnVarlen.crossattn_mask_type,
            )
        else: # otherwise we use flash attn varlen
            x_out = flash_attn_varlen_func(
                query, key, value,
                cu_seqlens_q=FlexFlashAttnVarlen.cu_seqlens_q,
                cu_seqlens_k=FlexFlashAttnVarlen.cu_seqlens_kv,
                max_seqlen_q=FlexFlashAttnVarlen.max_seqlen_q,
                max_seqlen_k=FlexFlashAttnVarlen.max_seqlen_kv,
                causal=FlexFlashAttnVarlen.is_causal_crossattn,
            )
        x_out = x_out[None, ...] # add batch_size dim
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_attn_params(
        n_seq: int = None,
        q_seq_len_list: List = None,
        kv_seq_len_list: List = None,
        is_causal_selfattn: bool = False,
        is_causal_crossattn: bool = False,
        torch_device: str = "cuda",
    ) -> None:
        if _use_magi(torch_device):
            # self attention
            selfattn_ranges = AttnRanges.from_ranges(FlexFlashAttnVarlen.compute_ranges(q_seq_len_list))
            if is_causal_selfattn:
                selfattn_mask_type = torch.ones(len(selfattn_ranges), dtype=torch.int32, device=torch_device)
            else:
                selfattn_mask_type = torch.zeros(len(selfattn_ranges), dtype=torch.int32, device=torch_device)
            FlexFlashAttnVarlen.selfattn_ranges = selfattn_ranges
            FlexFlashAttnVarlen.is_causal_selfattn = is_causal_selfattn
            FlexFlashAttnVarlen.selfattn_mask_type = selfattn_mask_type
            # cross attention
            crossattn_q_ranges = AttnRanges.from_ranges(FlexFlashAttnVarlen.compute_ranges(q_seq_len_list))
            crossattn_kv_ranges = AttnRanges.from_ranges(FlexFlashAttnVarlen.compute_ranges(kv_seq_len_list))
            if is_causal_crossattn:
                crossattn_mask_type = torch.ones(len(crossattn_q_ranges), dtype=torch.int32, device=torch_device)
            else:
                crossattn_mask_type = torch.zeros(len(crossattn_q_ranges), dtype=torch.int32, device=torch_device)
            FlexFlashAttnVarlen.crossattn_q_ranges = crossattn_q_ranges
            FlexFlashAttnVarlen.crossattn_kv_ranges = crossattn_kv_ranges
            FlexFlashAttnVarlen.crossattn_mask_type = crossattn_mask_type
        else: # otherwise we use flash attn varlen
            seqlen_q = torch.tensor(q_seq_len_list, dtype=torch.int32, device=torch_device)
            cu_seqlens_q = torch.zeros(n_seq + 1, dtype=torch.int32, device=torch_device)
            cu_seqlens_q[1:] = torch.cumsum(seqlen_q, dim=0)
            FlexFlashAttnVarlen.max_seqlen_q = max(seqlen_q)
            FlexFlashAttnVarlen.cu_seqlens_q = cu_seqlens_q

            seqlen_kv = torch.tensor(kv_seq_len_list, dtype=torch.int32, device=torch_device)
            cu_seqlens_kv = torch.zeros(n_seq + 1, dtype=torch.int32, device=torch_device)
            cu_seqlens_kv[1:] = torch.cumsum(seqlen_kv, dim=0)
            FlexFlashAttnVarlen.max_seqlen_kv = max(seqlen_kv)
            FlexFlashAttnVarlen.cu_seqlens_kv = cu_seqlens_kv

            FlexFlashAttnVarlen.is_causal_selfattn = is_causal_selfattn
            FlexFlashAttnVarlen.is_causal_crossattn = is_causal_crossattn

    @staticmethod
    def compute_ranges(seq_len_list):
        """Turn per-segment lengths into [start, end) pairs.

        [900, 1200]  ->  [[0, 900], [900, 2100]]     (same info as cu_seqlens [0, 900, 2100])
        """
        ranges = []
        start = 0
        for seq_len in seq_len_list:
            seq_len = int(seq_len)
            ranges.append([start, start + seq_len])
            start += seq_len
        return ranges


def build_attn_op(attn_type: str = "self-attention"):
    """Use MagiAttention when it is installed and the GPU's compute capability is 9.0 or
    higher (Hopper and newer); otherwise FlashAttention-3."""
    return FlexFlashAttnVarlen(attn_type)


def build_attn_mask(
    n_seq: int = None,
    q_seq_len_list: List = None,
    kv_seq_len_list: List = None,
    is_causal_selfattn: bool = False,
    is_causal_crossattn: bool = False,
    device: str = "cuda",
):
    """Compute this forward pass's segment boundaries into FlexFlashAttnVarlen's
    *class* attributes, where all 30 blocks read them from.

    Class-level state avoids threading cu_seqlens through block and processor
    signatures, but it is not reentrant: two model instances interleaving forward
    passes will clobber each other. Fine for one model running sequentially.
    """
    FlexFlashAttnVarlen.init_attn_params(
        n_seq, q_seq_len_list, kv_seq_len_list,
        is_causal_selfattn, is_causal_crossattn, device,
    )


def _get_qkv_projections(attn: "WanAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if attn.cross_attention_dim_head is None:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


def _get_added_kv_projections(attn: "WanAttention", encoder_hidden_states_img: torch.Tensor):
    if attn.fused_projections:
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


def apply_rotary_emb(x, freqs):
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(3)
    return x_out.to(x.dtype)


class WanAttnProcessor:
    def __init__(self, attn_type="self-attention"):
        self.attn_type = attn_type
        self.attn_op = build_attn_op(attn_type=attn_type)

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,  # complex RoPE frequencies
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # Image-context branch (unused here: added_kv_proj_dim is None)
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = self.attn_op(query, key_img, value_img)
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = self.attn_op(query, key, value)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class RMSNorm(nn.RMSNorm):
    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class WanAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WanAttnProcessor
    _available_processors = [WanAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        processor=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = RMSNorm(dim_head * heads, eps=eps)

        self.set_processor(processor)

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if self.cross_attention_dim_head is None:
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            concatenated_bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def init_weights(self):
        """Initialize weights for the attention module."""
        nn.init.normal_(self.to_q.weight, std=0.02)
        if self.to_q.bias is not None:
            nn.init.constant_(self.to_q.bias, 0.0)

        nn.init.normal_(self.to_k.weight, std=0.02)
        if self.to_k.bias is not None:
            nn.init.constant_(self.to_k.bias, 0.0)

        nn.init.normal_(self.to_v.weight, std=0.02)
        if self.to_v.bias is not None:
            nn.init.constant_(self.to_v.bias, 0.0)

        nn.init.normal_(self.to_out[0].weight, std=0.02)
        if self.to_out[0].bias is not None:
            nn.init.constant_(self.to_out[0].bias, 0.0)

        if self.add_k_proj is not None:
            nn.init.normal_(self.add_k_proj.weight, std=0.02)
            if self.add_k_proj.bias is not None:
                nn.init.constant_(self.add_k_proj.bias, 0.0)

        if self.add_v_proj is not None:
            nn.init.normal_(self.add_v_proj.weight, std=0.02)
            if self.add_v_proj.bias is not None:
                nn.init.constant_(self.add_v_proj.bias, 0.0)

        if hasattr(self, 'to_qkv'):
            nn.init.normal_(self.to_qkv.weight, std=0.02)
            if self.to_qkv.bias is not None:
                nn.init.constant_(self.to_qkv.bias, 0.0)

        if hasattr(self, 'to_kv'):
            nn.init.normal_(self.to_kv.weight, std=0.02)
            if self.to_kv.bias is not None:
                nn.init.constant_(self.to_kv.bias, 0.0)

        if hasattr(self, 'to_added_kv'):
            nn.init.normal_(self.to_added_kv.weight, std=0.02)
            if self.to_added_kv.bias is not None:
                nn.init.constant_(self.to_added_kv.bias, 0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,  # complex RoPE frequencies
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states, rotary_emb, **kwargs)
