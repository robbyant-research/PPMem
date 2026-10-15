import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear + low-rank adaptation. State-dict compatible with nn.Linear:
    the base `weight` / `bias` parameters are adopted from the original Linear
    (same Parameter objects), so ckpt keys `<name>.weight` / `<name>.bias` still
    match after wrapping. Adds `lora_A`, `lora_B` as extras.
    """

    def __init__(self, orig: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.weight = orig.weight
        self.bias = orig.bias if orig.bias is not None else None
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features, device=self.weight.device, dtype=self.weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, device=self.weight.device, dtype=self.weight.dtype)
        )
        nn.init.trunc_normal_(self.lora_A, std=0.02)
        self.lora_rank = rank
        self.lora_alpha = alpha
        self.lora_scale = alpha / rank
        self.enable_lora = True

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.enable_lora:
            lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scale
            out = out + lora.to(out.dtype)
        return out


def _wrap_attention(attn: nn.Module, rank: int, alpha: int) -> None:
    """Wrap attn's to_q/to_k/to_v/to_out[0] with LoRALinear. Works for both
    attn1 (self-attn) and attn2 (cross-attn)."""
    attn.to_q = LoRALinear(attn.to_q, rank, alpha)
    attn.to_k = LoRALinear(attn.to_k, rank, alpha)
    attn.to_v = LoRALinear(attn.to_v, rank, alpha)
    attn.to_out[0] = LoRALinear(attn.to_out[0], rank, alpha)


def _wrap_ffn(ffn: nn.Module, rank: int, alpha: int) -> None:
    """Wrap FeedForward's two linear layers: the up-projection inside the activation
    module (net[0].proj) and the down-projection (net[2])."""
    # Up-projection (inside the activation wrapper)
    if hasattr(ffn.net[0], 'proj'):
        ffn.net[0].proj = LoRALinear(ffn.net[0].proj, rank, alpha)
    # Down-projection
    if hasattr(ffn.net, '__getitem__') and len(ffn.net) > 2:
        ffn.net[2] = LoRALinear(ffn.net[2], rank, alpha)


def apply_lora_to_block(block: nn.Module, rank: int, alpha: int, targets: list) -> int:
    """Apply LoRA to the specified target modules in a DiT block.

    targets: subset of {"attn1", "attn2", "ffn"}. Returns number of LoRALinear
    instances created in this block.
    """
    n_before = sum(1 for m in block.modules() if isinstance(m, LoRALinear))
    if "attn1" in targets and hasattr(block, "attn1"):
        _wrap_attention(block.attn1, rank, alpha)
    if "attn2" in targets and hasattr(block, "attn2"):
        _wrap_attention(block.attn2, rank, alpha)
    if "ffn" in targets and hasattr(block, "ffn"):
        _wrap_ffn(block.ffn, rank, alpha)
    n_after = sum(1 for m in block.modules() if isinstance(m, LoRALinear))
    return n_after - n_before


def parse_lora_targets(spec: str) -> list:
    """Parse lora_targets spec string into a list.
    - ""  or "attn1" → ["attn1"] (default)
    - "all"           → ["attn1", "attn2", "ffn"]
    - "attn1,ffn"     → ["attn1", "ffn"]
    - "attn1,attn2"   → ["attn1", "attn2"]
    """
    s = (spec or "").strip()
    if not s or s == "attn1":
        return ["attn1"]
    if s == "all":
        return ["attn1", "attn2", "ffn"]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    valid = {"attn1", "attn2", "ffn"}
    invalid = [p for p in parts if p not in valid]
    if invalid:
        raise ValueError(f"Invalid lora_targets entries: {invalid}. Must be subset of {valid}, 'all', or empty")
    # Deduplicate preserving order
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def parse_block_range(spec: str, num_blocks: int) -> list[int]:
    spec = (spec or "").strip()
    if not spec:
        return []
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo_i, hi_i = int(lo), int(hi)
            for i in range(lo_i, hi_i + 1):
                if 0 <= i < num_blocks:
                    out.add(i)
        else:
            i = int(part)
            if 0 <= i < num_blocks:
                out.add(i)
    return sorted(out)


def set_lora_enabled(model: nn.Module, enabled: bool) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enable_lora = enabled
            n += 1
    return n
