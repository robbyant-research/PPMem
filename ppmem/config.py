"""Configuration: the model constants and the per-run settings.

The model's values are fixed by the released checkpoint and live in `MODEL`. The
settings a run can change are in two sections, set from a YAML file (`--config`) or
the command line (`--section.key value`, which wins over the YAML):

  run     sampling steps, CFG, flow shift, determinism, debug output
"""

import dataclasses
import sys
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import yaml

# The VAE z_dim. Named separately because in_channels is derived from it.
ARCH_OUT_CHANNELS = 48


@dataclass(frozen=True)
class Model:
    """The model's hyperparameters, all fixed by the released checkpoint. The model reads
    this object as `self.model_args`."""

    # architecture
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    out_channels: int = ARCH_OUT_CHANNELS
    out_channels_feat: Optional[int] = 240
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 14336
    num_layers: int = 30
    cross_attn_norm: bool = True
    eps: float = 1e-06
    added_kv_proj_dim: Optional[int] = None
    # Six concatenated latents: main, proj_rgb_1/2, proj_mask_1/2 and next_ar_depth.
    in_channels: int = ARCH_OUT_CHANNELS * 6
    decoder_hidden_dim: int = 1024

    # depth: the plane head predicts log depth in [min_depth, max_depth]; the depth fed
    # to the DiT is log-normalized over [depth_norm_range_min, depth_norm_range_max].
    depth_mode: str = "log"
    min_depth: float = 0.2
    max_depth: float = 277.0
    depth_norm_method: str = "log"
    depth_norm_range_min: float = 0.2
    depth_norm_range_max: float = 99.0

    # LoRA on the self-attention of every block
    lora_on_dit_blocks: str = "0-29"
    lora_rank: int = 256
    lora_alpha: int = 256
    lora_targets: str = "attn1"


MODEL = Model()


@dataclass
class Run:
    """Per-run knobs."""

    deterministic: bool = False
    """Enable deterministic algorithm settings; this may reduce performance."""

    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    flow_shift: float = 5.0
    """Shift of the flow-matching sampling schedule."""

    save_debug: bool = False
    """Also write debug outputs: world-space normals for every pose (pool/normal/); under
    pool/_current, rendered depth/normal/alpha images, raw plane parameters and PLY
    (overwritten by each chunk), and the plane-merge debug images (merge_debug/step{NNN}_*,
    kept for every chunk). The pipeline does not read them."""

@dataclass
class JobConfig:
    """Inference config."""

    run: Run = field(default_factory=Run)

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}
_NONE = {"none", "null", ""}

# The config file path is not a dataclass field; accept these three spellings.
_CONFIG_FILE_KEYS = {"--config", "--config_file", "--config-file"}


def _norm(name: str) -> str:
    return name.replace("-", "_")


def _schema() -> dict[str, dict[str, Any]]:
    """{section: {field: declared type}}, taken from the dataclass declarations."""
    return {
        sec.name: {f.name: f.type for f in dataclasses.fields(sec.type)}
        for sec in dataclasses.fields(JobConfig)
    }


def _coerce(raw: Any, typ: Any, where: str) -> Any:
    """Coerce to the declared type; raise rather than passing a value through as-is."""
    origin = typing.get_origin(typ)

    if origin is typing.Literal:
        allowed = typing.get_args(typ)
        for a in allowed:
            if raw == a or str(raw) == str(a):
                return a
        raise SystemExit(f"🔴 {where}: {raw!r} is not one of {list(allowed)}.")

    if origin in (typing.Union, types.UnionType):
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in _NONE):
            return None
        inner = [a for a in typing.get_args(typ) if a is not type(None)]
        return _coerce(raw, inner[0], where)

    if origin is list:
        if isinstance(raw, list):
            return [str(x) for x in raw]
        return [s for s in str(raw).split(",") if s != ""]

    if typ is bool:
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        raise SystemExit(f"🔴 {where}: {raw!r} is not true/false")

    if typ in (int, float, str):
        try:
            return typ(raw)
        except (TypeError, ValueError):
            raise SystemExit(f"🔴 {where}: cannot convert {raw!r} to {typ.__name__}")

    return raw


def _resolve(sec: str, fld: str, schema, where: str) -> tuple[str, str, Any]:
    """Resolve (section, field) to a real field. Returns (section, field, type)."""
    sec_n, fld_n = _norm(sec), _norm(fld)
    if sec_n not in schema:
        raise SystemExit(
            f"🔴 {where}: no section named `{sec_n}` (available: {', '.join(sorted(schema))})"
        )
    if fld_n not in schema[sec_n]:
        raise SystemExit(f"🔴 {where}: `{sec_n}` has no field named `{fld_n}`")
    return sec_n, fld_n, schema[sec_n][fld_n]


def _from_yaml(path: str, schema) -> dict[tuple[str, str], Any]:
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except OSError as e:
        raise SystemExit(f"🔴 config not readable: {path} ({e.strerror})")
    except yaml.YAMLError as e:
        raise SystemExit(f"🔴 {path} is not valid YAML: {str(e).splitlines()[0]}")
    if not isinstance(data, dict):
        raise SystemExit(f"🔴 {path}: the top level must be a mapping of sections")

    flat: dict[tuple[str, str], Any] = {}
    for sec, kv in data.items():
        if not isinstance(kv, dict):
            raise SystemExit(f"🔴 {path}: `{sec}` must contain key/value pairs, got {type(kv).__name__}")
        for key, val in kv.items():
            sec_n, fld_n, typ = _resolve(sec, key, schema, path)
            flat[(sec_n, fld_n)] = _coerce(val, typ, f"{path}: {sec}.{key}")
    return flat


def _from_cli(argv: list[str], schema) -> dict[tuple[str, str], Any]:
    flat: dict[tuple[str, str], Any] = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        i += 1

        if tok == "--set":
            if i >= len(argv):
                raise SystemExit("🔴 --set needs section.key=value after it")
            spec, i = argv[i], i + 1
            # `--set config=x.yaml` == `--config x.yaml`; _find_config_file took it already
            if "=" in spec and _norm(spec.split("=", 1)[0]) == "config":
                continue
            if "=" not in spec or "." not in spec.split("=", 1)[0]:
                raise SystemExit(f"🔴 --set {spec}: expected section.key=value")
            key, val = spec.split("=", 1)
            sec, fld = key.split(".", 1)
            sec_n, fld_n, typ = _resolve(sec, fld, schema, f"--set {spec}")
            flat[(sec_n, fld_n)] = _coerce(val, typ, f"--set {spec}")
            continue

        if not tok.startswith("--"):
            raise SystemExit(f"🔴 unrecognized argument: {tok}")

        # The config file path belongs to no section and was already taken by
        # _find_config_file; skip it or it falls into the --section.key error below.
        if tok in _CONFIG_FILE_KEYS:
            i += 1                      # skip the path that follows it too
            continue
        if "=" in tok and tok.split("=", 1)[0] in _CONFIG_FILE_KEYS:
            continue

        body = tok[2:]
        if "=" in body:
            key, val = body.split("=", 1)
            has_val = True
        else:
            key, val, has_val = body, None, False
        if "." not in key:
            raise SystemExit(f"🔴 {tok}: must be written as --section.key")

        sec, fld = key.split(".", 1)
        sec_n, fld_n, typ = _resolve(sec, fld, schema, tok)

        if typ is bool and not has_val:
            # A bool does not consume the next token.
            flat[(sec_n, fld_n)] = True
            continue
        if not has_val:
            if i >= len(argv):
                raise SystemExit(f"🔴 {tok} is missing a value")
            val, i = argv[i], i + 1
        flat[(sec_n, fld_n)] = _coerce(val, typ, tok)

    return flat


def _find_config_file(argv: list[str]) -> str | None:
    """Pre-scan argv for the YAML path.

    `--set config=x.yaml` must be recognised here too, otherwise it falls into the
    generic --set branch and errors out as "config is not a field of any section".
    """
    found = None
    for i, tok in enumerate(argv):
        if "=" in tok and tok.split("=", 1)[0] in _CONFIG_FILE_KEYS:
            found = tok.split("=", 1)[1]
        elif tok in _CONFIG_FILE_KEYS:
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise SystemExit(f"🔴 {tok} needs a path")
            found = argv[i + 1]
        elif tok == "--set" and i + 1 < len(argv) and "=" in argv[i + 1]:
            key, val = argv[i + 1].split("=", 1)
            if "--" + _norm(key) in _CONFIG_FILE_KEYS or _norm(key) == "config":
                found = val
        if found == "":
            raise SystemExit(f"🔴 {tok}: empty config path")
    # Last one wins, like every other flag here.
    return found


def _print_help(schema) -> None:
    print(__doc__)
    print("Available sections and fields:\n")
    for sec, flds in schema.items():
        print(f"  [{sec}]")
        for name, typ in flds.items():
            tn = getattr(typ, "__name__", str(typ).replace("typing.", ""))
            print(f"      {name}: {tn}")
        print()


def load_config(argv: list[str] | None = None) -> JobConfig:
    """Read the YAML, apply command-line overrides, return a JobConfig."""
    argv = list(sys.argv[1:] if argv is None else argv)
    schema = _schema()

    if "--help" in argv or "-h" in argv:
        _print_help(schema)
        raise SystemExit(0)

    flat: dict[tuple[str, str], Any] = {}
    path = _find_config_file(argv)
    if path:
        flat.update(_from_yaml(path, schema))
    flat.update(_from_cli(argv, schema))  # command line overrides YAML

    config = JobConfig()
    for (sec, fld), value in flat.items():
        setattr(getattr(config, sec), fld, value)

    return config
