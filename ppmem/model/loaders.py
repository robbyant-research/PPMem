"""Weight loading: the WAN VAE, text encoder and tokenizer, and the PPMem checkpoint.

Model weights are loaded on CPU, then moved to the requested device.
"""

import importlib.util
import json
from pathlib import Path

import torch

from transformers import T5TokenizerFast, UMT5EncoderModel

from .vae import AutoencoderKLWanSeparate

CHECKPOINT_INDEX = 'model.safetensors.index.json'
CHECKPOINT_SINGLE = 'model.safetensors'
SHARD_GLOB = 'model-*-of-*.safetensors'
# Published next to the weights, in the subdirectory names diffusers expects.
WAN_COMPONENTS = ('vae', 'text_encoder', 'tokenizer')


def locate_checkpoint(checkpoint_dir):
    """Locate checkpoint weights without loading tensors.

    Search order: sharded safetensors, then a single model.safetensors.
    Returns (kind, files, description, weight_map or None).
    Raises FileNotFoundError with the searched paths if no checkpoint is found.
    """
    ckpt = Path(checkpoint_dir)
    index = ckpt / CHECKPOINT_INDEX
    if index.exists():
        weight_map = json.loads(index.read_text())['weight_map']
        shards = sorted(set(weight_map.values()))
        absent = [s for s in shards if not (ckpt / s).exists()]
        if absent:
            raise FileNotFoundError(
                f"{index} lists {len(shards)} shards and {len(absent)} are not there: "
                f"{absent[:4]}. A download that stopped early leaves exactly this."
            )
        # Unreferenced shards may belong to a different checkpoint.
        stray = sorted({p.name for p in ckpt.glob(SHARD_GLOB)} - set(shards))
        if stray:
            raise ValueError(
                f"{index} names {len(shards)} shards but {len(stray)} more are in "
                f"{ckpt}: {stray[:4]}. Convert into an empty directory."
            )
        return 'shards', [ckpt / s for s in shards], f"{len(shards)} safetensors shards", weight_map

    single = ckpt / CHECKPOINT_SINGLE
    if single.exists():
        return 'single', [single], single.name, None

    raise FileNotFoundError(
        f"no checkpoint in {ckpt}. Looked for {CHECKPOINT_INDEX} (sharded), then "
        f"{CHECKPOINT_SINGLE}."
    )


def check_checkpoint_dir(checkpoint_dir):
    """Check for weights and the vae/, text_encoder/, and tokenizer/ directories.

    Returns a description of the weight format found.
    """
    _, _, detail, _ = locate_checkpoint(checkpoint_dir)
    ckpt = Path(checkpoint_dir)
    absent = [c for c in WAN_COMPONENTS if not (ckpt / c).is_dir()]
    if absent:
        raise FileNotFoundError(
            f"{ckpt} holds the weights but not {absent}. The published checkpoint carries "
            f"{list(WAN_COMPONENTS)} alongside them; copy or link them in if you assembled "
            f"this directory yourself."
        )
    return detail


def read_checkpoint(checkpoint_dir):
    """The model weights out of a checkpoint directory, whatever form they take.

    Returns (state_dict, what it read).
    """
    from safetensors.torch import load_file

    kind, files, detail, weight_map = locate_checkpoint(checkpoint_dir)

    if kind == 'shards':
        state_dict = {}
        for shard in files:
            state_dict.update(load_file(str(shard)))
        if set(state_dict) != set(weight_map):
            only_index = sorted(set(weight_map) - set(state_dict))[:4]
            only_shards = sorted(set(state_dict) - set(weight_map))[:4]
            raise ValueError(
                f"{Path(checkpoint_dir) / CHECKPOINT_INDEX} and the shards disagree on "
                f"which tensors exist (index only: {only_index}, "
                f"shards only: {only_shards})."
            )
        return state_dict, detail

    return load_file(str(files[0])), detail


def _sub(subfolder: str):
    """Pass subfolder separately for both local directories and HF repo IDs.

    An empty value preserves each loader's default: None for diffusers, "" for transformers.
    """
    return {"subfolder": subfolder} if subfolder else {}


def load_vae(
    vae_path: str,
    torch_dtype: torch.dtype = None,
    torch_device: torch.device = None,
    subfolder: str = "",
):
    """
    Load the Wan VAE, used per frame (each frame encoded and decoded on its own).

    Args:
        vae_path: checkpoint directory or HF repo ID
        torch_dtype: data type for the model
        torch_device: device to load the model on
        subfolder: subdirectory holding the VAE (the checkpoint's vae/)
    """
    vae = AutoencoderKLWanSeparate.from_pretrained(
        vae_path,
        **_sub(subfolder),
        torch_dtype=torch_dtype,
        # diffusers' default needs accelerate and warns when it is missing; the
        # fallback it then takes is the same as passing False.
        low_cpu_mem_usage=importlib.util.find_spec("accelerate") is not None,
    )
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path: str,
    torch_dtype: torch.dtype = None,
    torch_device: torch.device = None,
    subfolder: str = "",
):
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        **_sub(subfolder),
        dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(
    tokenizer_path: str,
    subfolder: str = "",
):
    tokenizer = T5TokenizerFast.from_pretrained(
        tokenizer_path,
        **_sub(subfolder),
    )
    return tokenizer
