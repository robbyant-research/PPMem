"""Resolve user-facing model choices without importing the inference stack."""
import os
from pathlib import Path

DEFAULT_CHECKPOINT = "Robbyant-Research/PPMem"
DEFAULT_MOGE_CHECKPOINT = "Ruicheng/moge-2-vitl-normal"


def _source(value, *, directory):
    value = os.fspath(value)
    if not value.strip():
        raise ValueError("Checkpoint must be a local path or a Hugging Face repository ID.")
    path = Path(value).expanduser()
    if path.exists():
        valid = path.is_dir() if directory else path.is_file()
        if not valid:
            kind = "directory" if directory else "file"
            raise ValueError(f"Expected a checkpoint {kind}: {path}")
        if directory and not any(path.iterdir()):
            raise FileNotFoundError(f"Checkpoint directory is empty: {path}")
        return str(path.resolve())
    if value.startswith(("/", ".", "~")) or "\\" in value or (not directory and path.suffix in {".pt", ".pth", ".bin", ".safetensors"}):
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    # An explicit ./ prefix distinguishes a missing relative path from a Hub ID.
    from huggingface_hub.utils import validate_repo_id
    validate_repo_id(value)
    return value


def checkpoint_source(checkpoint=None):
    """--checkpoint when given, else the public model."""
    if checkpoint is not None:
        return _source(checkpoint, directory=True)
    return DEFAULT_CHECKPOINT


def resolve_checkpoint(checkpoint=None):
    """The local checkpoint directory, downloading from the Hub if needed."""
    source = checkpoint_source(checkpoint)
    if Path(source).is_absolute():
        return source
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=source)


def moge_source(checkpoint=None):
    """--moge-checkpoint when given, else the public MoGe model."""
    if checkpoint is not None:
        return _source(checkpoint, directory=False)
    return DEFAULT_MOGE_CHECKPOINT
