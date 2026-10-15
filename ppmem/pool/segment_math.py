"""
Pure segment/chunk arithmetic for the incremental AR grid.

The overlap-by-1 chunk layout (segment_size/stride = 25/24): consecutive chunks share
one frame. The last chunk may be shorter, with whatever poses the trajectory has left.
No GPU or model deps.

"""

import random


def resolve_stride(segment_size, stride=None):
    """The pose advance per AR step: segment_size - 1 unless given. A shorter last chunk
    passes its own length as segment_size and the full stride."""
    return segment_size - 1 if stride is None else stride


def chunk_length(step, n_poses, segment_size):
    """Frames in chunk `step` of an n_poses trajectory: segment_size, or what is left
    for the last chunk (its first frame is the previous chunk's last)."""
    return min(segment_size, n_poses - step * (segment_size - 1))


def last_step(n_poses, segment_size):
    """Index of the last chunk: the one that reaches the trajectory's last pose."""
    stride = segment_size - 1
    return max(0, -(-(n_poses - segment_size) // stride))


def segment_pose_range(step, segment_size, stride=None):
    """
    Pose indices covered by chunk `step`: [step*stride, step*stride + segment_size).
    The first pose of chunk k+1 is the last pose of chunk k (inherited frame).
    """
    stride = resolve_stride(segment_size, stride)
    start = step * stride
    return start, start + segment_size


def chunk_seed(seed, step):
    """
    Per-chunk generation seed. One seed for every chunk would give them all identical
    initial noise; seed + step keeps the run deterministic while decorrelating chunks.
    """
    return seed + step


def sample_history_poses_per_chunk(step, segment_size, stride, max_history_frames=20, rng=None):
    """
    For each completed prior chunk except chunk 0, pick one random pose from each
    third of its pose range. Chunk 0 is excluded because its frames duplicate the
    first-frame and early context conditions. If the total exceeds max_history_frames,
    randomly subsample and sort the result.

    Returns a sorted-per-chunk list of pose indices (may be empty for step <= 1).

    """
    rng = rng if rng is not None else random.Random(0)
    sampled = []
    for j in range(1, step):
        poses = list(range(j * stride, j * stride + segment_size))
        n = len(poses)
        for part in range(3):
            start = int(part * n / 3)
            end = max(int((part + 1) * n / 3), start + 1)
            sampled.append(rng.choice(poses[start:end]))
    if len(sampled) > max_history_frames:
        sampled = sorted(rng.sample(sampled, max_history_frames))
    return sampled
