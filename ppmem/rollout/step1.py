"""
Latents for step k >= 1: encode what project() returns for the step, together with step 0's
state, into the metadata generate_video takes (the same format as step 0's).
"""

import random
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image

from ppmem.pool.segment_math import sample_history_poses_per_chunk
from ppmem.pool.layout import pool_rgb_files, pool_rgb_path


def proj_arrays(proj, h_tgt, w_tgt, num_context=5):
    """project()'s output as the arrays step1 encodes, at (h_tgt, w_tgt)."""
    def at_size(img):
        if img.shape[0] != h_tgt or img.shape[1] != w_tgt:
            img = cv2.resize(img, (w_tgt, h_tgt), interpolation=cv2.INTER_LINEAR)
        return img

    def frames(stack):
        return np.stack([at_size(f) for f in stack], axis=0)

    def mask_3ch(mask_u8):
        # 0..255 -> [-1, 1], repeated over 3 channels
        m = frames(mask_u8).astype(np.float32) / 255.0 * 2.0 - 1.0
        return np.repeat(m[..., None], 3, axis=-1)

    context_rgb = [at_size(f) for f in proj['context_rgb'][:num_context]]
    context_depth = [at_size(d) for d in proj['context_depth'][:num_context]]
    # Zero context frames need correctly shaped empty arrays; np.stack rejects an empty list.
    return {
        'proj_rgb_1': frames(proj['proj_rgb_1']),       # (segment_size, H, W, 3) uint8
        'proj_rgb_2': frames(proj['proj_rgb_2']),
        'proj_mask_1': mask_3ch(proj['proj_mask_1']),   # (segment_size, H, W, 3) in [-1, 1]
        'proj_mask_2': mask_3ch(proj['proj_mask_2']),
        'next_ar_depth': frames(proj['next_ar_depth']),
        'context_frames': (np.stack(context_rgb, axis=0) if context_rgb
                           else np.zeros((0, h_tgt, w_tgt, 3), dtype=np.uint8)),
        'context_depth': (np.stack(context_depth, axis=0) if context_depth
                          else np.zeros((0, h_tgt, w_tgt), dtype=np.float32)),
    }


def step1(
    incremental_dir,
    proj,
    step=1,
    prompt=None,
    encoder=None,
    num_context=5,
    segment_size=25,
    stride=None,
    history_mode='per_chunk',
    max_history_frames=20,
):
    """
    Build step k's latents from project()'s output `proj` (which carries step 0's
    state) and return them as generate_video's metadata.

    encoder: LatentEncoder (required).
    segment_size: Frames in this chunk (25, or fewer for a last partial chunk).
    stride: Pose advance per chunk (24); defaults to segment_size - 1.
    history_mode: 'per_chunk' = 3 poses per completed prior chunk excluding chunk 0,
        'uniform' = 5 poses spread uniformly over all prior poses.
    """
    pool_dir = Path(incremental_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    step0 = proj['step0']
    h_tgt = int(step0['video_height'])
    w_tgt = int(step0['video_width'])
    if stride is None:
        stride = segment_size - 1

    if prompt is None:
        prompt = step0['text']

    data = proj_arrays(proj, h_tgt, w_tgt, num_context)

    print("\nEncoding to latent...")
    assert encoder is not None, "encoder is required: LatentEncoder expects the caller to pass a VAE"

    # The prompt is not encoded here: the engine encodes it once for all chunks.
    images_dict = {
        'proj_rgb_1': data['proj_rgb_1'],
        'proj_rgb_2': data['proj_rgb_2'],
        'proj_mask_1': data['proj_mask_1'],
        'proj_mask_2': data['proj_mask_2'],
    }
    if num_context > 0:
        images_dict['context_frames'] = data['context_frames'][:num_context]
    latents_dict = encoder.encode_images_batch(
        images_dict, normalize=True, keys_no_normalize=['proj_mask_1', 'proj_mask_2'],
    )
    latents_dict = encoder.rearrange_latents(latents_dict)

    latent = latents_dict['proj_rgb_1']
    B, FHW, C = latent.shape
    latent_h = h_tgt // 16
    latent_w = w_tgt // 16
    # Latent frame count: the separate VAE is 1:1 per frame, so it equals segment_size.
    latent_f = segment_size
    if FHW != latent_f * latent_h * latent_w:
        raise ValueError(
            f"proj_rgb_1 latent has {FHW} patches but expected "
            f"F*H*W = {latent_f}*{latent_h}*{latent_w} = {latent_f * latent_h * latent_w} "
            f"(segment_size={segment_size})"
        )

    next_ar_depth = torch.from_numpy(data['next_ar_depth'].astype(np.float32))  # (segment_size, H, W)
    if num_context > 0:
        context_depth = torch.from_numpy(data['context_depth'][:num_context].astype(np.float32))
    else:
        context_depth = torch.zeros(0, h_tgt, w_tgt)

    meta_info = {
        'text': prompt,
        'step': step,
        'video_num_frames': segment_size,
        'segment_stride': stride,
        'video_height': h_tgt,
        'video_width': w_tgt,
        'latent_num_frames': latent_f,
        'latent_height': latent_h,
        'latent_width': latent_w,
        'context_num_frames': num_context,
        'intrinsics': step0['intrinsics'],
        'extrinsics_c2w': step0['extrinsics_c2w'],
        'next_ar_depth': next_ar_depth,
        'context_depth': context_depth,
    }
    for name in images_dict.keys():
        meta_info[f'latent_{name}'] = latents_dict[name].cpu().detach()[0]

    # Add very_first_frame (VF) and history
    # VF: inherit from step 0 (always the same first frame)
    meta_info['latent_very_first_frame'] = step0['latent_very_first_frame']
    meta_info['very_first_num_frames'] = 1
    # History from pool rgb (frames before current segment)
    pool_rgb_dir = pool_dir / 'rgb'
    current_start_pose = step * stride
    if history_mode == 'per_chunk':
        # Deterministic per step, decorrelated across steps
        # The prior chunks are all full-length: stride + 1 frames each.
        selected = sample_history_poses_per_chunk(
            step, stride + 1, stride,
            max_history_frames=max_history_frames, rng=random.Random(step),
        )
        missing = [i for i in selected if pool_rgb_path(pool_dir, i) is None]
        if missing:
            raise FileNotFoundError(
                f"History poses {missing} missing from {pool_rgb_dir} "
                f"(prior chunks should be complete)"
            )
    elif history_mode == 'uniform':
        available = sorted([int(p.stem) for p in pool_rgb_files(pool_dir)
                            if int(p.stem) < current_start_pose])
        max_hist = 5
        if len(available) > max_hist:
            indices = np.linspace(0, len(available) - 1, max_hist, dtype=int)
            selected = [available[i] for i in indices]
        else:
            selected = available
    else:
        raise ValueError(f"Unknown history_mode '{history_mode}' (per_chunk|uniform)")
    if selected:
        hist_imgs = np.stack([
            cv2.resize(np.array(Image.open(pool_rgb_path(pool_dir, idx)).convert('RGB')),
                       (w_tgt, h_tgt))
            for idx in selected
        ], axis=0)  # (N, H, W, 3)
        hist_latents = encoder.encode_images_batch({'history': hist_imgs}, normalize=True)
        hist_latents = encoder.rearrange_latents(hist_latents)
        meta_info['latent_history'] = hist_latents['history'].cpu().detach()[0]
        meta_info['history_num_frames'] = len(selected)
        # Pass the chosen pose indices downstream. Re-deriving them elsewhere can
        # yield the same count but different frames, which silently misaligns depth
        # against RGB.
        meta_info['history_poses'] = [int(x) for x in selected]
        print(f"History ({history_mode}): {len(selected)} frames from pool (poses {selected})")
    else:
        meta_info['latent_history'] = torch.zeros(0, C)
        meta_info['history_num_frames'] = 0
        meta_info['history_poses'] = []
        print("History: empty (no prior frames)")

    # First frame for i2v = inherited frame (first frame of step 1 segment)
    first_rgb = data['proj_rgb_1'][0]
    meta_info['first_frame_pil_images'] = Image.fromarray(first_rgb.astype(np.uint8))

    return meta_info
