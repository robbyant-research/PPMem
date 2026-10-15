"""
Step 1+ projection: no MoGe; depth = next_ar_depth; proj RGB/mask from self_ar_depth.
Source RGB is chosen by coverage over all previous steps' views.
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

from ppmem.pool.projector import reproject_to_target_views
from ppmem.pool.source_selection import select_best_source_views_globally
from ppmem.pool.layout import (
    find_array,
    load_array,
    load_step0_state,
    pool_rgb_path,
)


def load_pool_depths(pool_dir, num_frames, video_height, video_width):
    """Load depth from pool for pose indices 0..num_frames-1 (pool index = pose index).

    Source selection needs it for every frame; RGB is only needed for the few frames that
    get picked, so load_pool_rgb reads those on demand.
    Coverage is depth > 0: the renderer zeroes depth wherever its alpha is 0.
    """
    pool_dir = Path(pool_dir)
    depths = []
    for i in range(num_frames):
        rgb_path = pool_rgb_path(pool_dir, i)
        d_path = find_array(pool_dir / 'depth', f'{i:05d}')
        if rgb_path is None or d_path is None:
            raise FileNotFoundError(
                f"Pool missing frame {i}: need rgb/{i:05d}.png|.jpg and "
                f"depth/{i:05d}.npy under {pool_dir}"
            )
        d = load_array(d_path)
        # Resized together with the RGB frame; its header gives the size without decoding.
        with Image.open(str(rgb_path)) as im:
            rgb_w, rgb_h = im.size
        if rgb_h != video_height or rgb_w != video_width:
            d = cv2.resize(d, (video_width, video_height), interpolation=cv2.INTER_LINEAR)
        depths.append(d)
    return depths


def load_pool_rgb(pool_dir, i, video_height, video_width):
    """RGB frame of pose i from the pool, at video resolution."""
    rgb = np.array(Image.open(str(pool_rgb_path(Path(pool_dir), i))).convert('RGB'))
    if rgb.shape[0] != video_height or rgb.shape[1] != video_width:
        rgb = cv2.resize(rgb, (video_width, video_height), interpolation=cv2.INTER_LINEAR)
    return rgb


def _coverage_selection_inputs(src_depths, tgt_poses, K,
                               coverage_downscale=1, coverage_max_targets=None):
    """
    Optionally downsampled sources + subsampled probe targets for coverage selection.
    Both reduce the memory and time the selection takes; off by default. Turning either
    on can change which sources are selected.
    """
    if coverage_downscale > 1:
        ds = coverage_downscale
        h, w = src_depths[0].shape[:2]
        sel_w, sel_h = w // ds, h // ds
        sel_depths = [cv2.resize(d, (sel_w, sel_h), interpolation=cv2.INTER_NEAREST) for d in src_depths]
        sel_K = np.array(K, dtype=np.float32).copy()
        sel_K[0] /= ds
        sel_K[1] /= ds
    else:
        sel_depths, sel_K = src_depths, K
    tgt_poses = np.asarray(tgt_poses)
    if coverage_max_targets is not None and len(tgt_poses) > coverage_max_targets:
        probe = np.linspace(0, len(tgt_poses) - 1, coverage_max_targets, dtype=int)
        sel_tgt_poses = tgt_poses[probe]
    else:
        sel_tgt_poses = tgt_poses
    return sel_depths, sel_tgt_poses, sel_K


SUPPORTED_TOP_K = (1, 2)


def check_top_k(top_k):
    """Allow one or two sources for the two conditioning streams.

    Extra sources lack per-target compositing and would break frame alignment.
    """
    if top_k not in SUPPORTED_TOP_K:
        raise ValueError(
            f"--top_k must be one of {SUPPORTED_TOP_K}, got {top_k}. Two sources fill "
            f"the two projection streams; one source fills both with the same view."
        )


def project(
    incremental_dir,
    step=1,
    top_k=2,
    device='cuda',
    segment_size=25,
    stride=None,
    coverage_downscale=1,
    coverage_max_targets=None,
    num_context=5,
):
    """
    Step 1+ projection: use next_ar_depth as depth; proj RGB/mask from self_ar_depth
    with source views selected by select_best_source_views_globally.
    Reads from pool and returns the arrays step1 encodes; the pool merge also stores the
    projected RGB for the diagnostic grid. top_k is 1 or 2, see check_top_k.
    segment_size: frames in this chunk (25, or fewer for a last partial chunk); stride: 24.
    coverage_downscale/coverage_max_targets: resolution/probe-target reduction for
    the coverage selection only (final reprojection stays full-res, all targets).
    """
    check_top_k(top_k)
    pool_dir = Path(incremental_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)

    latent_data = load_step0_state(pool_dir)
    extrinsics_c2w = np.array(latent_data['extrinsics_c2w'])
    intrinsics_list = latent_data['intrinsics']
    video_height = int(latent_data['video_height'])
    video_width = int(latent_data['video_width'])
    if stride is None:
        stride = segment_size - 1
    segment_start = step * stride

    intr = intrinsics_list[0]
    if isinstance(intr, dict):
        K = np.array([
            [intr['fx'], 0.0, intr['cx']],
            [0.0, intr['fy'], intr['cy']],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
    else:
        K = np.array(intr, dtype=np.float32)

    # Source pool: pose indices 0..(num_src-1), every pose up to this chunk's first.
    # Pool index = pose index.
    num_src = step * stride + 1
    src_indices = list(range(num_src))  # pose indices

    src_depths = load_pool_depths(pool_dir, num_src, video_height, video_width)
    src_poses = [extrinsics_c2w[i] for i in src_indices]

    rgb_cache = {}

    def src_rgb(i):
        """RGB of pool frame i, read once."""
        if i not in rgb_cache:
            rgb_cache[i] = load_pool_rgb(pool_dir, i, video_height, video_width)
        return rgb_cache[i]

    tgt_poses = extrinsics_c2w[segment_start : segment_start + segment_size]
    if len(tgt_poses) < segment_size:
        raise ValueError(f"Trajectory has {len(extrinsics_c2w)} poses; need at least {segment_start + segment_size} for step {step}")

    # Coverage selection inputs (possibly downsampled; final projection is full-res)
    sel_depths, sel_tgt_poses, sel_K = _coverage_selection_inputs(
        src_depths, tgt_poses, K,
        coverage_downscale=coverage_downscale, coverage_max_targets=coverage_max_targets,
    )

    # Proj sources and context frames come from the same greedy selection: it is
    # deterministic with a total tie-break, so the first `top_k` picks of a `num_context`
    # run are exactly the picks of a `top_k` run. Run it once and slice.
    n_sel = max(top_k, num_context)
    print(f"\nSelecting best {n_sel} source views from {len(src_indices)} (step {step}, segment [{segment_start}..{segment_start+segment_size}))...")
    _, _, _order, _scores = select_best_source_views_globally(
        src_indices, sel_depths, src_poses,
        sel_tgt_poses, sel_K, top_k=n_sel, device=device,
    )

    # The picks are used in pose order: the earlier source feeds proj_rgb_1, and the
    # context frames enter the sequence in pose order.
    def _take(n):
        idx, sc = _order[:n], _scores[:n]
        if idx:
            o = sorted(range(len(idx)), key=lambda i: idx[i])
            idx, sc = [idx[i] for i in o], [sc[i] for i in o]
        return idx, sc

    best_src_indices, coverage_scores = _take(top_k)
    context_src_indices, context_coverage_scores = _take(num_context)
    print(f"Selected source views (greedy order): {list(_order)}")

    # First frame of segment = inherited (same as last frame of previous segment); pool index = pose index
    inherited_rgb = src_rgb(segment_start)
    inherited_mask = np.ones((video_height, video_width), dtype=np.float32)

    # Reproject from each selected source to each target; build proj_rgb_1/2, proj_mask_1/2
    proj_rgb_1 = [inherited_rgb]
    proj_rgb_2 = [inherited_rgb]
    proj_mask_1 = [inherited_mask]
    proj_mask_2 = [inherited_mask]

    # The first selected source feeds proj_rgb_1 and the second feeds proj_rgb_2.
    # One call per source covers targets 1..segment_size-1; each target is still computed
    # on its own inside, so this matches reprojecting them one at a time.
    for rank, best_idx in enumerate(best_src_indices):
        local = src_indices.index(best_idx)
        rgbs, masks = reproject_to_target_views(
            src_rgb(local), src_depths[local], src_poses[local], tgt_poses[1:segment_size], K,
            device=device,
        )
        if rank == 0:
            proj_rgb_1.extend(rgbs)
            proj_mask_1.extend(masks)
        else:
            proj_rgb_2.extend(rgbs)
            proj_mask_2.extend(masks)
    if len(best_src_indices) == 1:
        proj_rgb_2 = list(proj_rgb_1)
        proj_mask_2 = list(proj_mask_1)

    proj_rgb_1 = np.stack(proj_rgb_1, axis=0)   # (segment_size, H, W, 3)
    proj_rgb_2 = np.stack(proj_rgb_2, axis=0)
    proj_mask_1 = np.stack(proj_mask_1, axis=0) # (segment_size, H, W)
    proj_mask_2 = np.stack(proj_mask_2, axis=0)

    # Depth for this step = next_ar_depth from pool
    next_ar_depth_dir = pool_dir / 'next_ar_depth'
    if not next_ar_depth_dir.exists():
        raise FileNotFoundError(f"Expected next_ar_depth under pool {pool_dir}")
    next_ar_depths = []
    for t in range(segment_size):
        path = next_ar_depth_dir / f'depth_{t:04d}.npy'
        if not path.exists():
            raise FileNotFoundError(f"Expected {path}")
        d = np.load(path)
        if d.shape[0] != video_height or d.shape[1] != video_width:
            d = cv2.resize(d, (video_width, video_height), interpolation=cv2.INTER_LINEAR)
        next_ar_depths.append(d)
    next_ar_depth = np.stack(next_ar_depths, axis=0)

    # Quantized to 8 bits here, as the conditioning always has been: RGB to uint8, the
    # masks to 0..255.
    proj_rgb_1 = np.clip(proj_rgb_1, 0, 255).astype(np.uint8)
    proj_rgb_2 = np.clip(proj_rgb_2, 0, 255).astype(np.uint8)
    proj_mask_1 = (proj_mask_1 * 255).astype(np.uint8)
    proj_mask_2 = (proj_mask_2 * 255).astype(np.uint8)

    # Context frames, in pose order.
    context_rgb = [src_rgb(src_indices.index(i)).astype(np.uint8) for i in context_src_indices]
    context_depth = [src_depths[src_indices.index(i)] for i in context_src_indices]

    print(f"\nStep {step} projection done.")
    return {
        'step0': latent_data,           # step0_state.pt, so step1 need not read it again
        'proj_rgb_1': proj_rgb_1,       # (segment_size, H, W, 3) uint8
        'proj_rgb_2': proj_rgb_2,
        'proj_mask_1': proj_mask_1,     # (segment_size, H, W) uint8, 0 or 255
        'proj_mask_2': proj_mask_2,
        'next_ar_depth': next_ar_depth, # (segment_size, H, W)
        'context_rgb': context_rgb,     # num_context x (H, W, 3) uint8
        'context_depth': context_depth, # num_context x (H, W)
    }


def main():
    parser = argparse.ArgumentParser(description='Step 1+ projection (no MoGe; depth=next_ar_depth; proj from self_ar with source selection)')
    parser.add_argument('--incremental_dir', type=str, required=True, help='Pool root: read source views from the pool')
    parser.add_argument('--step', type=int, default=1, help='Segment step (default 1)')
    parser.add_argument('--top_k', type=int, default=2, help='Number of source views to select (default 2)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--segment_size', type=int, default=25, help='Frames per AR chunk')
    parser.add_argument('--coverage_downscale', type=int, default=1,
                        help='Downsample factor for coverage selection only (4 recommended at 576x1024)')
    parser.add_argument('--coverage_max_targets', type=int, default=None,
                        help='Probe-target count for coverage selection (None = all segment poses)')
    parser.add_argument('--num_context', type=int, default=5,
                        help='Context frames to write, ranks 1..N (0=none)')
    args = parser.parse_args()
    project(
        incremental_dir=args.incremental_dir,
        step=args.step,
        top_k=args.top_k,
        device=args.device,
        segment_size=args.segment_size,
        coverage_downscale=args.coverage_downscale,
        coverage_max_targets=args.coverage_max_targets,
        num_context=args.num_context,
    )


if __name__ == '__main__':
    main()
