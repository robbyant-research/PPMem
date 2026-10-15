"""
Source view selection for step1+ incremental generation.
Selects the best source views from all previous steps using greedy union coverage.
"""

import numpy as np
from .projector import reproject_coverage
from ppmem._logging import logger

def _pack(mask):
    """A bool mask as uint64 words, 64 pixels each; the zero padding adds no set bits."""
    packed = np.packbits(mask.reshape(-1))
    packed = np.concatenate([packed, np.zeros(-len(packed) % 8, dtype=np.uint8)])
    return packed.view(np.uint64)


def _popcount(words):
    """Number of set bits in a uint64 array."""
    if hasattr(np, "bitwise_count"):                       # numpy >= 2.0
        return int(np.bitwise_count(words).sum(dtype=np.int64))
    # numpy 1.x: SWAR popcount per word
    x = words - ((words >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int(((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).sum(dtype=np.int64))


def select_best_source_views_globally(
    src_indices,
    src_depths,
    src_poses,
    tgt_poses,
    K,
    top_k=2,
    device='cuda',
    top_k_then_min_idx=None,
    prefer_loop_indices=False,
    loop_gap=80,
    sort_selected_by_index=True,
):
    """
    Greedy select top_k source views that maximize union coverage over all target views.
    Each selected view is chosen to maximize the union of coverage with the ones
    already selected.

    Args:
        src_indices: List of source view indices; the returned picks are drawn from these,
            and ties go to the smaller index
        src_depths: List of (H, W) depth maps; a pixel is valid where depth > 0.01
        src_poses: List of (4, 4) c2w poses
        tgt_poses: List of (4, 4) c2w target poses (or (N,4,4) array)
        K: (3, 3) intrinsic matrix
        top_k: Number of sources to select (e.g. 2 for proj_rgb_1, proj_rgb_2)
        device: 'cuda' or 'cpu'
        top_k_then_min_idx: If set to an int (e.g. 5), in each round consider the top this-many
            candidates by coverage, then pick the one with the smallest index among them.
            If None, use default behavior (pick the single best by coverage).
        prefer_loop_indices: If True, when there exist "loop" views (idx <= max(src_indices) - loop_gap),
            restrict each round to only those loop indices (still by coverage order).
        loop_gap: Used when prefer_loop_indices is True. Loop = indices <= max(src_indices) - loop_gap.
        sort_selected_by_index: If True, after selection sort the result by index ascending (small first).

    Returns:
        best_src_indices: top_k values from src_indices, sorted by index when
            sort_selected_by_index
        coverage_scores: cumulative union coverage, aligned with best_src_indices
        greedy_order: the same values in the order the greedy picked them. The greedy
            is deterministic with a total tie-break, so the first n of this list are
            exactly what a top_k=n run would pick -- a caller that needs several k
            runs this once and slices.
        greedy_scores: coverage aligned with greedy_order
    """
    tgt_poses = np.asarray(tgt_poses)
    if tgt_poses.ndim == 2:
        tgt_poses = tgt_poses[np.newaxis, ...]

    # Per-source coverage of every target, bit-packed: (N_tgt * H * W) bools -> 1/8 the bytes.
    # Selection only ever counts pixels of unions, which packing leaves exact.
    src_masks = {}
    for src_idx, src_depth, src_pose in zip(src_indices, src_depths, src_poses):
        src_masks[src_idx] = _pack(reproject_coverage(src_depth, src_pose, tgt_poses, K, device=device))

    # Greedy selection: each round picks the candidate whose union with the views picked
    # so far covers the most pixels, ties going to the smaller index.
    #
    # Lazy evaluation: a candidate's gain over the current union, |m \ U|, can only shrink
    # as U grows, so the gain computed in an earlier round bounds this round's. Candidates
    # are visited in order of that bound, and the round stops as soon as the bound can no
    # longer beat (or tie with a smaller index) the best exact score found. The picks and
    # scores are exactly those of scoring every candidate every round.
    selected_indices = []
    cumulative_coverage_scores = []
    union = None
    union_count = 0
    union_buf = None
    available = set(src_indices)
    gain = {i: _popcount(m) for i, m in src_masks.items()}   # exact for the empty union

    for k in range(top_k):
        if not available:
            if selected_indices:
                selected_indices.append(selected_indices[0])
                cumulative_coverage_scores.append(cumulative_coverage_scores[-1])
                logger.warning(f"  [Selection {k+1}] No more sources, duplicating view {selected_indices[0]}")
            continue

        pool = available
        if prefer_loop_indices:
            loop = {i for i in available if i <= max(src_indices) - loop_gap}
            if loop:
                pool = loop

        def exact_score(i):
            if union is None:
                return gain[i]
            np.bitwise_or(union, src_masks[i], out=union_buf)
            score = _popcount(union_buf)
            gain[i] = score - union_count
            return score

        if top_k_then_min_idx is not None and top_k_then_min_idx > 0:
            # Needs the top-n by score, so every candidate is scored.
            candidates = sorted(((exact_score(i), i) for i in pool), key=lambda x: (-x[0], x[1]))
            best_increase, best_idx = min(candidates[:top_k_then_min_idx], key=lambda x: x[1])
        else:
            best_increase, best_idx = -1, None
            for i in sorted(pool, key=lambda i: (-gain[i], i)):
                bound = union_count + gain[i]
                if bound < best_increase or (bound == best_increase and i > best_idx):
                    break
                score = exact_score(i)
                if score > best_increase or (score == best_increase and i < best_idx):
                    best_increase, best_idx = score, i

        if union is None:
            union = src_masks[best_idx].copy()
            union_buf = np.empty_like(union)
        else:
            np.bitwise_or(union, src_masks[best_idx], out=union)
        union_count = best_increase
        selected_indices.append(best_idx)
        cumulative_coverage_scores.append(best_increase)
        available.discard(best_idx)
        logger.debug(f"  [Selection {k+1}] Selected view {best_idx} (cumulative union coverage: {best_increase:,} pixels)")

    greedy_order = list(selected_indices)
    greedy_scores = list(cumulative_coverage_scores)

    if sort_selected_by_index and selected_indices:
        order = sorted(range(len(selected_indices)), key=lambda i: selected_indices[i])
        selected_indices = [selected_indices[i] for i in order]
        cumulative_coverage_scores = [cumulative_coverage_scores[i] for i in order]

    return selected_indices, cumulative_coverage_scores, greedy_order, greedy_scores
