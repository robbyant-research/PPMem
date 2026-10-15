"""
Pool layout for incremental video generation.

Pool index = pose index (one pose stored once; overlap frame not duplicated).
With segment_size S and stride = S-1 (25/24):
  - Step 0: poses 0..S-1 -> pool 0..S-1 (S frames).
  - Step k>=1: poses (k*stride+1)..(k*stride+stride) -> pool same indices
    (stride new frames; pose k*stride already in pool).

Pool directory contains:
  - rgb/{i:05d}.png|.jpg     pose index i (jpg q92 with --rgb_format jpg; chunk
                             boundaries always png — see merge_video_output_to_pool)
  - depth/{i:05d}.npy        (coverage is depth > 0)
  - normal/{i:05d}.png       world-space normals, 16-bit: n = png / 65535 * 2 - 1
                             (--run.save_debug only)
  - next_ar_depth/           segment_size frames: the next step's depth input (depth_0000.npy ..)
  - proj_rgb_1/, proj_rgb_2/ the projected conditioning frames, for the diagnostic videos
  - step0_state.pt           what later chunks and resume need from step 0: trajectory,
                             intrinsics, video size, prompt, very-first-frame latent
  - input_depth/{i:05d}.png  depth the model was conditioned on at pose i, 16-bit
                             centimetres (depth = png / 100)
  - merged_plane_params.pt, merged_planes_world.ply   the plane pool
  - _current/                staging dir for video gen output (before merge)
"""

from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import cv2
import imageio

from .segment_math import resolve_stride, segment_pose_range
from ppmem._logging import logger


def ensure_pool_dirs(pool_dir):
    """Create pool subdirs if they don't exist."""
    pool_dir = Path(pool_dir)
    for name in ('rgb', 'depth', 'next_ar_depth', 'input_depth',
                 'proj_rgb_1', 'proj_rgb_2',
                 '_current'):
        (pool_dir / name).mkdir(parents=True, exist_ok=True)
    return pool_dir


STEP0_STATE = 'step0_state.pt'
STEP0_STATE_KEYS = ('text', 'intrinsics', 'extrinsics_c2w', 'video_height', 'video_width',
                    'latent_very_first_frame')


def save_step0_state(pool_dir, step0_meta):
    """Keep what later chunks and resume need from step 0's metadata."""
    import torch
    torch.save({k: step0_meta[k] for k in STEP0_STATE_KEYS}, Path(pool_dir) / STEP0_STATE)


def load_step0_state(pool_dir):
    import torch
    path = Path(pool_dir) / STEP0_STATE
    if not path.exists():
        raise FileNotFoundError(f"Expected {path} (written by step 0)")
    return torch.load(path, map_location='cpu', weights_only=False)


def save_input_depth(pool_dir, step, stride, next_ar_depth):
    """Store the depth chunk `step` was conditioned on, by pose, as 16-bit centimetres.

    A chunk's first pose is the previous chunk's last; that frame keeps the earlier
    chunk's depth, as the RGB does.
    """
    out_dir = Path(pool_dir) / 'input_depth'
    out_dir.mkdir(parents=True, exist_ok=True)
    depth = np.asarray(next_ar_depth, dtype=np.float32)
    for fi in range(1 if step > 0 else 0, depth.shape[0]):
        cm = np.clip(np.round(depth[fi] * 100.0), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(out_dir / f'{step * stride + fi:05d}.png'), cm)


def load_input_depth(pool_dir, pose):
    """Depth (metres) the model was conditioned on at `pose`, or None."""
    path = Path(pool_dir) / 'input_depth' / f'{pose:05d}.png'
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 100.0


# ---------------------------------------------------------------------------
# Pool RGB storage: png, or jpg (quality 92) with --rgb_format jpg. Chunk boundary
# frames stay png either way: each is the first frame the next chunk starts from.
# Other frames can also come back as conditioning (projection sources, context,
# history), and in jpg mode those inputs are lossy.
# ---------------------------------------------------------------------------

JPEG_QUALITY = 92


def find_image(dir_path, stem):
    """Resolve <dir>/<stem>.png or .jpg. Returns Path or None."""
    dir_path = Path(dir_path)
    for ext in ('.png', '.jpg'):
        p = dir_path / f'{stem}{ext}'
        if p.exists():
            return p
    return None


def save_image(dir_path, stem, rgb_uint8, fmt='png'):
    """Save (H, W, 3) uint8 RGB as <dir>/<stem>.png or .jpg (quality 92)."""
    dir_path = Path(dir_path)
    if fmt == 'png':
        out = dir_path / f'{stem}.png'
        Image.fromarray(rgb_uint8).save(out)
    elif fmt == 'jpg':
        out = dir_path / f'{stem}.jpg'
        Image.fromarray(rgb_uint8).save(out, quality=JPEG_QUALITY)
    else:
        raise ValueError(f"Unknown image format '{fmt}' (png|jpg)")
    return out


def pool_rgb_path(pool_dir, pose_idx):
    """Resolve pool/rgb frame for a pose index (png preferred, jpg fallback)."""
    return find_image(Path(pool_dir) / 'rgb', f'{pose_idx:05d}')


def pool_rgb_files(pool_dir):
    """All pool/rgb frames sorted by pose index, png + jpg (png wins on duplicates)."""
    rgb_dir = Path(pool_dir) / 'rgb'
    if not rgb_dir.exists():
        return []
    by_stem = {}
    for ext in ('.jpg', '.png'):  # png second so it overwrites a duplicate jpg
        for p in rgb_dir.glob(f'*{ext}'):
            if p.stem.isdigit():
                by_stem[p.stem] = p
    return [by_stem[s] for s in sorted(by_stem, key=int)]


def find_array(dir_path, stem):
    """Resolve <dir>/<stem>.npy. Returns Path or None.

    """
    p = Path(dir_path) / f'{stem}.npy'
    return p if p.exists() else None


def load_array(path):
    """Load a .npy array."""
    if path is None:
        raise FileNotFoundError("Array file missing (find_array returned None)")
    return np.load(path)


def merge_video_output_to_pool(current_output_dir, pool_dir, step, segment_size=25, stride=None,
                               rgb_format='png', step0_proj_rgb=None, proj_rgb=None):
    """
    Merge one step's video gen output into the pool. Pool index = pose index; overlap frame not duplicated.

    Step 0: write segment_size frames (poses 0..segment_size-1) to pool.
    Step k>=1: write stride frames (poses segment_start+1 .. segment_start+stride) to pool; skip first frame (inherited, already in pool).

    Reads from current_output_dir: generated_video.mp4, self_ar_depth/ (renderer: depth_{segment_start+1}..depth_{segment_start+segment_size}), next_ar_depth/.

    Args:
        current_output_dir: Path to dir containing generated_video.mp4, self_ar_depth/, next_ar_depth/
        pool_dir: Path to pool root
        step: Step index (0, 1, 2, ...); segment_start = step * stride
        segment_size: Frames per step (25)
        stride: Pose advance per step; defaults to segment_size - 1 (overlap by 1)
        rgb_format: 'png' (default) or 'jpg' (quality 92). Frame 0 and
            chunk-boundary frames (pose % stride == 0) stay png: they are re-consumed
            as conditioning (inherited frame / anchor), so they stay lossless
            back into the loop.
        step0_proj_rgb: step 0 only, the (segment_size - 1, H, W, 3) uint8 projected frames
            from step0(); written to proj_rgb_1/2 for the diagnostic grid.
        proj_rgb: step k >= 1 only, project()'s (proj_rgb_1, proj_rgb_2), each
            (segment_size, H, W, 3) uint8; written to proj_rgb_1/2 by pose index.
    """
    stride = resolve_stride(segment_size, stride)
    current_output_dir = Path(current_output_dir)
    pool_dir = Path(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    for name in ('rgb', 'depth', 'next_ar_depth'):
        (pool_dir / name).mkdir(exist_ok=True)

    segment_start, _ = segment_pose_range(step, segment_size, stride)
    if step == 0:
        num_write = segment_size
        video_range = (0, segment_size)  # video frame indices to write
        pool_pose_start = 0
        # Renderer wrote depth_0001..depth_{segment_size:04d} (start_idx=0)
        depth_file_offset = 1  # first renderer file index
    else:
        num_write = stride
        video_range = (1, segment_size)  # skip first (inherited), write the rest
        pool_pose_start = segment_start + 1
        # Renderer wrote depth_{segment_start+1}..depth_{segment_start+segment_size}; we skip the inherited first
        depth_file_offset = segment_start + 2

    # 1. Extract video frames to pool/rgb (pose indices)
    video_path = current_output_dir / 'generated_video.mp4'
    if not video_path.exists():
        raise FileNotFoundError(f"Expected {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total != segment_size:
        cap.release()
        raise ValueError(
            f"Video has {total} frames but segment_size={segment_size}; "
            f"refusing to subsample/pad ({video_path})"
        )
    # Read in order (seeking decodes from the previous keyframe every time); frames before
    # video_range[0] are read and dropped.
    for _ in range(video_range[0]):
        cap.read()
    for i in range(video_range[0], video_range[1]):
        frame_idx = i
        ret, bgr = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
        pose_idx = pool_pose_start + (i - video_range[0])
        # Chunk boundaries (frame 0 included) stay lossless: the next chunk starts from them
        fmt = 'png' if pose_idx % stride == 0 else rgb_format
        save_image(pool_dir / 'rgb', f'{pose_idx:05d}', bgr[:, :, ::-1].copy(), fmt=fmt)
    cap.release()

    # 2. Copy self_ar_depth and normal -> pool by pose index.
    # The renderer zeroes depth wherever alpha is 0, so coverage is depth > 0.
    self_ar_depth_dir = current_output_dir / 'self_ar_depth' / 'depth'
    self_ar_normal_dir = current_output_dir / 'self_ar_depth' / 'normal'
    if self_ar_depth_dir.exists():
        for j in range(num_write):
            r = depth_file_offset + j
            src_d = self_ar_depth_dir / f'depth_{r:04d}.npy'
            if not src_d.exists():
                continue
            pose_idx = pool_pose_start + j
            shutil.copy2(src_d, pool_dir / 'depth' / f'{pose_idx:05d}.npy')
            if self_ar_normal_dir.exists():   # --run.save_debug
                src_n = self_ar_normal_dir / f'normal_{r:04d}.png'
                if src_n.exists():
                    (pool_dir / 'normal').mkdir(exist_ok=True)
                    shutil.copy2(src_n, pool_dir / 'normal' / f'{pose_idx:05d}.png')
    else:
        logger.warning(f"  No self_ar_depth found, skipping depth/normal copy")

    # 3. Copy next_ar_depth (absent after the last chunk of the trajectory)
    # Files without a matching source remain unchanged.
    next_ar_dir = current_output_dir / 'next_ar_depth' / 'depth'
    if next_ar_dir.exists():
        # As many frames as the next chunk has: stride + 1, or fewer for a last partial chunk.
        copied = 0
        for t in range(stride + 1):
            frame_idx = segment_start + stride + t
            src = next_ar_dir / f'depth_{frame_idx+1:04d}.npy'
            if not src.exists():
                break
            shutil.copy2(src, pool_dir / 'next_ar_depth' / f'depth_{t:04d}.npy')
            copied += 1
        if copied == 0:
            logger.warning(f"  next_ar_depth: {next_ar_dir} holds no frames for the next chunk.")
    else:
        logger.info(f"  No next_ar_depth to copy (last chunk of the trajectory)")

    # 4. Persist proj_rgb_1/2 to pool (for diagnostic visualization)
    for k, proj_name in enumerate(('proj_rgb_1', 'proj_rgb_2')):
        pool_proj_dir = pool_dir / proj_name
        pool_proj_dir.mkdir(exist_ok=True)
        if step == 0:
            # Step 0: proj_rgb_1 == proj_rgb_2 (both are the projection of frame 0)
            # Frame 0 from pool/rgb, frames 1.. from step0_proj_rgb, which is at the
            # projection resolution and may need resizing to the video's
            rgb0_path = pool_rgb_path(pool_dir, 0)  # pose 0 is always png
            if rgb0_path is not None:
                shutil.copy2(rgb0_path, pool_proj_dir / f'00000{rgb0_path.suffix}')
            if step0_proj_rgb is not None and rgb0_path is not None:
                # Get target size from pool rgb frame 0
                ref_img = Image.open(rgb0_path)
                tgt_w, tgt_h = ref_img.size
                for t, frame in enumerate(step0_proj_rgb[:segment_size - 1], start=1):
                    img = Image.fromarray(frame)
                    if img.size != (tgt_w, tgt_h):
                        img = img.resize((tgt_w, tgt_h), Image.BILINEAR)
                    save_image(pool_proj_dir, f'{t:05d}', np.array(img), fmt=rgb_format)
        else:
            # Step k: frames 0..segment_size-1 of project()'s proj_rgb_{1,2}
            if proj_rgb is not None:
                for t, frame in enumerate(proj_rgb[k][:segment_size]):
                    pose_idx = segment_start + t
                    if rgb_format == 'png':
                        cv2.imwrite(str(pool_proj_dir / f'{pose_idx:05d}.png'),
                                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    else:
                        save_image(pool_proj_dir, f'{pose_idx:05d}', frame, fmt=rgb_format)

    print(f"Merged step {step} to pool: pose indices [{pool_pose_start}..{pool_pose_start+num_write}) ({num_write} frames)"
          + (", next_ar_depth updated." if next_ar_dir.exists() else "."))


def clear_chunk_staging(staging_dir):
    """Drop a chunk's staging renders once merge_video_output_to_pool has copied them."""
    for name in ('self_ar_depth', 'next_ar_depth'):
        shutil.rmtree(Path(staging_dir) / name, ignore_errors=True)


def encode_pool_rgb_to_video(pool_dir, output_path=None, fps=15):
    """
    Encode all pool/rgb frames (sorted by pose index) into a single MP4: the video result up to the current step.
    Uses imageio (RGB frames, no BGR/cv2 channel swap). Call after merge_video_output_to_pool.
    """
    pool_dir = Path(pool_dir)
    rgb_dir = pool_dir / 'rgb'
    if not rgb_dir.exists():
        raise FileNotFoundError(f"Expected {rgb_dir}")
    frame_paths = pool_rgb_files(pool_dir)
    if not frame_paths:
        raise ValueError(f"No frames in {rgb_dir}")
    out = output_path if output_path is not None else pool_dir / 'generated_full.mp4'
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # (H, W, 3) uint8 RGB -- imageio writes RGB directly, no BGR swap. Frame by frame, so
    # memory does not grow with the length of the run.
    with imageio.v2.get_writer(str(out), fps=fps, codec="libx264", quality=8) as writer:
        for p in frame_paths:
            writer.append_data(np.array(Image.open(p).convert('RGB')))
    print(f"Encoded pool RGB to video: {len(frame_paths)} frames -> {out}")
    return out


def _depth_to_colormap(depth, vmin=0.0, vmax=10.0):
    """Convert depth array to RGB colormap image (turbo)."""
    import matplotlib.cm as cm
    d = np.clip(depth, vmin, vmax)
    d = (d - vmin) / (vmax - vmin + 1e-8)
    colored = (cm.turbo(d)[:, :, :3] * 255).astype(np.uint8)
    return colored


def encode_pool_diagnostic_grid(pool_dir, stride, fps=15, depth_vmax=10.0):
    """
    Encode a single grid video combining all diagnostics:
      Top row:    generated_rgb | proj_rgb_1 | proj_rgb_2
      Bottom row: depth_inferred | next_ar_depth_input | (empty)
    Each cell is the original resolution. Output: diagnostic_grid.mp4
    """
    pool_dir = Path(pool_dir)
    rgb_dir = pool_dir / 'rgb'
    if not rgb_dir.exists():
        return
    pose_files = pool_rgb_files(pool_dir)
    num_poses = len(pose_files)
    if num_poses == 0:
        return

    sample = np.array(Image.open(pose_files[0]).convert('RGB'))
    h, w = sample.shape[:2]

    blank = np.zeros((h, w, 3), dtype=np.uint8)

    def rgb_frame(src_dir, i):
        path = find_image(src_dir, f'{i:05d}')
        if path is None:
            return blank
        img = Image.open(path).convert('RGB')
        if img.size != (w, h):
            img = img.resize((w, h), Image.BILINEAR)
        return np.array(img)

    def depth_frame(src_dir, i):
        path = find_array(src_dir, f'{i:05d}')
        return blank if path is None else _depth_to_colormap(load_array(path), vmax=depth_vmax)

    def add_label(img, text):
        out = img.copy()
        cv2.putText(out, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(out, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        return out

    # Built and written one frame at a time, so memory does not grow with the run.
    out = pool_dir / 'diagnostic_grid.mp4'
    with imageio.v2.get_writer(str(out), fps=fps, codec="libx264", quality=8) as writer:
        for i in range(num_poses):
            nar_depth = load_input_depth(pool_dir, i)
            top = np.concatenate([
                add_label(rgb_frame(rgb_dir, i), "generated"),
                add_label(rgb_frame(pool_dir / 'proj_rgb_1', i), "proj_rgb_1"),
                add_label(rgb_frame(pool_dir / 'proj_rgb_2', i), "proj_rgb_2"),
            ], axis=1)
            bot = np.concatenate([
                add_label(depth_frame(pool_dir / 'depth', i), "depth_inferred"),
                add_label(blank if nar_depth is None else _depth_to_colormap(nar_depth, vmax=depth_vmax),
                          "next_ar_depth"),
                blank,
            ], axis=1)
            writer.append_data(np.concatenate([top, bot], axis=0))
    print(f"Encoded diagnostic_grid: {num_poses} frames ({3*w}x{2*h}) -> {out}")


IN_PROGRESS = '.chunk_in_progress'


def mark_chunk_in_progress(pool_dir, step):
    """Mark a chunk before writing pool state; clear only after all writes finish.

    merged_plane_params.pt and next_ar_depth/ are overwritten in place, so file
    counts cannot identify a partial write. A remaining marker blocks resume.
    The marker is not fsynced and may be lost on power failure.
    """
    pool_dir = Path(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    (pool_dir / IN_PROGRESS).write_text(f"{step}\n")


def clear_chunk_in_progress(pool_dir):
    """Clear the marker only after all chunk writes and finalization succeed."""
    (Path(pool_dir) / IN_PROGRESS).unlink(missing_ok=True)


def check_fresh_state(pool_dir):
    """Reject existing pool state before starting a new run.

    Empty directories and logs are allowed. Resuming uses --start_step and
    check_resume_state.
    """
    pool_dir = Path(pool_dir)
    found = []
    if (pool_dir / 'merged_plane_params.pt').exists():
        found.append('merged_plane_params.pt')
    if (pool_dir / STEP0_STATE).exists():
        found.append(STEP0_STATE)
    if pool_rgb_files(pool_dir):
        found.append(f"{len(pool_rgb_files(pool_dir))} frames in rgb/")
    n_next = len(list((pool_dir / 'next_ar_depth').glob('depth_*.npy')))
    if n_next:
        found.append(f"{n_next} frames in next_ar_depth/")
    if (pool_dir / IN_PROGRESS).exists():
        found.append(IN_PROGRESS)
    if found:
        raise ValueError(
            f"{pool_dir} already holds a run: {', '.join(found)}. "
            f"Choose an empty pool directory, or pass --start_step to resume the existing run."
        )


def check_resume_state(pool_dir, start_step, segment_size, stride, extrinsic_c2w=None,
                       next_frames=None):
    """Reject an unfinished chunk, then validate the pool for resuming at start_step.

    Check the marker first, followed by plane step, RGB/depth counts, and optional
    trajectory consistency.
    """
    import torch

    pool_dir = Path(pool_dir)
    expect_frames = (start_step - 1) * stride + segment_size

    flag = pool_dir / IN_PROGRESS
    if flag.exists():
        was = flag.read_text().strip() or '?'
        raise ValueError(
            f"{flag} says step {was} was still being written when the previous run "
            f"stopped. The plane pool and next_ar_depth are overwritten in place, so "
            f"what is on disk cannot be told apart from a finished chunk by counting. "
            f"Start a fresh run."
        )

    merged = pool_dir / 'merged_plane_params.pt'
    if not merged.exists():
        raise FileNotFoundError(
            f"--start_step {start_step} resumes an existing pool, but {merged} is not "
            f"there. Only a run that completed a chunk can be resumed."
        )

    pool = torch.load(merged, map_location='cpu', weights_only=False)
    if 'plane_step_created' not in pool:
        raise ValueError(
            f"{merged} has no 'plane_step_created', so which chunks it holds cannot be "
            f"established (keys: {sorted(pool.keys())}). Start a fresh run instead of "
            f"resuming onto a pool that cannot be checked."
        )
    # The pool must not already hold planes from the chunk about to run. Written as "<"
    # rather than "== start_step - 1" on purpose: a chunk whose planes were all filtered
    # out leaves the maximum lower, and that is a legal state, not a mixed one.
    pool_step = int(pool['plane_step_created'].max())
    if pool_step >= start_step:
        raise ValueError(
            f"{merged} already holds planes from step {pool_step}, and --start_step "
            f"{start_step} would add step {start_step}'s planes on top of them. The "
            f"previous run stopped after writing the pool and before committing that "
            f"chunk's frames. Resume from step {pool_step + 1}, or start fresh."
        )

    n_frames = len(pool_rgb_files(pool_dir))
    if n_frames != expect_frames:
        raise ValueError(
            f"{pool_dir}/rgb holds {n_frames} frames, and chunks 0..{start_step - 1} "
            f"come to {expect_frames} ({start_step - 1} x {stride} + {segment_size}). "
            f"The previous run did not finish committing a chunk. Start fresh."
        )

    n_next = len(list((pool_dir / 'next_ar_depth').glob('depth_*.npy'))) \
        if (pool_dir / 'next_ar_depth').exists() else 0
    # Chunk start_step takes its first next_frames of them (fewer for a last partial chunk).
    next_frames = segment_size if next_frames is None else next_frames
    if n_next < next_frames:
        raise ValueError(
            f"{pool_dir}/next_ar_depth holds {n_next} frames, expected {next_frames} "
            f"-- step {start_step} takes its depth from there, and the previous chunk "
            f"renders it as its last act. Start fresh."
        )

    # The trajectory step 0 stored is the one every later chunk projects with; a new
    # --extrinsic is recorded in run_meta.json but never reaches the projection.
    if extrinsic_c2w is not None:
        latent0 = pool_dir / STEP0_STATE
        if not latent0.exists():
            raise FileNotFoundError(f"{latent0} is not there, so the trajectory this pool "
                                    f"was built with cannot be compared.")
        stored = np.asarray(load_step0_state(pool_dir)['extrinsics_c2w'])
        given = np.asarray(extrinsic_c2w)
        if stored.shape != given.shape or not np.array_equal(stored, given):
            raise ValueError(
                f"--extrinsic does not match the trajectory in {latent0} "
                f"(stored {stored.shape}, given {given.shape}). Every chunk after the "
                f"first projects with the stored one, so a different --extrinsic would "
                f"be recorded but not used. Pass the original, or start fresh."
            )

    logger.info(f"[resume] pool holds chunks 0..{pool_step}, {n_frames} frames, "
                f"{n_next} next_ar_depth -- resuming at step {start_step}")
