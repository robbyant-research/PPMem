"""Load frame 0's depth and matching intrinsics from --init_geometry.

The .npz stores a (H, W) depth map in metres and fx, fy, cx, cy at that resolution.
Loading disables pickle. Without --init_geometry, MoGe estimates both from the image.
"""

from math import gcd

import cv2
import numpy as np

from ppmem._logging import logger

REQUIRED_KEYS = ('depth', 'fx', 'fy', 'cx', 'cy')
SUPPORTED_UNITS = ('m',)


def aspect_crop_box(w_src, h_src, h_tgt, w_tgt):
    """Return (crop_w, crop_h) for the largest crop with the exact target aspect ratio.

    Shared by the RGB and depth centre-crop paths.
    """
    g = gcd(w_tgt, h_tgt)
    aw, ah = w_tgt // g, h_tgt // g
    k = min(w_src // aw, h_src // ah)
    if k < 1:
        raise ValueError(f"{w_src}x{h_src} is smaller than one {aw}:{ah} unit")
    return aw * k, ah * k


def intrinsics_to_K(intrinsics):
    """Return frame 0's intrinsics as a 3x3 float32 matrix.

    Accept a dict of fx/fy/cx/cy or a per-frame sequence of such dicts.
    For a sequence, only the first entry is used.
    """
    entry = intrinsics if isinstance(intrinsics, dict) else None
    if entry is None:
        try:
            entry = intrinsics[0]
        except (TypeError, IndexError, KeyError) as e:
            raise ValueError(
                f"intrinsics is {type(intrinsics).__name__} with no frame 0 ({e}). "
                f"Expected one dict of fx/fy/cx/cy per frame."
            )
    if not isinstance(entry, dict):
        raise ValueError(
            f"intrinsics[0] is a {type(entry).__name__}, expected a dict of fx/fy/cx/cy."
        )
    absent = [k for k in ('fx', 'fy', 'cx', 'cy') if k not in entry]
    if absent:
        raise ValueError(f"intrinsics[0] is a dict missing {absent} (has {sorted(entry)}).")
    return np.array([[entry['fx'], 0.0, entry['cx']],
                     [0.0, entry['fy'], entry['cy']],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


class _Missing(Exception):
    """Carries which keys are absent out of the `with np.load(...)` block, so that a
    missing key and a file numpy could not read get different messages."""

    def __init__(self, absent, present):
        super().__init__(str(absent))
        self.absent, self.present = absent, present


def load_init_geometry(path, hw_target, hw_image=None):
    """Read an --init_geometry .npz and return (depth, K).

    depth is (H, W) float32 in metres; K is a 3x3 float32 matrix at the same resolution.
    """
    # Attach the file path to NumPy loading errors.
    try:
        with np.load(str(path)) as z:      # allow_pickle=False by default
            present = set(z.files)
            absent = [k for k in REQUIRED_KEYS if k not in present]
            if absent:
                raise _Missing(absent, sorted(present))
            raw = {k: z[k] for k in REQUIRED_KEYS}
            raw['units'] = z['units'] if 'units' in present else None
    except _Missing as m:
        raise ValueError(
            f"{path} is missing {m.absent}; --init_geometry needs "
            f"{list(REQUIRED_KEYS)} (found {m.present})."
        )
    except Exception as e:
        raise ValueError(
            f"{path} could not be read as an .npz ({type(e).__name__}: {e}). Write it "
            f"with np.savez(path, depth=..., fx=..., fy=..., cx=..., cy=...); an .npz "
            f"holding pickled objects is refused on purpose."
        )

    depth = np.asarray(raw['depth'], dtype=np.float32)
    fx, fy, cx, cy = (float(raw[k]) for k in ('fx', 'fy', 'cx', 'cy'))
    units = 'm' if raw['units'] is None else str(raw['units'])

    if units not in SUPPORTED_UNITS:
        raise ValueError(
            f"{path} declares units={units!r}; only {list(SUPPORTED_UNITS)} is supported. "
            f"Convert the depth before storing it -- trajectory translation is in the same "
            f"units, so a silent mismatch moves the camera by the wrong amount."
        )
    if depth.ndim != 2:
        raise ValueError(
            f"{path}: depth has shape {depth.shape}, expected (H, W). Only frame 0 is ever "
            f"used, so store that one frame -- a leading frame axis would be indexed as a "
            f"single row."
        )

    h, w = depth.shape

    # Focal-length positivity checks alone do not reject NaN.
    bad = [k for k, v in (('fx', fx), ('fy', fy), ('cx', cx), ('cy', cy))
           if not np.isfinite(v)]
    if bad:
        raise ValueError(f"{path}: {', '.join(bad)} is not finite ({bad[0]}={dict(fx=fx, fy=fy, cx=cx, cy=cy)[bad[0]]}).")
    if fx <= 0 or fy <= 0:
        raise ValueError(f"{path}: fx={fx}, fy={fy} must both be positive.")

    # Replace non-finite depth before interpolation to prevent NaN propagation.
    n_bad = int((~np.isfinite(depth)).sum())
    if n_bad:
        if n_bad == depth.size:
            raise ValueError(f"{path}: every one of the {depth.size} depth values is NaN or Inf.")
        logger.warning(
            f"{path}: {n_bad} of {depth.size} depth values ({100 * n_bad / depth.size:.2f}%) "
            f"are NaN or Inf; treating them as invalid (0), the same as sky."
        )
        depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)

    # Resize depth and scale K to the target video resolution.
    h_t, w_t = int(hw_target[0]), int(hw_target[1])
    if abs((w / h) - (w_t / h_t)) > 1e-3:
        # Apply the image's centre crop only when the source dimensions match.
        if hw_image is not None and (h, w) == (int(hw_image[0]), int(hw_image[1])):
            cw, ch = aspect_crop_box(w, h, h_t, w_t)
            x0, y0 = (w - cw) // 2, (h - ch) // 2
            depth = depth[y0:y0 + ch, x0:x0 + cw]
            cx, cy = cx - x0, cy - y0
            h, w = ch, cw
            logger.info(
                f"init_geometry: cropped {hw_image[1]}x{hw_image[0]} -> {cw}x{ch} at "
                f"({x0},{y0}) for {w_t}:{h_t}; step 0 crops the image with the same box"
            )
        else:
            raise ValueError(
                f"{path}: depth is {w}x{h}, aspect {w / h:.4f}, but the video is "
                f"{w_t}x{h_t}, aspect {w_t / h_t:.4f}"
                + (f" and the image is {hw_image[1]}x{hw_image[0]}, a third size"
                   if hw_image is not None else "")
                + f". Which part of the depth the video covers is not knowable from "
                  f"here. Store the depth at the image's size, or crop it to "
                  f"{w_t}:{h_t} yourself."
            )
    if (h, w) != (h_t, w_t):
        sx, sy = w_t / w, h_t / h
        # INTER_LINEAR to match the depth resize step 0 already does on this path
        # (ppmem/rollout/step0.py, depth_frame0 and the projected frames).
        depth = cv2.resize(depth, (w_t, h_t), interpolation=cv2.INTER_LINEAR)
        fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
        logger.info(f"init_geometry: scaled {w}x{h} -> {w_t}x{h_t} (depth and K together)")

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    logger.info(
        f"init_geometry: depth {h_t}x{w_t} in [{depth.min():.3f}, {depth.max():.3f}] {units}, "
        f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"
    )
    return depth, K
