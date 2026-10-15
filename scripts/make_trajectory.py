#!/usr/bin/env python3
"""Synthesize a camera trajectory for the single-image demo.

Writes an (N, 4, 4) float32 array of c2w poses for `scripts/run_demo.sh --extrinsic`.

Camera convention (see ppmem/pool/projector.py):
  xyz_cam = K^-1 @ [u,v,1] * depth ; xyz_world = c2w @ xyz_cam
OpenCV axes: +X right, +Y down, +Z forward along the optical axis.
R[:,2] of c2w is forward in world space; turns rotate about Y.

World units match frame 0's depth map. With MoGe-2, --dist is in estimated metres.

Three shapes:
  dolly     go forward, then reverse back to the start without turning
  pendulum  swing back and forth, optionally several times
  spin      rotate in place -- no translation

dolly and pendulum translate forward/backward only. spin does not translate.
The helper checks motion against fixed per-frame guidelines.

Any frame count from 2 up works: chunks have 25 frames and share one with the previous
chunk, and the last chunk takes whatever frames are left.
See the camera trajectory section of docs/inference.md for guidance on distance and frame count.
"""
import argparse
import numpy as np


def _pose(R, t):
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _Ry(theta):
    """Rotate about the world Y axis. Y points down in OpenCV, so this is turning in place."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=np.float64)


def traj_dolly(dist, n):
    """Go out dist and reverse back to 0, orientation fixed throughout."""
    half = (n - 1) / 2.0
    zs = [dist * (i / half) if i <= half else dist * (2.0 - i / half) for i in range(n)]
    return np.stack([_pose(np.eye(3), np.array([0.0, 0.0, z])) for z in zs])


def traj_pendulum(dist, n, cycles=2):
    """Swing back and forth `cycles` times, orientation fixed. Cosine easing, so it
    slows at each end instead of turning sharply."""
    i = np.arange(n, dtype=np.float64)
    zs = dist * (1.0 - np.cos(2.0 * np.pi * cycles * i / (n - 1))) / 2.0
    return np.stack([_pose(np.eye(3), np.array([0.0, 0.0, z])) for z in zs])


def traj_spin(dist, n, turns=1.0):
    """Rotate in place by `turns` revolutions; `dist` is ignored.

    Sample n evenly spaced angles, excluding the endpoint. For integer turns,
    wrapping to the first pose follows the same angular increment.
    """
    return np.stack([_pose(_Ry(2.0 * np.pi * turns * i / n), np.zeros(3)) for i in range(n)])


SHAPES = {"dolly": traj_dolly, "pendulum": traj_pendulum, "spin": traj_spin}


def path_length(poses):
    return float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, choices=sorted(SHAPES))
    ap.add_argument("--dist", type=float, required=True, help="Forward distance, in MoGe metres")
    ap.add_argument("--frames", type=int, required=True, help="Total frames, at least 2")
    ap.add_argument("--cycles", type=int, default=2, help="pendulum only: how many round trips")
    ap.add_argument("--turns", type=float, default=1.0,
                    help="spin only: how many full circles (1.0 = one turn)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.frames < 2:
        raise SystemExit(f"--frames must be at least 2, got {a.frames}")

    kw = ({"cycles": a.cycles} if a.shape == "pendulum" else
          {"turns": a.turns} if a.shape == "spin" else {})
    poses = SHAPES[a.shape](a.dist, a.frames, **kw)

    # Check the pose count, identity start pose, and return to the starting position.
    assert poses.shape == (a.frames, 4, 4), poses.shape
    assert np.allclose(poses[0], np.eye(4), atol=1e-9), "frame 0 must be the identity pose"
    end_t = poses[-1, :3, 3]
    assert np.linalg.norm(end_t) < 1e-6, f"every shape should return to the start, got endpoint {end_t}"

    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    total = path_length(poses)
    max_steps = max(0, -(-(a.frames - 25) // 24))   # the last chunk index
    out = a.out if a.out.endswith(".npy") else a.out + ".npy"   # the name np.save writes
    np.save(out, poses.astype(np.float32))
    extra = ""
    if a.shape == "spin":
        extra = f" deg_per_frame={360.0 * a.turns / a.frames:.2f}"
    # Measure the rotation between consecutive poses.
    R = poses[:, :3, :3]
    rel = np.einsum("nij,nkj->nik", R[1:], R[:-1])
    rot = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)))
    rot_max = float(rot.max()) if len(rot) else 0.0
    print(f"TRAJ_OK shape={a.shape} dist={a.dist} frames={a.frames} max_steps={max_steps} "
          f"path_len={total:.2f} step_med={np.median(steps):.4f} step_max={steps.max():.4f} "
          f"rot_max={rot_max:.2f}deg{extra} -> {out}")

    # Fixed guidelines for translation (metres) and rotation (degrees) per frame.
    STEP_BUDGET, ROT_BUDGET = 0.20, 2.9
    warn = []
    if steps.max() > STEP_BUDGET:
        warn.append(f"max per-frame displacement {steps.max():.3f} m > {STEP_BUDGET}: "
                    f"consider more frames to reduce the displacement per frame.")
    if rot_max > ROT_BUDGET:
        need = int(np.ceil(a.frames * rot_max / ROT_BUDGET))
        msg = (f"max per-frame rotation {rot_max:.2f} deg > {ROT_BUDGET}: "
               f"large rotations may reduce projected coverage. "
               f"Estimated frames for this rotation guideline: >={need}")
        if a.shape == "spin":
            msg += ". Rotation may expose areas not yet covered by the plane pool."
        warn.append(msg)
    for w in warn:
        print(f"  \u26a0\ufe0f  {w}")
    if warn:
        print("  These are fixed motion guidelines; suitable motion depends on scene "
              "depth and camera field of view.")


if __name__ == "__main__":
    main()
