#!/usr/bin/env python3
"""Summarize a finished run's plane pool: which chunk its planes come from, and what it looks like.

Reads a run directory (the `--out` of `scripts/run_demo.sh`) and writes, next to it:

    pool_report/
      pool_growth.png        planes in the final pool by the chunk that created them, and
                             the running total (planes later deleted are not counted)
      pool_topdown_color.png top-down view of the pool, coloured by the chunk that
                             added each plane
      pool_topdown_gray.png  the same view in grayscale, shaded by depth from the
                             first camera
      pool_report.json       the same numbers as data

Colour and grayscale are both produced because the two answer different questions:
colour shows *when* a region entered the pool, grayscale shows *where* it is.

    python scripts/pool_report.py <run_dir> [--out <dir>]
"""
import argparse
import json
import pathlib
import sys

import numpy as np


def load_pool(run_dir: pathlib.Path):
    import torch
    p = run_dir / "pool" / "merged_plane_params.pt"
    if not p.exists():
        raise SystemExit(f"🔴 no pool at {p}")
    d = torch.load(p, map_location="cpu", weights_only=False)
    centers = d["plane_center_world"].float().numpy()
    step = d["plane_step_created"].float().numpy().astype(int)
    return centers, step


def first_camera(run_dir: pathlib.Path):
    """Camera centre of pose 0, for the grayscale depth shading."""
    import torch
    state = run_dir / "pool" / "step0_state.pt"
    if state.exists():
        e = torch.load(state, map_location="cpu", weights_only=False)["extrinsics_c2w"][0]
        return np.asarray(e)[:3, 3].astype(float)
    return np.zeros(3)


def chunk_log(run_dir: pathlib.Path):
    """keep/new per chunk, straight out of run.log."""
    import re
    log = run_dir / "run.log"
    if not log.exists():
        return []
    pat = re.compile(r"\[PlaneMerger\] step=(\d+): (\d+) new planes .* keep (\d+)")
    out = []
    for line in log.read_text(errors="replace").replace("\r", "\n").split("\n"):
        m = pat.search(line)
        if m:
            out.append({"step": int(m.group(1)), "new": int(m.group(2)),
                        "keep": int(m.group(3))})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--px", type=int, default=900, help="output image size")
    a = ap.parse_args()

    run_dir = pathlib.Path(a.run_dir)
    out_dir = pathlib.Path(a.out) if a.out else run_dir / "pool_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    centers, step = load_pool(run_dir)
    total = centers.shape[0]

    # Exclude non-finite centers from plots and report their count.
    finite = np.isfinite(centers).all(axis=1)
    n_nan = int((~finite).sum())
    cen, stp = centers[finite], step[finite]

    steps = sorted(set(stp.tolist()))
    per_step = {int(s): int((stp == s).sum()) for s in steps}
    report = {
        "run": str(run_dir),
        "planes_total": total,
        "planes_nan": n_nan,
        "planes_usable": int(finite.sum()),
        "added_per_chunk": per_step,
        "merger_log": chunk_log(run_dir),
    }

    if finite.sum() == 0:
        (out_dir / "pool_report.json").write_text(json.dumps(report, indent=2))
        print(f"  🔴 every plane has NaN geometry ({total}); nothing to draw")
        print(f"  wrote {out_dir/'pool_report.json'}")
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── growth ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=130)
    ks = sorted(per_step)
    vals = [per_step[k] for k in ks]
    ax.bar([str(k) for k in ks], vals, color="#4C78A8")
    ax.set_xlabel("chunk that added the plane")
    ax.set_ylabel("planes kept in the final pool")
    ax2 = ax.twinx()
    ax2.plot([str(k) for k in ks], np.cumsum(vals), color="#E45756", marker="o")
    ax2.set_ylabel("running total", color="#E45756")
    ax.set_title(f"{run_dir.name}: {int(finite.sum())} planes"
                 + (f"  ({n_nan} dropped as NaN)" if n_nan else ""))
    fig.tight_layout()
    fig.savefig(out_dir / "pool_growth.png")
    plt.close(fig)

    # ── top-down: x-z plane, the one a forward/back trajectory moves in ───────
    # Cropping affects the scatter only; pool_report.json counts the full pool.
    x, z = cen[:, 0], cen[:, 2]
    x_lo, x_hi = np.percentile(x, [2, 98])
    z_lo, z_hi = np.percentile(z, [2, 98])
    rad = np.linalg.norm(cen[:, [0, 2]] - cen[0, [0, 2]], axis=1)
    r_cap = float(np.percentile(rad, 95))
    keep = ((x >= x_lo) & (x <= x_hi) & (z >= z_lo) & (z <= z_hi) & (rad <= r_cap))
    xs, zs, ss = x[keep], z[keep], stp[keep]
    # Re-fit the axes to what survived, so the cap actually tightens the view.
    if xs.size:
        x_lo, x_hi = float(xs.min()), float(xs.max())
        z_lo, z_hi = float(zs.min()), float(zs.max())
    n_hidden = int((~keep).sum())

    # Keep the scene's proportions, but let the figure follow them instead of
    # squeezing a long corridor into a square.
    span_x = max(x_hi - x_lo, 1e-6)
    span_z = max(z_hi - z_lo, 1e-6)
    ratio = float(np.clip(span_z / span_x, 0.4, 2.5))
    figsize = (6.4, 6.4 * ratio) if ratio >= 1 else (6.4 / ratio, 6.4)
    pt = float(np.clip(40000.0 / max(len(xs), 1), 0.3, 6.0))

    def scatter(color, cmap, cbar_label, path, discrete=False):
        fig, ax = plt.subplots(figsize=figsize, dpi=130)
        kw = {}
        if discrete:
            lev = sorted(set(np.asarray(color).tolist()))
            kw["norm"] = matplotlib.colors.BoundaryNorm(
                [lev[0] - 0.5] + [v + 0.5 for v in lev], len(lev))
            cmap = matplotlib.colors.ListedColormap(
                plt.get_cmap("viridis")(np.linspace(0.15, 0.95, len(lev))))
        sc = ax.scatter(xs, zs, c=color, s=pt, cmap=cmap, linewidths=0, **kw)
        ax.set_aspect("equal")
        ax.set_xlabel("x (world)")
        ax.set_ylabel("z (world, forward)")
        ax.set_title(f"{run_dir.name}: pool, top-down"
                     + (f"  ({n_hidden} planes outside view)" if n_hidden else ""))
        cb = fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.75)
        if discrete:
            cb.set_ticks(sorted(set(np.asarray(color).tolist())))
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    scatter(ss, "viridis", "added by chunk", out_dir / "pool_topdown_color.png",
            discrete=True)

    cam0 = first_camera(run_dir)
    dist = np.linalg.norm(cen[keep] - cam0[None, :], axis=1)
    scatter(dist, "gray_r", "distance from pose 0 (m)", out_dir / "pool_topdown_gray.png")

    (out_dir / "pool_report.json").write_text(json.dumps(report, indent=2))
    print(f"  {run_dir.name}: {total} planes, {n_nan} NaN, {int(finite.sum())} drawn")
    for k in ks:
        print(f"     chunk {k}: +{per_step[k]}")
    print(f"  wrote {out_dir}/ (growth + topdown color + topdown gray + json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
