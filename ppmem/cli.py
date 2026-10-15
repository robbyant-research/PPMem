"""Incremental video generation pipeline, in a single process.

Loads the models once, then generates chunks 0..max_steps.

    ppmem --config configs/inference.yaml \
          --checkpoint <checkpoint dir or HF repo ID> \
          --image input.png --extrinsic trajectory.npy \
          --output_root runs/demo --max_steps 5

--output_root <run_root> layout: the pool lives in <run_root>/pool, per-chunk mp4s in
<run_root>/chunks/, the settings used in run_meta.json.

Any extra --section.key arguments are passed through to load_config().
"""

import os
import sys
from pathlib import Path

# Let the CUDA allocator grow its segments instead of fragmenting into fixed blocks, so
# reserved memory stays close to what is in use. It is read when torch first allocates
# on the GPU; a value the user has set wins. run_demo.sh sets the same default.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ppmem._logging import logger


def _input_check(check, *args, **kwargs):
    """Run a check on the user's input. Its ValueError or FileNotFoundError already says
    what to fix, so print that alone instead of a traceback."""
    try:
        return check(*args, **kwargs)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(str(e)) from None


def _build_parser(argparse):
    """The CLI flags. Shared by main() and --help so the two cannot drift apart."""
    parser = argparse.ArgumentParser(
        description="Generate a camera-controlled video from one image, chunk by chunk, "
                    "while building a persistent plane memory of the scene."
    )
    parser.add_argument("--output_root", type=str, default=None,
                        help="Run root (required): the pool under pool/, per-chunk archives "
                             "under chunks/, run_meta.json.")
    parser.add_argument("--image", type=str, default=None, help="Input image (required for step 0)")
    parser.add_argument("--extrinsic", type=str, default=None, help="Extrinsic .npy (required for step 0)")
    parser.add_argument("--prompt", type=str, default="A video of indoor scene", help="Text prompt")
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="CFG scale; overrides run.guidance_scale from the config file.")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Last chunk index to run (0..max_steps). Default: as many as "
                             "the trajectory has poses for.")
    parser.add_argument("--start_step", type=int, default=0,
                        help="Chunk index to start from; 1 or more resumes an existing run.")
    parser.add_argument("--top_k", type=int, default=2,
                        help="Source views per target frame for step 1+ projection: 1 or 2.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the initial noise; chunk k uses seed + k.")
    parser.add_argument("--checkpoint", default=None,
                        help="PPMem local checkpoint directory or Hugging Face repo ID. "
                             "Default: Robbyant-Research/PPMem.")
    parser.add_argument("--moge-checkpoint", default=None,
                        help="MoGe local .pt file or Hugging Face repo ID. Defaults to "
                             "Ruicheng/moge-2-vitl-normal; unused with --init_geometry.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on. Default: cuda:0; use "
                             "CUDA_VISIBLE_DEVICES to pick another GPU.")
    parser.add_argument("--h_tgt", type=int, default=None,
                        help="Video height. Default: 576 (the training resolution).")
    parser.add_argument("--w_tgt", type=int, default=None,
                        help="Video width. Default: 1024 (the training resolution).")
    parser.add_argument("--num_context", type=int, default=5, help="Number of context frames (0=none)")
    parser.add_argument("--init_geometry", type=str, default=None,
                        help="Frame 0's depth and the intrinsics that describe it, as an .npz "
                             "with depth/fx/fy/cx/cy. When omitted, MoGe estimates both from "
                             "the input image.")
    parser.add_argument("--history_mode", type=str, default="per_chunk",
                        choices=["per_chunk", "uniform"],
                        help="History frame sampling for step 1+: per_chunk = 3 per completed "
                             "prior chunk excluding chunk 0, cap 20; "
                             "uniform = up to 5 evenly spaced frames across prior history.")
    parser.add_argument("--coverage_downscale", type=int, default=None,
                        help="Downsample factor for source-selection coverage only. Default: 1.")
    parser.add_argument("--coverage_max_targets", type=int, default=None,
                        help="Probe-target count for source-selection coverage (<=0 = all). "
                             "Default: all.")
    parser.add_argument("--rgb_format", type=str, default=None, choices=["png", "jpg"],
                        help="Pool RGB storage. Default: png. "
                             "Frame 0 + chunk boundaries are always png.")
    parser.add_argument("--segment_size", type=int, default=None,
                        help="Frames per AR chunk. Default: 25. Consecutive chunks share one "
                             "frame, so each chunk advances segment_size - 1 poses.")
    return parser


def main():
    import argparse

    # Flush every line even when stdout is a file or a pipe, so a log shows progress
    # as it happens (run_demo.sh also sets PYTHONUNBUFFERED).
    sys.stdout.reconfigure(line_buffering=True)

    # --help must be cheap and must also list the config sections. Do it before the
    # heavy imports below, and before argparse gets a chance to exit on its own.
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        from ppmem.config import load_config
        _build_parser(argparse).print_help()
        print()
        load_config(["--help"])          # raises SystemExit(0)

    import shutil
    import torch
    import numpy as np
    from ppmem.rollout.video import VideoGenerator
    from ppmem.pool.layout import (
        check_fresh_state,
        check_resume_state,
        clear_chunk_in_progress,
        clear_chunk_staging,
        mark_chunk_in_progress,
        merge_video_output_to_pool,
        ensure_pool_dirs,
        encode_pool_rgb_to_video,
        encode_pool_diagnostic_grid,
        save_input_depth,
    )
    import time, json
    from ppmem.rollout.moge import MoGeEstimator
    from ppmem.rollout.latents import LatentEncoder
    from ppmem.pool.segment_math import chunk_seed, chunk_length, last_step
    from ppmem.config import load_config
    from ppmem._logging import init_logger

    from ppmem.rollout.step0 import step0
    from ppmem.rollout.project import project, check_top_k
    from ppmem.rollout.step1 import step1

    init_logger()

    parser = _build_parser(argparse)

    parsed, unknown = parser.parse_known_args()

    if parsed.output_root is None:
        raise SystemExit("--output_root is required.")
    output_root = Path(parsed.output_root)
    pool_dir = output_root / "pool"
    output_root.mkdir(parents=True, exist_ok=True)
    chunks_dir = output_root / "chunks"
    ensure_pool_dirs(pool_dir)

    if parsed.start_step <= 0 and (not parsed.image or not parsed.extrinsic):
        raise SystemExit("For step 0, --image and --extrinsic are required.")

    job_config = load_config(unknown)

    num_inference_steps = job_config.run.num_inference_steps
    guidance_scale = parsed.guidance_scale if parsed.guidance_scale is not None \
        else job_config.run.guidance_scale

    # AR chunk grid.
    segment_size = parsed.segment_size if parsed.segment_size is not None else 25
    if segment_size < 2:
        raise SystemExit(f"--segment_size {segment_size} must be >= 2: consecutive chunks share a frame.")
    # Poses each chunk advances: the last frame of chunk k is the first of chunk k+1.
    stride = segment_size - 1
    print(f"Mode: segment_size={segment_size}, stride={stride} "
          f"(latent frames per chunk: {segment_size})")

    # project() checks this too, but it only runs at step 1+, which is after the
    # checkpoint is loaded and step 0 has generated a chunk.
    _input_check(check_top_k, parsed.top_k)

    # Default resolution for the released checkpoint.
    h_tgt = parsed.h_tgt if parsed.h_tgt is not None else 576
    w_tgt = parsed.w_tgt if parsed.w_tgt is not None else 1024
    print(f"Resolution: {h_tgt}x{w_tgt}")

    rgb_format = parsed.rgb_format if parsed.rgb_format is not None else "png"
    print(f"Output: {output_root} (pool RGB as {rgb_format})")

    coverage_downscale = parsed.coverage_downscale if parsed.coverage_downscale is not None else 1
    if parsed.coverage_max_targets is None:
        coverage_max_targets = None
    else:
        coverage_max_targets = parsed.coverage_max_targets if parsed.coverage_max_targets > 0 else None
    logger.info(f"Coverage selection: downscale={coverage_downscale}, "
          f"max_targets={coverage_max_targets or 'all'}")

    # Load frame-0 depth and intrinsics for step-0 condition projection.
    init_depth = None
    init_K = None
    if parsed.init_geometry:
        from ppmem.pool.init_geometry import load_init_geometry
        # The image's size lets the loader crop the depth when the two are the same
        # frame; reading the header does not decode the pixels.
        hw_image = None
        if parsed.image:
            from PIL import Image as _Image
            with _input_check(_Image.open, parsed.image) as _im:
                hw_image = (_im.size[1], _im.size[0])
        # Validate initial geometry before loading the model.
        init_depth, init_K = _input_check(load_init_geometry, parsed.init_geometry, (h_tgt, w_tgt), hw_image)

    # Load the trajectory to determine the default max_steps.
    extrinsic_c2w = None
    if parsed.extrinsic:
        extrinsic_c2w = _input_check(np.load, parsed.extrinsic)
        if extrinsic_c2w.ndim != 3 or extrinsic_c2w.shape[1:] != (4, 4):
            raise SystemExit(
                f"--extrinsic expects an (N, 4, 4) array of c2w poses, got "
                f"{extrinsic_c2w.shape}. scripts/make_trajectory.py generates a valid one."
            )

    # The trajectory length decides the chunks: each has segment_size frames except a
    # shorter last one, which takes the poses that are left.
    if extrinsic_c2w is not None:
        n_poses = extrinsic_c2w.shape[0]
    elif parsed.start_step > 0:
        from ppmem.pool.layout import load_step0_state
        n_poses = len(_input_check(load_step0_state, pool_dir)['extrinsics_c2w'])
    else:
        raise SystemExit("--extrinsic is required.")
    if n_poses < 2:
        raise SystemExit(f"--extrinsic has {n_poses} poses; a chunk needs at least 2.")
    most = last_step(n_poses, segment_size)
    max_steps = parsed.max_steps if parsed.max_steps is not None else most

    # Validate before loading the checkpoint or writing a chunk marker.
    # max_steps is the last chunk index, inclusive.
    if max_steps < 0:
        raise SystemExit(f"--max_steps {max_steps} must be >= 0 (it is the last chunk index, inclusive).")
    if max_steps > most:
        raise SystemExit(
            f"--max_steps {max_steps} goes past the trajectory: its {n_poses} poses end in "
            f"chunk {most}."
        )
    last_len = chunk_length(max_steps, n_poses, segment_size)
    print(f"Steps: 0..{max_steps} ({n_poses} poses"
          + (f"; chunk {max_steps} has {last_len} frames)" if last_len < segment_size else ")"))

    # Resuming carries the previous chunks' state forward, so that state has to be
    # exactly "chunks 0..start_step-1, done". Checked before the checkpoint loads.
    if parsed.start_step > 0:
        _input_check(check_resume_state, pool_dir, parsed.start_step, segment_size, stride,
                     extrinsic_c2w=extrinsic_c2w,
                     next_frames=chunk_length(parsed.start_step, n_poses, segment_size))
    else:
        _input_check(check_fresh_state, pool_dir)

    # Record generation parameters before loading the model.
    run_meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": parsed.prompt,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "seed": parsed.seed,
        "per_chunk_seed": "seed + step",
        "segment_size": segment_size,
        "stride": stride,
        "h_tgt": h_tgt,
        "w_tgt": w_tgt,
        "history_mode": parsed.history_mode,
        "coverage_downscale": coverage_downscale,
        "coverage_max_targets": coverage_max_targets,
        "top_k": parsed.top_k,
        "num_context": parsed.num_context,
        "image": parsed.image,
        "extrinsic": parsed.extrinsic,
        "init_geometry": parsed.init_geometry,
        "step0_depth": "supplied" if parsed.init_geometry else "moge",
        "start_step": parsed.start_step,
        "max_steps": max_steps,
        "pool_dir": str(pool_dir),
        "output_root": str(output_root),
        "rgb_format": rgb_format,
        "config_args": unknown,
    }
    # Save resume parameters separately to preserve the original run metadata.
    meta_path = (output_root / f"run_meta_resume_{parsed.start_step:03d}.json"
                 if parsed.start_step > 0 else output_root / "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"Run metadata -> {meta_path}")
    print(f"Prompt: {parsed.prompt}")
    print(f"Sampling: {num_inference_steps} steps, guidance scale {guidance_scale}")

    def _check_step0_intrinsics(metadata, expected_K=None):
        """Check frame bounds and agreement with the loader's K, when supplied.

        A valid principal point may be off-centre.
        """
        from ppmem.pool.init_geometry import intrinsics_to_K
        K = intrinsics_to_K(metadata["intrinsics"])
        h, w = int(metadata["video_height"]), int(metadata["video_width"])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        if not (0 <= cx <= w and 0 <= cy <= h):
            raise ValueError(
                f"Step 0 wrote cx={cx:.1f}, cy={cy:.1f}, outside a {w}x{h} frame."
            )
        if expected_K is not None and not np.allclose(K, expected_K, atol=1e-3):
            raise ValueError(
                f"Step 0's intrinsics differ from the ones --init_geometry supplied "
                f"(already scaled to {w}x{h} by the loader).\n"
                f"  loader: fx={expected_K[0, 0]:.3f} fy={expected_K[1, 1]:.3f} "
                f"cx={expected_K[0, 2]:.3f} cy={expected_K[1, 2]:.3f}\n"
                f"  step 0: fx={K[0, 0]:.3f} fy={K[1, 1]:.3f} cx={cx:.3f} cy={cy:.3f}"
            )
        return K

    def _finish_chunk(step):
        """Log peak VRAM and archive the chunk video."""
        if not job_config.run.save_debug:
            clear_chunk_staging(pool_dir / "_current")   # the pool merge has taken what it needs
        logger.info(f"[VRAM] chunk {step}: peak allocated "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        torch.cuda.reset_peak_memory_stats()
        chunk_dir = chunks_dir / f"chunk_{step:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_mp4 = pool_dir / "_current" / "generated_video.mp4"
        if chunk_mp4.exists():
            shutil.move(str(chunk_mp4), str(chunk_dir / "generated_video.mp4"))

    print("\n========== Loading model ==========")
    # Resolve and validate the checkpoint before constructing the model.
    from ppmem.model.loaders import check_checkpoint_dir
    from ppmem.checkpoints import checkpoint_source, resolve_checkpoint, moge_source
    moge_checkpoint = (_input_check(moge_source, parsed.moge_checkpoint)
                       if parsed.start_step <= 0 and not parsed.init_geometry else None)
    # Paths and repo IDs are checked here; a Hub download reports its own errors.
    _input_check(checkpoint_source, parsed.checkpoint)
    checkpoint_dir = resolve_checkpoint(parsed.checkpoint)
    checkpoint_found = _input_check(check_checkpoint_dir, checkpoint_dir)
    print(f"Checkpoint: {checkpoint_dir} ({checkpoint_found})")
    run_meta["checkpoint_directory"] = checkpoint_dir
    run_meta["moge_checkpoint"] = moge_checkpoint
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)

    generator = VideoGenerator(job_config=job_config, prompt=parsed.prompt,
                               checkpoint_dir=checkpoint_dir, device=parsed.device)
    # Use the engine-resolved device so rollout tensors share the model's device.
    device = generator.device

    moge = None
    encoder = None

    # step-0 depth anchor; --init_geometry supplies it instead.
    if parsed.start_step <= 0 and not parsed.init_geometry:
        moge = MoGeEstimator(model_type="moge-2-vitl-normal", device=device,
                             checkpoint=moge_checkpoint)

    t_pipeline_start = time.time()

    if parsed.start_step <= 0:
        mark_chunk_in_progress(pool_dir, 0)
        print("\n========== Step 0: latent ==========")
        encoder = LatentEncoder(
            device=device,
            # Match input tensors to the VAE's dtype.
            dtype=generator.vae.dtype,
            vae=generator.vae,   # reuse the already-loaded VAE, don't load a second one
        )
        t = time.time()
        step0_meta = step0(
            image_path=parsed.image,
            extrinsic_c2w=extrinsic_c2w,
            prompt=parsed.prompt,
            incremental_dir=str(pool_dir),
            h_tgt=h_tgt,
            w_tgt=w_tgt,
            device=device,
            moge=moge,
            encoder=encoder,
            num_context=parsed.num_context,
            segment_size=chunk_length(0, n_poses, segment_size),
            init_depth=init_depth,
            init_K=init_K,
        )
        logger.info(f"[PROFILE] step0_latent: {time.time()-t:.1f}s")
        _check_step0_intrinsics(step0_meta, expected_K=init_K)
        step0_proj_rgb = step0_meta.pop('proj_rgb_frames')   # only the pool merge uses it
        # MoGe only estimates frame 0; free its GPU memory before generation starts.
        moge = None
        torch.cuda.empty_cache()

        print("\n========== Step 0: video + merge ==========")
        save_input_depth(pool_dir, 0, stride, step0_meta['next_ar_depth'])
        output_staging = pool_dir / "_current"
        t = time.time()
        generator.generate_video(
            metadata=step0_meta,
            output_dir=str(output_staging),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=chunk_seed(parsed.seed, 0),
            pool_dir=pool_dir,
        )
        logger.info(f"[PROFILE] step0_video_gen: {time.time()-t:.1f}s")
        t = time.time()
        merge_video_output_to_pool(output_staging, pool_dir, step=0,
                                   segment_size=chunk_length(0, n_poses, segment_size),
                                   stride=stride,
                                   rgb_format=rgb_format,
                                   step0_proj_rgb=step0_proj_rgb)
        del step0_proj_rgb
        logger.info(f"[PROFILE] step0_merge: {time.time()-t:.1f}s")
        _finish_chunk(0)
        clear_chunk_in_progress(pool_dir)
        print("Step 0 done.")

    if encoder is None and max_steps >= 1:
        encoder = LatentEncoder(
            device=device,
            dtype=generator.vae.dtype,
            vae=generator.vae,   # reuse the already-loaded VAE, don't load a second one
        )

    for step in range(1, max_steps + 1):
        if step < parsed.start_step:
            continue
        try:
            t_step = time.time()
            chunk_len = chunk_length(step, n_poses, segment_size)   # shorter for a last partial chunk
            mark_chunk_in_progress(pool_dir, step)
            print(f"\n========== Step {step}: proj ==========")
            t = time.time()
            proj = project(
                    incremental_dir=str(pool_dir),
                    step=step,
                    top_k=parsed.top_k,
                    device=device,
                    segment_size=chunk_len,
                    stride=stride,
                    coverage_downscale=coverage_downscale,
                    coverage_max_targets=coverage_max_targets,
                    num_context=parsed.num_context,
                )
            logger.info(f"[PROFILE] step{step}_proj: {time.time()-t:.1f}s")

            print(f"\n========== Step {step}: latent ==========")
            t = time.time()
            step_meta = step1(
                incremental_dir=str(pool_dir),
                proj=proj,
                step=step,
                prompt=parsed.prompt,
                encoder=encoder,
                num_context=parsed.num_context,
                segment_size=chunk_len,
                stride=stride,
                history_mode=parsed.history_mode,
            )
            proj_rgb = (proj['proj_rgb_1'], proj['proj_rgb_2'])   # only the pool merge uses them now
            del proj
            logger.info(f"[PROFILE] step{step}_latent: {time.time()-t:.1f}s")

            print(f"\n========== Step {step}: video + merge ==========")
            save_input_depth(pool_dir, step, stride, step_meta['next_ar_depth'])
            output_staging = pool_dir / "_current"
            t = time.time()
            generator.generate_video(
                metadata=step_meta,
                output_dir=str(output_staging),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                # Chunk k uses seed + k: deterministic, and the chunks do not all start
                # from the same initial noise.
                seed=chunk_seed(parsed.seed, step),
                pool_dir=pool_dir,
            )
            logger.info(f"[PROFILE] step{step}_video_gen: {time.time()-t:.1f}s")
            t = time.time()
            merge_video_output_to_pool(output_staging, pool_dir, step=step,
                                       segment_size=chunk_len, stride=stride,
                                       rgb_format=rgb_format, proj_rgb=proj_rgb)
            del proj_rgb
            logger.info(f"[PROFILE] step{step}_merge: {time.time()-t:.1f}s")
            _finish_chunk(step)
            clear_chunk_in_progress(pool_dir)
            logger.info(f"[PROFILE] step{step}_total: {time.time()-t_step:.1f}s")
            print(f"Step {step} done.")
        except Exception as e:
            logger.error(f"Step {step} FAILED ({type(e).__name__}): {e}")
            raise

    t_total = time.time() - t_pipeline_start
    logger.info(f"[PROFILE] pipeline_total: {t_total:.1f}s")

    # Encode the videos
    t = time.time()
    encode_pool_rgb_to_video(pool_dir)
    encode_pool_diagnostic_grid(pool_dir, stride)
    logger.info(f"[PROFILE] encode_all_videos: {time.time()-t:.1f}s")

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
