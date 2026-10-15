"""Video Generator for Incremental Video Generation"""

import shutil
import torch
import numpy as np
from pathlib import Path
from einops import rearrange

from diffusers.utils import export_to_video
from ppmem._logging import logger
from ppmem.config import MODEL


class VideoGenerator:
    """Generate each chunk from its latent data, then merge its planes into the pool."""

    def __init__(
        self,
        job_config,
        prompt,
        checkpoint_dir,
        device=None,
    ):
        """
        Initialize video generator

        Args:
            job_config: JobConfig object (from load_config)
            prompt: Text prompt of every chunk. It is encoded once while the model loads
                and the text encoder is then freed, so generate_video only accepts
                latents made with this prompt.
            checkpoint_dir: Local checkpoint directory (see ppmem.checkpoints).
            device: Device to run on. None lets the engine resolve it.
        """
        self._requested_device = device
        self.job_config = job_config
        self.prompt = prompt
        self.checkpoint_dir = checkpoint_dir

        from ppmem.engine.builder import InferenceEngine
        from ppmem.model.loaders import read_checkpoint

        self.read_checkpoint = read_checkpoint
        self.InferenceEngine = InferenceEngine

        self.pipeline, self.vae, self.engine = self._load_model()

    def _load_model(self):
        """Load model and create pipeline"""
        import time as _time
        engine = self.InferenceEngine(self.job_config, self.prompt, self.checkpoint_dir,
                                      device=self._requested_device)
        self.device = engine.device

        ckpt_path = self.checkpoint_dir

        t_load = _time.time()
        state_dict, read_as = self.read_checkpoint(ckpt_path)
        print(f"Loading checkpoint from {ckpt_path} ({read_as})")
        missing, unexpected = engine.model_parts[0].load_state_dict(state_dict, strict=False)
        # VAE keys are absent from the main checkpoint because vae/ is loaded separately.
        from_vae = [k for k in missing if k.startswith("vae.")]
        missing = [k for k in missing if not k.startswith("vae.")]
        logger.info(f"[PROFILE] checkpoint_load ({read_as}): {_time.time()-t_load:.1f}s; "
              f"missing={len(missing)}, unexpected={len(unexpected)}, "
              f"read from vae/={len(from_vae)}")
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint does not match the model that was built: "
                f"{len(missing)} missing, {len(unexpected)} unexpected "
                f"(both must be 0).\n"
                f"  first missing:    {missing[:5]}\n"
                f"  first unexpected: {unexpected[:5]}\n"
                f"The model is built for the released PPMem checkpoint; pass that checkpoint."
            )

        vae = engine.vae

        # The engine has already encoded the prompt and freed the text encoder.
        pipeline = engine.pipeline_cls(
            tokenizer=None,
            text_encoder=None,
            transformer=engine.model_parts[0],
            vae=vae,
            scheduler=engine.eval_scheduler,
            pipe_dtype=engine.model_dtype,
        )

        print(f"Model loaded successfully")

        return pipeline, vae, engine

    @torch.no_grad()
    def generate_video(
        self,
        metadata,
        output_dir,
        num_inference_steps=50,
        guidance_scale=5.0,
        seed=42,
        pool_dir=None,
    ):
        """
        Generate video from latent data
        
        Args:
            metadata: the chunk's latent data, as step0() / step1() return it
            output_dir: Output directory
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            seed: Random seed
            pool_dir: Pool root (required). The chunk's planes are merged into
                      pool/merged_plane_params.pt, and next_ar_depth is rendered from the merged pool.
        
        Returns:
            output_paths: Dict of generated file paths
        """
        if not pool_dir:
            raise ValueError("generate_video needs pool_dir: the pool holds the history and "
                             "very-first depths and accumulates the planes")
        self._validate_metadata(metadata)
        if metadata['text'] != self.prompt:
            raise ValueError(
                f"This chunk's latent data was made with a different prompt than the one "
                f"this VideoGenerator encoded: {metadata['text']!r}"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # The staging renders are named by pose, so an earlier chunk's would otherwise pile
        # up here -- and the pool merge must only ever see this chunk's.
        for name in ('self_ar_depth', 'next_ar_depth'):
            shutil.rmtree(output_dir / name, ignore_errors=True)
        save_debug = self.job_config.run.save_debug

        generator = torch.manual_seed(seed)

        latent_height = metadata['latent_height']
        latent_width = metadata['latent_width']
        latent_num_frames = metadata['latent_num_frames']

        func_input = {
            'prompt_embeds': self.engine.prompt_embeds,
            'negative_prompt_embeds': self.engine.negative_prompt_embeds,
            'height': metadata['video_height'],
            'width': metadata['video_width'],
            'num_frames': metadata['video_num_frames'],
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'generator': generator,
        }

        # The first frame the video starts from
        assert 'first_frame_pil_images' in metadata, "Missing first_frame_pil_images"
        func_input['image'] = metadata['first_frame_pil_images'].resize(
            (func_input['width'], func_input['height'])
        )

        # Add context frames (optional: skip if context_num_frames == 0)
        num_ctx = metadata.get('context_num_frames', 0)
        if num_ctx > 0 and 'latent_context_frames' in metadata:
            context_latents = rearrange(
                metadata['latent_context_frames'],
                "(f h w) c -> c f h w",
                f=num_ctx,
                h=latent_height,
                w=latent_width,
            ).to(self.device)
            func_input['latents_context'] = context_latents
            logger.debug(f"Loaded context frames: {num_ctx} frames")
        else:
            logger.info(f"No context frames (context_num_frames={num_ctx})")

        # Add very_first / history frames
        vf_data = metadata["latent_very_first_frame"]
        n_vf = vf_data.shape[0] // (latent_height * latent_width)
        func_input['latents_very_first'] = rearrange(
            vf_data, "(f h w) c -> c f h w", f=n_vf, h=latent_height, w=latent_width,
        ).to(self.device)
        logger.debug(f"Loaded very_first frames: {n_vf} frames")
        if "latent_history" in metadata:
            hist_data = metadata["latent_history"]
            n_hist = hist_data.shape[0] // (latent_height * latent_width)
            if n_hist > 0:
                func_input['latents_history'] = rearrange(
                    hist_data, "(f h w) c -> c f h w", f=n_hist, h=latent_height, w=latent_width,
                ).to(self.device)
                logger.debug(f"Loaded history frames: {n_hist} frames")

        # Add proj_rgb and proj_mask

        for suffix in ['1', '2']:
            for prefix in ['proj_rgb', 'proj_mask']:
                key = f'latent_{prefix}_{suffix}'
                assert key in metadata, f"Missing {key}"
                latent = rearrange(
                    metadata[key],
                    "(f h w) c -> c f h w",
                    f=latent_num_frames,
                    h=latent_height,
                    w=latent_width,
                ).to(self.device)
                func_input[f'latent_{prefix}_{suffix}'] = latent

        # Add depth

        depth_norm_method = MODEL.depth_norm_method
        depth_range = (MODEL.depth_norm_range_min, MODEL.depth_norm_range_max)

        assert 'next_ar_depth' in metadata, "Missing next_ar_depth"
        # The VAE encodes each depth frame on its own, one latent frame per frame.
        # Depth token fusion also takes the raw depth, one map per main frame.
        depth_latent = self._encode_depth_to_latent(
            self.engine,
            metadata['next_ar_depth'].to(self.device),
            norm_method=depth_norm_method,
            depth_range=depth_range,
        ).squeeze(0)
        # Depth latents for the history and very-first frames (hv_ = history + very first).
        hv_n_hist = int(metadata.get('history_num_frames', 0) or 0)
        hv_maps = []
        if hv_n_hist > 0:
            # The history poses step1 chose for this chunk
            hv_poses = metadata['history_poses']
            assert len(hv_poses) == hv_n_hist, (
                f"metadata has {len(hv_poses)} history_poses but "
                f"history_num_frames={hv_n_hist}")
            for hv_p in hv_poses:
                hv_maps.append(np.load(f"{pool_dir}/depth/{hv_p:05d}.npy").astype(np.float32))
        # The very-first frame is frame 0 of the whole sequence, the same frame as
        # latent_very_first_frame. At step 0 pool/depth does not exist yet
        # (it appears after the merge), so fall back to next_ar_depth[0].
        hv_vf_path = Path(pool_dir) / 'depth' / '00000.npy'
        if hv_vf_path.exists():
            hv_maps.append(np.load(hv_vf_path).astype(np.float32))
        else:
            hv_maps.append(metadata['next_ar_depth'][0].cpu().numpy().astype(np.float32))
        hv_depth = torch.from_numpy(np.stack(hv_maps))   # history first, then very first
        func_input['latents_hv_depth'] = self._encode_depth_to_latent(
            self.engine, hv_depth.to(self.device),
            norm_method=depth_norm_method, depth_range=depth_range,
        ).squeeze(0)
        func_input['latents_depth'] = depth_latent

        # Pass raw depth for depth token fusion (metric depth shortcut)
        func_input['raw_next_ar_depth'] = [metadata['next_ar_depth'].to(self.device)]

        # Add context depth (optional: skip if no context frames)
        if num_ctx > 0 and 'context_depth' in metadata and metadata['context_depth'].numel() > 0:
            context_depth_latent = self._encode_depth_to_latent(
                self.engine,
                metadata['context_depth'].to(self.device),
                norm_method=depth_norm_method,
                depth_range=depth_range,
            ).squeeze(0)
            func_input['latents_context_depth'] = context_depth_latent
        assert 'intrinsics' in metadata, "Missing intrinsics"
        assert 'extrinsics_c2w' in metadata, "Missing extrinsics_c2w"

        # Convert intrinsics to tensor
        if isinstance(metadata['intrinsics'], list):
            K_list = []
            for intr in metadata['intrinsics']:
                if isinstance(intr, dict):
                    K_list.append(torch.tensor([
                        [intr['fx'], 0.0, intr['cx']],
                        [0.0, intr['fy'], intr['cy']],
                        [0.0, 0.0, 1.0],
                    ], dtype=torch.float32))
                else:
                    K_list.append(torch.tensor(intr, dtype=torch.float32))
            intrinsics = torch.stack(K_list).to(self.device)
        else:
            intrinsics = torch.tensor(metadata['intrinsics'], dtype=torch.float32).to(self.device)

        extrinsics_c2w_full = torch.tensor(metadata['extrinsics_c2w'], dtype=torch.float32).to(self.device)
        step = metadata.get('step', 0)
        segment_size = metadata['video_num_frames']
        stride = metadata.get('segment_stride', segment_size - 1)  # overlap by 1
        segment_start = step * stride  # step0=[0..seg), step1=[stride..stride+seg), ...
        # Pipeline gets current segment only (segment_size poses)
        extrinsics_c2w = extrinsics_c2w_full[segment_start : segment_start + segment_size]
        # One per frame of this chunk; step 0 stored them for a full-length chunk.
        func_input['intrinsics'] = intrinsics[:segment_size]
        func_input['extrinsics_c2w'] = extrinsics_c2w

        print(f"Generating video...")
        print(f"  Step: {step} (segment pose indices [{segment_start}..{segment_start+segment_size}))")

        # The pipeline derives the latent frame count as (num_frames - 1) // 4 + 1, the Wan
        # VAE's temporal factor, but this VAE maps frames 1:1. Pass the frame count that
        # maps back to segment_size latent frames.
        func_input['num_frames'] = (func_input['num_frames'] - 1) * 4 + 1

        # Run pipeline
        pipeline_output = self.pipeline(**func_input)
        videos = pipeline_output.frames
        plane_params = getattr(pipeline_output, 'plane_params', None)

        # Save video
        video = videos[0]
        video_path = output_dir / 'generated_video.mp4'
        export_to_video(video, str(video_path), fps=15)
        print(f"Saved video to: {video_path}")

        output_paths = {'video': str(video_path)}

        # Save plane params and render depth
        if plane_params is not None:
            from ppmem.model.planes import (
                transform_planes_to_world,
                planes_to_mesh,
                save_planes_as_ply,
                save_depth_maps,
                save_normal_maps,
            )

            plane_save_path = output_dir / 'plane_params.pt'
            planes_to_save = dict(plane_params)

            # Transform to world space
            extrinsics = plane_params.get('extrinsics_c2w', None)
            merged_world_planes = []
            if extrinsics is not None:
                for pp in plane_params['plane_params_per_video']:
                    world_pp = transform_planes_to_world(pp, extrinsics)
                    merged_world_planes.append(world_pp)

                planes_to_save['plane_params_world'] = merged_world_planes

            # This chunk's raw planes, as PLY, per-camera maps and a .pt. The pool keeps
            # the merged planes; these are for inspection only.
            if save_debug and extrinsics is not None:
                # Save world-space PLY
                for vid_idx, pp in enumerate(plane_params['plane_params_per_video']):
                    verts_w, faces_w, colors_w, vnormals_w = planes_to_mesh(pp, extrinsics)
                    ply_path = output_dir / f'planes_world.ply'
                    save_planes_as_ply(str(ply_path), verts_w, faces_w, colors_w, vnormals_w)
                    logger.info(f"Saved world-space planes to: {ply_path}")
                    output_paths['planes_world_ply'] = str(ply_path)

                # Per-cam depth and normal maps (camera-space, from plane_params_per_video)
                for vid_idx, pp in enumerate(plane_params['plane_params_per_video']):
                    depth_dir = output_dir / f"cam_{vid_idx}_depth"
                    save_depth_maps(str(depth_dir), pp)
                    logger.info(f"Depth maps (cam {vid_idx}) → {depth_dir}")
                    normal_dir = output_dir / f"cam_{vid_idx}_normals"
                    save_normal_maps(str(normal_dir), pp)
                    logger.info(f"Normal maps (cam {vid_idx}) → {normal_dir}")

            if save_debug:
                torch.save(planes_to_save, plane_save_path)
                logger.info(f"Saved plane params to: {plane_save_path}")
                output_paths['plane_params'] = str(plane_save_path)

            # Render depth maps from plane params (using world-space planes)
            if merged_world_planes:
                self._render_with_pool_merge(
                    merged_world_planes, metadata, output_dir, Path(pool_dir), output_paths,
                )
            else:
                logger.warning("No world-space planes available; skipping depth rendering.")

        return output_paths

    def _validate_metadata(self, metadata):
        """Validate that all required fields exist"""
        required_fields = [
            'text',
            'video_height',
            'video_width',
            'video_num_frames',
            'latent_height',
            'latent_width',
            'latent_num_frames',
            'latent_proj_rgb_1',
            'latent_proj_rgb_2',
            'latent_proj_mask_1',
            'latent_proj_mask_2',
            'next_ar_depth',
            'intrinsics',
            'extrinsics_c2w',
        ]

        missing = [f for f in required_fields if f not in metadata]
        if missing:
            raise ValueError(f"Missing required fields in metadata: {missing}")

    def _encode_depth_to_latent(self, engine, depth, norm_method='linear', depth_range=(0.2, 10.0)):
        """Encode depth to latent space"""
        return engine.encode_depth_to_latent(depth, norm_method, depth_range)

    # ------------------------------------------------------------------
    # Render + merge into the pool
    # ------------------------------------------------------------------

    def _render_with_pool_merge(self, merged_world_planes, metadata, output_dir,
                                pool_path, output_paths):
        """
        Three phases:
          A. Render self_ar from new planes → depth/alpha/normal tensors.
          B. PlaneMerger: delete bad pool planes, filter new planes, concat.
          C. Render next_ar from merged pool.
        Saves updated pool to pool_path/merged_plane_params.pt and exports PLY.
        """
        from ppmem.pool.depth_renderer import DepthRenderer
        from ppmem.pool.merger import PlaneMerger
        from ppmem.model.planes import (
            planes_to_mesh, save_planes_as_ply, quat_to_rot,
        )

        step         = metadata.get('step', 0)
        segment_size = metadata['video_num_frames']
        stride       = metadata.get('segment_stride', segment_size - 1)
        segment_start = step * stride
        video_height = metadata['video_height']
        video_width  = metadata['video_width']

        extr_full  = torch.tensor(metadata['extrinsics_c2w'],
                                  dtype=torch.float32).to(self.device)
        intrinsics = self._parse_intrinsics(metadata, extr_full.shape[0])

        self_ar_extr = extr_full[segment_start: segment_start + segment_size]
        self_ar_intr = intrinsics[:segment_size]

        renderer = DepthRenderer(image_height=video_height, image_width=video_width,
                                 device=self.device)
        world_pp = merged_world_planes[0]

        # ---- A: render self_ar (new planes only) ----------------------------
        print(f"\n[PoolMerge] Rendering self_ar (new planes, step {step})...")
        self_ar_maps = renderer.render_depth(world_pp, self_ar_intr, self_ar_extr)
        save_debug = self.job_config.run.save_debug
        renderer.save_depth_maps(self_ar_maps, self_ar_extr,
                                 output_dir / 'self_ar_depth', start_idx=segment_start,
                                 save_normal=save_debug, save_alpha=save_debug,
                                 save_png=save_debug)

        # Process allmap → tensors for PlaneMerger
        proc = [renderer.process_allmap_to_depth(m, e)
                for m, e in zip(self_ar_maps, self_ar_extr)]
        depth_T  = torch.stack([r['depth']        for r in proc])  # [T, H, W]

        # Load prev-pool coverage from pool/next_ar_depth/ (written by previous step).
        # depth_{t:04d}.npy covers exactly the current segment's T cameras.
        # At step 0 the files don't exist → alpha_prev stays None → dist filter keeps all far planes.
        alpha_prev = self._load_pool_next_ar_alpha(pool_path, segment_size,
                                                   depth_T.shape[-2], depth_T.shape[-1])

        # ---- B: merge -------------------------------------------------------
        print(f"\n[PoolMerge] Running PlaneMerger...")
        prev_pool = None
        merged_path = pool_path / 'merged_plane_params.pt'
        if merged_path.exists():
            prev_pool = torch.load(merged_path, weights_only=False)
            print(f"  Loaded prev pool: {prev_pool.get('planes_per_frame', 0)} planes")

        merger = PlaneMerger()
        updated_pool, debug_info = merger.merge(
            new_planes     = world_pp,
            pool_planes    = prev_pool,
            cam_extrinsics = self_ar_extr,
            intrinsics     = self_ar_intr,
            depth_T        = depth_T,
            alpha_prev     = alpha_prev,
            step           = step,
            debug          = save_debug,
        )
        if save_debug:
            merger.save_visualizations(debug_info, depth_T, self_ar_extr, self_ar_intr,
                                       output_dir, step=step)

        # Prune against the next step's poses while the pool is still in memory —
        # what gets saved below is what step+1 will load. The next chunk is full-length
        # (stride + 1 frames) or the trajectory's shorter tail; it needs a new pose.
        need_next = min(segment_start + 2 * stride + 1, extr_full.shape[0])
        has_next = need_next - (segment_start + stride) >= 2
        if has_next:
            next_extr = extr_full[segment_start + stride: need_next]
            updated_pool = merger.forward_prune(
                updated_pool, next_extr,
                next_intrinsics=self_ar_intr[:next_extr.shape[0]],
                image_height=depth_T.shape[-2], image_width=depth_T.shape[-1],
            )

        torch.save(updated_pool, merged_path)
        _pc = updated_pool.get('plane_center_world')
        _nan = int(torch.isnan(_pc.float()).any(dim=-1).sum()) if _pc is not None else 0
        _tot = int(_pc.shape[0]) if _pc is not None else 0
        print(f"  Saved merged pool: {updated_pool.get('planes_per_frame', 0)} planes "
              f"→ {merged_path}")
        if _nan:
            logger.warning(
                f"[PoolMerge] {_nan}/{_tot} pool planes ({_nan * 100.0 / max(_tot, 1):.1f}%) "
                f"have NaN geometry. The rasterizer sanitizes them, but they are stored raw: "
                f"merged_planes_world.ply carries them and plane_obs_score is NaN for them."
            )

        # Export merged PLY for visualisation
        P_total = updated_pool.get('planes_per_frame', 0)
        if P_total > 0 and 'plane_center_world' in updated_pool:
            norm_w = updated_pool.get('plane_normal_world')
            if norm_w is None:
                norm_w = quat_to_rot(updated_pool['plane_rot_q'])[:, :, 2]
            pp_ply = {
                'plane_center_camera': updated_pool['plane_center_world'],
                'plane_normal_camera': norm_w,
                'plane_radii':         updated_pool['plane_radii'],
                'plane_rot_q':         updated_pool['plane_rot_q'],
                'num_frames':          1,
                'planes_per_frame':    P_total,
            }
            verts_m, faces_m, colors_m, norms_m = planes_to_mesh(pp_ply, extrinsics_c2w=None)
            ply_path = pool_path / 'merged_planes_world.ply'
            save_planes_as_ply(str(ply_path), verts_m, faces_m, colors_m, norms_m)
            logger.info(f"  Saved merged PLY → {ply_path}")

        # ---- C: render next_ar from merged pool -----------------------------
        if has_next and P_total > 0:
            print(f"\n[PoolMerge] Rendering next_ar (merged pool, poses "
                  f"[{segment_start+stride}..{need_next}))...")
            next_extr = extr_full[segment_start + stride: need_next]
            n_next    = next_extr.shape[0]
            next_maps = renderer.render_depth(updated_pool, self_ar_intr[:n_next], next_extr)
            # The pool merge reads only the depth of these.
            renderer.save_depth_maps(next_maps, next_extr,
                                     output_dir / 'next_ar_depth',
                                     start_idx=segment_start + stride,
                                     save_normal=save_debug, save_alpha=save_debug,
                                     save_png=save_debug)
        elif not has_next:
            # No next_ar depth is needed after the trajectory ends.
            logger.info(f"[PoolMerge] last chunk of the trajectory: no next_ar depth to render.")
        else:
            # An empty pool cannot provide next_ar depth; stop before old files are reused.
            raise RuntimeError(
                f"[PoolMerge] Cannot render next_ar depth for poses "
                f"[{segment_start + stride}, {need_next}): merged pool is empty. "
                f"Output: {pool_path / 'next_ar_depth'}."
            )

        output_paths['self_ar_depth_dir'] = str(output_dir / 'self_ar_depth')
        output_paths['next_ar_depth_dir'] = str(output_dir / 'next_ar_depth')

    def _load_pool_next_ar_alpha(self, pool_path, n_frames, H, W):
        """
        Load pool/next_ar_depth/depth_{t:04d}.npy for t=0..n_frames-1 and convert to
        a binary alpha tensor [n_frames, H, W] (1 where depth > 0.01, 0 elsewhere).
        Returns None if no accumulated pool exists yet (step 0 or fresh run).

        Only reads next_ar_depth when merged_plane_params.pt exists, i.e. when there is
        an accumulated pool from a previous step.
        """
        if not (pool_path / 'merged_plane_params.pt').exists():
            return None

        next_ar_dir = pool_path / 'next_ar_depth'
        frames = []
        any_found = False
        for t in range(n_frames):
            p = next_ar_dir / f'depth_{t:04d}.npy'
            if p.exists():
                d = np.load(str(p))
                frames.append(torch.from_numpy(d > 0.01).float())
                any_found = True
            else:
                frames.append(torch.zeros(H, W, dtype=torch.float32))
        if not any_found:
            return None
        return torch.stack(frames)  # [T, H, W]

    def _parse_intrinsics(self, metadata, n_frames):
        """Parse intrinsics from metadata dict → [n_frames, 3, 3] on self.device."""
        intr_raw = metadata['intrinsics']
        if isinstance(intr_raw, list) and len(intr_raw) > 0 and isinstance(intr_raw[0], dict):
            K_list = []
            for intr in intr_raw:
                K_list.append(torch.tensor([
                    [intr['fx'], 0.0, intr['cx']],
                    [0.0, intr['fy'], intr['cy']],
                    [0.0, 0.0, 1.0],
                ], dtype=torch.float32))
            return torch.stack(K_list).to(self.device)
        elif isinstance(intr_raw, list):
            return torch.stack(
                [torch.tensor(k, dtype=torch.float32) for k in intr_raw]
            ).to(self.device)
        else:
            k = torch.tensor(intr_raw, dtype=torch.float32).to(self.device)
            if k.dim() == 2:
                k = k.unsqueeze(0).expand(n_frames, -1, -1)
            return k
