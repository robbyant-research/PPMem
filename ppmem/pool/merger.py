"""
PlaneMerger: maintains the plane pool across chunks.

For each new chunk, after self_ar depth has been rendered from its planes:
  1. Delete pool planes that float in front of the new predicted depth.
  2. Filter the new planes:
       2a. Distance filter (>5 m, only if the rendered region is not empty).
       2b. Depth-edge filter on the log-depth grid.
       2b'. Dedup within the chunk: where frames overlap, keep the better-observed plane.
       2c. Observation-quality filter against the pool, both ways: a new plane is dropped
           where the pool already covers the region better, and a pool plane is replaced
           where the new one is better.
  3. Add the surviving new planes (with metadata) to the pool.
Before the next chunk, forward_prune drops pool planes none of its cameras can see.

"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from pathlib import Path
from ppmem._logging import logger


# Quaternion / normal helper

def _get_plane_normals(planes: dict, device=None) -> torch.Tensor:
    """Return [P, 3] float normals. Uses stored field; falls back to rot_q z-axis."""
    if 'plane_normal_world' in planes:
        n = planes['plane_normal_world']
    else:
        # quat convention [w, x, y, z]; normal = R[:,2] (z-column)
        q = planes['plane_rot_q']
        qr, qi, qj, qk = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        nx = 2 * (qi * qk + qr * qj)
        ny = 2 * (qj * qk - qi * qr)
        nz = 1 - 2 * (qi ** 2 + qj ** 2)
        n = torch.stack([nx, ny, nz], dim=-1)
    if device is not None:
        n = n.to(device)
    return n.float()


# PlaneMerger

class PlaneMerger:
    """
    Manages the cumulative plane pool for incremental video generation.

    Pool dict schema (saved as merged_plane_params.pt):
        plane_center_world  [P, 3]
        plane_radii         [P, 4]
        plane_rot_q         [P, 4]
        plane_normal_world  [P, 3]
        plane_obs_score     [P]      dist / |cos_angle| at creation (lower = better)
        plane_dist_created  [P]      distance to creating camera
        plane_step_created  [P]      step index when this plane was added
        planes_per_frame    int = P
        num_frames          int = 1
    """

    def __init__(
        self,
        max_dist: float = 5.0,
        far_depth_ignore: float = 1e9,
        depth_edge_thresh: float = 1.0,
        occlude_margin: float = 0.1,
        vote_ratio: float = 0.2,
        obs_same_region_thresh: float = 0.5,
        alpha_empty_thresh: float = 0.3,
        intra_step_dedup: bool = True,
        bidirectional_quality: bool = True,
        forward_prune_ratio: float = 0.3,
    ):
        """
        Args:
            max_dist: Planes farther than this (m) are candidates for distance filter.
            far_depth_ignore: deletion rule applies only where new depth <= this (m).
                1e9 lets it apply everywhere.
            depth_edge_thresh: RelSobel magnitude threshold for depth discontinuity.
            occlude_margin: Pool plane flagged if depth_pool < depth_new * (1 - margin).
            vote_ratio: Fraction of cameras that must agree to delete a pool plane.
            obs_same_region_thresh: 3-D centre distance (m) for "same region" check.
            alpha_empty_thresh: Alpha below which a pixel is considered empty.
            intra_step_dedup: Within one step, drop planes that a later frame
                re-observes from closer.
            bidirectional_quality: Also delete the pool plane that a better new
                plane replaces, instead of only skipping worse new planes.
            forward_prune_ratio: Drop pool planes the next step will view from
                1/ratio times closer than they were created. 0 disables.
        """
        self.max_dist = max_dist
        self.far_depth_ignore = far_depth_ignore
        self.depth_edge_thresh = depth_edge_thresh
        self.occlude_margin = occlude_margin
        self.vote_ratio = vote_ratio
        self.obs_same_region_thresh = obs_same_region_thresh
        self.alpha_empty_thresh = alpha_empty_thresh
        self.intra_step_dedup = intra_step_dedup
        self.bidirectional_quality = bidirectional_quality
        self.forward_prune_ratio = forward_prune_ratio

    # Main entry

    def merge(
        self,
        new_planes: dict,
        pool_planes: Optional[dict],
        cam_extrinsics: torch.Tensor,   # [T, 4, 4] c2w for current segment
        intrinsics: torch.Tensor,       # [T, 3, 3]
        depth_T: torch.Tensor,           # [T, H, W]  self_ar rendered depth (current planes)
        alpha_prev: Optional[torch.Tensor] = None,  # [T, H, W] prev-pool coverage at current cams
                                                     # (from pool/next_ar_depth/); None → step 0
        step: int = 0,
        debug: bool = False,
    ):
        """
        Merge new segment's planes with the pool. Returns (updated pool, debug_info); debug_info
        feeds save_visualizations and is only built when `debug` is set (None otherwise).

        new_planes: output of transform_planes_to_world — contains frame_indices [T*HW]
                    that map each plane to its creating camera in cam_extrinsics.
        pool_planes: flat pool from merged_plane_params.pt (num_frames=1), or None.
        """
        device = cam_extrinsics.device
        T, H, W = depth_T.shape

        def _dev(d):
            return {k: v.to(device).float() if isinstance(v, torch.Tensor) else v
                    for k, v in d.items()}

        new = _dev(new_planes)

        # ---- Step 1: delete bad pool planes ---------------------------------
        pool_clean = None
        pool_delete_mask = None
        pool_centers_orig = None
        if pool_planes is not None and pool_planes.get('plane_center_world',
                                                        torch.empty(0)).shape[0] > 0:
            pool_centers_orig = pool_planes['plane_center_world'].cpu()
            pool_clean, pool_delete_mask = self._delete_occluded(
                _dev(pool_planes), cam_extrinsics, intrinsics, depth_T, H, W
            )

        # ---- Per-plane obs metrics for new planes ---------------------------
        obs_scores, min_dists = self._compute_obs_metrics(new, cam_extrinsics)

        # ---- Step 2a: distance filter ---------------------------------------
        # Use prev-pool coverage (alpha_prev) so we don't self-check against the
        # current step's own render.  None → step 0, treat everything as empty.
        _alpha_cov = alpha_prev.to(device) if alpha_prev is not None \
                     else torch.zeros(T, H, W, device=device)
        dist_keep = self._distance_mask(
            new, cam_extrinsics, intrinsics, _alpha_cov, min_dists, H, W
        )

        # ---- Step 2b: log-depth edge filter ---------------------------------
        edge_keep = self._depth_edge_mask(new, cam_extrinsics, device)

        # ---- Step 2b': dedup within the step --------------------------------
        # Runs on the edge survivors only. The contest is decided by obs_score, so
        # a plane the edge filter already rejected must not be in it to win.
        intra_keep = torch.ones_like(edge_keep)
        if self.intra_step_dedup:
            e_idx = edge_keep.nonzero(as_tuple=False).squeeze(1)
            intra_keep[e_idx] = self._intra_step_dedup(
                _index(new, e_idx), obs_scores[e_idx], device
            )

        keep = dist_keep & edge_keep & intra_keep  # [P_new]

        # ---- Step 2c: obs quality filter ----
        quality_keep = torch.ones_like(keep)
        if pool_clean is not None and pool_clean.get('plane_center_world',
                                                      torch.empty(0)).shape[0] > 0:
            # Only planes still standing may evict a pool plane, so the contest runs
            # on the survivors rather than on every candidate.
            s_idx = keep.nonzero(as_tuple=False).squeeze(1)
            sub = _index(new, s_idx)
            if self.bidirectional_quality:
                sub_keep, pool_replaced = self._obs_quality_mask_bidir(
                    sub, obs_scores[s_idx], pool_clean, device
                )
                if bool(pool_replaced.any()):
                    pool_clean = _index(pool_clean, ~pool_replaced.to(
                        pool_clean['plane_center_world'].device))
            else:
                sub_keep = self._obs_quality_mask(
                    sub, obs_scores[s_idx], pool_clean, device
                )
            quality_keep[s_idx] = sub_keep
            keep = keep & quality_keep

        # ---- Collect surviving planes + add metadata ------------------------
        P_new = new['plane_center_world'].shape[0]
        surv_idx = keep.nonzero(as_tuple=False).squeeze(1)
        logger.info(
            f"[PlaneMerger] step={step}: {P_new} new planes → keep {surv_idx.shape[0]} "
            f"(dist_keep={dist_keep.sum().item()}, edge_keep={edge_keep.sum().item()}, "
            f"final={keep.sum().item()})"
        )

        new_surv = _index(new, surv_idx)
        new_surv = self._add_metadata(new_surv, obs_scores[surv_idx], min_dists[surv_idx], step)
        new_surv.pop('frame_indices', None)

        updated_pool = _concat(pool_clean, new_surv)

        if not debug:
            return updated_pool, None
        debug_info = {
            'depth_grad':       _sobel_magnitude(depth_T).cpu(),  # [T, H, W]
            'dist_keep':        dist_keep.cpu(),
            'edge_keep':        edge_keep.cpu(),
            'quality_keep':     quality_keep.cpu(),
            'final_keep':       keep.cpu(),
            'new_centers':      new['plane_center_world'].cpu(),
            'pool_centers_orig': pool_centers_orig,             # [P_pool, 3] or None
            'pool_delete_mask': pool_delete_mask.cpu() if pool_delete_mask is not None else None,
        }
        return updated_pool, debug_info

    # Forward-looking prune

    def forward_prune(self, pool, next_cam_extrinsics, next_intrinsics=None,
                      image_height=576, image_width=1024):
        """Drop pool planes the next step will view from much closer than they were created.

        A plane created from far away is about to be re-created at better quality,
        so keeping it only adds a worse duplicate. Only planes actually inside a
        next-step frustum are dropped — something close but behind the camera will
        not be re-observed, so it stays.

        Args:
            pool: merged pool dict; needs plane_dist_created.
            next_cam_extrinsics: [T_next, 4, 4] c2w for the next step's cameras.
            next_intrinsics: [T_next, 3, 3] or [3, 3]. None skips the frustum check,
                so every plane close enough is pruned whether or not it is in view.
            image_height, image_width: render resolution the frustum check uses.
        """
        if next_cam_extrinsics is None or self.forward_prune_ratio <= 0:
            return pool

        centers = pool.get('plane_center_world')
        dist_created = pool.get('plane_dist_created')
        if centers is None or dist_created is None or centers.shape[0] == 0:
            return pool

        centers_f = centers.float()
        next_extr = next_cam_extrinsics.float()
        cam_pos = next_extr[:, :3, 3]
        T_next = next_extr.shape[0]

        P = centers_f.shape[0]
        min_dist_next = torch.full((P,), float('inf'))
        in_any_frustum = torch.zeros(P, dtype=torch.bool)
        chunk = 8192

        K = None
        if next_intrinsics is not None:
            K = next_intrinsics.float()
            if K.dim() == 2:
                K = K.unsqueeze(0).expand(T_next, -1, -1)

        w2c = torch.inverse(next_extr).cpu().float()
        R_all, t_all = w2c[:, :3, :3], w2c[:, :3, 3]
        if K is not None:
            K = K.cpu().float()

        for s in range(0, P, chunk):
            e = min(s + chunk, P)
            c = centers_f[s:e].cpu()
            d = torch.cdist(c, cam_pos.cpu())          # [n, T_next]

            if K is None:
                # Without intrinsics, use the minimum distance across all cameras.
                min_dist_next[s:e] = d.min(dim=1).values
                continue

            # Measure distance only from cameras that can see the plane, so pruning
            # requires a view that can re-observe it from sufficiently close up.
            vis = torch.zeros_like(d, dtype=torch.bool)   # [n, T_next]
            for t in range(T_next):
                p_cam = (R_all[t] @ c.T).T + t_all[t]
                z = p_cam[:, 2]
                u = p_cam[:, 0] / z.clamp(min=1e-6) * K[t, 0, 0] + K[t, 0, 2]
                v = p_cam[:, 1] / z.clamp(min=1e-6) * K[t, 1, 1] + K[t, 1, 2]
                vis[:, t] = ((z > 0.01) & (u >= 0) & (u < image_width)
                             & (v >= 0) & (v < image_height))
            # Planes outside every frustum get infinite distance and are retained.
            min_dist_next[s:e] = d.masked_fill(~vis, float('inf')).min(dim=1).values
            in_any_frustum[s:e] = vis.any(dim=1)

        ratio = min_dist_next.to(dist_created.device) / dist_created.float().clamp(min=1e-6)
        prune = ratio < self.forward_prune_ratio
        if K is not None:
            # The visibility mask is on CPU; prune can be on CUDA when a merge
            # retains the existing pool without adding any new planes.
            prune = prune & in_any_frustum.to(prune.device)
        n_pruned = int(prune.sum())

        if n_pruned > 0:
            pool = _index(pool, (~prune).nonzero(as_tuple=False).squeeze(1))
            logger.debug(f"[PlaneMerger] forward_prune: {n_pruned}/{P} planes removed "
                        f"(next step views them {1/self.forward_prune_ratio:.0f}x closer "
                        f"than they were created)")
        return pool

    # Visualization

    def save_visualizations(self, debug_info, depth_T, cam_extrinsics, intrinsics,
                            output_dir, step=0, vis_cams=None, max_pool_dots=300):
        """
        Save merge debug images to output_dir/merge_debug/.

        Per selected camera:
          plane_keep.png — depth colormap with new-plane dots (by keep/drop category)
                           and pool-plane dots (kept=blue circle, deleted=red cross).
          depth_edge.png — Sobel gradient heatmap.
          stats.json     — merge statistics.

        Dot colours for new planes:
          Green  — kept
          Gold   — dropped by obs-quality filter
          Orange — dropped by depth-edge filter
          Red    — dropped by distance filter
        """
        import cv2
        import json

        output_dir = Path(output_dir)
        vis_dir    = output_dir / 'merge_debug'
        vis_dir.mkdir(parents=True, exist_ok=True)

        T, H, W = depth_T.shape
        if vis_cams is None:
            n_vis    = min(5, T)
            vis_cams = [int(i * (T - 1) / max(n_vis - 1, 1)) for i in range(n_vis)]

        new_centers      = debug_info.get('new_centers')        # [P_new, 3] cpu
        dist_keep        = debug_info.get('dist_keep')          # [P_new] bool cpu
        edge_keep        = debug_info.get('edge_keep')
        quality_keep     = debug_info.get('quality_keep')
        final_keep       = debug_info.get('final_keep')
        pool_centers_orig = debug_info.get('pool_centers_orig') # [P_pool, 3] cpu or None
        pool_delete_mask  = debug_info.get('pool_delete_mask')  # [P_pool] bool cpu or None
        depth_grad        = debug_info.get('depth_grad')        # [T, H, W] cpu or None

        device = cam_extrinsics.device

        def project(centers_cpu, t):
            """Project [P, 3] cpu centers to (u_np, v_np, valid_np) for camera t."""
            if centers_cpu is None or centers_cpu.shape[0] == 0:
                return None
            c  = centers_cpu.to(device).float()
            w2c = torch.inverse(cam_extrinsics[t])       # [4, 4]
            p   = (w2c[:3, :3] @ c.T).T + w2c[:3, 3]   # [P, 3]
            z   = p[:, 2].clamp(min=1e-6)
            fx, fy = intrinsics[t, 0, 0], intrinsics[t, 1, 1]
            cx, cy = intrinsics[t, 0, 2], intrinsics[t, 1, 2]
            u = (p[:, 0] / z * fx + cx).long().cpu().numpy()
            v = (p[:, 1] / z * fy + cy).long().cpu().numpy()
            valid = ((p[:, 2] > 0.01) & torch.tensor(
                (u >= 0) & (u < W) & (v >= 0) & (v < H), device=device
            )).cpu().numpy()
            return u, v, valid

        def depth_colormap(d_np):
            """[H,W] float → (H,W,3) BGR uint8 (TURBO, 0–10 m)."""
            normed = np.clip(d_np / 10.0, 0, 1)
            gray   = (normed * 255).astype(np.uint8)
            return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

        # Subsample pool centers for display (avoid thousands of dots)
        if pool_centers_orig is not None and pool_centers_orig.shape[0] > max_pool_dots:
            perm = torch.randperm(pool_centers_orig.shape[0])[:max_pool_dots]
            pool_sub_centers = pool_centers_orig[perm]
            pool_sub_delete  = pool_delete_mask[perm] if pool_delete_mask is not None else None
        else:
            pool_sub_centers = pool_centers_orig
            pool_sub_delete  = pool_delete_mask

        plane_frames = []
        edge_frames  = []

        for t in vis_cams:
            depth_np = depth_T[t].cpu().numpy()
            vis      = depth_colormap(depth_np)

            # ---- draw new plane dots ----------------------------------------
            if new_centers is not None and new_centers.shape[0] > 0:
                proj = project(new_centers, t)
                if proj is not None:
                    u_arr, v_arr, valid = proj
                    for p in range(u_arr.shape[0]):
                        if not valid[p]:
                            continue
                        up, vp = int(u_arr[p]), int(v_arr[p])
                        if dist_keep is not None and not dist_keep[p].item():
                            col = (0,   0, 220)   # red  — distance dropped
                        elif edge_keep is not None and not edge_keep[p].item():
                            col = (0, 140, 255)   # orange — edge dropped
                        elif quality_keep is not None and not quality_keep[p].item():
                            col = (0, 200, 255)   # gold — quality dropped
                        else:
                            col = (0, 200,   0)   # green — kept
                        cv2.circle(vis, (up, vp), 3, col, -1)

            # ---- draw pool plane dots (sampled) -----------------------------
            if pool_sub_centers is not None and pool_sub_centers.shape[0] > 0:
                proj_pool = project(pool_sub_centers, t)
                if proj_pool is not None:
                    u_arr, v_arr, valid = proj_pool
                    for p in range(u_arr.shape[0]):
                        if not valid[p]:
                            continue
                        up, vp = int(u_arr[p]), int(v_arr[p])
                        if pool_sub_delete is not None and pool_sub_delete[p].item():
                            cv2.drawMarker(vis, (up, vp), (0, 0, 200),
                                           cv2.MARKER_CROSS, markerSize=7, thickness=1)
                        else:
                            cv2.circle(vis, (up, vp), 2, (200, 80, 0), 1)  # blue ring

            cv2.putText(vis, f'step{step} cam{t}', (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            plane_frames.append(vis)

            # ---- depth gradient vis -----------------------------------------
            if depth_grad is not None:
                g_np  = depth_grad[t].numpy()
                scale = max(self.depth_edge_thresh * 3, 1e-6)
                gray  = np.clip(g_np / scale, 0, 1)
                gbgr  = cv2.applyColorMap((gray * 255).astype(np.uint8), cv2.COLORMAP_HOT)
                # Mark edge-triggered plane centers in white
                if new_centers is not None and edge_keep is not None:
                    proj = project(new_centers, t)
                    if proj is not None:
                        u_arr, v_arr, valid = proj
                        for p in range(u_arr.shape[0]):
                            if valid[p] and not edge_keep[p].item():
                                cv2.circle(gbgr, (int(u_arr[p]), int(v_arr[p])), 3,
                                           (255, 255, 255), -1)
                cv2.putText(gbgr, f'edge cam{t}', (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                edge_frames.append(gbgr)

        if plane_frames:
            cv2.imwrite(str(vis_dir / f'step{step:03d}_plane_keep.png'),
                        np.concatenate(plane_frames, axis=1))
        if edge_frames:
            cv2.imwrite(str(vis_dir / f'step{step:03d}_depth_edge.png'),
                        np.concatenate(edge_frames, axis=1))

        # ---- stats JSON -------------------------------------------------------
        stats = {
            'step':               step,
            'n_pool_orig':        int(pool_centers_orig.shape[0]) if pool_centers_orig is not None else 0,
            'n_pool_deleted':     int(pool_delete_mask.sum().item()) if pool_delete_mask is not None else 0,
            'n_new_total':        int(new_centers.shape[0]) if new_centers is not None else 0,
            'n_new_dist_keep':    int(dist_keep.sum().item()) if dist_keep is not None else 0,
            'n_new_edge_keep':    int(edge_keep.sum().item()) if edge_keep is not None else 0,
            'n_new_quality_keep': int(quality_keep.sum().item()) if quality_keep is not None else 0,
            'n_new_final_keep':   int(final_keep.sum().item()) if final_keep is not None else 0,
        }
        with open(vis_dir / f'step{step:03d}_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"[MergeVis] Saved debug images → {vis_dir}")

    # Step 2b': dedup within the chunk

    def _intra_step_dedup(self, new_planes, obs_scores, device, chunk_size: int = 4096):
        """Drop planes a later frame of this step already covers from closer.

        Frames are walked in order into an accumulator. For each frame, a plane
        landing within obs_same_region_thresh of an accumulated one is resolved by
        obs_score: the loser is dropped, whichever side it is on.
        """
        frame_idx = new_planes.get('frame_indices')
        if frame_idx is None:
            P = new_planes['plane_center_world'].shape[0]
            return torch.ones(P, dtype=torch.bool, device=device)

        comp_device = device if device.type == 'cuda' else torch.device('cpu')
        centers = new_planes['plane_center_world'].to(comp_device).float()
        scores = obs_scores.to(comp_device).float()
        frame_idx_d = frame_idx.to(comp_device)
        P = centers.shape[0]
        keep = torch.ones(P, dtype=torch.bool, device=comp_device)
        frames = sorted(frame_idx_d.unique().tolist())
        if len(frames) <= 1:
            return keep.to(device)

        thresh = self.obs_same_region_thresh
        acc_indices = []
        for f in frames:
            f_global = (frame_idx_d == f).nonzero(as_tuple=False).squeeze(1)
            if len(acc_indices) == 0:
                acc_indices = f_global.tolist()
                continue

            acc_idx_t = torch.tensor(acc_indices, dtype=torch.long, device=comp_device)
            acc_centers = centers[acc_idx_t]
            acc_scores = scores[acc_idx_t]
            f_centers = centers[f_global]
            f_scores = scores[f_global]
            killed_acc = set()

            for s in range(0, f_global.shape[0], chunk_size):
                e = min(s + chunk_size, f_global.shape[0])
                min_d, min_i = torch.cdist(f_centers[s:e], acc_centers).min(dim=1)
                nearby = min_d < thresh
                if not nearby.any():
                    continue
                nb_idx = nearby.nonzero(as_tuple=False).squeeze(1)
                new_wins = f_scores[s:e][nb_idx] < acc_scores[min_i[nb_idx]]

                kill_old_pos = min_i[nb_idx[new_wins]]
                keep[acc_idx_t[kill_old_pos]] = False
                killed_acc.update(kill_old_pos.tolist())
                keep[f_global[s + nb_idx[~new_wins]]] = False

            if killed_acc:
                acc_indices = [i for pos, i in enumerate(acc_indices) if pos not in killed_acc]
            acc_indices.extend(f_global[keep[f_global]].tolist())

        n_removed = int((~keep).sum())
        if n_removed:
            logger.debug(f"[PlaneMerger] intra_step_dedup: removed {n_removed}/{P} "
                        f"redundant planes across {len(frames)} frames")
        return keep.to(device)

    # Step 1: delete occluded pool planes

    def _delete_occluded(self, pool, cam_extrinsics, intrinsics, depth_T, H, W):
        centers = pool['plane_center_world']  # [P, 3]
        P = centers.shape[0]
        if P == 0:
            return pool, torch.zeros(0, dtype=torch.bool, device=centers.device)

        uvs, depths_cam = _project(centers, cam_extrinsics, intrinsics)
        # uvs [P, T, 2], depths_cam [P, T]
        depth_at = _sample(depth_T, uvs)  # [P, T]

        in_front  = depths_cam > 0.01
        has_depth = depth_at > 0.01
        not_far   = depth_at < self.far_depth_ignore
        in_image  = (uvs[..., 0] >= 0) & (uvs[..., 0] < W) & \
                    (uvs[..., 1] >= 0) & (uvs[..., 1] < H)
        # Pool plane is closer than new depth → floating in front → bad
        floats    = depths_cam < depth_at * (1.0 - self.occlude_margin)

        votes     = (in_front & has_depth & not_far & floats & in_image).float()  # [P, T]
        vis_count = in_image.float().sum(1).clamp(min=1)                          # [P]
        delete    = (votes.sum(1) / vis_count) > self.vote_ratio

        n_del = delete.sum().item()
        logger.debug(f"[PlaneMerger] delete_occluded: {n_del}/{P} pool planes removed")
        return _index(pool, (~delete).nonzero(as_tuple=False).squeeze(1)), delete

    # Step 2a: distance filter

    def _distance_mask(self, planes, cam_extrinsics, intrinsics, alpha_T, min_dists, H, W):
        """Keep if dist <= max_dist, OR if dist > max_dist but rendered region is empty."""
        within  = min_dists <= self.max_dist  # [P]
        far_idx = (~within).nonzero(as_tuple=False).squeeze(1)
        if far_idx.numel() == 0:
            return within

        # For far planes: keep only where the previous pool leaves the pixel uncovered
        far_centers = planes['plane_center_world'][far_idx]
        uvs, _ = _project(far_centers, cam_extrinsics, intrinsics)
        alpha_at  = _sample(alpha_T, uvs)                               # [P_far, T]
        in_image  = (uvs[..., 0] >= 0) & (uvs[..., 0] < W) & \
                    (uvs[..., 1] >= 0) & (uvs[..., 1] < H)
        alpha_vis = torch.where(in_image, alpha_at, torch.zeros_like(alpha_at))
        max_alpha = alpha_vis.max(dim=1).values                         # [P_far]
        bg_empty  = max_alpha < self.alpha_empty_thresh                 # True → keep

        keep = within.clone()
        keep[far_idx] = bg_empty
        return keep

    # Step 2b: log-depth edge filter

    def _depth_edge_mask(self, planes, cam_extrinsics, device):
        """Discard planes sitting on a depth discontinuity.

        Sobel magnitude of log-depth on the per-plane depth grid. Log-depth makes the
        threshold scale-invariant, and the plane grid has none of the grid artifacts a
        rendered depth image would add.
        """
        P = planes['plane_center_world'].shape[0]
        T = cam_extrinsics.shape[0]

        frame_idx = planes.get(
            'frame_indices',
            torch.zeros(P, dtype=torch.long, device=device)
        ).to(device).long().clamp(0, T - 1)

        h_patch = planes.get('h_patch')
        w_patch = planes.get('w_patch')
        if h_patch is None or w_patch is None:
            raise RuntimeError(
                "_depth_edge_mask needs the plane grid shape (h_patch, w_patch). "
                "extract_plane_params_from_pred_prim sets both, so a dict without "
                "them did not come from there."
            )

        P_full = T * h_patch * w_patch
        if ('plane_center_camera' in planes
                and planes['plane_center_camera'].shape[0] == P_full):
            cam_z = planes['plane_center_camera'][:, 2].to(device).float()
        elif P == P_full:
            w2c = torch.inverse(cam_extrinsics)
            R_all, t_all = w2c[:, :3, :3], w2c[:, :3, 3]
            centers = planes['plane_center_world'].to(device).float()
            p_cam = (R_all[frame_idx] @ centers.unsqueeze(-1)).squeeze(-1) + t_all[frame_idx]
            cam_z = p_cam[:, 2]
        else:
            raise RuntimeError(
                f"_depth_edge_mask needs the full plane grid ({P_full} planes) but was "
                f"given {P}. It has to run before anything that drops planes."
            )

        patch_depth = cam_z.reshape(T, h_patch, w_patch)
        valid = patch_depth > 0.01
        log_depth = torch.zeros_like(patch_depth)
        log_depth[valid] = torch.log(patch_depth[valid])

        grad_flat = _sobel_magnitude(log_depth).reshape(P_full)
        discard = (grad_flat > self.depth_edge_thresh) & valid.reshape(-1)

        logger.debug(f"[PlaneMerger] depth_edge_filter: {int(discard.sum())}/{P_full} "
                    f"new planes discarded")
        return (~discard).to(device)

    def _obs_quality_mask(self, new_planes, new_obs_scores, pool, device,
                          chunk_size: int = 512):
        """Skip new plane if pool already has a nearby plane with better obs_score.

        Uses chunked CPU cdist to avoid [P_new × P_pool] GPU allocation.
        Peak memory ≈ chunk_size × P_pool × 4 bytes on CPU.
        """
        new_c  = new_planes['plane_center_world'].cpu().float()  # [P_new, 3]
        pool_c = pool['plane_center_world'].cpu().float()        # [P_pool, 3]
        P_new  = new_c.shape[0]

        if 'plane_obs_score' not in pool or pool_c.shape[0] == 0:
            return torch.ones(P_new, dtype=torch.bool, device=device)

        pool_scores = pool['plane_obs_score'].cpu().float()   # [P_pool]
        min_d, min_i = _nearest_in_pool(new_c, pool_c, chunk_size)

        nearby    = min_d < self.obs_same_region_thresh
        pool_best = pool_scores[min_i]
        keep      = ~nearby | (new_obs_scores.cpu() < pool_best)

        n_skip = (~keep).sum().item()
        logger.debug(f"[PlaneMerger] obs_quality_filter: {n_skip}/{P_new} new planes skipped "
              f"(pool covers region with better obs)")
        return keep.to(device)

    def _obs_quality_mask_bidir(self, new_planes, new_obs_scores, pool, device,
                                chunk_size: int = 512):
        """Both directions: skip the worse new plane, and evict the pool plane a
        better new one replaces.

        Returns (new_keep [P_new], pool_delete [P_pool]). A score tie resolves in
        favour of the new plane, so the pool entry is the one that goes.
        """
        new_c  = new_planes['plane_center_world'].cpu().float()
        pool_c = pool['plane_center_world'].cpu().float()
        P_new  = new_c.shape[0]
        P_pool = pool_c.shape[0]

        if 'plane_obs_score' not in pool or P_pool == 0:
            return (torch.ones(P_new, dtype=torch.bool, device=device),
                    torch.zeros(P_pool, dtype=torch.bool, device=device))

        pool_scores = pool['plane_obs_score'].cpu().float()
        new_scores = new_obs_scores.cpu().float()
        min_d, min_i = _nearest_in_pool(new_c, pool_c, chunk_size)

        nearby = min_d < self.obs_same_region_thresh
        pool_wins = nearby & (pool_scores[min_i] < new_scores)
        new_wins = nearby & ~pool_wins

        new_keep = ~pool_wins
        pool_delete = torch.zeros(P_pool, dtype=torch.bool)
        pool_delete[min_i[new_wins]] = True

        logger.debug(f"[PlaneMerger] obs_quality_bidir: {int(pool_wins.sum())}/{P_new} "
                    f"new skipped, {int(pool_delete.sum())}/{P_pool} pool replaced")
        return new_keep.to(device), pool_delete

    # Obs metrics

    def _compute_obs_metrics(self, planes, cam_extrinsics):
        """
        For each new plane, compute obs_score = dist / max(|cos_angle|, ε)
        using the plane's creating camera (from frame_indices).
        Returns obs_scores [P], min_dists [P].
        """
        device    = cam_extrinsics.device
        centers   = planes['plane_center_world'].float()   # [P, 3]
        normals   = _get_plane_normals(planes, device)     # [P, 3]
        P         = centers.shape[0]
        T         = cam_extrinsics.shape[0]

        frame_idx = planes.get(
            'frame_indices',
            torch.zeros(P, dtype=torch.long, device=device)
        ).to(device).long().clamp(0, T - 1)

        cam_centers = cam_extrinsics[frame_idx, :3, 3]   # [P, 3]
        diffs       = centers - cam_centers               # [P, 3]
        dists       = diffs.norm(dim=-1)                  # [P]
        dirs        = diffs / dists.unsqueeze(-1).clamp(min=1e-6)
        cos_angle   = (normals * dirs).sum(-1).abs()      # [P]
        obs_scores  = dists / cos_angle.clamp(min=1e-4)   # [P]

        return obs_scores, dists

    # Metadata

    @staticmethod
    def _add_metadata(planes, obs_scores, min_dists, step):
        P = planes['plane_center_world'].shape[0]
        planes['plane_obs_score']    = obs_scores.cpu()
        planes['plane_dist_created'] = min_dists.cpu()
        planes['plane_step_created'] = torch.full((P,), step, dtype=torch.long)
        return planes


# Module-level pure helpers

def _project(centers, cam_extrinsics, intrinsics):
    """
    Project [P, 3] world centers to all T cameras.
    Returns uvs [P, T, 2] (float), depths_cam [P, T].
    """

    w2c    = torch.inverse(cam_extrinsics)    # [T, 4, 4]
    R      = w2c[:, :3, :3]                  # [T, 3, 3]
    t_vec  = w2c[:, :3, 3]                   # [T, 3]
    fx     = intrinsics[:, 0, 0]
    fy     = intrinsics[:, 1, 1]
    cx     = intrinsics[:, 0, 2]
    cy     = intrinsics[:, 1, 2]

    # p_cam [P, T, 3]
    p_cam = (R.unsqueeze(0) @ centers.unsqueeze(1).unsqueeze(-1)).squeeze(-1) \
            + t_vec.unsqueeze(0)
    z      = p_cam[..., 2]                   # [P, T]
    z_safe = z.clamp(min=1e-6)
    u = p_cam[..., 0] / z_safe * fx.unsqueeze(0) + cx.unsqueeze(0)
    v = p_cam[..., 1] / z_safe * fy.unsqueeze(0) + cy.unsqueeze(0)

    return torch.stack([u, v], dim=-1), z     # uvs [P, T, 2], depths [P, T]


def _sample(maps, uvs):
    """
    Sample [T, H, W] maps at [P, T, 2] pixel coords (nearest neighbour, 0 outside).
    Returns [P, T].
    """
    T, H, W = maps.shape
    P       = uvs.shape[0]
    device  = maps.device

    u_int  = uvs[..., 0].long()   # [P, T]
    v_int  = uvs[..., 1].long()
    in_img = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)

    t_idx  = torch.arange(T, device=device).unsqueeze(0).expand(P, -1)
    u_c    = u_int.clamp(0, W - 1)
    v_c    = v_int.clamp(0, H - 1)
    vals   = maps[t_idx, v_c, u_c]
    return torch.where(in_img, vals, torch.zeros_like(vals))


def _sobel_magnitude(depth):
    """Sobel gradient magnitude for [T, H, W] → [T, H, W]."""
    d  = depth.unsqueeze(1).float()   # [T, 1, H, W]
    # Replicate, not zero: zero padding reads as "depth 0 just outside the image"
    # and puts a huge false gradient all along the border.
    d  = F.pad(d, [1, 1, 1, 1], mode='replicate')
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=torch.float32, device=depth.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(d, kx)
    gy = F.conv2d(d, ky)
    return (gx ** 2 + gy ** 2).sqrt().squeeze(1)   # [T, H, W]



_PLANE_FIELDS = frozenset({
    'plane_center_world', 'plane_radii', 'plane_rot_q',
    'plane_normal_world', 'plane_obs_score', 'plane_dist_created',
    'plane_step_created', 'frame_indices',
})


def _nearest_in_pool(new_c, pool_c, chunk_size):
    """(min_dist, min_index) from each new plane to the pool, chunked on CPU."""
    P_new = new_c.shape[0]
    min_d = torch.empty(P_new, dtype=torch.float32)
    min_i = torch.empty(P_new, dtype=torch.long)
    for s in range(0, P_new, chunk_size):
        e = min(s + chunk_size, P_new)
        md, mi = torch.cdist(new_c[s:e], pool_c).min(dim=1)
        min_d[s:e] = md
        min_i[s:e] = mi
    return min_d, min_i


def _index(planes, idx):
    """Index all per-plane tensor fields with idx (LongTensor or BoolTensor)."""
    result = {}
    for k, v in planes.items():
        if isinstance(v, torch.Tensor) and k in _PLANE_FIELDS:
            result[k] = v[idx]
        else:
            result[k] = v
    P = result['plane_center_world'].shape[0] if 'plane_center_world' in result else 0
    result['planes_per_frame'] = P
    result['num_frames']       = 1
    return result


def _concat(pool, new):
    """
    Concatenate two pool dicts on the plane axis. Either may be None / empty.

    The result carries the pool schema and nothing else. A passenger array the
    caller happened to include (camera-space fields, the patch grid shape) would
    keep its pre-filter row count — _index only slices _PLANE_FIELDS — and from
    then on its row i is a different plane from plane_center_world's row i.
    """
    _CONCAT_FIELDS = _PLANE_FIELDS - {'frame_indices'}

    def _empty(d):
        return d is None or d.get('plane_center_world', torch.empty(0)).shape[0] == 0

    def _schema(d):
        # On the CPU like the concatenated case below, so the saved pool loads anywhere.
        out = {k: v.cpu() for k, v in d.items() if k in _CONCAT_FIELDS}
        out['planes_per_frame'] = d['plane_center_world'].shape[0]
        out['num_frames'] = 1
        return out

    if _empty(pool) and _empty(new):
        return {'plane_center_world': torch.empty(0, 3), 'planes_per_frame': 0, 'num_frames': 1}
    if _empty(pool):
        return _schema(new)
    if _empty(new):
        return _schema(pool)

    P_pool = pool['plane_center_world'].shape[0]
    P_new  = new['plane_center_world'].shape[0]
    result = {}
    # Sorted so the saved pool's key order does not depend on per-process string hashing
    # (a frozenset iterates in that order).
    for k in sorted(_CONCAT_FIELDS):
        vp, vn = pool.get(k), new.get(k)
        if vp is None and vn is None:
            continue
        if vp is not None and vn is not None:
            result[k] = torch.cat([vp.cpu(), vn.cpu()], dim=0)
        elif vp is not None:
            fill = torch.zeros((P_new,) + vp.shape[1:], dtype=vp.dtype)
            result[k] = torch.cat([vp.cpu(), fill], dim=0)
        else:
            fill = torch.zeros((P_pool,) + vn.shape[1:], dtype=vn.dtype)
            result[k] = torch.cat([fill, vn.cpu()], dim=0)

    result['planes_per_frame'] = P_pool + P_new
    result['num_frames']       = 1
    return result
