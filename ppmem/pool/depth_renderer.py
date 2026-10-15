"""Depth Renderer for Plane Parameters"""

import math
import torch
import numpy as np
from typing import Dict
from pathlib import Path
from ppmem._logging import logger

try:
    from diff_rect_rasterization import RectRasterizationSettings, RectRasterizer
    RASTERIZER_AVAILABLE = True
except ImportError:
    logger.warning("diff_rect_rasterization not available; DepthRenderer will not work.")
    RASTERIZER_AVAILABLE = False


def _focal2fov(focal: float, pixels: int) -> float:
    """FOV in radians from focal length and image dimension."""
    return 2.0 * math.atan(pixels / (2.0 * focal))


def _get_projection_matrix(znear: float, zfar: float, fovX: float, fovY: float, device: torch.device) -> torch.Tensor:
    """4x4 perspective projection with depth mapping."""
    tanHalfFovY = math.tan(fovY / 2.0)
    tanHalfFovX = math.tan(fovX / 2.0)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    z_sign = 1.0
    P = torch.zeros(4, 4, device=device, dtype=torch.float32)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class DepthRenderer:
    """
    Render depth maps from plane parameters using rectangular Gaussian splatting.
    
    Settings come from RectRasterizationSettings; the RectRasterizer forward returns
    allmap with depth at [0], alpha at [1] and normal at [2:5].
    """

    def __init__(
        self,
        image_height: int = 576,
        image_width: int = 1024,
        device: str = 'cuda',
    ):
        """
        Initialize depth renderer
        
        Args:
            image_height: Target render height
            image_width: Target render width
            device: Device to render on
        """
        if not RASTERIZER_AVAILABLE:
            raise RuntimeError("diff_rect_rasterization is required for DepthRenderer")

        self.H = image_height
        self.W = image_width
        self.device = device
        self.bg = torch.tensor([0., 0., 0.], device=device)

    @torch.no_grad()
    def render_depth(
        self,
        plane_params: Dict,
        intrinsics: torch.Tensor,
        extrinsics_c2w: torch.Tensor,
        splat_weight: float = 300.0,
    ):
        """
        Render depth maps from WORLD-SPACE plane parameters.
        
        IMPORTANT: Input planes must be in world space (transformed via transform_planes_to_world).
        Planes are stored in world coordinates.
        
        Args:
            plane_params: Dict with keys (from transform_planes_to_world):
                'plane_center_world': [T*HW, 3] - centers in world space
                'plane_normal_world': [T*HW, 3] - normals in world space
                'plane_radii': [T*HW, 4]
                'plane_rot_q': [T*HW, 4]
                'num_frames': int (T), or 1 to render the same planes from every camera
                'planes_per_frame': int (HW)
            intrinsics: [T, 3, 3] or [3, 3] camera intrinsics
            extrinsics_c2w: [T, 4, 4] camera-to-world transforms
            splat_weight: Splat weight for rasterization (default 300.0)
        
        Returns:
            allmaps: List of [5, H, W] tensors (T frames)
                [0]: depth
                [1]: visibility/alpha
                [2:5]: normal (in camera frame)
        """
        T = extrinsics_c2w.shape[0]
        HW = plane_params['planes_per_frame']
        # One set of planes shared by every camera (the merged pool), or one set per frame.
        shared = plane_params['num_frames'] == 1
        if not shared and plane_params['num_frames'] != T:
            raise ValueError(f"{plane_params['num_frames']} frames of planes for {T} cameras")

        # Extract plane parameters (WORLD SPACE)
        plane_center_world = plane_params['plane_center_world'].float().to(self.device)  # [T*HW, 3]
        plane_radii_all = plane_params['plane_radii'].float().to(self.device)  # [T*HW, 4]
        plane_rot_q_all = plane_params['plane_rot_q'].float().to(self.device)  # [T*HW, 4]

        if intrinsics.dim() == 2:
            intrinsics = intrinsics.unsqueeze(0).expand(T, -1, -1)
        intrinsics = intrinsics.to(self.device).float()  # [T, 3, 3]
        extrinsics_c2w = extrinsics_c2w.to(self.device).float()  # [T, 4, 4]

        allmaps_list = []

        for t in range(T):
            # Get plane params for this frame (WORLD SPACE)
            s = 0 if shared else t * HW
            e = s + HW
            plane_center = plane_center_world[s:e]  # [HW, 3] in world space
            plane_radii = plane_radii_all[s:e]  # [HW, 4]
            plane_rot_q = plane_rot_q_all[s:e]  # [HW, 4]

            # Get camera params for this frame
            K = intrinsics[t]  # [3, 3]
            c2w = extrinsics_c2w[t]  # [4, 4]
            w2c = torch.inverse(c2w)  # [4, 4]

            # Rasterization params
            raster_cam_w2c = w2c.transpose(0, 1)  # [4, 4] transposed (GS convention)
            raster_cam_center = c2w[:3, 3]  # [3] camera center in world

            # FOV from intrinsics
            fovX = _focal2fov(K[0, 0].item(), self.W)
            fovY = _focal2fov(K[1, 1].item(), self.H)
            tanfovx = math.tan(fovX / 2.0)
            tanfovy = math.tan(fovY / 2.0)
            # Full projection matrix with znear/zfar (required for correct depth in rasterizer)
            znear, zfar = 0.01, 10.0
            P = _get_projection_matrix(znear, zfar, fovX, fovY, self.device)
            # full_proj so C++ gets P @ w2c in column-major: pass (P @ w2c).T
            raster_cam_fullproj = (P @ w2c).transpose(0, 1)
            raster_img_center = torch.tensor([K[0, 2].item(), K[1, 2].item()], device=self.device, dtype=torch.float32)

            # Rasterization settings
            raster_settings = RectRasterizationSettings(
                image_height=self.H,
                image_width=self.W,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=self.bg,
                scale_modifier=1.0,
                viewmatrix=raster_cam_w2c,
                projmatrix=raster_cam_fullproj,
                sh_degree=0,
                campos=raster_cam_center,
                prefiltered=False,
                debug=False,
                lambdaw=splat_weight * 5.0,
                image_center=raster_img_center,
                scales2=plane_radii[:, :2].contiguous(),  # [HW, 2]
            )

            # Rasterize (planes in world space, rasterizer transforms to camera)
            screenspace_points = torch.zeros_like(plane_center, dtype=plane_center.dtype, requires_grad=False, device=self.device)
            rasterizer = RectRasterizer(raster_settings=raster_settings)

            rgb, _, allmap = rasterizer(
                means3D=plane_center,  # World-space centers
                means2D=screenspace_points,
                shs=None,
                colors_precomp=torch.rand_like(plane_center),  # Random colors (not used)
                opacities=torch.ones_like(plane_center)[:, :1],
                scales=plane_radii,
                rotations=plane_rot_q,
                cov3D_precomp=None,
            )

            allmaps_list.append(allmap)

        return allmaps_list

    def process_allmap_to_depth(
        self,
        allmap: torch.Tensor,
        extrinsic_c2w: torch.Tensor,
        alpha_threshold: float = 0.5,
    ):
        """
        Process allmap to extract depth, normal, and alpha.
        
        Args:
            allmap: [5, H, W] rendered map
            extrinsic_c2w: [4, 4] camera-to-world transform for normal transform
            alpha_threshold: Threshold for alpha mask
        
        Returns:
            dict with keys:
                'depth': [H, W] metric depth (masked by alpha)
                'normal_world': [H, W, 3] normal in world frame (masked by alpha)
                'alpha': [H, W] alpha mask (binary)
        """
        # Extract channels (allmap: [0]=depth_raw, [1]=alpha, [2:5]=normal)
        depth_raw = allmap[0]  # [H, W]
        alpha = allmap[1]  # [H, W]
        nch = allmap.shape[0]
        if nch < 5:
            logger.warning(f"[DepthRenderer] process_allmap_to_depth: allmap has {nch} channels, expected >=5")
        normal_local = allmap[2:5]  # [3, H, W]

        alpha_mask = alpha > alpha_threshold  # [H, W]

        # Process depth: divide by alpha to get metric depth, then apply mask
        depth = depth_raw / (alpha + 1e-8)  # [H, W]
        depth_masked = torch.where(alpha_mask, depth, torch.zeros_like(depth))

        # Transform normal to world frame [H, W, 3]
        w2c = torch.inverse(extrinsic_c2w).to(self.device).float()
        raster_cam_w2c = w2c.transpose(0, 1)  # Transposed (GS convention)
        normal_world = (normal_local.permute(1, 2, 0) @ (raster_cam_w2c[:3, :3].T))  # [H, W, 3]

        alpha_mask_3d = alpha_mask.unsqueeze(-1).expand_as(normal_world)
        normal_masked = torch.where(alpha_mask_3d, normal_world, torch.zeros_like(normal_world))

        normal_norm = torch.norm(normal_masked, dim=-1, keepdim=True)  # [H, W, 1]
        normal_norm = torch.where(normal_norm > 1e-6, normal_norm, torch.ones_like(normal_norm))
        normal_normalized = normal_masked / normal_norm
        normal_normalized = torch.clamp(normal_normalized, -1.0, 1.0)

        alpha_binary = alpha_mask.float()

        return {
            'depth': depth_masked,
            'normal_world': normal_normalized,
            'alpha': alpha_binary,
        }

    def save_depth_maps(
        self,
        allmaps_list,
        extrinsics_c2w: torch.Tensor,
        output_dir: Path,
        start_idx: int = 0,
        alpha_threshold: float = 0.5,
        save_normal: bool = True,
        save_alpha: bool = True,
        save_png: bool = True,
    ):
        """
        Save rendered depth maps to numpy and PNG files.

        Args:
            allmaps_list: List of [5, H, W] tensors (T frames)
            extrinsics_c2w: [T, 4, 4] camera-to-world transforms
            output_dir: Output directory
            start_idx: Starting frame index (for naming)
            alpha_threshold: Threshold for alpha mask
            save_normal: Also write the world-space normals, as 16-bit PNG: each channel
                (n + 1) / 2 * 65535, so n = png / 65535 * 2 - 1 (error below 0.03 degrees)
            save_alpha: Also write the alpha arrays (alpha > 0.5 exactly where depth > 0)
            save_png: Also write PNG visualizations of what is saved
        """
        import cv2

        output_dir = Path(output_dir)
        depth_dir = output_dir / 'depth'
        normal_dir = output_dir / 'normal'
        alpha_dir = output_dir / 'alpha'
        depth_dir.mkdir(parents=True, exist_ok=True)
        if save_normal:
            normal_dir.mkdir(parents=True, exist_ok=True)
        if save_alpha:
            alpha_dir.mkdir(parents=True, exist_ok=True)

        for t, allmap in enumerate(allmaps_list):
            frame_idx = start_idx + t
            c2w = extrinsics_c2w[t]

            result = self.process_allmap_to_depth(allmap, c2w, alpha_threshold)

            depth_np = result['depth'].cpu().numpy()

            # Save as numpy arrays (1-based indexing)
            np.save(depth_dir / f'depth_{frame_idx+1:04d}.npy', depth_np)
            if save_png:
                depth_vis = np.clip(depth_np, 0, 10)
                depth_vis = (depth_vis / 10.0 * 255).astype(np.uint8)
                cv2.imwrite(str(depth_dir / f'depth_{frame_idx+1:04d}.png'), depth_vis)
            if save_normal:
                normal_np = result['normal_world'].cpu().numpy()
                normal_u16 = np.round((np.clip(normal_np, -1.0, 1.0) + 1.0) / 2.0 * 65535).astype(np.uint16)
                cv2.imwrite(str(normal_dir / f'normal_{frame_idx+1:04d}.png'),
                            cv2.cvtColor(normal_u16, cv2.COLOR_RGB2BGR))
            if save_alpha:
                alpha_np = result['alpha'].cpu().numpy()
                np.save(alpha_dir / f'alpha_{frame_idx+1:04d}.npy', alpha_np)
                if save_png:
                    # Alpha: grayscale
                    alpha_vis = (alpha_np * 255).astype(np.uint8)
                    cv2.imwrite(str(alpha_dir / f'alpha_{frame_idx+1:04d}.png'), alpha_vis)

        logger.info(f"Saved {len(allmaps_list)} depth maps to: {output_dir}")
