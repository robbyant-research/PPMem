"""Projection Module for Incremental Video Generation"""

import torch


def _as_device_tensor(x, device):
    """numpy arrays become float32 tensors; tensors keep their dtype."""
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()
    return x.to(device)


def _source_points_world(src_depth, src_pose, K, device):
    """World coordinates (4, H*W) of every source pixel, and which pixels are valid."""
    H, W = src_depth.shape
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    ones = torch.ones_like(u)

    valid_mask = src_depth > 0.01

    uv1 = torch.stack([u, v, ones], dim=-1).reshape(-1, 3).T  # (3, H*W)
    K_inv = torch.inverse(K)
    xyz_cam_src = K_inv @ uv1  # (3, H*W)

    depth_flat = src_depth.reshape(-1)  # (H*W,)
    xyz_cam_src = xyz_cam_src * depth_flat.unsqueeze(0)  # (3, H*W)

    xyz_cam_src_h = torch.cat(
        [xyz_cam_src, torch.ones((1, xyz_cam_src.shape[1]), device=device)],
        dim=0
    )  # (4, H*W)

    xyz_world = src_pose @ xyz_cam_src_h  # (4, H*W)
    return xyz_world, valid_mask.reshape(-1)


def _project_to_target(xyz_world, valid_mask_flat, tgt_pose, K, H, W):
    """Pixel coordinates and depth of every source point in one target view, and which
    of them land inside it."""
    tgt_pose_inv = torch.inverse(tgt_pose)
    xyz_cam_tgt = tgt_pose_inv @ xyz_world  # (4, H*W)

    xyz_cam_tgt_3d = xyz_cam_tgt[:3, :]  # (3, H*W)
    uv1_tgt = K @ xyz_cam_tgt_3d  # (3, H*W)

    depths_tgt = uv1_tgt[2, :]
    u_tgt = uv1_tgt[0, :] / (depths_tgt + 1e-8)
    v_tgt = uv1_tgt[1, :] / (depths_tgt + 1e-8)

    valid = (u_tgt >= 0) & (u_tgt < W-1) & (v_tgt >= 0) & (v_tgt < H-1) & \
            (depths_tgt > 0) & valid_mask_flat
    return u_tgt, v_tgt, depths_tgt, valid


def reproject_coverage(src_depth, src_pose, tgt_poses, intrinsic, device='cuda'):
    """Where reproject_to_target_views' mask would be > 0, without the colour and depth splats.

    Source selection only needs this coverage. A target pixel is covered when a valid source
    point gives it a positive bilinear weight -- the same points, corners and weights as the
    full reprojection, so the result matches its `mask > 0` exactly.

    Returns:
        coverage: (N, H, W) bool numpy array
    """
    src_depth = _as_device_tensor(src_depth, device)
    src_pose = _as_device_tensor(src_pose, device)
    tgt_poses = _as_device_tensor(tgt_poses, device)
    K = _as_device_tensor(intrinsic, device)

    H, W = src_depth.shape
    N = tgt_poses.shape[0]
    xyz_world, valid_mask_flat = _source_points_world(src_depth, src_pose, K, device)

    coverage = torch.zeros((N, H * W), dtype=torch.bool, device=device)
    for i in range(N):
        u_tgt, v_tgt, _, valid = _project_to_target(xyz_world, valid_mask_flat, tgt_poses[i], K, H, W)
        u_tgt_valid = u_tgt[valid]
        v_tgt_valid = v_tgt[valid]
        if len(u_tgt_valid) == 0:
            continue
        u0 = torch.floor(u_tgt_valid).long()
        v0 = torch.floor(v_tgt_valid).long()
        wu = u_tgt_valid - u0.float()
        wv = v_tgt_valid - v0.float()
        idx00 = v0 * W + u0
        for idx, weight in ((idx00, (1 - wu) * (1 - wv)), (idx00 + 1, wu * (1 - wv)),
                            (idx00 + W, (1 - wu) * wv), (idx00 + W + 1, wu * wv)):
            coverage[i, idx[weight > 0]] = True
    return coverage.view(N, H, W).cpu().numpy()


def reproject_to_target_views(
    src_rgb, src_depth, src_pose, tgt_poses, intrinsic,
    device='cuda', return_depth=False
):
    """
    Reproject source view to multiple target views using PyTorch (GPU accelerated).
    
    Args:
        src_rgb: (H, W, 3) source RGB image, numpy array [0-255]
        src_depth: (H, W) source depth map, numpy array; only pixels with depth > 0.01 are projected
        src_pose: (4, 4) source c2w pose matrix, numpy array
        tgt_poses: (N, 4, 4) target c2w pose matrices, numpy array
        intrinsic: (3, 3) camera intrinsic matrix, numpy array
        device: Device to run on
        return_depth: If True, also return reprojected depth maps
    
    Returns:
        reprojected_rgbs: (N, H, W, 3) reprojected RGB images, numpy array [0-255]
        masks: (N, H, W) valid pixel masks [0-1], numpy array
        reprojected_depths (optional): (N, H, W) reprojected depth maps, numpy array
    """
    src_rgb = _as_device_tensor(src_rgb, device)
    src_depth = _as_device_tensor(src_depth, device)
    src_pose = _as_device_tensor(src_pose, device)
    tgt_poses = _as_device_tensor(tgt_poses, device)
    K = _as_device_tensor(intrinsic, device)

    H, W = src_depth.shape
    N = tgt_poses.shape[0]  # Number of target views

    xyz_world, valid_mask_flat = _source_points_world(src_depth, src_pose, K, device)

    reprojected_rgbs = []
    masks = []
    reprojected_depths = [] if return_depth else None

    for i in range(N):
        u_tgt, v_tgt, depths_tgt, valid = _project_to_target(
            xyz_world, valid_mask_flat, tgt_poses[i], K, H, W)

        reprojected_rgb = torch.zeros((H, W, 3), dtype=torch.float64, device=device)
        reprojected_depth = torch.zeros((H, W), dtype=torch.float64, device=device) if return_depth else None
        mask = torch.zeros((H, W), dtype=torch.float32, device=device)

        src_rgb_flat = src_rgb.reshape(-1, 3).double()

        u_tgt_valid = u_tgt[valid]
        v_tgt_valid = v_tgt[valid]
        src_rgb_valid = src_rgb_flat[valid]
        depths_tgt_valid = depths_tgt[valid]

        if len(u_tgt_valid) == 0:
            reprojected_rgbs.append(reprojected_rgb.float())
            masks.append(mask)
            if return_depth:
                reprojected_depths.append(reprojected_depth.float())
            continue

        u0 = torch.floor(u_tgt_valid).long()
        v0 = torch.floor(v_tgt_valid).long()
        u1 = u0 + 1
        v1 = v0 + 1

        wu = u_tgt_valid - u0.float()
        wv = v_tgt_valid - v0.float()

        wu_d = wu.double()
        wv_d = wv.double()

        clamped_depth = torch.clamp(depths_tgt_valid, min=0).double()
        median_depth = torch.median(clamped_depth)
        if median_depth > 1e-7:
            normalized_depth = clamped_depth / median_depth
        else:
            normalized_depth = clamped_depth

        log_depth = torch.log1p(normalized_depth)
        depth_weight_scale = 50.0
        max_log_depth = log_depth.max()

        if max_log_depth > 1e-7:
            exponent = (log_depth / (max_log_depth + 1e-7)) * depth_weight_scale
        else:
            exponent = torch.zeros_like(log_depth)

        min_exponent = exponent.min()
        exponent_normalized = exponent - min_exponent
        max_range = 15.0
        current_range = exponent_normalized.max()
        if current_range > max_range:
            exponent_normalized = exponent_normalized * (max_range / current_range)

        inv_depth_weights_d = torch.exp(-exponent_normalized)

        weight_buffer = torch.zeros((H, W), dtype=torch.float64, device=device)
        # Accumulate in float64 like the colour, so the float32 result does not depend on the
        # order the atomic adds land in.
        depth_weight_buffer = torch.zeros((H, W), dtype=torch.float64, device=device) if return_depth else None
        mask_buffer = torch.zeros((H, W), dtype=torch.float64, device=device)

        w00_bilinear = ((1 - wu) * (1 - wv))
        w10_bilinear = (wu * (1 - wv))
        w01_bilinear = ((1 - wu) * wv)
        w11_bilinear = (wu * wv)

        # Depth-weighted bilinear weights
        w00_d = ((1 - wu_d) * (1 - wv_d)) * inv_depth_weights_d
        w10_d = (wu_d * (1 - wv_d)) * inv_depth_weights_d
        w01_d = ((1 - wu_d) * wv_d) * inv_depth_weights_d
        w11_d = (wu_d * wv_d) * inv_depth_weights_d

        idx00 = v0 * W + u0
        for c in range(3):
            reprojected_rgb.view(-1, 3)[:, c].scatter_add_(
                0, idx00, src_rgb_valid[:, c] * w00_d
            )
        weight_buffer.view(-1).scatter_add_(0, idx00, w00_d)
        mask_buffer.view(-1).scatter_add_(0, idx00, w00_bilinear.double())
        if return_depth:
            reprojected_depth.view(-1).scatter_add_(0, idx00, depths_tgt_valid.double() * w00_d)
            depth_weight_buffer.view(-1).scatter_add_(0, idx00, w00_d)

        idx10 = v0 * W + u1
        for c in range(3):
            reprojected_rgb.view(-1, 3)[:, c].scatter_add_(
                0, idx10, src_rgb_valid[:, c] * w10_d
            )
        weight_buffer.view(-1).scatter_add_(0, idx10, w10_d)
        mask_buffer.view(-1).scatter_add_(0, idx10, w10_bilinear.double())
        if return_depth:
            reprojected_depth.view(-1).scatter_add_(0, idx10, depths_tgt_valid.double() * w10_d)
            depth_weight_buffer.view(-1).scatter_add_(0, idx10, w10_d)

        idx01 = v1 * W + u0
        for c in range(3):
            reprojected_rgb.view(-1, 3)[:, c].scatter_add_(
                0, idx01, src_rgb_valid[:, c] * w01_d
            )
        weight_buffer.view(-1).scatter_add_(0, idx01, w01_d)
        mask_buffer.view(-1).scatter_add_(0, idx01, w01_bilinear.double())
        if return_depth:
            reprojected_depth.view(-1).scatter_add_(0, idx01, depths_tgt_valid.double() * w01_d)
            depth_weight_buffer.view(-1).scatter_add_(0, idx01, w01_d)

        idx11 = v1 * W + u1
        for c in range(3):
            reprojected_rgb.view(-1, 3)[:, c].scatter_add_(
                0, idx11, src_rgb_valid[:, c] * w11_d
            )
        weight_buffer.view(-1).scatter_add_(0, idx11, w11_d)
        mask_buffer.view(-1).scatter_add_(0, idx11, w11_bilinear.double())
        if return_depth:
            reprojected_depth.view(-1).scatter_add_(0, idx11, depths_tgt_valid.double() * w11_d)
            depth_weight_buffer.view(-1).scatter_add_(0, idx11, w11_d)

        valid_weights = weight_buffer > 1e-8
        reprojected_rgb[valid_weights] = reprojected_rgb[valid_weights] / \
            weight_buffer[valid_weights].unsqueeze(-1)

        if return_depth:
            valid_depth_weights = depth_weight_buffer > 1e-8
            reprojected_depth[valid_depth_weights] = reprojected_depth[valid_depth_weights] / depth_weight_buffer[valid_depth_weights]

        reprojected_rgb = reprojected_rgb.float()
        mask = torch.clamp(mask_buffer, 0, 1).float()

        reprojected_rgbs.append(reprojected_rgb)
        masks.append(mask)
        if return_depth:
            reprojected_depths.append(reprojected_depth.float())

    reprojected_rgbs = torch.stack(reprojected_rgbs, dim=0).cpu().numpy()  # (N, H, W, 3)
    masks = torch.stack(masks, dim=0).cpu().numpy()  # (N, H, W)

    if return_depth:
        reprojected_depths = torch.stack(reprojected_depths, dim=0).cpu().numpy()  # (N, H, W)
        return reprojected_rgbs, masks, reprojected_depths
    else:
        return reprojected_rgbs, masks
