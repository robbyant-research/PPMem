import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """
    Normalize quaternion (supports any batch dimensions)
    Input shape: 
      - [4]           (single quaternion)
      - [n, 4]        (batch quaternions)
      - [batch, ..., 4] (high-dim batch)
    Output shape: same as input
    """
    import logging
    _logger = logging.getLogger(__name__)
    dim = -1
    norms = torch.linalg.norm(q, ord=2, dim=dim, keepdim=True)
    near_zero = (norms < 1e-6).any()
    if near_zero:
        _logger.warning(
            f"[normalize_quaternion] Near-zero quaternion norm detected! "
            f"min_norm={norms.min().item():.2e}, count={int((norms < 1e-6).sum().item())}/{norms.numel()}"
        )
    epsilon = torch.tensor(1e-6, device=q.device, dtype=q.dtype)
    normalized_q = q / torch.maximum(norms, epsilon)
    return normalized_q


def rot_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion [w, x, y, z].
    Numerically stable for all rotations (including ~180°).
    R: [3, 3] or [N, 3, 3]
    Returns: [4] or [N, 4]
    """
    if R.dim() == 2:
        R = R.unsqueeze(0)
    # R [N, 3, 3]
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    eps = 1e-6
    w = torch.sqrt(torch.clamp(1.0 + trace, min=eps)) / 2.0
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4.0 * w + eps)
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4.0 * w + eps)
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4.0 * w + eps)
    # When trace <= 0, w is small; use largest diagonal to pick a stable branch
    R00, R11, R22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    cond0 = (trace <= 0) & (R00 >= R11) & (R00 >= R22)
    s0 = 2.0 * torch.sqrt(torch.clamp(1.0 + R00 - R11 - R22, min=eps))
    w0 = (R[:, 2, 1] - R[:, 1, 2]) / (s0 + eps)
    x0 = 0.25 * s0
    y0 = (R[:, 0, 1] + R[:, 1, 0]) / (s0 + eps)
    z0 = (R[:, 0, 2] + R[:, 2, 0]) / (s0 + eps)
    cond1 = (trace <= 0) & (R11 > R00) & (R11 >= R22)
    s1 = 2.0 * torch.sqrt(torch.clamp(1.0 - R00 + R11 - R22, min=eps))
    w1 = (R[:, 0, 2] - R[:, 2, 0]) / (s1 + eps)
    x1 = (R[:, 0, 1] + R[:, 1, 0]) / (s1 + eps)
    y1 = 0.25 * s1
    z1 = (R[:, 1, 2] + R[:, 2, 1]) / (s1 + eps)
    cond2 = (trace <= 0) & (R22 > R00) & (R22 > R11)
    s2 = 2.0 * torch.sqrt(torch.clamp(1.0 - R00 - R11 + R22, min=eps))
    w2 = (R[:, 1, 0] - R[:, 0, 1]) / (s2 + eps)
    x2 = (R[:, 0, 2] + R[:, 2, 0]) / (s2 + eps)
    y2 = (R[:, 1, 2] + R[:, 2, 1]) / (s2 + eps)
    z2 = 0.25 * s2
    w = torch.where(cond0, w0, torch.where(cond1, w1, torch.where(cond2, w2, w)))
    x = torch.where(cond0, x0, torch.where(cond1, x1, torch.where(cond2, x2, x)))
    y = torch.where(cond0, y0, torch.where(cond1, y1, torch.where(cond2, y2, y)))
    z = torch.where(cond0, z0, torch.where(cond1, z1, torch.where(cond2, z2, z)))
    q = torch.stack([w, x, y, z], dim=-1)
    q = F.normalize(q, dim=-1, eps=eps)
    if R.shape[0] == 1:
        return q.squeeze(0)
    return q


def quat_mult(qa: torch.Tensor, qb: torch.Tensor) -> torch.Tensor:
    """
    Quaternion multiplication: qa * qb such that R(qa*qb) = R(qa) @ R(qb).
    qa, qb: [..., 4] in [w, x, y, z]
    Returns: [..., 4] broadcast-compatible.
    """
    # (w1,x1,y1,z1)*(w2,x2,y2,z2) = (w1w2 - x1x2 - y1y2 - z1z2, w1x2+x1w2+y1z2-z1y2, ...)
    w1, x1, y1, z1 = qa[..., 0], qa[..., 1], qa[..., 2], qa[..., 3]
    w2, x2, y2, z2 = qb[..., 0], qb[..., 1], qb[..., 2], qb[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    q = torch.stack([w, x, y, z], dim=-1)
    return F.normalize(q, dim=-1, eps=1e-6)


def quat_to_rot(q):
    """
    Convert quaternion to rotation matrix
    q: [bs, 4] quaternion
    Returns: [bs, 3, 3] rotation matrix
    """
    assert isinstance(q, torch.Tensor)
    assert q.shape[-1] == 4
    if q.dim() == 1:
        q = q.unsqueeze(0)  # 1, 4
    elif q.dim() == 2:
        pass  # bs, 4
    else:
        raise NotImplementedError

    batch_size, _ = q.shape
    q = F.normalize(q, dim=1, eps=1e-6)
    R = torch.ones((batch_size, 3, 3), device=q.device)

    qr = q[:, 0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj * qi - qk * qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1 - 2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2 * (qj * qk - qi * qr)
    R[:, 2, 0] = 2 * (qk * qi - qj * qr)
    R[:, 2, 1] = 2 * (qj * qk + qi * qr)
    R[:, 2, 2] = 1 - 2 * (qi**2 + qj**2)

    return R


def extract_plane_params_from_pred_prim(pred_prim: dict) -> dict:
    """
    Extract clean plane parameters from model's get_primitives output.
    Squeezes batch dimension and computes metadata for per-frame reshaping.

    Args:
        pred_prim: dict from get_primitives with keys like 'pred_plane_center_camera' etc.
            Each value has shape [1, T*H*W, ...] (batch dim = 1 from packed sequence).

    Returns:
        dict with flat plane params and metadata:
            'plane_center_camera': [T*HW, 3]
            'plane_normal_camera': [T*HW, 3]
            'plane_depth':         [T*HW, 1]
            'plane_radii':         [T*HW, 4]
            'plane_rot_q':         [T*HW, 4]
            'intrinsics_patches':  [T, 3, 3]
            'num_frames':          int (T)
            'planes_per_frame':    int (HW)
            'h_patch':             int (H)
            'w_patch':             int (W)
    """
    center = pred_prim['pred_plane_center_camera'].squeeze(0)   # [T*HW, 3]
    normal = pred_prim['pred_plane_normal_camera'].squeeze(0)   # [T*HW, 3]
    depth  = pred_prim['pred_depth_patches'].squeeze(0)         # [T*HW, 1]
    radii  = pred_prim['pred_plane_radii_padding'].squeeze(0)   # [T*HW, 4]
    rot_q  = pred_prim['pred_plane_rot_q_normed'].squeeze(0)    # [T*HW, 4]
    intr   = pred_prim['intrinsics_patches']                    # [T, 3, 3]

    # Extract spatial dims from boundary_mask [B, T, H, W]
    bmask  = pred_prim['boundary_mask'].squeeze(0)              # [T, H, W]
    num_frames = bmask.shape[0]
    h_patch    = bmask.shape[1]
    w_patch    = bmask.shape[2]
    planes_per_frame = h_patch * w_patch

    return {
        'plane_center_camera': center,
        'plane_normal_camera': normal,
        'plane_depth': depth,
        'plane_radii': radii,
        'plane_rot_q': rot_q,
        'intrinsics_patches': intr,
        'num_frames': num_frames,
        'planes_per_frame': planes_per_frame,
        'h_patch': h_patch,
        'w_patch': w_patch,
    }


def transform_planes_to_world(
    plane_params: dict,
    extrinsics_c2w: torch.Tensor,
) -> dict:
    """
    Transform per-frame camera-space planes to world space using extrinsics,
    then concatenate all frames together.

    Args:
        plane_params: dict from extract_plane_params_from_pred_prim with keys:
            'plane_center_camera': [T*HW, 3]
            'plane_normal_camera': [T*HW, 3]
            'num_frames': int (T)
            'planes_per_frame': int (HW)
            ... (other keys are passed through unchanged)
        extrinsics_c2w: [T, 4, 4] per-frame camera-to-world transforms.
            Convention: p_world = R_c2w @ p_camera + t_c2w

    Returns:
        dict with all original keys plus:
            'plane_center_world': [T*HW, 3]
            'plane_normal_world': [T*HW, 3]
            'frame_indices':      [T*HW] (long) — which frame each plane belongs to
    """
    T = plane_params['num_frames']
    HW = plane_params['planes_per_frame']

    assert extrinsics_c2w.shape[0] == T, (
        f"extrinsics_c2w has {extrinsics_c2w.shape[0]} frames but plane_params has {T}"
    )

    centers_cam = plane_params['plane_center_camera'].reshape(T, HW, 3)
    normals_cam = plane_params['plane_normal_camera'].reshape(T, HW, 3)
    rot_q_cam = plane_params['plane_rot_q'].reshape(T, HW, 4)

    R_all = extrinsics_c2w[:, :3, :3]  # [T, 3, 3]
    t_all = extrinsics_c2w[:, :3, 3]   # [T, 3]

    # Batched transform: centers_world = centers_cam @ R.T + t
    centers_world = torch.bmm(centers_cam, R_all.transpose(1, 2)) + t_all.unsqueeze(1)
    normals_world = torch.bmm(normals_cam, R_all.transpose(1, 2))
    normals_world = F.normalize(normals_world, dim=-1)

    # Transform quaternions to world: q_world = q_c2w * q_cam (so R_world = R_c2w @ R_cam)
    q_c2w = rot_to_quat(R_all)  # [T, 4]
    rot_q_world = quat_mult(q_c2w.unsqueeze(1), rot_q_cam)  # [T, HW, 4]
    rot_q_world = rot_q_world.reshape(T * HW, 4)

    frame_indices = torch.arange(T, device=centers_cam.device).unsqueeze(1).expand(T, HW).reshape(-1)

    result = dict(plane_params)
    result['plane_center_world'] = centers_world.reshape(T * HW, 3)
    result['plane_normal_world'] = normals_world.reshape(T * HW, 3)
    result['plane_rot_q'] = rot_q_world
    result['frame_indices'] = frame_indices
    return result


def planes_to_mesh(
    plane_params: dict,
    extrinsics_c2w: torch.Tensor = None,
) -> tuple:
    """
    Convert plane parameters into a triangle mesh (each plane → 1 quad → 2 triangles).

    The quad corners are computed in camera space using the quaternion-derived local
    frame, then optionally transformed to world space per-frame.

    Args:
        plane_params: dict from extract_plane_params_from_pred_prim, must contain:
            'plane_center_camera': [N, 3]
            'plane_normal_camera': [N, 3]
            'plane_radii':         [N, 4]  (only first 2 values used as half-extents)
            'plane_rot_q':         [N, 4]
            'num_frames':          int
            'planes_per_frame':    int
        extrinsics_c2w: [T, 4, 4] optional. If given, output mesh is in world space.

    Returns:
        (vertices, faces, vertex_colors, vertex_normals):
            vertices:       [N*4, 3] float32
            faces:          [N*2, 3] int64
            vertex_colors:  [N*4, 3] uint8  (normal-mapped RGB)
            vertex_normals: [N*4, 3] float32
    """
    centers = plane_params['plane_center_camera'].float()   # [N, 3]
    normals = plane_params['plane_normal_camera'].float()   # [N, 3]
    radii   = plane_params['plane_radii'][:, :2].float()    # [N, 2]
    rot_q   = plane_params['plane_rot_q'].float()           # [N, 4]
    T       = plane_params['num_frames']
    HW      = plane_params['planes_per_frame']
    N       = centers.shape[0]

    R = quat_to_rot(rot_q)              # [N, 3, 3]
    local_x = R[:, :, 0]               # [N, 3]
    local_y = R[:, :, 1]               # [N, 3]

    r1 = radii[:, 0:1]                 # [N, 1]
    r2 = radii[:, 1:2]                 # [N, 1]
    dx = local_x * r1                  # [N, 3]
    dy = local_y * r2                  # [N, 3]

    # Quad corners in camera space: [N, 4, 3]
    verts = torch.stack([
        centers + dx + dy,
        centers - dx + dy,
        centers - dx - dy,
        centers + dx - dy,
    ], dim=1)

    # Transform to world space per-frame
    if extrinsics_c2w is not None:
        assert extrinsics_c2w.shape[0] == T
        verts = verts.reshape(T, HW, 4, 3)
        normals = normals.reshape(T, HW, 3)

        R_c2w = extrinsics_c2w[:, :3, :3].float()  # [T, 3, 3]
        t_c2w = extrinsics_c2w[:, :3, 3].float()   # [T, 3]

        verts_out = []
        normals_out = []
        for t in range(T):
            v = verts[t].reshape(-1, 3) @ R_c2w[t].T + t_c2w[t]   # [HW*4, 3]
            n = normals[t] @ R_c2w[t].T                            # [HW, 3]
            verts_out.append(v.reshape(HW, 4, 3))
            normals_out.append(F.normalize(n, dim=-1))

        verts   = torch.cat(verts_out, dim=0)     # [N, 4, 3]
        normals = torch.cat(normals_out, dim=0)    # [N, 3]

    vertices = verts.reshape(-1, 3)                # [N*4, 3]

    # Faces: two triangles per quad
    offsets = torch.arange(N, device=centers.device, dtype=torch.long) * 4
    f0 = torch.stack([offsets, offsets + 1, offsets + 2], dim=1)
    f1 = torch.stack([offsets, offsets + 2, offsets + 3], dim=1)
    faces = torch.cat([f0, f1], dim=0)             # [2N, 3]

    # Per-vertex normal (shared within quad)
    vertex_normals = normals.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3)

    # Vertex colors: normal-mapped RGB
    vertex_colors = ((normals + 1.0) * 0.5 * 255).clamp(0, 255).byte()
    vertex_colors = vertex_colors.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3)

    return vertices, faces, vertex_colors, vertex_normals


def save_planes_as_ply(
    filepath: str,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_colors: torch.Tensor = None,
    vertex_normals: torch.Tensor = None,
):
    """
    Save mesh to binary little-endian PLY file.

    Args:
        filepath: output path (should end with .ply)
        vertices:      [V, 3] float
        faces:         [F, 3] int
        vertex_colors: [V, 3] uint8, optional
        vertex_normals:[V, 3] float, optional
    """
    V = vertices.shape[0]
    F_count = faces.shape[0]
    verts_np = vertices.cpu().float().numpy()
    faces_np = faces.cpu().numpy().astype(np.int32)

    # --- header ---
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {V}",
        "property float x", "property float y", "property float z",
    ]
    if vertex_normals is not None:
        lines += ["property float nx", "property float ny", "property float nz"]
    if vertex_colors is not None:
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
    lines += [
        f"element face {F_count}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    header = "\n".join(lines) + "\n"

    # --- build vertex blob via structured array ---
    fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
    if vertex_normals is not None:
        fields += [('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')]
    if vertex_colors is not None:
        fields += [('r', 'u1'), ('g', 'u1'), ('b', 'u1')]

    vert_arr = np.empty(V, dtype=np.dtype(fields))
    vert_arr['x'] = verts_np[:, 0]
    vert_arr['y'] = verts_np[:, 1]
    vert_arr['z'] = verts_np[:, 2]
    if vertex_normals is not None:
        normals_np = vertex_normals.cpu().float().numpy()
        vert_arr['nx'] = normals_np[:, 0]
        vert_arr['ny'] = normals_np[:, 1]
        vert_arr['nz'] = normals_np[:, 2]
    if vertex_colors is not None:
        colors_np = vertex_colors.cpu().numpy().astype(np.uint8)
        vert_arr['r'] = colors_np[:, 0]
        vert_arr['g'] = colors_np[:, 1]
        vert_arr['b'] = colors_np[:, 2]

    # --- build face blob ---
    face_dtype = np.dtype([('count', 'u1'), ('v0', '<i4'), ('v1', '<i4'), ('v2', '<i4')])
    face_arr = np.empty(F_count, dtype=face_dtype)
    face_arr['count'] = 3
    face_arr['v0'] = faces_np[:, 0]
    face_arr['v1'] = faces_np[:, 1]
    face_arr['v2'] = faces_np[:, 2]

    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(vert_arr.tobytes())
        f.write(face_arr.tobytes())


# Depth / normal map visualization

_TURBO_KEYS = np.array([
    [0.18995, 0.07176, 0.23217],
    [0.19483, 0.26149, 0.57569],
    [0.09140, 0.58840, 0.76200],
    [0.19659, 0.80780, 0.59240],
    [0.39024, 0.90091, 0.29988],
    [0.66936, 0.93868, 0.16663],
    [0.97176, 0.78170, 0.07763],
    [0.94277, 0.47587, 0.08384],
    [0.79573, 0.17700, 0.10167],
    [0.70567, 0.01555, 0.15023],
], dtype=np.float64)
_TURBO_POS = np.linspace(0.0, 1.0, len(_TURBO_KEYS))


def _apply_turbo(values: np.ndarray) -> np.ndarray:
    """Map [H, W] float in [0,1] to [H, W, 3] uint8 via turbo-like colormap."""
    flat = values.ravel()
    r = np.interp(flat, _TURBO_POS, _TURBO_KEYS[:, 0])
    g = np.interp(flat, _TURBO_POS, _TURBO_KEYS[:, 1])
    b = np.interp(flat, _TURBO_POS, _TURBO_KEYS[:, 2])
    rgb = np.stack([r, g, b], axis=-1).reshape(*values.shape, 3)
    return (rgb * 255).clip(0, 255).astype(np.uint8)


def save_depth_maps(
    out_dir: str,
    plane_params: dict,
    prefix: str = "depth",
    global_normalize: bool = True,
):
    """
    Save per-frame depth maps as colored PNG images.

    Args:
        out_dir: directory (created if missing).
        plane_params: dict with 'plane_depth' [T*HW, 1], 'h_patch', 'w_patch', 'num_frames'.
        prefix: filename prefix.
        global_normalize: if True, normalize across all frames for consistent scale.
    """
    os.makedirs(out_dir, exist_ok=True)
    T  = plane_params['num_frames']
    H  = plane_params['h_patch']
    W  = plane_params['w_patch']
    depth_all = plane_params['plane_depth'].squeeze(-1).float().cpu().numpy()  # [T*HW]
    depth_all = depth_all.reshape(T, H, W)

    if global_normalize:
        vmin = np.nanmin(depth_all)
        vmax = np.nanmax(depth_all)
    else:
        vmin = vmax = None

    for t in range(T):
        d = depth_all[t]
        if not global_normalize:
            vmin = np.nanmin(d)
            vmax = np.nanmax(d)
        norm = (d - vmin) / (vmax - vmin + 1e-8)
        norm = np.clip(norm, 0.0, 1.0)
        rgb = _apply_turbo(norm)
        Image.fromarray(rgb).save(os.path.join(out_dir, f"{prefix}_{t:03d}.png"))


def save_normal_maps(
    out_dir: str,
    plane_params: dict,
    prefix: str = "normal",
):
    """
    Save per-frame normal maps as RGB PNG images (normal-mapped: (n+1)/2*255).

    Args:
        out_dir: directory (created if missing).
        plane_params: dict with 'plane_normal_camera' [T*HW, 3], 'h_patch', 'w_patch', 'num_frames'.
        prefix: filename prefix.
    """
    os.makedirs(out_dir, exist_ok=True)
    T  = plane_params['num_frames']
    H  = plane_params['h_patch']
    W  = plane_params['w_patch']
    normals = plane_params['plane_normal_camera'].float().cpu().numpy()  # [T*HW, 3]
    normals = normals.reshape(T, H, W, 3)

    for t in range(T):
        rgb = ((normals[t] + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(os.path.join(out_dir, f"{prefix}_{t:03d}.png"))
